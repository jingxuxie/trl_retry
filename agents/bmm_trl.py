import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    GCActor,
    GCDiscreteActor,
    GCDiscreteCritic,
    GCValue,
)


BMM_DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def config_budgets(config):
    """Return sorted BMM budgets from a config-like object."""
    budgets = tuple(int(x) for x in config.get("budgets", BMM_DEFAULT_BUDGETS))
    max_budget = int(config.get("max_budget", max(budgets)))
    budgets = tuple(sorted(set(x for x in budgets if 1 <= x <= max_budget)))
    if len(budgets) == 0 or budgets[0] != 1:
        budgets = tuple(sorted(set((1, *budgets))))
    if budgets[-1] != max_budget:
        budgets = tuple(sorted(set((*budgets, max_budget))))
    return budgets


def normalize_budget(budget, max_budget):
    """Map positive step budgets to a scalar log feature in [0, 1]."""
    budget = jnp.asarray(budget, dtype=jnp.float32)
    budget = jnp.maximum(budget, 1.0)
    denom = jnp.maximum(jnp.log2(float(max_budget)), 1.0)
    return jnp.log2(budget) / denom


def _broadcast_budget_like_goal(value, goal):
    if value.ndim == len(goal.shape[:-1]) + 1 and value.shape[-1] == 1:
        value = jnp.squeeze(value, axis=-1)
    if value.ndim == 0:
        value = jnp.full(goal.shape[:-1], value, dtype=goal.dtype)
    else:
        value = jnp.broadcast_to(value, goal.shape[:-1]).astype(goal.dtype)
    return value


def augment_goal_with_budget(
    goal,
    budget,
    max_budget,
    budgets=None,
    budget_feature="log_scalar",
    offset=None,
    oracle_offset_feature=False,
):
    """Append budget features to vector goals."""
    if isinstance(goal, (dict, flax.core.FrozenDict)):
        raise ValueError("BMM-TRL first pass supports vector goals only, not dict goals.")

    goal = jnp.asarray(goal)
    if goal.ndim not in (2, 3):
        raise ValueError(
            "BMM-TRL first pass supports vector goals with shape [B, G] or "
            f"[N, B, G], got shape {goal.shape}."
        )

    b = normalize_budget(budget, max_budget)
    b = _broadcast_budget_like_goal(b, goal)
    features = [goal, b[..., None]]

    if oracle_offset_feature:
        if offset is None:
            offset_feature = jnp.zeros(goal.shape[:-1], dtype=goal.dtype)
        else:
            offset_feature = normalize_budget(offset, max_budget)
            offset_feature = _broadcast_budget_like_goal(offset_feature, goal)
        features.append(offset_feature[..., None])

    if budget_feature == "log_scalar_onehot":
        if budgets is None:
            budgets = BMM_DEFAULT_BUDGETS
        budgets_arr = jnp.asarray(tuple(int(x) for x in budgets), dtype=jnp.float32)
        budget_idxs = jnp.searchsorted(
            budgets_arr, jnp.asarray(budget, dtype=jnp.float32), side="left"
        )
        budget_idxs = jnp.clip(budget_idxs, 0, len(tuple(budgets)) - 1)
        budget_idxs = _broadcast_budget_like_goal(budget_idxs, goal).astype(jnp.int32)
        features.append(
            jax.nn.one_hot(budget_idxs, len(tuple(budgets)), dtype=goal.dtype)
        )
    elif budget_feature != "log_scalar":
        raise ValueError(f"Unsupported BMM budget_feature: {budget_feature}")

    return jnp.concatenate(features, axis=-1)


def actor_budget(goals, max_budget):
    """Return a max-budget array matching the leading dimensions of goals."""
    return jnp.full(goals.shape[:-1], max_budget, dtype=jnp.float32)


def masked_mean(x, mask, eps=1e-8):
    """Mean of x under a mask that broadcasts over leading ensemble axes."""
    mask = jnp.asarray(mask, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask[None, ...]
    mask = jnp.broadcast_to(mask, x.shape)
    return (x * mask).sum() / (mask.sum() + eps)


class BMMTRLAgent(flax.struct.PyTreeNode):
    """Budgeted Max-Min TRL prototype agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def maybe_actions(self, actions):
        if self.config.get("diagnostic_critic_mode", "action") == "state":
            return None
        return actions

    def augment_goal(self, goal, budget, offset=None):
        return augment_goal_with_budget(
            goal,
            budget,
            self.config["max_budget"],
            budgets=config_budgets(self.config),
            budget_feature=self.config["budget_feature"],
            offset=offset,
            oracle_offset_feature=self.config.get("oracle_offset_feature", False),
        )

    def augment_goal_for_critic(self, observations, goals, budgets, offsets=None):
        aug_goals = self.augment_goal(goals, budgets, offsets)
        if self.config.get("critic_absdiff_goal_feature", False):
            if observations.shape[-1] != goals.shape[-1]:
                raise ValueError(
                    "critic_absdiff_goal_feature requires matching observation "
                    f"and goal dims, got {observations.shape[-1]} and {goals.shape[-1]}."
                )
            absdiff = jnp.abs(jnp.asarray(observations) - jnp.asarray(goals))
            aug_goals = jnp.concatenate([aug_goals, absdiff], axis=-1)
        return aug_goals

    def critic_logits_for(
        self,
        observations,
        actions,
        goals,
        budgets,
        offsets=None,
        grad_params=None,
        target=False,
    ):
        module = "target_critic" if target else "critic"
        aug_goals = self.augment_goal_for_critic(
            observations, goals, budgets, offsets
        )
        maybe_actions = self.maybe_actions(actions)
        if grad_params is None or target:
            return self.network.select(module)(
                observations,
                goals=aug_goals,
                actions=maybe_actions,
            )
        return self.network.select(module)(
            observations,
            goals=aug_goals,
            actions=maybe_actions,
            params=grad_params,
        )

    def critic_logits_for_pair_grid(
        self,
        observations,
        actions,
        goals,
        budgets,
        offsets=None,
        grad_params=None,
        target=False,
    ):
        budgets = jnp.asarray(budgets)
        leading_shape = budgets.shape
        leading_ndim = budgets.ndim
        flat_observations = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[leading_ndim:]), observations
        )
        flat_actions = jnp.reshape(actions, (-1,) + actions.shape[leading_ndim:])
        flat_goals = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[leading_ndim:]), goals
        )
        flat_budgets = jnp.reshape(budgets, (-1,))
        flat_offsets = None if offsets is None else jnp.reshape(offsets, (-1,))
        flat_logits = self.critic_logits_for(
            flat_observations,
            flat_actions,
            flat_goals,
            flat_budgets,
            offsets=flat_offsets,
            grad_params=grad_params,
            target=target,
        )
        return jnp.reshape(flat_logits, flat_logits.shape[:1] + leading_shape)

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def critic_loss(self, batch, grad_params):
        if self.config["oracle_distill"]:
            raise ValueError("BMM-TRL prototype does not support oracle_distill=True.")

        goal_key = "value_goals"
        r_logits = self.critic_logits_for(
            batch["observations"],
            batch["actions"],
            batch[goal_key],
            batch["value_budgets"],
            offsets=batch.get("value_offsets"),
            grad_params=grad_params,
        )
        r = jax.nn.sigmoid(r_logits)

        compute_trans = float(self.config.get("lambda_trans", 0.0)) != 0.0
        compute_pos = float(self.config.get("lambda_pos", 0.0)) != 0.0
        compute_budget_neg = float(self.config.get("lambda_budget_neg", 0.0)) != 0.0
        compute_hard_neg = float(self.config.get("lambda_hard_neg", 0.0)) != 0.0
        compute_rand_hinge = float(self.config.get("lambda_rand_hinge", 0.0)) != 0.0
        compute_mono = float(self.config.get("lambda_mono", 0.0)) != 0.0
        compute_sup = float(self.config.get("lambda_sup", 0.0)) != 0.0
        compute_rank = float(self.config.get("lambda_rank", 0.0)) != 0.0
        zero_scalar = jnp.asarray(0.0, dtype=r_logits.dtype)
        zero_r = jnp.zeros_like(r)

        if compute_trans:
            if batch["value_midpoint_goals"].ndim == batch[goal_key].ndim + 1:
                witness_goals = batch["value_midpoint_goals"]
                witness_shape = witness_goals.shape[:-1]
                parent_observations = jnp.broadcast_to(
                    batch["observations"][None, ...],
                    witness_shape + batch["observations"].shape[-1:],
                )
                parent_actions = jnp.broadcast_to(
                    batch["actions"][None, ...],
                    witness_shape + batch["actions"].shape[-1:],
                )
                parent_goals = jnp.broadcast_to(
                    batch[goal_key][None, ...],
                    witness_shape + batch[goal_key].shape[-1:],
                )
                first_logits = self.critic_logits_for_pair_grid(
                    parent_observations,
                    parent_actions,
                    witness_goals,
                    batch["value_left_budgets"],
                    offsets=batch.get(
                        "value_midpoint_offsets", batch["value_left_budgets"]
                    ),
                    target=True,
                )
                second_logits = self.critic_logits_for_pair_grid(
                    batch["value_midpoint_observations"],
                    batch["value_midpoint_actions"],
                    parent_goals,
                    batch["value_right_budgets"],
                    offsets=batch["value_right_budgets"],
                    target=True,
                )
                first_r = jax.nn.sigmoid(first_logits)
                second_r = jax.nn.sigmoid(second_logits)
                witness_valids = jnp.asarray(
                    batch["trans_valids"], dtype=first_r.dtype
                )
                y_candidates = jnp.minimum(first_r, second_r)
                y_candidates = jnp.where(
                    witness_valids[None, ...] > 0, y_candidates, -1.0
                )
                y_trans = jax.lax.stop_gradient(jnp.max(y_candidates, axis=1))
                trans_valids = (witness_valids.max(axis=0) > 0).astype(
                    r_logits.dtype
                )
            else:
                first_logits = self.critic_logits_for(
                    batch["observations"],
                    batch["actions"],
                    batch["value_midpoint_goals"],
                    batch["value_left_budgets"],
                    offsets=batch.get(
                        "value_midpoint_offsets", batch["value_left_budgets"]
                    ),
                    target=True,
                )
                first_r = jax.nn.sigmoid(first_logits)

                second_logits = self.critic_logits_for(
                    batch["value_midpoint_observations"],
                    batch["value_midpoint_actions"],
                    batch[goal_key],
                    batch["value_right_budgets"],
                    offsets=batch["value_right_budgets"],
                    target=True,
                )
                second_r = jax.nn.sigmoid(second_logits)

                witness_valids = jnp.asarray(batch["trans_valids"], dtype=first_r.dtype)
                y_trans = jax.lax.stop_gradient(jnp.minimum(first_r, second_r))
                trans_valids = witness_valids

            target_mask = trans_valids
            while target_mask.ndim < y_trans.ndim:
                target_mask = target_mask[None, ...]
            y_trans = jnp.clip(y_trans, 0.0, 1.0)
            y_trans = jnp.where(target_mask > 0, y_trans, 0.0)
            loss_trans = masked_mean(self.bce_loss(r_logits, y_trans), trans_valids)
        else:
            first_r = zero_r
            second_r = zero_r
            witness_valids = jnp.zeros_like(batch["value_budgets"], dtype=r_logits.dtype)
            trans_valids = witness_valids
            y_trans = zero_r
            loss_trans = zero_scalar
        loss_pos = (
            self.bce_loss(r_logits, jnp.ones_like(r_logits)).mean()
            if compute_pos
            else zero_scalar
        )

        if compute_budget_neg:
            neg_logits = self.critic_logits_for(
                batch["observations"],
                batch["actions"],
                batch[goal_key],
                batch["value_neg_budgets"],
                offsets=batch.get("value_offsets"),
                grad_params=grad_params,
            )
            neg_r = jax.nn.sigmoid(neg_logits)
            loss_budget_neg = masked_mean(
                self.bce_loss(neg_logits, jnp.zeros_like(neg_logits)),
                batch["value_neg_valids"],
            )
        else:
            neg_r = zero_r
            loss_budget_neg = zero_scalar

        if compute_hard_neg:
            hard_neg_logits = self.critic_logits_for(
                batch["observations"],
                batch["actions"],
                batch["value_hard_neg_goals"],
                batch["value_hard_neg_budgets"],
                offsets=batch.get("value_hard_neg_offsets"),
                grad_params=grad_params,
            )
            hard_neg_r = jax.nn.sigmoid(hard_neg_logits)
            loss_hard_neg = masked_mean(
                self.bce_loss(hard_neg_logits, jnp.zeros_like(hard_neg_logits)),
                batch["value_hard_neg_valids"],
            )
        else:
            hard_neg_r = zero_r
            loss_hard_neg = zero_scalar

        if compute_rand_hinge:
            rand_logits = self.critic_logits_for(
                batch["observations"],
                batch["actions"],
                batch["value_random_goals"],
                batch["value_random_budgets"],
                grad_params=grad_params,
            )
            rand_r = jax.nn.sigmoid(rand_logits)
            loss_rand_hinge = jnp.mean(
                jnp.maximum(rand_r - self.config["rand_hinge_rho"], 0.0) ** 2
            )
        else:
            rand_r = zero_r
            loss_rand_hinge = zero_scalar

        if compute_mono:
            low_logits = self.critic_logits_for(
                batch["observations"],
                batch["actions"],
                batch[goal_key],
                batch["mono_low_budgets"],
                offsets=batch.get("value_offsets"),
                grad_params=grad_params,
            )
            high_logits = self.critic_logits_for(
                batch["observations"],
                batch["actions"],
                batch[goal_key],
                batch["mono_high_budgets"],
                offsets=batch.get("value_offsets"),
                grad_params=grad_params,
            )
            low_r = jax.nn.sigmoid(low_logits)
            high_r = jax.nn.sigmoid(high_logits)
            loss_mono = jnp.mean(jnp.maximum(low_r - high_r, 0.0) ** 2)
        else:
            low_r = zero_r
            high_r = zero_r
            loss_mono = zero_scalar

        if compute_sup and "value_sup_goals" in batch:
            sup_logits = self.critic_logits_for_pair_grid(
                batch["value_sup_observations"],
                batch["value_sup_actions"],
                batch["value_sup_goals"],
                batch["value_sup_budgets"],
                offsets=batch["value_sup_offsets"],
                grad_params=grad_params,
            )
            sup_labels = jnp.asarray(batch["value_sup_labels"], dtype=sup_logits.dtype)
            sup_valids = jnp.asarray(batch["value_sup_valids"], dtype=sup_logits.dtype)
            sup_r = jax.nn.sigmoid(sup_logits)
            loss_sup = masked_mean(self.bce_loss(sup_logits, sup_labels), sup_valids)
        else:
            sup_logits = jnp.zeros((1, 1, 1), dtype=r_logits.dtype)
            sup_labels = jnp.zeros((1, 1), dtype=r_logits.dtype)
            sup_valids = jnp.zeros((1, 1), dtype=r_logits.dtype)
            sup_r = jnp.zeros_like(sup_logits)
            loss_sup = jnp.asarray(0.0, dtype=r_logits.dtype)

        if compute_rank and "value_rank_goals" in batch:
            rank_pos_logits = self.critic_logits_for_pair_grid(
                batch["value_rank_observations"],
                batch["value_rank_actions"],
                batch["value_rank_goals"],
                batch["value_rank_pos_budgets"],
                offsets=batch["value_rank_offsets"],
                grad_params=grad_params,
            )
            rank_neg_logits = self.critic_logits_for_pair_grid(
                batch["value_rank_observations"],
                batch["value_rank_actions"],
                batch["value_rank_goals"],
                batch["value_rank_neg_budgets"],
                offsets=batch["value_rank_offsets"],
                grad_params=grad_params,
            )
            rank_valids = jnp.asarray(
                batch["value_rank_valids"], dtype=rank_pos_logits.dtype
            )
            rank_gap = rank_pos_logits - rank_neg_logits
            loss_rank = masked_mean(
                jax.nn.softplus(self.config["rank_margin"] - rank_gap), rank_valids
            )
            rank_gap_mean = masked_mean(rank_gap, rank_valids)
        else:
            rank_valids = jnp.zeros((1, 1), dtype=r_logits.dtype)
            rank_gap = jnp.zeros((1, 1, 1), dtype=r_logits.dtype)
            loss_rank = jnp.asarray(0.0, dtype=r_logits.dtype)
            rank_gap_mean = jnp.asarray(0.0, dtype=r_logits.dtype)

        loss_trans_over_sup = loss_trans / jnp.maximum(loss_sup, 1e-8)

        total_loss = (
            self.config["lambda_trans"] * loss_trans
            + self.config["lambda_pos"] * loss_pos
            + self.config["lambda_budget_neg"] * loss_budget_neg
            + self.config["lambda_hard_neg"] * loss_hard_neg
            + self.config["lambda_rand_hinge"] * loss_rand_hinge
            + self.config["lambda_mono"] * loss_mono
            + self.config["lambda_sup"] * loss_sup
            + self.config["lambda_rank"] * loss_rank
        )

        hard_neg_budgets = jnp.asarray(
            batch["value_hard_neg_budgets"], dtype=jnp.float32
        )
        hard_neg_offsets = jnp.asarray(
            batch["value_hard_neg_offsets"], dtype=jnp.float32
        )
        hard_neg_valids = batch["value_hard_neg_valids"]
        hard_neg_offset_over_budget = hard_neg_offsets / jnp.maximum(
            hard_neg_budgets, 1.0
        )

        info = {
            "total_loss": total_loss,
            "loss_trans": loss_trans,
            "loss_pos": loss_pos,
            "loss_budget_neg": loss_budget_neg,
            "loss_hard_neg": loss_hard_neg,
            "loss_rand_hinge": loss_rand_hinge,
            "loss_mono": loss_mono,
            "loss_sup": loss_sup,
            "loss_rank": loss_rank,
            "loss_trans_over_sup": loss_trans_over_sup,
            "r_mean": r.mean(),
            "r_min": r.min(),
            "r_max": r.max(),
            "y_trans_mean": masked_mean(y_trans, trans_valids),
            "first_r_mean": first_r.mean(),
            "second_r_mean": second_r.mean(),
            "pos_r_mean": r.mean(),
            "neg_r_mean": neg_r.mean(),
            "hard_neg_r_mean": masked_mean(hard_neg_r, hard_neg_valids),
            "rand_r_mean": rand_r.mean(),
            "mono_violation": (low_r > high_r).mean(),
            "trans_valid_frac": trans_valids.mean(),
            "neg_valid_frac": batch["value_neg_valids"].mean(),
            "hard_neg_valid_frac": hard_neg_valids.mean(),
            "hard_neg_budget_mean": masked_mean(hard_neg_budgets, hard_neg_valids),
            "hard_neg_offset_mean": masked_mean(hard_neg_offsets, hard_neg_valids),
            "hard_neg_offset_over_budget_mean": masked_mean(
                hard_neg_offset_over_budget, hard_neg_valids
            ),
            "sup_r_mean": masked_mean(sup_r, sup_valids),
            "sup_pos_r_mean": masked_mean(sup_r, sup_valids * sup_labels),
            "sup_neg_r_mean": masked_mean(sup_r, sup_valids * (1.0 - sup_labels)),
            "sup_valid_frac": sup_valids.mean(),
            "sup_pos_frac": masked_mean(sup_labels, sup_valids),
            "rank_gap_mean": rank_gap_mean,
            "rank_valid_frac": rank_valids.mean(),
        }

        if compute_sup and "value_sup_goals" in batch:
            for budget in config_budgets(self.config):
                budget_mask = (
                    jnp.asarray(batch["value_sup_budgets"]) == int(budget)
                ).astype(sup_valids.dtype)
                pos_mask = sup_valids * budget_mask * sup_labels
                neg_mask = sup_valids * budget_mask * (1.0 - sup_labels)
                info[f"sup_pos_count_H={int(budget)}"] = pos_mask.sum()
                info[f"sup_neg_count_H={int(budget)}"] = neg_mask.sum()
                info[f"sup_pos_r_mean_H={int(budget)}"] = masked_mean(sup_r, pos_mask)
                info[f"sup_neg_r_mean_H={int(budget)}"] = masked_mean(sup_r, neg_mask)
                info[f"sup_valid_frac_H={int(budget)}"] = (
                    sup_valids * budget_mask
                ).mean()
                info[f"loss_sup_H={int(budget)}"] = masked_mean(
                    self.bce_loss(sup_logits, sup_labels), sup_valids * budget_mask
                )

        for budget in config_budgets(self.config):
            budget_mask = (
                jnp.asarray(batch["value_budgets"]) == int(budget)
            ).astype(r_logits.dtype)
            trans_mask = trans_valids * budget_mask
            if witness_valids.ndim == 2:
                witness_budget_mask = (
                    jnp.asarray(batch["value_budgets"])[None, :] == int(budget)
                ).astype(first_r.dtype)
                branch_mask = witness_valids * witness_budget_mask
            else:
                branch_mask = trans_mask
            loss_trans_h = masked_mean(
                self.bce_loss(r_logits, y_trans), trans_mask
            )
            info[f"y_trans_mean_H={int(budget)}"] = masked_mean(y_trans, trans_mask)
            info[f"first_r_mean_H={int(budget)}"] = masked_mean(first_r, branch_mask)
            info[f"second_r_mean_H={int(budget)}"] = masked_mean(second_r, branch_mask)
            info[f"parent_r_mean_H={int(budget)}"] = masked_mean(r, trans_mask)
            info[f"loss_trans_H={int(budget)}"] = loss_trans_h
            if f"loss_sup_H={int(budget)}" in info:
                info[f"loss_trans_over_sup_H={int(budget)}"] = loss_trans_h / jnp.maximum(
                    info[f"loss_sup_H={int(budget)}"], 1e-8
                )

        if "trans_parent_oracle_labels" in batch:
            info["trans_parent_oracle_label_mean"] = masked_mean(
                jnp.asarray(batch["trans_parent_oracle_labels"], dtype=r_logits.dtype),
                trans_valids,
            )
        if "trans_branch_oracle_valids" in batch:
            branch_oracle_valids = jnp.asarray(
                batch["trans_branch_oracle_valids"], dtype=first_r.dtype
            )
            info["trans_branch_oracle_valid_mean"] = masked_mean(
                branch_oracle_valids,
                witness_valids if witness_valids.ndim == branch_oracle_valids.ndim else trans_valids,
            )

        return total_loss, info

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "rpg":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            if pe_info["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            aug_actor_goals = self.augment_goal(
                batch["actor_goals"],
                actor_budget(batch["actor_goals"], self.config["max_budget"]),
            )
            q1, q2 = self.network.select("critic")(
                batch["observations"], aug_actor_goals, q_actions
            )
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(pe_info["alpha"] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        elif self.config["pe_type"] == "discrete":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )

            n_actions = jnp.repeat(
                jnp.expand_dims(jnp.arange(0, pe_info["action_ct"]), 1),
                self.config["batch_size"],
                axis=1,
            )
            n_obs = jnp.repeat(
                jnp.expand_dims(batch["observations"], 0), pe_info["action_ct"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(batch["actor_goals"], 0), pe_info["action_ct"], axis=0
            )
            aug_n_goals = self.augment_goal(
                n_goals,
                actor_budget(n_goals, self.config["max_budget"]),
            )

            q = self.network.select("critic")(n_obs, aug_n_goals, n_actions).mean(axis=0)
            v = jnp.sum(q * dist.probs.T, axis=0)
            q_loss = -v.mean()

            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(pe_info["alpha"] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
            }

        elif self.config["pe_type"] == "frs":
            batch_size, action_dim = batch["actions"].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch["actions"]
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], x_t, t, params=grad_params
            )
            actor_loss = jnp.mean((pred - y) ** 2)

            return actor_loss, {
                "actor_loss": actor_loss,
            }

        raise ValueError(f"Unsupported pe_type: {self.config['pe_type']}")

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        if self.config.get("value_only", False):
            return critic_loss, info

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "frs":
            if goals is None:
                raise ValueError("BMM-TRL FRS sampling requires goals.")
            n_observations = jnp.repeat(
                jnp.expand_dims(observations, 0), pe_info["num_samples"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(goals, 0), pe_info["num_samples"], axis=0
            )

            n_actions = jax.random.normal(
                seed,
                (
                    pe_info["num_samples"],
                    *observations.shape[:-1],
                    self.config["action_dim"],
                ),
            )
            for i in range(pe_info["flow_steps"]):
                t = jnp.full(
                    (pe_info["num_samples"], *observations.shape[:-1], 1),
                    i / pe_info["flow_steps"],
                )
                vels = self.network.select("actor")(
                    n_observations, n_goals, n_actions, t
                )
                n_actions = n_actions + vels / pe_info["flow_steps"]
            n_actions = jnp.clip(n_actions, -1, 1)

            if self.config.get("actor_budget_mode", "max") == "scan":
                budgets = jnp.asarray(config_budgets(self.config), dtype=jnp.float32)
                candidate_shape = n_actions.shape[:-1]
                scan_shape = (budgets.shape[0],) + candidate_shape
                scan_observations = jnp.broadcast_to(
                    n_observations[None, ...],
                    scan_shape + observations.shape[-1:],
                )
                scan_goals = jnp.broadcast_to(
                    n_goals[None, ...],
                    scan_shape + goals.shape[-1:],
                )
                scan_actions = jnp.broadcast_to(
                    n_actions[None, ...],
                    scan_shape + n_actions.shape[-1:],
                )
                scan_budgets = jnp.broadcast_to(
                    budgets.reshape((budgets.shape[0],) + (1,) * len(candidate_shape)),
                    scan_shape,
                )
                logits = self.critic_logits_for_pair_grid(
                    scan_observations,
                    scan_actions,
                    scan_goals,
                    scan_budgets,
                )
                probs = jax.nn.sigmoid(logits)
                probs = jnp.min(probs, axis=0)
                threshold = float(self.config.get("actor_budget_threshold", 0.5))
                fallback_budget = float(self.config["max_budget"]) * 2.0
                feasible_budgets = jnp.where(
                    probs >= threshold,
                    scan_budgets,
                    fallback_budget,
                )
                first_feasible_budget = feasible_budgets.min(axis=0)
                best_prob = probs.max(axis=0)
                prob_weight = float(
                    self.config.get("actor_budget_scan_prob_weight", 0.01)
                )
                q = -first_feasible_budget / float(self.config["max_budget"])
                q = q + prob_weight * best_prob
            else:
                aug_n_goals = self.augment_goal(
                    n_goals,
                    actor_budget(n_goals, self.config["max_budget"]),
                )
                q = self.network.select("critic")(
                    n_observations, goals=aug_n_goals, actions=n_actions
                )
                q = jnp.min(q, axis=0)

            if len(observations.shape) == 2:
                actions = n_actions[
                    jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])
                ]
            else:
                actions = n_actions[jnp.argmax(q)]

            return actions

        else:
            dist = self.network.select("actor")(
                observations, goals, temperature=temperature
            )
            actions = dist.sample(seed=seed)

            if self.config["pe_type"] != "discrete":
                actions = jnp.clip(actions, -1, 1)

            return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        if config["oracle_distill"]:
            raise ValueError("BMM-TRL prototype does not support oracle_distill=True.")
        if config["budget_feature"] not in ("log_scalar", "log_scalar_onehot"):
            raise ValueError(
                "BMM-TRL supports budget_feature='log_scalar' or "
                "'log_scalar_onehot'."
            )
        if config.get("diagnostic_critic_mode", "action") not in ("action", "state"):
            raise ValueError(
                "BMM-TRL supports diagnostic_critic_mode='action' or 'state'."
            )
        if (
            config.get("diagnostic_critic_mode", "action") == "state"
            and not config.get("value_only", False)
        ):
            raise ValueError(
                "diagnostic_critic_mode='state' is diagnostic-only and requires "
                "value_only=True."
            )
        if config["split_mode"] != "half":
            raise ValueError("BMM-TRL prototype only supports split_mode='half'.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["actor_goals"]
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = config[config["pe_type"]]

        critic_mode = config.get("diagnostic_critic_mode", "action")
        if config["pe_type"] == "discrete" and critic_mode == "action":
            critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
                action_dim=config["discrete"]["action_ct"],
            )
        else:
            critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
            )

        if config["pe_type"] == "frs":
            actor_def = ActorVectorField(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions, ex_times)
        elif config["pe_type"] == "discrete":
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=config["discrete"]["action_ct"],
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
                state_dependent_std=False,
                const_std=pe_info["const_std"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)

        ex_critic_goals = augment_goal_with_budget(
            ex_goals,
            actor_budget(ex_goals, config["max_budget"]),
            config["max_budget"],
            budgets=config_budgets(config),
            budget_feature=config["budget_feature"],
            oracle_offset_feature=config.get("oracle_offset_feature", False),
        )
        if config.get("critic_absdiff_goal_feature", False):
            if ex_observations.shape[-1] != ex_goals.shape[-1]:
                raise ValueError(
                    "critic_absdiff_goal_feature requires matching observation "
                    f"and goal dims, got {ex_observations.shape[-1]} and "
                    f"{ex_goals.shape[-1]}."
                )
            ex_critic_goals = jnp.concatenate(
                [ex_critic_goals, jnp.abs(ex_observations - ex_goals)], axis=-1
            )
        critic_in = (
            (ex_observations, ex_critic_goals)
            if critic_mode == "state"
            else (ex_observations, ex_critic_goals, ex_actions)
        )
        network_info = dict(
            critic=(critic_def, critic_in),
            target_critic=(copy.deepcopy(critic_def), critic_in),
            actor=(actor_def, ex_actor_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="bmm_trl",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            oracle_distill=False,
            pe_type="frs",  # frs (flow rejection sampling), rpg (reparameterized grads), discrete
            frs=mlc.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=mlc.ConfigDict(dict(alpha=0.03, const_std=True)),
            discrete=mlc.ConfigDict(dict(alpha=0.03, action_ct=0)),
            budgets=BMM_DEFAULT_BUDGETS,
            max_budget=1024,
            use_budget_goal_aug=True,
            budget_feature="log_scalar",
            split_mode="half",
            split_min_frac=0.25,
            num_split_samples=1,
            num_witnesses=1,
            lambda_trans=0.0,
            lambda_pos=0.0,
            lambda_budget_neg=0.0,
            lambda_hard_neg=0.0,
            lambda_rand_hinge=0.0,
            lambda_mono=0.05,
            lambda_sup=1.0,
            lambda_rank=0.5,
            num_sup_pairs=8,
            num_rank_pairs=8,
            rank_margin=1.0,
            budget_neg_frac=0.5,
            hard_neg_min_factor=1.25,
            hard_neg_max_factor=4.0,
            sup_pos_boundary_frac=0.5,
            sup_neg_min_factor=1.0,
            sup_neg_max_factor=2.0,
            sup_include_far_negs=True,
            rand_hinge_rho=0.3,
            actor_budget_mode="max",
            actor_budget_threshold=0.5,
            actor_budget_scan_prob_weight=0.01,
            diagnostic_critic_mode="action",
            value_only=False,
            oracle_offset_feature=False,
            critic_absdiff_goal_feature=False,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.0,
                    value_p_trajgoal=1.0,
                    value_p_randomgoal=0.0,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=0.5,
                    actor_p_randomgoal=0.5,
                    actor_geom_sample=True,
                )
            ),
        )
    )
    return config
