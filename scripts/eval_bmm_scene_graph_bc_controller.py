#!/usr/bin/env python
"""Scene-Play graph-subgoal smoke with a fixed oracle-goal BC controller."""

import argparse
import json
from pathlib import Path
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from agents.bmm_trl import get_config
from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_subgoal_bc_controller as bc_eval
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from utils.pointmaze_graph import load_graph_distance_matrix_npz, load_graph_npz


def parse_int_list(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_tuple(value):
    return bc_eval.parse_tuple(value)


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_task_set(value):
    if value is None or str(value).strip() == "":
        return set()
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def parse_task_float_map(value):
    out = {}
    if value is None or str(value).strip() == "":
        return out
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Expected task override in TASK:VALUE form, got {item!r}"
            )
        task, raw_value = item.split(":", 1)
        out[int(task.strip())] = float(raw_value.strip())
    return out


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def unwrap_env(env):
    cur = env
    for _ in range(16):
        if hasattr(cur, "compute_oracle_observation"):
            return cur
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            cur = cur.unwrapped
            continue
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        break
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "compute_oracle_observation"):
        return env.unwrapped
    raise ValueError("Environment does not expose compute_oracle_observation().")


def configure_value_agent(args, train_dataset):
    budgets = tuple(parse_int_list(args.budgets))
    config = get_config()
    config.budgets = budgets
    config.max_budget = max(budgets)
    config.batch_size = 1
    config.diagnostic_critic_mode = "state"
    config.value_only = True
    config.budget_feature = str(args.budget_feature)
    config.actor_hidden_dims = parse_tuple(args.actor_hidden_dims)
    config.value_hidden_dims = parse_tuple(args.value_hidden_dims)
    config.layer_norm = parse_bool(args.layer_norm)
    config.critic_absdiff_goal_feature = bool(args.critic_absdiff_goal_feature)
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_sup_pairs = 0
    config.num_rank_pairs = 0
    config.dataset.reachability_label_type = "graph"
    config.dataset.graph_path = str(args.graph_path)
    gc_train = GCDataset(Dataset.create(**train_dataset), config)
    example_idxs = gc_train.dataset.get_random_idxs(1)
    example_batch = gc_train.sample(1, example_idxs)
    if args.critic_obs_rep_key:
        if args.critic_obs_rep_key != "oracle_reps":
            raise ValueError("--critic_obs_rep_key currently supports oracle_reps only.")
        example_batch["observations"] = np.asarray(
            train_dataset[args.critic_obs_rep_key], dtype=np.float32
        )[example_idxs]
    agent = agents[config["agent_name"]].create(args.seed, example_batch, config)
    restored = restore_agent(agent, args.value_restore_path, args.value_restore_epoch)
    return restored.replace(config=agent.config)


def representative_train_indices(train_state_to_bin, reps, mode="center"):
    train_state_to_bin = np.asarray(train_state_to_bin, dtype=np.int32)
    num_bins = int(train_state_to_bin.max()) + 1
    out = np.full(num_bins, -1, dtype=np.int32)
    if mode == "first":
        for idx, b in enumerate(train_state_to_bin):
            if int(b) >= 0 and out[int(b)] < 0:
                out[int(b)] = int(idx)
        return out
    reps = np.asarray(reps, dtype=np.float32)
    for b in range(num_bins):
        idxs = np.nonzero(train_state_to_bin == b)[0]
        if len(idxs) == 0:
            continue
        center = reps[idxs].mean(axis=0)
        out[b] = int(idxs[int(np.argmin(np.linalg.norm(reps[idxs] - center[None, :], axis=1)))])
    return out


@partial(jax.jit, static_argnames=("action_dim", "flow_steps", "zero_init"))
def sample_gcfbc_flow_action(
    network,
    observation,
    goal,
    seed,
    temperature: float,
    action_dim: int,
    flow_steps: int,
    zero_init: bool,
):
    if zero_init:
        actions = jnp.zeros((*observation.shape[:-1], action_dim), dtype=jnp.float32)
    else:
        actions = temperature * jax.random.normal(
            seed,
            (*observation.shape[:-1], action_dim),
        )
    for i in range(flow_steps):
        t = jnp.full((*observation.shape[:-1], 1), i / flow_steps)
        vels = network.select("actor_flow")(observation, goal, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1, 1)


def restore_controller_agent(train_dataset, args):
    if args.controller_agent_restore_path is None:
        raise ValueError(
            "--controller_agent_restore_path is required for --controller_type=agent."
        )
    if args.controller_agent_restore_epoch is None:
        raise ValueError(
            "--controller_agent_restore_epoch is required for --controller_type=agent."
        )
    agent_name, config, flags = bc_eval.load_agent_config(
        args.controller_agent_restore_path
    )
    controller_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = controller_dataset.sample(1)
    agent = agents[agent_name].create(int(args.seed), example_batch, config)
    agent = restore_agent(
        agent,
        args.controller_agent_restore_path,
        int(args.controller_agent_restore_epoch),
    )
    return agent, agent_name, flags


class SceneGraphPolicy:
    def __init__(
        self,
        value_agent,
        bc_model,
        bc_params,
        train_dataset,
        graph,
        distances,
        args,
        controller_agent=None,
    ):
        self.value_agent = value_agent
        self.bc_model = bc_model
        self.bc_params = bc_params
        self.controller_agent = controller_agent
        self.train_dataset = train_dataset
        self.graph = graph
        self.distances = np.asarray(distances, dtype=np.float32)
        self.args = args
        self.controller_type = str(getattr(args, "controller_type", "bc"))
        self.controller_temperature = float(getattr(args, "controller_temperature", 0.0))
        self.controller_flow_sample_mode = str(
            getattr(args, "controller_flow_sample_mode", "agent")
        )
        self.controller_rng = jax.random.PRNGKey(int(args.seed) + 991)
        self.selector = str(args.selector)
        self.left_budget = int(args.left_budget)
        self.right_budget = int(args.right_budget)
        self.num_candidates = int(args.num_subgoal_candidates)
        self.score_batch_size = int(args.score_batch_size)
        self.rng = np.random.default_rng(int(args.seed) + 7723)
        self.train_obs = np.asarray(train_dataset["observations"], dtype=np.float32)
        self.train_reps = np.asarray(train_dataset["oracle_reps"], dtype=np.float32)
        self.critic_obs_rep_key = str(args.critic_obs_rep_key or "")
        self.rep_dims = tuple(int(x) for x in graph["metadata"].get("rep_dims", ()))
        self.bin_centers = np.asarray(graph["bin_centers"], dtype=np.float32)
        self.bin_coords = np.asarray(graph["bin_coords"], dtype=np.int32)
        self.bin_size = float(graph["metadata"]["bin_size"])
        self.origin = (
            self.bin_centers[0]
            - (self.bin_coords[0].astype(np.float32) + 0.5) * self.bin_size
        )
        self.coord_to_bin = {
            tuple(coord.tolist()): int(idx) for idx, coord in enumerate(self.bin_coords)
        }
        self.rep_idx_by_bin = representative_train_indices(
            graph["train_state_to_bin"], self.train_reps, mode=args.subgoal_rep_mode
        )
        self.valid_rep_bins = np.nonzero(self.rep_idx_by_bin >= 0)[0].astype(np.int32)
        self.action_dim = int(np.asarray(train_dataset["actions"]).shape[-1])
        self.bc_layer_norm = bool(args.bc_layer_norm)
        self.map_fallback_count = 0
        self.map_fail_count = 0

    def rep_to_bin(self, rep):
        rep = np.asarray(rep, dtype=np.float32)
        coord = np.floor((rep - self.origin) / self.bin_size).astype(np.int32)
        b = self.coord_to_bin.get(tuple(coord.tolist()), -1)
        if b >= 0:
            return int(b)
        if self.args.online_bin_mode == "strict":
            self.map_fail_count += 1
            return -1
        d = np.linalg.norm(self.bin_centers - rep[None, :], axis=1)
        nearest = int(np.argmin(d))
        max_d = float(self.args.nearest_bin_max_dist)
        if max_d <= 0.0:
            max_d = 1.5 * self.bin_size
        if float(d[nearest]) <= max_d:
            self.map_fallback_count += 1
            return nearest
        self.map_fail_count += 1
        return -1

    def graph_distance(self, src, dst):
        if src < 0 or dst < 0:
            return float("inf")
        return float(self.distances[int(src), int(dst)])

    def current_rep(self, env, observation=None):
        try:
            return np.asarray(unwrap_env(env).compute_oracle_observation(), dtype=np.float32)
        except ValueError:
            if observation is None:
                raise
            if not self.rep_dims:
                raise
            observation = np.asarray(observation, dtype=np.float32)
            return observation[np.asarray(self.rep_dims, dtype=np.int32)].astype(np.float32)

    def bmm_scores(self, source_obs, source_rep, goal_rep, candidate_bins, left_budget, right_budget):
        candidate_bins = np.asarray(candidate_bins, dtype=np.int32)
        num_real = int(len(candidate_bins))
        if num_real == 0:
            return np.zeros((0,), dtype=np.float32)
        idxs = self.rep_idx_by_bin[candidate_bins]
        subgoal_obs = self.train_obs[idxs]
        subgoal_reps = self.train_reps[idxs]
        source_critic_obs = np.asarray(source_obs, dtype=np.float32)
        subgoal_critic_obs = subgoal_obs
        if self.critic_obs_rep_key:
            if self.critic_obs_rep_key != "oracle_reps":
                raise ValueError("--critic_obs_rep_key currently supports oracle_reps only.")
            source_critic_obs = np.asarray(source_rep, dtype=np.float32)
            subgoal_critic_obs = subgoal_reps
        source_obs_batch = np.repeat(source_critic_obs[None, :], num_real, axis=0)
        goal_reps = np.repeat(np.asarray(goal_rep, dtype=np.float32)[None, :], num_real, axis=0)
        zeros = np.zeros((num_real, self.action_dim), dtype=np.float32)
        left = np.full(num_real, int(left_budget), dtype=np.int32)
        right = np.full(num_real, int(right_budget), dtype=np.int32)
        chunks = []
        batch_size = max(1, int(self.score_batch_size))
        for start in range(0, num_real, batch_size):
            end = min(start + batch_size, num_real)
            real_count = end - start
            pad_count = batch_size - real_count
            if pad_count > 0 and bool(getattr(self.args, "pad_score_batches", False)):
                # Pad with a real row so JAX sees a stable scoring shape. Padded
                # scores are discarded before candidate ranking.
                source_obs_chunk = np.concatenate(
                    [
                        source_obs_batch[start:end],
                        np.repeat(source_obs_batch[end - 1 : end], pad_count, axis=0),
                    ],
                    axis=0,
                )
                zeros_chunk = np.concatenate(
                    [zeros[start:end], np.repeat(zeros[end - 1 : end], pad_count, axis=0)],
                    axis=0,
                )
                subgoal_reps_chunk = np.concatenate(
                    [
                        subgoal_reps[start:end],
                        np.repeat(subgoal_reps[end - 1 : end], pad_count, axis=0),
                    ],
                    axis=0,
                )
                subgoal_obs_chunk = np.concatenate(
                    [
                        subgoal_critic_obs[start:end],
                        np.repeat(subgoal_critic_obs[end - 1 : end], pad_count, axis=0),
                    ],
                    axis=0,
                )
                goal_reps_chunk = np.concatenate(
                    [
                        goal_reps[start:end],
                        np.repeat(goal_reps[end - 1 : end], pad_count, axis=0),
                    ],
                    axis=0,
                )
                left_chunk = np.full(batch_size, int(left_budget), dtype=np.int32)
                right_chunk = np.full(batch_size, int(right_budget), dtype=np.int32)
            else:
                source_obs_chunk = source_obs_batch[start:end]
                zeros_chunk = zeros[start:end]
                subgoal_reps_chunk = subgoal_reps[start:end]
                subgoal_obs_chunk = subgoal_critic_obs[start:end]
                goal_reps_chunk = goal_reps[start:end]
                left_chunk = left[start:end]
                right_chunk = right[start:end]
            left_logits = self.value_agent.critic_logits_for(
                source_obs_chunk,
                zeros_chunk,
                subgoal_reps_chunk,
                left_chunk,
                offsets=left_chunk,
            )
            right_logits = self.value_agent.critic_logits_for(
                subgoal_obs_chunk,
                zeros_chunk,
                goal_reps_chunk,
                right_chunk,
                offsets=right_chunk,
            )
            left_probs = np.asarray(jax.nn.sigmoid(left_logits)).min(axis=0)
            right_probs = np.asarray(jax.nn.sigmoid(right_logits)).min(axis=0)
            chunks.append(np.minimum(left_probs, right_probs)[:real_count])
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)

    def direct_reachability_score(self, source_obs, source_rep, goal_rep, budget):
        source_critic_obs = np.asarray(source_obs, dtype=np.float32)
        if self.critic_obs_rep_key:
            if self.critic_obs_rep_key != "oracle_reps":
                raise ValueError("--critic_obs_rep_key currently supports oracle_reps only.")
            source_critic_obs = np.asarray(source_rep, dtype=np.float32)
        budgets = np.full(1, int(budget), dtype=np.int32)
        logits = self.value_agent.critic_logits_for(
            source_critic_obs[None, :],
            np.zeros((1, self.action_dim), dtype=np.float32),
            np.asarray(goal_rep, dtype=np.float32)[None, :],
            budgets,
            offsets=budgets,
        )
        probs = np.asarray(jax.nn.sigmoid(logits)).min(axis=0)
        return float(probs[0])

    def candidate_bins(self, source_bin, goal_bin):
        if source_bin < 0 or goal_bin < 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        left = self.distances[int(source_bin)]
        right = self.distances[:, int(goal_bin)]
        source_goal = float(self.distances[int(source_bin), int(goal_bin)])
        valid = np.isfinite(left) & np.isfinite(right) & (self.rep_idx_by_bin >= 0)
        valid &= left <= float(self.left_budget)
        if np.isfinite(source_goal):
            valid &= right < source_goal - 1e-6
        bins = np.nonzero(valid)[0].astype(np.int32)
        if len(bins) == 0:
            valid = np.isfinite(left) & np.isfinite(right) & (self.rep_idx_by_bin >= 0)
            bins = np.nonzero(valid)[0].astype(np.int32)
        if len(bins) == 0:
            return bins, np.zeros((0,), dtype=np.float32)
        target_left = min(float(self.left_budget), source_goal) if np.isfinite(source_goal) else float(self.left_budget)
        path_slack = np.abs(left[bins] + right[bins] - source_goal) if np.isfinite(source_goal) else 0.0
        cost = (
            np.abs(left[bins] - target_left) / max(float(self.left_budget), 1.0)
            + right[bins] / max(float(self.right_budget), source_goal if np.isfinite(source_goal) else 1.0, 1.0)
            + path_slack / max(source_goal if np.isfinite(source_goal) else 1.0, 1.0)
        )
        order = np.argsort(cost)
        bins = bins[order[: self.num_candidates]]
        cost = cost[order[: self.num_candidates]]
        return bins, cost.astype(np.float32)

    def select_goal(
        self,
        observation,
        current_rep,
        final_goal_rep,
        final_goal_switch_distance=None,
        selector_override=None,
    ):
        selector = str(selector_override or self.selector)
        if selector == "support_then_bmm":
            selector = "support_path_only"
        elif selector == "bmm_then_support":
            selector = "BMM_support_path"
        elif selector in (
            "start_distance_gate_bmm_support",
            "start_distance_deltay_gate_bmm_support",
            "start_distance_cross_gate_bmm_support",
            "source_x_gate_bmm_support",
        ):
            selector = "BMM_support_path"
        source_bin = self.rep_to_bin(current_rep)
        goal_bin = self.rep_to_bin(final_goal_rep)
        source_goal_d = self.graph_distance(source_bin, goal_bin)
        switch_distance = (
            float(self.args.final_goal_switch_distance)
            if final_goal_switch_distance is None
            else float(final_goal_switch_distance)
        )
        direct_confidence = float("nan")
        force_direct = selector == "direct_goal" or not np.isfinite(source_goal_d)
        if not force_direct and source_goal_d <= switch_distance:
            if (
                str(getattr(self.args, "final_goal_switch_mode", "distance")) == "value_confidence"
                and source_goal_d > float(self.args.confidence_min_direct_distance)
            ):
                direct_confidence = self.direct_reachability_score(
                    observation,
                    current_rep,
                    final_goal_rep,
                    int(self.args.direct_confidence_budget),
                )
                force_direct = direct_confidence >= float(self.args.direct_confidence_threshold)
            else:
                force_direct = True
        if force_direct:
            return dict(
                goal_rep=np.asarray(final_goal_rep, dtype=np.float32),
                source_bin=source_bin,
                goal_bin=goal_bin,
                subgoal_bin=goal_bin,
                source_to_subgoal=source_goal_d,
                subgoal_to_goal=0.0,
                source_to_goal=source_goal_d,
                score=float("nan"),
                bmm_score=float("nan"),
                path_cost=float("nan"),
                direct_confidence=direct_confidence,
                selector_used=selector,
                direct=True,
            )
        bins, path_cost = self.candidate_bins(source_bin, goal_bin)
        if len(bins) == 0:
            return dict(
                goal_rep=np.asarray(final_goal_rep, dtype=np.float32),
                source_bin=source_bin,
                goal_bin=goal_bin,
                subgoal_bin=goal_bin,
                source_to_subgoal=source_goal_d,
                subgoal_to_goal=0.0,
                source_to_goal=source_goal_d,
                score=float("nan"),
                bmm_score=float("nan"),
                path_cost=float("nan"),
                direct_confidence=direct_confidence,
                selector_used=selector,
                direct=True,
            )
        if selector == "random_support":
            chosen = int(self.rng.integers(len(bins)))
            bmm = np.full(len(bins), np.nan, dtype=np.float32)
            scores = -path_cost
        elif selector == "support_path_only":
            bmm = np.full(len(bins), np.nan, dtype=np.float32)
            scores = -path_cost
            chosen = int(np.argmax(scores))
        elif selector == "support_saturation_bmm":
            support_scores = -path_cost
            support_chosen = int(np.argmax(support_scores))
            support_bin = int(bins[support_chosen])
            support_left = self.graph_distance(source_bin, support_bin)
            support_right = self.graph_distance(support_bin, goal_bin)
            use_bmm = (
                np.isfinite(support_left)
                and np.isfinite(support_right)
                and support_left
                >= float(self.left_budget) - float(self.args.support_saturation_source_margin)
                and support_right >= float(self.args.support_saturation_min_right)
            )
            if use_bmm:
                bmm = self.bmm_scores(
                    observation,
                    current_rep,
                    final_goal_rep,
                    bins,
                    self.left_budget,
                    self.right_budget,
                )
                scores = -path_cost + float(self.args.bmm_tiebreak_weight) * bmm
                chosen = int(np.argmax(scores))
            else:
                bmm = np.full(len(bins), np.nan, dtype=np.float32)
                scores = support_scores
                chosen = support_chosen
        elif selector == "BMM_support_path":
            bmm = self.bmm_scores(
                observation,
                current_rep,
                final_goal_rep,
                bins,
                self.left_budget,
                self.right_budget,
            )
            scores = -path_cost + float(self.args.bmm_tiebreak_weight) * bmm
            chosen = int(np.argmax(scores))
        else:
            raise ValueError(f"Unknown selector {selector!r}.")
        subgoal_bin = int(bins[chosen])
        subgoal_idx = int(self.rep_idx_by_bin[subgoal_bin])
        return dict(
            goal_rep=self.train_reps[subgoal_idx].astype(np.float32),
            source_bin=source_bin,
            goal_bin=goal_bin,
            subgoal_bin=subgoal_bin,
            source_to_subgoal=self.graph_distance(source_bin, subgoal_bin),
            subgoal_to_goal=self.graph_distance(subgoal_bin, goal_bin),
            source_to_goal=source_goal_d,
            score=float(scores[chosen]),
            bmm_score=float(bmm[chosen]) if np.isfinite(bmm[chosen]) else float("nan"),
            path_cost=float(path_cost[chosen]),
            direct_confidence=direct_confidence,
            selector_used=selector,
            direct=False,
        )

    def action_for_goal(self, observation, goal_rep):
        if self.controller_type == "agent":
            if self.controller_flow_sample_mode != "agent":
                if str(self.controller_agent.config.get("agent_name", "")) != "gcfbc":
                    raise ValueError(
                        "--controller_flow_sample_mode is only supported for GCFBC controllers."
                    )
                zero_init = self.controller_flow_sample_mode == "zero"
                if zero_init:
                    key = jax.random.PRNGKey(0)
                else:
                    self.controller_rng, key = jax.random.split(self.controller_rng)
                action = sample_gcfbc_flow_action(
                    self.controller_agent.network,
                    jnp.asarray(np.asarray(observation, dtype=np.float32)[None, :]),
                    jnp.asarray(np.asarray(goal_rep, dtype=np.float32)[None, :]),
                    key,
                    float(self.controller_temperature),
                    int(self.controller_agent.config["action_dim"]),
                    int(self.controller_agent.config["flow_steps"]),
                    bool(zero_init),
                )
            else:
                self.controller_rng, key = jax.random.split(self.controller_rng)
                action = self.controller_agent.sample_actions(
                    np.asarray(observation, dtype=np.float32)[None, :],
                    goals=np.asarray(goal_rep, dtype=np.float32)[None, :],
                    seed=key,
                    temperature=self.controller_temperature,
                )
            return np.asarray(action)[0].astype(np.float32)
        action = bc_eval.bc_apply_numpy(
            self.bc_params,
            np.asarray(observation, dtype=np.float32)[None, :],
            np.asarray(goal_rep, dtype=np.float32)[None, :],
            layer_norm=self.bc_layer_norm,
        )[0]
        return np.asarray(action, dtype=np.float32)


def finite_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def aggregate(rows):
    keys = [
        "success",
        "start_graph_d",
        "final_graph_d",
        "graph_improve",
        "start_rep_d",
        "final_rep_d",
        "rep_improve",
        "selected_source_to_subgoal",
        "selected_subgoal_to_goal",
        "selected_bmm_score",
        "selected_path_cost",
        "selected_direct_confidence",
        "subgoal_valid",
        "map_fallback_frac",
        "map_fail_frac",
        "direct_recovery_triggers",
        "support_recovery_triggers",
        "direct_choice_frac",
        "route_bmm_frac",
    ]
    out = {}
    for key in keys:
        out[key] = finite_mean([row.get(key, float("nan")) for row in rows])
    out["episodes"] = len(rows)
    out["wall_s_per_ep"] = finite_mean([row.get("wall_s", float("nan")) for row in rows])
    out["steps_per_s"] = float(
        np.sum([row.get("steps", 0) for row in rows])
        / max(np.sum([row.get("wall_s", 0.0) for row in rows]), 1e-9)
    )
    return out


def evaluate_selector(env, policy, task_ids, episodes_per_task, max_steps, args):
    rows = []
    task_switch_overrides = parse_task_float_map(args.task_final_goal_switch_distances)
    task_commit_overrides = {
        int(task_id): int(round(value))
        for task_id, value in parse_task_float_map(args.task_subgoal_commit_steps).items()
    }
    reset_controller_rng_task_ids = parse_task_set(args.reset_controller_rng_task_ids)
    for task_id in task_ids:
        commit_steps = max(
            1,
            int(task_commit_overrides.get(int(task_id), int(args.subgoal_commit_steps))),
        )
        if bool(args.reset_controller_rng_each_task) or int(task_id) in reset_controller_rng_task_ids:
            policy.controller_rng = jax.random.PRNGKey(int(args.seed) + 991)
        for ep in range(int(episodes_per_task)):
            episode_id = int(args.episode_offset) + int(ep)
            t0 = time.time()
            reset_seed = int(args.seed) + 1009 * int(task_id) + episode_id
            if bool(args.seed_global_reset_noise):
                np.random.seed(reset_seed)
            obs, info = env.reset(seed=reset_seed, options={"task_id": int(task_id)})
            if bool(args.reset_controller_rng_each_episode):
                policy.controller_rng = jax.random.PRNGKey(reset_seed + 991)
            final_goal_rep = np.asarray(info["goal"], dtype=np.float32)
            start_rep = policy.current_rep(env, obs)
            start_bin = policy.rep_to_bin(start_rep)
            goal_bin = policy.rep_to_bin(final_goal_rep)
            start_graph_d = policy.graph_distance(start_bin, goal_bin)
            recovery_allowed = (
                float(args.direct_recovery_min_start_graph_d) <= 0.0
                or (
                    np.isfinite(start_graph_d)
                    and start_graph_d >= float(args.direct_recovery_min_start_graph_d)
                )
            )
            start_rep_d = float(np.linalg.norm(start_rep - final_goal_rep))
            active = None
            active_until = -1
            start_route_selector = None
            if policy.selector in (
                "start_distance_gate_bmm_support",
                "start_distance_deltay_gate_bmm_support",
                "start_distance_cross_gate_bmm_support",
                "source_x_gate_bmm_support",
            ):
                force_bmm_tasks = parse_task_set(args.route_force_bmm_task_ids)
                use_bmm_route = (
                    np.isfinite(start_graph_d)
                    and start_graph_d >= float(args.route_bmm_min_start_graph_d)
                )
                if policy.selector == "source_x_gate_bmm_support":
                    use_bmm_route = float(start_rep[0]) >= float(args.route_bmm_min_source_x)
                if policy.selector in (
                    "start_distance_deltay_gate_bmm_support",
                    "start_distance_cross_gate_bmm_support",
                ):
                    start_to_goal = final_goal_rep - start_rep
                    if policy.selector == "start_distance_deltay_gate_bmm_support":
                        use_bmm_route = use_bmm_route or (
                            float(start_to_goal[1]) >= float(args.route_bmm_min_delta_y)
                        )
                    else:
                        use_bmm_route = use_bmm_route or (
                            float(start_rep[0]) >= float(args.route_bmm_min_source_x)
                            and float(start_to_goal[1]) >= float(args.route_bmm_min_delta_y)
                        )
                start_route_selector = (
                    "BMM_support_path"
                    if (
                        int(task_id) in force_bmm_tasks
                        or use_bmm_route
                    )
                    else "support_path_only"
                )
            recovery_switch_active = False
            direct_streak_active = False
            direct_streak_start_step = -1
            direct_streak_start_graph_d = float("inf")
            direct_best_graph_d = float("inf")
            direct_recovery_triggers = 0
            support_recovery_active = False
            support_recovery_triggers = 0
            support_best_graph_d = float("inf")
            support_last_improve_step = 0
            saw_direct_choice = False
            choices = []
            choice_trace = []
            success = 0.0
            terminated = truncated = False
            map_queries0 = policy.map_fallback_count + policy.map_fail_count
            for step in range(int(max_steps)):
                current_rep = policy.current_rep(env, obs)
                current_bin = policy.rep_to_bin(current_rep)
                current_goal_d = policy.graph_distance(current_bin, goal_bin)
                need_new = active is None or step >= active_until
                if (
                    policy.selector in ("support_then_bmm", "bmm_then_support")
                    and np.isfinite(current_goal_d)
                ):
                    min_delta = float(args.support_recovery_min_delta)
                    if (
                        not np.isfinite(support_best_graph_d)
                        or current_goal_d < support_best_graph_d - min_delta
                    ):
                        support_best_graph_d = current_goal_d
                        support_last_improve_step = step
                    if (
                        int(args.support_recovery_patience_steps) > 0
                        and recovery_allowed
                        and not support_recovery_active
                        and step - support_last_improve_step
                        >= int(args.support_recovery_patience_steps)
                        and current_goal_d > float(args.support_recovery_switch_distance)
                    ):
                        support_recovery_active = True
                        support_recovery_triggers += 1
                        need_new = True
                    if (
                        policy.selector == "bmm_then_support"
                        and int(args.bmm_no_direct_patience_steps) > 0
                        and recovery_allowed
                        and not support_recovery_active
                        and not saw_direct_choice
                        and step >= int(args.bmm_no_direct_patience_steps)
                        and current_goal_d > float(args.support_recovery_switch_distance)
                    ):
                        support_recovery_active = True
                        support_recovery_triggers += 1
                        need_new = True
                if active is not None and float(args.subgoal_replan_distance) >= 0.0:
                    if policy.graph_distance(current_bin, active["subgoal_bin"]) <= float(args.subgoal_replan_distance):
                        need_new = True
                if need_new:
                    switch_distance = (
                        float(args.direct_recovery_switch_distance)
                        if recovery_switch_active
                        else float(
                            task_switch_overrides.get(
                                int(task_id), float(args.final_goal_switch_distance)
                            )
                        )
                    )
                    selector_override = None
                    if policy.selector == "support_then_bmm":
                        selector_override = (
                            "BMM_support_path"
                            if support_recovery_active
                            else "support_path_only"
                        )
                    elif policy.selector == "bmm_then_support":
                        selector_override = (
                            "support_path_only"
                            if support_recovery_active
                            else "BMM_support_path"
                        )
                    elif policy.selector in (
                        "start_distance_gate_bmm_support",
                        "start_distance_deltay_gate_bmm_support",
                        "start_distance_cross_gate_bmm_support",
                        "source_x_gate_bmm_support",
                    ):
                        selector_override = start_route_selector
                    active = policy.select_goal(
                        obs,
                        current_rep,
                        final_goal_rep,
                        final_goal_switch_distance=switch_distance,
                        selector_override=selector_override,
                    )
                    active_until = step + commit_steps
                    choices.append(active)
                    saw_direct_choice = saw_direct_choice or bool(active.get("direct", False))
                    if bool(args.record_choices):
                        choice_trace.append(
                            dict(
                                step=int(step),
                                switch_distance=float(switch_distance),
                                direct=bool(active.get("direct", False)),
                                selector_used=str(active.get("selector_used", policy.selector)),
                                source_to_goal=float(active["source_to_goal"]),
                                source_to_subgoal=float(active["source_to_subgoal"]),
                                subgoal_to_goal=float(active["subgoal_to_goal"]),
                                bmm_score=float(active["bmm_score"]),
                                path_cost=float(active["path_cost"]),
                                direct_confidence=float(active["direct_confidence"]),
                            )
                        )
                if active.get("direct", False):
                    if not direct_streak_active:
                        direct_streak_active = True
                        direct_streak_start_step = step
                        direct_streak_start_graph_d = current_goal_d
                        direct_best_graph_d = current_goal_d
                    direct_best_graph_d = min(direct_best_graph_d, current_goal_d)
                    direct_elapsed = step - direct_streak_start_step
                    direct_improve = direct_streak_start_graph_d - direct_best_graph_d
                    if (
                        int(args.direct_recovery_patience_steps) > 0
                        and recovery_allowed
                        and not recovery_switch_active
                        and direct_elapsed >= int(args.direct_recovery_patience_steps)
                        and current_goal_d > float(args.direct_recovery_switch_distance)
                        and direct_improve < float(args.direct_recovery_min_improve)
                    ):
                        recovery_switch_active = True
                        direct_recovery_triggers += 1
                        active = policy.select_goal(
                            obs,
                            current_rep,
                            final_goal_rep,
                            final_goal_switch_distance=float(args.direct_recovery_switch_distance),
                            selector_override=(
                                "BMM_support_path"
                                if (
                                    policy.selector == "support_then_bmm"
                                    and support_recovery_active
                                )
                                else "support_path_only"
                                if (
                                    policy.selector == "bmm_then_support"
                                    and support_recovery_active
                                )
                                else None
                            ),
                        )
                        active_until = step + commit_steps
                        choices.append(active)
                        saw_direct_choice = saw_direct_choice or bool(active.get("direct", False))
                        if bool(args.record_choices):
                            choice_trace.append(
                                dict(
                                    step=int(step),
                                    switch_distance=float(args.direct_recovery_switch_distance),
                                    direct=bool(active.get("direct", False)),
                                    selector_used=str(active.get("selector_used", policy.selector)),
                                    source_to_goal=float(active["source_to_goal"]),
                                    source_to_subgoal=float(active["source_to_subgoal"]),
                                    subgoal_to_goal=float(active["subgoal_to_goal"]),
                                    bmm_score=float(active["bmm_score"]),
                                    path_cost=float(active["path_cost"]),
                                    direct_confidence=float(active["direct_confidence"]),
                                    recovery=True,
                                )
                            )
                        if active.get("direct", False):
                            direct_streak_start_step = step
                            direct_streak_start_graph_d = current_goal_d
                            direct_best_graph_d = current_goal_d
                        else:
                            direct_streak_active = False
                            direct_streak_start_step = -1
                            direct_streak_start_graph_d = float("inf")
                            direct_best_graph_d = float("inf")
                else:
                    direct_streak_active = False
                    direct_streak_start_step = -1
                    direct_streak_start_graph_d = float("inf")
                    direct_best_graph_d = float("inf")
                action = policy.action_for_goal(obs, active["goal_rep"])
                obs, reward, terminated, truncated, step_info = env.step(action)
                success = float(step_info.get("success", False))
                if success or terminated or truncated:
                    break
            final_rep = policy.current_rep(env, obs)
            final_bin = policy.rep_to_bin(final_rep)
            final_graph_d = policy.graph_distance(final_bin, goal_bin)
            final_rep_d = float(np.linalg.norm(final_rep - final_goal_rep))
            map_queries1 = policy.map_fallback_count + policy.map_fail_count
            map_total = max(map_queries1 - map_queries0, 1)
            row = dict(
                selector=policy.selector,
                task=int(task_id),
                episode=episode_id,
                success=float(success),
                steps=int(step + 1),
                terminated=bool(terminated),
                truncated=bool(truncated),
                start_graph_d=start_graph_d,
                final_graph_d=final_graph_d,
                graph_improve=start_graph_d - final_graph_d if np.isfinite(start_graph_d) and np.isfinite(final_graph_d) else float("nan"),
                start_rep_d=start_rep_d,
                final_rep_d=final_rep_d,
                rep_improve=start_rep_d - final_rep_d,
                selected_source_to_subgoal=finite_mean([c["source_to_subgoal"] for c in choices]),
                selected_subgoal_to_goal=finite_mean([c["subgoal_to_goal"] for c in choices]),
                selected_bmm_score=finite_mean([c["bmm_score"] for c in choices]),
                selected_path_cost=finite_mean([c["path_cost"] for c in choices]),
                selected_direct_confidence=finite_mean([c["direct_confidence"] for c in choices]),
                subgoal_valid=finite_mean(
                    [
                        float(
                            np.isfinite(c["source_to_subgoal"])
                            and c["source_to_subgoal"] <= float(args.left_budget)
                            and np.isfinite(c["subgoal_to_goal"])
                        )
                        for c in choices
                    ]
                ),
                direct_recovery_triggers=float(direct_recovery_triggers),
                support_recovery_triggers=float(support_recovery_triggers),
                direct_choice_frac=finite_mean([float(c["direct"]) for c in choices]),
                route_bmm_frac=finite_mean(
                    [
                        float(c.get("selector_used") == "BMM_support_path")
                        for c in choices
                    ]
                ),
                subgoal_commit_steps=float(commit_steps),
                map_fallback_frac=float((policy.map_fallback_count) / max(map_queries1, 1)),
                map_fail_frac=float((policy.map_fail_count) / max(map_queries1, 1)),
                wall_s=float(time.time() - t0),
            )
            if bool(args.record_choices):
                row["choices"] = choice_trace
            rows.append(row)
            print(
                "selector={selector} task={task} ep={episode} success={success:.1f} "
                "steps={steps} final_graph_d={final_graph_d:.1f} "
                "final_rep_d={final_rep_d:.3f} improve={graph_improve:.1f}".format(
                    **row
                ),
                flush=True,
            )
    return rows


def markdown_summary(result):
    lines = [
        "# Scene graph-subgoal controller smoke",
        "",
        f"env: `{result['env_name']}`",
        f"graph: `{result['graph_path']}`",
        f"value: `{result['value_restore_path']}:{result['value_restore_epoch']}`",
        f"controller type: `{result.get('controller_type', 'bc')}`",
        f"controller agent: `{result.get('controller_agent_name')}`",
        f"controller restore: `{result.get('controller_agent_restore_path')}`",
        f"controller flow sample mode: `{result.get('controller_flow_sample_mode', 'agent')}`",
        f"controller temperature: `{result.get('controller_temperature', 0.0)}`",
        f"bc steps: `{result['bc_info'].get('bc_steps')}`",
        f"tasks: `{result['task_ids']}`, episodes/task: `{result['episodes_per_task']}`",
        f"max steps: `{result['max_steps']}`",
        "",
        "| selector | success | final_graph_d | graph_improve | final_rep_d | rep_improve | src_to_subgoal | right_d | bmm_score | path_cost | subgoal_valid | wall_s/ep | steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {success:.4f} | {final_graph_d:.4f} | {graph_improve:.4f} | "
            "{final_rep_d:.4f} | {rep_improve:.4f} | {selected_source_to_subgoal:.4f} | "
            "{selected_subgoal_to_goal:.4f} | {selected_bmm_score:.4f} | "
            "{selected_path_cost:.4f} | {subgoal_valid:.4f} | {wall_s_per_ep:.4f} | "
            "{steps_per_s:.4f} |".format(name=row["name"], **m)
        )
    lines.extend(["", "Per-task:", "", "| selector | task | success | final_graph_d | final_rep_d | graph_improve |", "|---|---:|---:|---:|---:|---:|"])
    for row in result["selectors"]:
        for task in result["task_ids"]:
            eps = [ep for ep in row["episodes"] if ep["task"] == task]
            m = aggregate(eps)
            lines.append(
                "| {name} | {task} | {success:.4f} | {final_graph_d:.4f} | "
                "{final_rep_d:.4f} | {graph_improve:.4f} |".format(
                    name=row["name"], task=task, **m
                )
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="scene-play-oraclerep-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--distance_matrix_path", default=None)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--budgets", default="16,32,64,128")
    parser.add_argument("--budget_feature", default="log_scalar_onehot")
    parser.add_argument("--critic_obs_rep_key", default=None)
    parser.add_argument("--critic_absdiff_goal_feature", action="store_true")
    parser.add_argument("--actor_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--value_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--layer_norm", default="True")
    parser.add_argument("--selectors", default="direct_goal,support_path_only,BMM_support_path")
    parser.add_argument("--left_budget", type=int, default=32)
    parser.add_argument("--right_budget", type=int, default=128)
    parser.add_argument("--num_subgoal_candidates", type=int, default=64)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--pad_score_batches", action="store_true")
    parser.add_argument("--bmm_tiebreak_weight", type=float, default=0.05)
    parser.add_argument("--subgoal_commit_steps", type=int, default=20)
    parser.add_argument(
        "--task_subgoal_commit_steps",
        default="",
        help=(
            "Optional comma-separated TASK:STEPS overrides for subgoal commit "
            "lengths, e.g. '4:20,5:10'. Empty keeps --subgoal_commit_steps."
        ),
    )
    parser.add_argument("--subgoal_replan_distance", type=float, default=16.0)
    parser.add_argument("--final_goal_switch_distance", type=float, default=16.0)
    parser.add_argument(
        "--task_final_goal_switch_distances",
        default="",
        help=(
            "Optional comma-separated TASK:DIST overrides for the final-goal "
            "switch distance, e.g. '4:128,5:64'."
        ),
    )
    parser.add_argument(
        "--final_goal_switch_mode",
        choices=("distance", "value_confidence"),
        default="distance",
    )
    parser.add_argument("--confidence_min_direct_distance", type=float, default=128.0)
    parser.add_argument("--direct_confidence_budget", type=int, default=256)
    parser.add_argument("--direct_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--direct_recovery_patience_steps", type=int, default=0)
    parser.add_argument("--direct_recovery_switch_distance", type=float, default=128.0)
    parser.add_argument("--direct_recovery_min_improve", type=float, default=32.0)
    parser.add_argument("--direct_recovery_min_start_graph_d", type=float, default=0.0)
    parser.add_argument("--support_recovery_patience_steps", type=int, default=0)
    parser.add_argument("--support_recovery_switch_distance", type=float, default=128.0)
    parser.add_argument("--support_recovery_min_delta", type=float, default=32.0)
    parser.add_argument("--support_saturation_source_margin", type=float, default=0.0)
    parser.add_argument("--support_saturation_min_right", type=float, default=0.0)
    parser.add_argument("--bmm_no_direct_patience_steps", type=int, default=0)
    parser.add_argument("--route_bmm_min_start_graph_d", type=float, default=1480.0)
    parser.add_argument("--route_bmm_min_source_x", type=float, default=50.0)
    parser.add_argument("--route_bmm_min_delta_y", type=float, default=35.0)
    parser.add_argument("--route_force_bmm_task_ids", default="")
    parser.add_argument("--online_bin_mode", choices=("strict", "nearest"), default="nearest")
    parser.add_argument("--nearest_bin_max_dist", type=float, default=0.0)
    parser.add_argument("--subgoal_rep_mode", choices=("first", "center"), default="center")
    parser.add_argument("--task_ids", default="1,2,3,4,5")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed_global_reset_noise",
        action="store_true",
        help=(
            "Seed global NumPy before env.reset. OGBench locomaze goal/start "
            "noise uses np.random directly, so Gym's reset seed alone does not "
            "make task resets paired across methods."
        ),
    )
    parser.add_argument("--bc_restore_path", default=None)
    parser.add_argument("--bc_save_path", default=None)
    parser.add_argument("--bc_steps", type=int, default=2000)
    parser.add_argument("--bc_offsets", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--bc_batch_size", type=int, default=512)
    parser.add_argument("--bc_hidden_dims", default="(512,512,512)")
    parser.add_argument("--bc_layer_norm", action="store_true")
    parser.add_argument("--bc_lr", type=float, default=3e-4)
    parser.add_argument("--bc_log_interval", type=int, default=500)
    parser.add_argument("--bc_goal_rep", choices=("oracle",), default="oracle")
    parser.add_argument("--controller_type", choices=("bc", "agent"), default="bc")
    parser.add_argument("--controller_agent_restore_path", default=None)
    parser.add_argument("--controller_agent_restore_epoch", type=int, default=None)
    parser.add_argument("--controller_temperature", type=float, default=0.0)
    parser.add_argument(
        "--controller_flow_sample_mode",
        default="agent",
        choices=("agent", "zero", "temperature_scaled"),
        help=(
            "GCFBC controller sampling mode. 'agent' preserves the checkpoint API; "
            "'zero' uses deterministic zero-noise flow extraction; "
            "'temperature_scaled' scales the flow prior by controller_temperature."
        ),
    )
    parser.add_argument(
        "--reset_controller_rng_each_episode",
        action="store_true",
        help=(
            "Reset stochastic controller sampling RNG from the paired episode "
            "seed. Useful when controller_temperature > 0 so results do not "
            "depend on task/evaluation order."
        ),
    )
    parser.add_argument(
        "--reset_controller_rng_each_task",
        action="store_true",
        help=(
            "Reset stochastic controller sampling RNG at the start of each task "
            "block while preserving the within-task action-sampling stream."
        ),
    )
    parser.add_argument(
        "--reset_controller_rng_task_ids",
        default="",
        help=(
            "Comma-separated task IDs for task-block controller RNG reset. This "
            "allows task-specific stochastic-controller extraction without "
            "resetting every task."
        ),
    )
    parser.add_argument("--record_choices", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(int(args.seed))
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path_from_dir(args.dataset_dir)
    )
    graph = load_graph_npz(args.graph_path)
    distance_path = args.distance_matrix_path or str(Path(args.graph_path).with_name(f"{Path(args.graph_path).stem}_distance_matrix.npz"))
    distances = load_graph_distance_matrix_npz(distance_path, graph)
    if distances is None:
        raise FileNotFoundError(f"Missing or incompatible graph distance matrix: {distance_path}")
    value_agent = configure_value_agent(args, train_dataset)
    controller_agent = None
    controller_agent_name = None
    controller_agent_flags = None
    if args.controller_type == "agent":
        controller_agent, controller_agent_name, controller_agent_flags = (
            restore_controller_agent(train_dataset, args)
        )
        bc_model = None
        bc_params = None
        bc_info = {}
        print(
            "Restored agent controller "
            f"{controller_agent_name} from {args.controller_agent_restore_path}",
            flush=True,
        )
    else:
        if args.bc_restore_path:
            bc_model, bc_params, bc_info = bc_eval.load_bc(
                args.bc_restore_path, train_dataset, args
            )
            print(f"Restored BC controller from {args.bc_restore_path}", flush=True)
        else:
            bc_model, bc_params, bc_info = bc_eval.train_bc(train_dataset, args, rng)
            if args.bc_save_path:
                bc_eval.save_bc(args.bc_save_path, bc_params, args, bc_info)
                print(f"Saved BC controller to {args.bc_save_path}", flush=True)

    task_ids = parse_int_list(args.task_ids)
    selector_results = []
    for selector in [part.strip() for part in args.selectors.split(",") if part.strip()]:
        args.selector = selector
        print(f"Starting selector={selector}", flush=True)
        policy = SceneGraphPolicy(
            value_agent,
            bc_model,
            bc_params,
            train_dataset,
            graph,
            distances,
            args,
            controller_agent=controller_agent,
        )
        rows = evaluate_selector(
            env, policy, task_ids, args.episodes_per_task, args.max_steps, args
        )
        agg = aggregate(rows)
        print(
            f"Finished selector={selector} success={agg['success']:.4f} "
            f"final_graph_d={agg['final_graph_d']:.4f} final_rep_d={agg['final_rep_d']:.4f}",
            flush=True,
        )
        selector_results.append(dict(name=selector, aggregate=agg, episodes=rows))

    result = dict(
        env_name=args.env_name,
        graph_path=str(args.graph_path),
        distance_matrix_path=str(distance_path),
        graph_metadata=graph["metadata"],
        value_restore_path=str(args.value_restore_path),
        value_restore_epoch=int(args.value_restore_epoch),
        budget_feature=str(args.budget_feature),
        budgets=parse_int_list(args.budgets),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        score_batch_size=int(args.score_batch_size),
        pad_score_batches=bool(args.pad_score_batches),
        bmm_tiebreak_weight=float(args.bmm_tiebreak_weight),
        subgoal_commit_steps=int(args.subgoal_commit_steps),
        task_subgoal_commit_steps=str(args.task_subgoal_commit_steps),
        subgoal_replan_distance=float(args.subgoal_replan_distance),
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        episode_offset=int(args.episode_offset),
        max_steps=int(args.max_steps),
        final_goal_switch_distance=float(args.final_goal_switch_distance),
        task_final_goal_switch_distances=str(args.task_final_goal_switch_distances),
        final_goal_switch_mode=str(args.final_goal_switch_mode),
        confidence_min_direct_distance=float(args.confidence_min_direct_distance),
        direct_confidence_budget=int(args.direct_confidence_budget),
        direct_confidence_threshold=float(args.direct_confidence_threshold),
        direct_recovery_patience_steps=int(args.direct_recovery_patience_steps),
        direct_recovery_switch_distance=float(args.direct_recovery_switch_distance),
        direct_recovery_min_improve=float(args.direct_recovery_min_improve),
        direct_recovery_min_start_graph_d=float(args.direct_recovery_min_start_graph_d),
        support_recovery_patience_steps=int(args.support_recovery_patience_steps),
        support_recovery_switch_distance=float(args.support_recovery_switch_distance),
        support_recovery_min_delta=float(args.support_recovery_min_delta),
        bmm_no_direct_patience_steps=int(args.bmm_no_direct_patience_steps),
        route_bmm_min_start_graph_d=float(args.route_bmm_min_start_graph_d),
        route_bmm_min_source_x=float(args.route_bmm_min_source_x),
        route_bmm_min_delta_y=float(args.route_bmm_min_delta_y),
        route_force_bmm_task_ids=str(args.route_force_bmm_task_ids),
        record_choices=bool(args.record_choices),
        seed_global_reset_noise=bool(args.seed_global_reset_noise),
        controller_type=str(args.controller_type),
        controller_agent_name=controller_agent_name,
        controller_agent_restore_path=args.controller_agent_restore_path,
        controller_agent_restore_epoch=args.controller_agent_restore_epoch,
        controller_temperature=float(args.controller_temperature),
        controller_flow_sample_mode=str(args.controller_flow_sample_mode),
        reset_controller_rng_each_episode=bool(args.reset_controller_rng_each_episode),
        reset_controller_rng_each_task=bool(args.reset_controller_rng_each_task),
        reset_controller_rng_task_ids=str(args.reset_controller_rng_task_ids),
        controller_agent_env_name=(
            None
            if controller_agent_flags is None
            else controller_agent_flags.get("env_name")
        ),
        bc_info=bc_info,
        selectors=selector_results,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    md_path = out.with_suffix(".md")
    md_path.write_text(markdown_summary(result))
    print(f"Wrote Scene-Play graph BC report to {out}", flush=True)
    print(f"Wrote markdown summary to {md_path}", flush=True)


if __name__ == "__main__":
    main()
