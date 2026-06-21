import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


BMM_DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def get_bmm_budgets(config):
    """Return sorted positive BMM budgets no larger than max_budget."""
    budgets = np.asarray(config.get("budgets", BMM_DEFAULT_BUDGETS), dtype=np.int32)
    max_budget = int(config.get("max_budget", int(budgets.max())))
    if max_budget < 1:
        raise ValueError("BMM-TRL requires max_budget >= 1.")

    budgets = np.unique(budgets[(budgets >= 1) & (budgets <= max_budget)])
    if len(budgets) == 0 or budgets[0] != 1:
        budgets = np.unique(np.concatenate([np.asarray([1], dtype=np.int32), budgets]))
    if budgets[-1] != max_budget:
        budgets = np.unique(
            np.concatenate([budgets, np.asarray([max_budget], dtype=np.int32)])
        )
    if not np.all(budgets[1:] > budgets[:-1]):
        raise ValueError(f"BMM-TRL budgets must be strictly increasing: {budgets}")
    return budgets.astype(np.int32)


def sample_bmm_positive_budgets(offsets, budgets):
    """Sample a budget H with H >= each observed trajectory offset."""
    min_budget_idxs = np.searchsorted(budgets, offsets, side="left")
    min_budget_idxs = np.minimum(min_budget_idxs, len(budgets) - 1)
    spans = len(budgets) - min_budget_idxs
    sampled_idxs = min_budget_idxs + np.floor(np.random.rand(len(offsets)) * spans).astype(
        np.int32
    )
    return budgets[sampled_idxs]


def sample_bmm_negative_budgets(offsets, budgets, neg_frac):
    """Sample weak negative budgets H < neg_frac * offset where possible."""
    counts = np.searchsorted(budgets, neg_frac * offsets, side="left")
    valids = (counts > 0).astype(np.float32)
    safe_counts = np.maximum(counts, 1)
    sampled_idxs = np.floor(np.random.rand(len(offsets)) * safe_counts).astype(np.int32)
    return budgets[sampled_idxs], valids


def sample_bmm_hard_negative_pairs(
    idxs,
    final_state_idxs,
    budgets,
    min_factor,
    max_factor,
):
    """Sample same-trajectory goals that are beyond their sampled budget."""
    batch_size = len(idxs)
    remaining = (final_state_idxs - idxs).astype(np.int32)
    hard_budgets = budgets[np.random.randint(len(budgets), size=batch_size)].astype(
        np.int32
    )
    hard_offsets = np.minimum(np.maximum(remaining, 1), hard_budgets).astype(np.int32)
    hard_valids = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        valid_budget_idxs = []
        lows = []
        highs = []
        for budget_idx, budget in enumerate(budgets):
            budget = int(budget)
            low = max(budget + 1, int(np.ceil(min_factor * budget)))
            high = min(int(remaining[i]), int(np.floor(max_factor * budget)))
            if low <= high:
                valid_budget_idxs.append(budget_idx)
                lows.append(low)
                highs.append(high)

        if len(valid_budget_idxs) == 0:
            continue

        # Bias toward large budgets; the hard-negative loss is mainly guarding
        # against high-H all-ones collapse.
        valid_budgets = budgets[np.asarray(valid_budget_idxs, dtype=np.int32)].astype(
            np.float64
        )
        probs = valid_budgets / valid_budgets.sum()
        choice = int(np.searchsorted(np.cumsum(probs), np.random.rand(), side="right"))
        choice = min(choice, len(valid_budget_idxs) - 1)
        hard_budgets[i] = budgets[valid_budget_idxs[choice]]
        hard_offsets[i] = lows[choice] + int(
            np.floor(np.random.rand() * (highs[choice] - lows[choice] + 1))
        )
        hard_valids[i] = 1.0

    hard_goal_idxs = idxs + hard_offsets
    return hard_goal_idxs.astype(np.int32), hard_offsets, hard_budgets, hard_valids


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

        (self.terminal_locs,) = np.nonzero(self["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if hasattr(self, "valid_idxs"):
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        elif "valids" in self._dict:
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        defaults = {
            "value_p_curgoal": 0.0,
            "value_p_trajgoal": 1.0,
            "value_p_randomgoal": 0.0,
            "value_geom_sample": True,
            "actor_p_curgoal": 0.0,
            "actor_p_trajgoal": 0.5,
            "actor_p_randomgoal": 0.5,
            "actor_geom_sample": True,
            "gc_negative": True,
            "reachability_label_type": "logged_offset",
            "graph_path": "exp/bmm_pointmaze_graph.npz",
            "geodesic_distance_source": "grid",
        }

        self.config["dataset"] = ConfigDict(
            defaults | dict(self.config.get("dataset", {}))
        )

        dataset_config = self.config["dataset"]

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            dataset_config["value_p_curgoal"]
            + dataset_config["value_p_trajgoal"]
            + dataset_config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            dataset_config["actor_p_curgoal"]
            + dataset_config["actor_p_trajgoal"]
            + dataset_config["actor_p_randomgoal"],
            1.0,
        )

        # Set valid_idxs to exclude the final state in each trajectory.
        cur_idx = 0
        valid_idxs = []
        for terminal_idx in self.terminal_locs:
            valid_idxs.append(np.arange(cur_idx, terminal_idx))
            cur_idx = terminal_idx + 1
        self.dataset.valid_idxs = np.concatenate(valid_idxs)

    def sample(self, batch_size, idxs=None, evaluation=False):
        """
        Sample a batch of transitions with goals.

        batch_size: Batch size.
        idxs: Indices of the transitions to sample. If None, random indices are sampled.
        evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)

        batch = self.dataset.sample(batch_size, idxs)
        dataset_config = self.config["dataset"]

        if self.config.get("agent_name") == "bmm_trl":
            value_goal_idxs = self.sample_bmm_value_goals(idxs)
        else:
            value_goal_idxs = self.sample_goals(
                idxs,
                dataset_config["value_p_curgoal"],
                dataset_config["value_p_trajgoal"],
                dataset_config["value_p_randomgoal"],
                dataset_config["value_geom_sample"],
            )
        actor_goal_idxs = self.sample_goals(
            idxs,
            dataset_config["actor_p_curgoal"],
            dataset_config["actor_p_trajgoal"],
            dataset_config["actor_p_randomgoal"],
            dataset_config["actor_geom_sample"],
        )

        if "oracle_reps" in self.dataset:
            batch["value_goals"] = self.dataset["oracle_reps"][value_goal_idxs]
            batch["actor_goals"] = self.dataset["oracle_reps"][actor_goal_idxs]
        else:
            batch["value_goals"] = self.get_observations(value_goal_idxs)
            batch["actor_goals"] = self.get_observations(actor_goal_idxs)
        batch["value_goal_observations"] = self.get_observations(value_goal_idxs)
        batch["actor_goal_observations"] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if dataset_config["gc_negative"] else 0.0)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        if self.config.get("agent_name") in ["trl"]:
            assert (idxs != final_state_idxs).all() and (idxs != value_goal_idxs).all()

            value_middle_goal_idxs = np.random.randint(idxs, value_goal_idxs)

            batch["value_offsets"] = value_goal_idxs - idxs
            batch["value_midpoint_offsets"] = value_middle_goal_idxs - idxs
            batch["value_midpoint_observations"] = self.get_observations(
                value_middle_goal_idxs
            )
            batch["value_midpoint_actions"] = self.dataset["actions"][
                value_middle_goal_idxs
            ]
            batch["next_actions"] = self.dataset["actions"][idxs + 1]

            if "oracle_reps" in self.dataset:
                batch["value_midpoint_goals"] = self.dataset["oracle_reps"][
                    value_middle_goal_idxs
                ]
                batch["value_cur_goals"] = self.dataset["oracle_reps"][idxs]
                batch["value_next_goals"] = self.dataset["oracle_reps"][idxs + 1]
            else:
                batch["value_midpoint_goals"] = self.get_observations(
                    value_middle_goal_idxs
                )
                batch["value_cur_goals"] = self.get_observations(idxs)
                batch["value_next_goals"] = self.get_observations(idxs + 1)
        elif self.config.get("agent_name") == "bmm_trl":
            assert (idxs != final_state_idxs).all() and (idxs != value_goal_idxs).all()
            self.add_bmm_fields(batch, idxs, value_goal_idxs, final_state_idxs)

        return batch

    def sample_bmm_value_goals(self, idxs):
        """Sample future same-trajectory value goals with offsets covered by BMM budgets."""
        batch_size = len(idxs)
        dataset_config = self.config["dataset"]
        budgets = get_bmm_budgets(self.config)
        max_offset = int(budgets[-1])

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        max_goal_idxs = np.minimum(final_state_idxs, idxs + max_offset)
        spans = max_goal_idxs - idxs
        if not np.all(spans >= 1):
            raise ValueError("BMM-TRL sampled a terminal index; valid_idxs are malformed.")

        if dataset_config["value_geom_sample"]:
            offsets = np.random.geometric(1 - self.config["discount"], size=batch_size)
            offsets = np.minimum(offsets, spans)
        else:
            offsets = 1 + np.floor(np.random.rand(batch_size) * spans).astype(np.int32)

        return idxs + offsets

    def add_bmm_fields(self, batch, idxs, value_goal_idxs, final_state_idxs):
        """Add BMM-TRL budget, witness, weak-negative, and monotonicity fields."""
        batch_size = len(idxs)
        budgets = get_bmm_budgets(self.config)
        split_mode = self.config.get("split_mode", "half")
        if split_mode != "half":
            raise ValueError("BMM-TRL prototype only supports split_mode='half'.")

        offsets = (value_goal_idxs - idxs).astype(np.int32)
        value_budgets = sample_bmm_positive_budgets(offsets, budgets).astype(np.int32)

        left_budgets = np.where(value_budgets > 1, value_budgets // 2, 1).astype(
            np.int32
        )
        right_budgets = np.where(
            value_budgets > 1, value_budgets - left_budgets, 1
        ).astype(np.int32)

        midpoint_los = np.maximum(idxs + 1, value_goal_idxs - right_budgets)
        midpoint_his = np.minimum(value_goal_idxs - 1, idxs + left_budgets)
        trans_valids = ((offsets > 1) & (midpoint_los <= midpoint_his)).astype(
            np.float32
        )
        midpoint_widths = np.maximum(midpoint_his - midpoint_los + 1, 1)
        sampled_midpoints = midpoint_los + np.floor(
            np.random.rand(batch_size) * midpoint_widths
        ).astype(np.int32)
        fallback_midpoints = np.minimum(idxs + 1, final_state_idxs)
        value_midpoint_idxs = np.where(
            trans_valids > 0, sampled_midpoints, fallback_midpoints
        )

        neg_budgets, neg_valids = sample_bmm_negative_budgets(
            offsets, budgets, float(self.config.get("budget_neg_frac", 0.5))
        )
        (
            hard_neg_goal_idxs,
            hard_neg_offsets,
            hard_neg_budgets,
            hard_neg_valids,
        ) = sample_bmm_hard_negative_pairs(
            idxs,
            final_state_idxs,
            budgets,
            float(self.config.get("hard_neg_min_factor", 1.25)),
            float(self.config.get("hard_neg_max_factor", 4.0)),
        )
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        random_budgets = budgets[np.random.randint(len(budgets), size=batch_size)]

        if len(budgets) > 1:
            mono_low_idxs = np.random.randint(len(budgets) - 1, size=batch_size)
            mono_low_budgets = budgets[mono_low_idxs]
            mono_high_budgets = budgets[mono_low_idxs + 1]
        else:
            mono_low_budgets = np.full(batch_size, budgets[0], dtype=np.int32)
            mono_high_budgets = np.full(batch_size, budgets[0], dtype=np.int32)

        batch["value_offsets"] = offsets
        batch["value_budgets"] = value_budgets
        batch["value_left_budgets"] = left_budgets
        batch["value_right_budgets"] = right_budgets
        batch["value_midpoint_offsets"] = (value_midpoint_idxs - idxs).astype(np.int32)
        batch["value_midpoint_observations"] = self.get_observations(
            value_midpoint_idxs
        )
        batch["value_midpoint_actions"] = self.dataset["actions"][value_midpoint_idxs]
        batch["next_actions"] = self.dataset["actions"][idxs + 1]
        batch["trans_valids"] = trans_valids
        batch["value_neg_budgets"] = neg_budgets.astype(np.int32)
        batch["value_neg_valids"] = neg_valids
        batch["value_hard_neg_offsets"] = hard_neg_offsets.astype(np.int32)
        batch["value_hard_neg_budgets"] = hard_neg_budgets.astype(np.int32)
        batch["value_hard_neg_valids"] = hard_neg_valids
        batch["value_random_budgets"] = random_budgets.astype(np.int32)
        batch["value_random_goal_observations"] = self.get_observations(
            random_goal_idxs
        )
        batch["mono_low_budgets"] = mono_low_budgets.astype(np.int32)
        batch["mono_high_budgets"] = mono_high_budgets.astype(np.int32)

        if "oracle_reps" in self.dataset:
            batch["value_midpoint_goals"] = self.dataset["oracle_reps"][
                value_midpoint_idxs
            ]
            batch["value_cur_goals"] = self.dataset["oracle_reps"][idxs]
            batch["value_next_goals"] = self.dataset["oracle_reps"][idxs + 1]
            batch["value_random_goals"] = self.dataset["oracle_reps"][random_goal_idxs]
            batch["value_hard_neg_goals"] = self.dataset["oracle_reps"][
                hard_neg_goal_idxs
            ]
        else:
            batch["value_midpoint_goals"] = self.get_observations(value_midpoint_idxs)
            batch["value_cur_goals"] = self.get_observations(idxs)
            batch["value_next_goals"] = self.get_observations(idxs + 1)
            batch["value_random_goals"] = self.get_observations(random_goal_idxs)
            batch["value_hard_neg_goals"] = self.get_observations(hard_neg_goal_idxs)

        self.add_bmm_supervised_fields(batch, budgets)
        self.add_bmm_rank_fields(batch, budgets)

    def get_goal_vectors(self, idxs):
        """Return goals in the same vector representation as value_goals."""
        if "oracle_reps" in self.dataset:
            return self.dataset["oracle_reps"][idxs]
        return self.get_observations(idxs)

    def add_bmm_supervised_fields(self, batch, budgets):
        """Add balanced per-budget supervised reachability pairs."""
        batch_size = len(batch["actions"])
        num_pairs = int(self.config.get("num_sup_pairs", 0))
        if num_pairs <= 0:
            return

        valid_idxs = self.dataset.valid_idxs.astype(np.int32)
        valid_remaining = (
            self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)]
            - valid_idxs
        ).astype(np.int32)
        pos_boundary_frac = float(self.config.get("sup_pos_boundary_frac", 0.5))
        neg_min_factor = float(self.config.get("sup_neg_min_factor", 1.0))
        neg_max_factor = float(self.config.get("sup_neg_max_factor", 2.0))
        include_far_negs = bool(self.config.get("sup_include_far_negs", True))

        fallback_idx = int(valid_idxs[0])
        source_idxs = np.full((num_pairs, batch_size), fallback_idx, dtype=np.int32)
        goal_idxs = np.full((num_pairs, batch_size), fallback_idx + 1, dtype=np.int32)
        sup_budgets = np.zeros((num_pairs, batch_size), dtype=np.int32)
        sup_offsets = np.ones((num_pairs, batch_size), dtype=np.int32)
        sup_labels = np.zeros((num_pairs, batch_size), dtype=np.float32)
        sup_valids = np.zeros((num_pairs, batch_size), dtype=np.float32)

        pos_eligible = {}
        neg_eligible = {}
        for budget in budgets:
            budget = int(budget)
            min_pos_remaining = max(1, int(np.ceil(pos_boundary_frac * budget)))
            eligible = valid_idxs[valid_remaining >= min_pos_remaining]
            if len(eligible) == 0:
                eligible = valid_idxs[valid_remaining >= 1]
            pos_eligible[budget] = eligible
            neg_eligible[budget] = valid_idxs[valid_remaining >= budget + 1]

        for pair_idx in range(num_pairs):
            sampled_budgets = budgets[
                np.random.randint(len(budgets), size=batch_size)
            ].astype(np.int32)
            labels = (np.random.rand(batch_size) < 0.5).astype(np.float32)
            sup_budgets[pair_idx] = sampled_budgets
            sup_labels[pair_idx] = labels

            for budget in budgets:
                budget = int(budget)
                pos_cols = np.nonzero(
                    (sampled_budgets == budget) & (labels == 1.0)
                )[0]
                if len(pos_cols) > 0 and len(pos_eligible[budget]) > 0:
                    eligible = pos_eligible[budget]
                    srcs = eligible[np.random.randint(len(eligible), size=len(pos_cols))]
                    rem = (
                        self.terminal_locs[np.searchsorted(self.terminal_locs, srcs)]
                        - srcs
                    ).astype(np.int32)
                    hi = np.minimum(budget, rem)
                    lo = np.minimum(
                        max(1, int(np.ceil(pos_boundary_frac * budget))), hi
                    )
                    valid = hi >= 1
                    widths = np.maximum(hi - lo + 1, 1)
                    sampled_offsets = lo + np.floor(
                        np.random.rand(len(pos_cols)) * widths
                    ).astype(np.int32)
                    cols = pos_cols[valid]
                    source_idxs[pair_idx, cols] = srcs[valid]
                    goal_idxs[pair_idx, cols] = srcs[valid] + sampled_offsets[valid]
                    sup_offsets[pair_idx, cols] = sampled_offsets[valid]
                    sup_valids[pair_idx, cols] = 1.0

                neg_cols = np.nonzero(
                    (sampled_budgets == budget) & (labels == 0.0)
                )[0]
                if len(neg_cols) > 0 and len(neg_eligible[budget]) > 0:
                    eligible = neg_eligible[budget]
                    srcs = eligible[np.random.randint(len(eligible), size=len(neg_cols))]
                    rem = (
                        self.terminal_locs[np.searchsorted(self.terminal_locs, srcs)]
                        - srcs
                    ).astype(np.int32)
                    near_lo = max(
                        budget + 1, int(np.floor(neg_min_factor * budget)) + 1
                    )
                    near_hi = np.minimum(
                        rem, int(np.floor(neg_max_factor * budget))
                    ).astype(np.int32)
                    lo = np.full(len(neg_cols), near_lo, dtype=np.int32)
                    hi = near_hi
                    if include_far_negs:
                        far_hi = np.minimum(rem, int(np.floor(4.0 * budget))).astype(
                            np.int32
                        )
                        use_far = (np.random.rand(len(neg_cols)) < 0.25) & (
                            far_hi > near_hi
                        )
                        lo = np.where(use_far, near_hi + 1, lo)
                        hi = np.where(use_far, far_hi, hi)
                    fallback = lo > hi
                    lo = np.where(fallback, budget + 1, lo)
                    hi = np.where(fallback, rem, hi)
                    valid = lo <= hi
                    widths = np.maximum(hi - lo + 1, 1)
                    sampled_offsets = lo + np.floor(
                        np.random.rand(len(neg_cols)) * widths
                    ).astype(np.int32)
                    cols = neg_cols[valid]
                    source_idxs[pair_idx, cols] = srcs[valid]
                    goal_idxs[pair_idx, cols] = srcs[valid] + sampled_offsets[valid]
                    sup_offsets[pair_idx, cols] = sampled_offsets[valid]
                    sup_valids[pair_idx, cols] = 1.0

        batch["value_sup_observations"] = self.get_observations(source_idxs)
        batch["value_sup_actions"] = self.dataset["actions"][source_idxs]
        batch["value_sup_goals"] = self.get_goal_vectors(goal_idxs)
        batch["value_sup_budgets"] = sup_budgets
        batch["value_sup_offsets"] = sup_offsets
        batch["value_sup_labels"] = sup_labels
        batch["value_sup_valids"] = sup_valids

    def add_bmm_rank_fields(self, batch, budgets):
        """Add same-pair budget-threshold ranking examples."""
        batch_size = len(batch["actions"])
        num_pairs = int(self.config.get("num_rank_pairs", 0))
        if num_pairs <= 0 or len(budgets) < 2:
            return

        rank_cache = getattr(self, "_bmm_rank_valid_cache", None)
        if rank_cache is None:
            valid_idxs = self.dataset.valid_idxs.astype(np.int32)
            valid_remaining = (
                self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)]
                - valid_idxs
            ).astype(np.int32)
            order = np.argsort(valid_remaining, kind="stable")
            rank_cache = (
                valid_idxs[order],
                valid_remaining[order],
                int(valid_idxs[0]),
            )
            self._bmm_rank_valid_cache = rank_cache
        sorted_valid_idxs, sorted_remaining, fallback_idx = rank_cache

        pos_budget_idxs = 1 + np.random.randint(
            len(budgets) - 1, size=(num_pairs, batch_size)
        )
        pos_budgets = budgets[pos_budget_idxs].astype(np.int32)
        neg_budgets = budgets[pos_budget_idxs - 1].astype(np.int32)
        widths = np.maximum(pos_budgets - neg_budgets, 1)
        offsets = (
            neg_budgets
            + 1
            + np.floor(np.random.rand(num_pairs, batch_size) * widths).astype(np.int32)
        )

        starts = np.searchsorted(sorted_remaining, offsets.reshape(-1), side="left")
        counts = len(sorted_valid_idxs) - starts
        valid_flat = counts > 0

        sample_positions = np.zeros_like(starts)
        sample_positions[valid_flat] = starts[valid_flat] + np.floor(
            np.random.rand(valid_flat.sum()) * counts[valid_flat]
        ).astype(np.int32)
        sampled_sources = np.full_like(starts, fallback_idx, dtype=np.int32)
        sampled_sources[valid_flat] = sorted_valid_idxs[sample_positions[valid_flat]]

        source_idxs = sampled_sources.reshape(num_pairs, batch_size)
        goal_idxs = source_idxs + offsets
        valids = valid_flat.reshape(num_pairs, batch_size).astype(np.float32)
        goal_idxs = np.where(valids > 0, goal_idxs, fallback_idx + 1).astype(np.int32)

        batch["value_rank_observations"] = self.get_observations(source_idxs)
        batch["value_rank_actions"] = self.dataset["actions"][source_idxs]
        batch["value_rank_goals"] = self.get_goal_vectors(goal_idxs)
        batch["value_rank_pos_budgets"] = pos_budgets
        batch["value_rank_neg_budgets"] = neg_budgets
        batch["value_rank_offsets"] = offsets
        batch["value_rank_valids"] = valids

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None
    ):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config["discount"]

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - discount, size=batch_size
            )  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(
            lambda arr: arr[idxs], self.dataset["observations"]
        )
