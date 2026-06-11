#!/usr/bin/env python
"""Synthetic tests for BMM reachability diagnostic sampling."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import (
    augment_goal_with_budget,
    config_budgets,
    get_config,
    normalize_budget,
)
from scripts.bmm_reachability_utils import (
    make_pair_batch,
    sample_balanced_budget_pairs,
    sample_fixed_offset_pairs,
    sample_random_pairs,
)
from utils.datasets import Dataset, GCDataset


def make_fake_dataset(num_trajs=3, traj_len=20, obs_dim=4, action_dim=2):
    size = num_trajs * traj_len
    observations = np.arange(size * obs_dim, dtype=np.float32).reshape(size, obs_dim)
    observations = observations / observations.max()
    actions = np.linspace(-1.0, 1.0, size * action_dim, dtype=np.float32).reshape(
        size, action_dim
    )
    terminals = np.zeros(size, dtype=np.float32)
    terminals[traj_len - 1 :: traj_len] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def make_config():
    config = get_config()
    config.discount = 0.9
    config.budgets = (1, 2, 4, 8)
    config.max_budget = 8
    return config


def main():
    rng = np.random.default_rng(0)
    config = make_config()
    dataset = GCDataset(make_fake_dataset(), config)

    idxs, goal_idxs, offsets = sample_random_pairs(dataset, 64, 16, rng)
    pair_batch = make_pair_batch(dataset, idxs, goal_idxs, offsets, budget=4)
    assert np.array_equal(pair_batch["labels"], (pair_batch["offsets"] <= 4).astype(float))
    assert pair_batch["goals"].shape[-1] == pair_batch["observations"].shape[-1]

    generated = {}
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        offset = max(1, int(round(4 * multiplier)))
        fixed = sample_fixed_offset_pairs(dataset, offset, 8, rng)
        if fixed is not None:
            bucket_batch = make_pair_batch(dataset, *fixed, budget=4)
            generated[multiplier] = int(bucket_batch["offsets"][0])
            assert np.all(bucket_batch["offsets"] == offset)
            assert np.all(bucket_batch["labels"] == (offset <= 4))

    assert generated == {0.25: 1, 0.5: 2, 1.0: 4, 2.0: 8, 4.0: 16}

    augmented = augment_goal_with_budget(
        pair_batch["goals"],
        pair_batch["budgets"],
        config.max_budget,
    )
    assert augmented.shape[-1] == pair_batch["goals"].shape[-1] + 1
    expected_budget_feature = np.asarray(
        normalize_budget(pair_batch["budgets"], config.max_budget)
    )
    assert np.allclose(np.asarray(augmented[:, -1]), expected_budget_feature)

    balanced = sample_balanced_budget_pairs(dataset, budget=4, num_pairs=32, rng=rng)
    assert balanced is not None
    assert np.array_equal(
        balanced["labels"], (balanced["offsets"] <= balanced["budgets"]).astype(float)
    )
    assert balanced["labels"].sum() > 0
    assert (balanced["labels"] == 0).sum() > 0
    assert np.all(balanced["offsets"][balanced["labels"] == 1] <= 4)
    assert np.all(balanced["offsets"][balanced["labels"] == 0] > 4)

    budgets = config_budgets(config)
    onehot_augmented = augment_goal_with_budget(
        pair_batch["goals"],
        pair_batch["budgets"],
        config.max_budget,
        budgets=budgets,
        budget_feature="log_scalar_onehot",
    )
    assert onehot_augmented.shape[-1] == pair_batch["goals"].shape[-1] + 1 + len(
        budgets
    )

    oracle_augmented = augment_goal_with_budget(
        pair_batch["goals"],
        pair_batch["budgets"],
        config.max_budget,
        offset=pair_batch["offsets"],
        oracle_offset_feature=True,
    )
    assert oracle_augmented.shape[-1] == pair_batch["goals"].shape[-1] + 2
    expected_offset_feature = np.asarray(
        normalize_budget(pair_batch["offsets"], config.max_budget)
    )
    assert np.allclose(np.asarray(oracle_augmented[:, -1]), expected_offset_feature)

    print("BMM reachability diagnostic sampler checks passed.")


if __name__ == "__main__":
    main()
