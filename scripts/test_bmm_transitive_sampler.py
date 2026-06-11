#!/usr/bin/env python
"""Synthetic checks for geodesic V-transitive witness sampling modes."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_bmm_geodesic_value as diag
from utils.datasets import Dataset


def make_line_context(num_cells=12):
    positions = np.arange(num_cells, dtype=np.float32)
    observations = np.stack(
        [positions, np.zeros_like(positions), positions / max(num_cells - 1, 1)],
        axis=-1,
    ).astype(np.float32)
    actions = np.zeros((num_cells, 2), dtype=np.float32)
    terminals = np.zeros(num_cells, dtype=np.float32)
    terminals[-1] = 1.0
    dataset = Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )
    cells = np.arange(num_cells, dtype=np.int32)
    cell_distances = np.abs(cells[:, None] - cells[None, :]).astype(np.int32)
    state_to_cell = cells.copy()
    goal_by_cell = [np.asarray([idx], dtype=np.int32) for idx in cells]
    context = dict(
        kind="grid_geodesic",
        geodesic_budget_unit="grid_cells",
        cell_distances=cell_distances,
        steps_per_cell=1.0,
        label_distance_scale=1.0,
        train_state_to_cell=state_to_cell,
        train_goal_by_cell=goal_by_cell,
    )
    return dataset, context


def assert_valid_transitive_batch(batch, budget, batch_size, num_witnesses):
    required_keys = {
        "trans_witness_cell_counts",
        "trans_witness_candidate_counts",
        "trans_effective_unique_witness_counts",
        "trans_replacement_used",
        "trans_witness_fallback_used",
    }
    missing = required_keys - set(batch)
    assert not missing, f"Missing sampler diagnostics: {sorted(missing)}"
    assert batch["trans_valids"].shape == (num_witnesses, batch_size)
    assert batch["trans_effective_unique_witness_counts"].shape == (batch_size,)
    assert np.all(batch["trans_effective_unique_witness_counts"] >= 1.0)
    assert np.all(batch["trans_effective_unique_witness_counts"] <= num_witnesses)
    assert np.all(batch["trans_witness_candidate_counts"] <= batch["trans_witness_cell_counts"])
    assert np.any(batch["trans_replacement_used"] > 0.0)

    left_budget = budget // 2
    right_budget = budget - left_budget
    valid = batch["trans_valids"] > 0
    assert np.all(batch["value_budgets"] == budget)
    assert np.all(batch["trans_parent_distances"] <= budget + 1e-6)
    assert np.all(batch["trans_left_distances"][valid] <= left_budget + 1e-6)
    assert np.all(batch["trans_right_distances"][valid] <= right_budget + 1e-6)
    assert np.all(batch["trans_branch_oracle_valids"][valid] == 1.0)


def sample_for_mode(mode):
    dataset, context = make_line_context()
    diag.FLAGS.trans_witness_mode = mode
    diag.FLAGS.num_trans_witnesses = 4
    diag.FLAGS.trans_pos_boundary_frac = 0.5
    diag.FLAGS.trans_endpoint_epsilon = 1e-6
    diag.FLAGS.trans_boundary_beta = 0.5
    return diag.sample_grid_transitive_v_pairs(
        dataset,
        context,
        "train",
        budgets=(4,),
        batch_size=64,
        rng=np.random.default_rng(10),
    )


def main():
    if not diag.FLAGS.is_parsed():
        diag.FLAGS([sys.argv[0]])

    for mode in ("uniform_valid", "avoid_endpoints", "slack_balanced", "boundary_balanced"):
        batch = sample_for_mode(mode)
        assert_valid_transitive_batch(batch, budget=4, batch_size=64, num_witnesses=4)
        summary = diag.summarize_transitive_batch(batch, budgets=(4,))
        assert summary["trans_witness_mode"] == mode
        row = summary["budget_rows"][0]
        assert "effective_unique_witness_count_mean" in row
        assert "replacement_used_frac" in row
        assert summary["histograms"]["4"]["parent_distance_over_H"]["total"] == 64
        if mode != "uniform_valid":
            assert row["zero_left_frac"] == 0.0
            assert row["zero_right_frac"] == 0.0

    print("BMM transitive sampler checks passed.")


if __name__ == "__main__":
    main()
