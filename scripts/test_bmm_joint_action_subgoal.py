#!/usr/bin/env python
"""Synthetic checks for BMM joint action-subgoal diagnostics."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_joint_action_subgoal as joint


def main():
    batch = {
        "source_observations": np.zeros((2, 3), dtype=np.float32),
        "candidate_observations": np.zeros((2, 2, 3), dtype=np.float32),
        "actions": np.zeros((2, 2, 2), dtype=np.float32),
        "goals": np.zeros((2, 3), dtype=np.float32),
        "subgoal_observations": np.zeros((2, 3, 3), dtype=np.float32),
        "subgoal_cells": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "source_distances": np.asarray(
            [[80.0, 20.0, 120.0], [30.0, 80.0, 140.0]], dtype=np.float32
        ),
        "next_distances": np.asarray(
            [
                [[79.0, 19.0, 119.0], [100.0, 10.0, 20.0]],
                [[29.0, 79.0, 139.0], [100.0, 20.0, 79.0]],
            ],
            dtype=np.float32,
        ),
        "right_distances": np.asarray(
            [[80.0, 140.0, 20.0], [130.0, 80.0, 20.0]], dtype=np.float32
        ),
        "direct_source_distances": np.asarray([160.0, 160.0], dtype=np.float32),
        "direct_next_distances": np.asarray(
            [[159.0, 150.0], [159.0, 120.0]], dtype=np.float32
        ),
        "state_valids": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        "action_valids": np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        "budget": 160,
        "left_budget": 80,
        "right_budget": 80,
        "candidate_action_count": 2,
        "num_subgoals": 3,
    }

    assert joint.pair_shape(batch) == (2, 2, 3)
    assert joint.source_mode_inputs(batch).shape == (2, 2, 3, 3)
    assert joint.own_state_inputs(batch).shape == (2, 2, 3, 3)
    assert joint.pair_actions(batch).shape == (2, 2, 3, 2)
    assert joint.pair_subgoals(batch).shape == (2, 2, 3, 3)

    baselines = joint.oracle_baselines(batch, np.random.default_rng(0))
    oracle_metrics = joint.selection_metrics(
        baselines["oracle_action_valid_midpoint"], batch
    )
    assert np.isclose(oracle_metrics["action_valid_frac"], 1.0)
    assert oracle_metrics["selected_unique_action_slots"] == 1
    assert np.isclose(oracle_metrics["selected_nonlogged_action_frac"], 0.0)

    action_1_scores = np.zeros((2, 2, 3), dtype=np.float32)
    action_1_scores[:, 1, 2] = 1.0
    action_1_metrics = joint.selection_metrics(action_1_scores, batch)
    assert np.isclose(action_1_metrics["selected_nonlogged_action_frac"], 1.0)
    assert np.isclose(action_1_metrics["action_valid_frac"], 1.0)
    assert action_1_metrics["selected_unique_subgoal_cells"] == 2

    random_metrics = joint.selection_metrics(np.zeros((2, 2, 3)), batch)
    assert np.isclose(random_metrics["state_valid_frac"], 0.5)

    coverage = joint.joint_candidate_coverage(
        batch,
        {
            "oracle_best_selected_distance_mean": 1.0,
            "next_distance_spread_mean": 2.0,
            "unique_next_cell_count_mean": 2.0,
        },
    )
    assert np.isclose(coverage["oracle_any_action_valid_frac"], 1.0)
    assert np.isclose(coverage["oracle_any_state_valid_frac"], 1.0)
    assert coverage["oracle_best_action_midpoint_error"] >= 0.0
    assert "next_distance_spread_mean" in coverage

    print("BMM joint action-subgoal diagnostic checks passed.")


if __name__ == "__main__":
    main()
