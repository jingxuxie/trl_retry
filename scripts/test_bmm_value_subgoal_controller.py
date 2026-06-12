#!/usr/bin/env python
"""Synthetic checks for the value-subgoal controller diagnostic."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_value_subgoal_controller as diag


def main():
    batch = {
        "source_cells": np.asarray([0, 0], dtype=np.int32),
        "goal_cells": np.asarray([2, 2], dtype=np.int32),
        "subgoal_cells": np.asarray([[1, 2], [2, 1]], dtype=np.int32),
    }
    step_distances = np.asarray(
        [
            [0.0, 10.0, 8.0],
            [10.0, 0.0, 4.0],
            [8.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    controller = {
        "step_distances": step_distances,
        "state_to_cell": np.asarray([0, 2, 0, 2], dtype=np.int32),
        "neighbor_pools": [
            np.asarray([0, 2], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
        ],
    }
    scores = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = diag.nn_controller_metrics(scores, batch, controller)
    assert np.isclose(metrics["nn_valid_frac"], 1.0)
    assert np.isclose(metrics["nn_query_source_distance"], 10.0)
    assert np.isclose(metrics["nn_next_distance"], 4.0)
    assert np.isclose(metrics["nn_query_improvement"], 6.0)
    assert np.isclose(metrics["nn_query_reduces_frac"], 1.0)

    missing_controller = dict(controller)
    missing_controller["neighbor_pools"] = [np.asarray([], dtype=np.int32)] * 3
    missing = diag.nn_controller_metrics(scores, batch, missing_controller)
    assert np.isclose(missing["nn_valid_frac"], 0.0)
    assert np.isnan(missing["nn_query_improvement"])

    low_level = diag.low_level_controller_metrics(scores, batch, controller)
    assert np.isclose(low_level["local_progress_max_valid_frac"], 1.0)
    assert np.isclose(low_level["local_progress_max_subgoal_improvement"], 6.0)
    assert np.isclose(low_level["local_progress_max_goal_improvement"], 8.0)
    assert np.isclose(low_level["local_progress_max_subgoal_reduces_frac"], 1.0)
    assert np.isclose(low_level["direct_goal_same_cell_goal_improvement"], 8.0)
    assert np.isclose(low_level["random_same_cell_subgoal_improvement"], 6.0)

    print("BMM value-subgoal controller diagnostic checks passed.")


if __name__ == "__main__":
    main()
