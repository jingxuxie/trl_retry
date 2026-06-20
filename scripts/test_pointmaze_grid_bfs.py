#!/usr/bin/env python
"""Synthetic checks for PointMaze grid BFS utilities."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.datasets import Dataset
from utils.pointmaze_grid import (
    free_cell_to_state_indices,
    free_cell_distance_matrix,
    grid_distance_statistics,
    ij_to_xy,
    sample_grid_budget_pairs,
    sample_grid_budget_q_pairs,
    state_to_free_cell_indices,
    xy_pair_grid_distances,
)


def main():
    maze_map = np.asarray(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    free_cells, cell_distances = free_cell_distance_matrix(maze_map)
    stats = grid_distance_statistics(cell_distances, steps_per_cell=10.0)
    assert stats["max_cells"] == 6
    assert stats["max_steps"] == 60.0

    starts = np.asarray([[-4.0 + 1 * 4.0, -4.0 + 1 * 4.0]], dtype=np.float32)
    goals = np.asarray([[-4.0 + 3 * 4.0, -4.0 + 1 * 4.0]], dtype=np.float32)
    distances = xy_pair_grid_distances(
        starts,
        goals,
        maze_map,
        free_cells,
        cell_distances,
        steps_per_cell=1.0,
    )
    assert distances.shape == (1,)
    assert distances[0] == 6.0

    observations = ij_to_xy(free_cells).astype(np.float32)
    actions = np.zeros((len(observations), 2), dtype=np.float32)
    terminals = np.zeros(len(observations), dtype=np.float32)
    terminals[-1] = 1.0
    valids = np.ones(len(observations), dtype=np.float32)
    dataset = Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
        valids=valids,
    )
    state_to_cell = state_to_free_cell_indices(dataset, maze_map, free_cells)
    goal_by_cell = free_cell_to_state_indices(state_to_cell, len(free_cells))
    pair_batch = sample_grid_budget_pairs(
        dataset,
        state_to_cell,
        goal_by_cell,
        cell_distances,
        steps_per_cell=1.0,
        budget=2,
        num_pairs=64,
        rng=np.random.default_rng(0),
    )
    assert pair_batch is not None
    assert pair_batch["labels"].sum() > 0
    assert (pair_batch["labels"] == 0).sum() > 0
    assert pair_batch["source_idxs"].shape == pair_batch["labels"].shape
    assert pair_batch["goal_idxs"].shape == pair_batch["labels"].shape
    assert np.all(pair_batch["grid_distances"][pair_batch["labels"] == 1] <= 2)
    assert np.all(pair_batch["grid_distances"][pair_batch["labels"] == 0] > 2)

    q_pair_batch = sample_grid_budget_q_pairs(
        dataset,
        state_to_cell,
        goal_by_cell,
        cell_distances,
        steps_per_cell=1.0,
        budget=3,
        num_pairs=64,
        rng=np.random.default_rng(1),
    )
    assert q_pair_batch is not None
    assert q_pair_batch["observations"].shape == q_pair_batch["goals"].shape
    assert q_pair_batch["actions"].shape[0] == len(q_pair_batch["labels"])
    assert np.all(q_pair_batch["source_idxs"] + 1 < len(observations))
    assert np.allclose(
        q_pair_batch["next_observations"],
        observations[q_pair_batch["source_idxs"] + 1],
    )
    assert np.all(q_pair_batch["grid_distances"][q_pair_batch["labels"] == 1] <= 2)
    assert np.all(q_pair_batch["grid_distances"][q_pair_batch["labels"] == 0] > 2)
    assert q_pair_batch["labels"].sum() > 0
    assert (q_pair_batch["labels"] == 0).sum() > 0

    print("PointMaze grid BFS checks passed.")


if __name__ == "__main__":
    main()
