#!/usr/bin/env python
"""Synthetic checks for value-subgoal policy-smoke selector comparison."""

from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_subgoal_bc_controller as bc_eval
from scripts import eval_bmm_value_subgoal_policy_smoke as smoke


def make_policy(selector):
    policy = object.__new__(smoke.ValueSubgoalNNPolicy)
    policy.selector = smoke.normalize_selector(selector)
    policy.left_budget = 10
    policy.right_budget = 10
    policy.step_distances = np.asarray(
        [
            [0.0, 10.0, 2.0, 8.0],
            [10.0, 0.0, 8.0, 2.0],
            [2.0, 8.0, 0.0, 6.0],
            [8.0, 2.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    policy.rng = np.random.default_rng(0)
    policy.budgets = (10,)
    policy.value_gate_threshold = 0.5
    finite_steps = policy.step_distances[
        np.isfinite(policy.step_distances) & (policy.step_distances > 0.0)
    ]
    policy.min_step_distance = float(finite_steps.min())
    return policy


def assert_numpy_bc_matches_jax(layer_norm):
    class Args:
        bc_hidden_dims = "(8, 6)"
        bc_layer_norm = layer_norm
        bc_goal_rep = "observation"

    train_dataset = dict(
        observations=np.zeros((2, 4), dtype=np.float32),
        actions=np.zeros((2, 3), dtype=np.float32),
    )
    model, obs_dim, goal_dim, _ = bc_eval.build_bc_model(train_dataset, Args)
    params = model.init(
        jax.random.PRNGKey(0),
        np.zeros((1, obs_dim), dtype=np.float32),
        np.zeros((1, goal_dim), dtype=np.float32),
    )["params"]
    rng = np.random.default_rng(123)
    observations = rng.normal(size=(5, obs_dim)).astype(np.float32)
    goals = rng.normal(size=(5, goal_dim)).astype(np.float32)
    jax_actions = np.asarray(bc_eval.make_bc_apply(model)(params, observations, goals))
    numpy_actions = bc_eval.bc_apply_numpy(
        params, observations, goals, layer_norm=layer_norm
    )
    assert np.allclose(jax_actions, numpy_actions, atol=1e-5)


def main():
    candidate_cells = np.asarray([1, 2, 3], dtype=np.int32)
    subgoals = np.asarray(
        [
            [10.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=np.float32,
    )
    observation = np.asarray([0.0, 0.0], dtype=np.float32)
    goal = np.asarray([10.0, 0.0], dtype=np.float32)

    geometric = make_policy("geometric")
    geometric_scores = geometric.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(geometric_scores)) == 1

    oracle = make_policy("oracle_midpoint")
    oracle_scores = oracle.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(oracle_scores)) == 0

    random = make_policy("random")
    random_scores = random.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert random_scores.shape == (3,)

    assert smoke.normalize_selector("bmm-v") == "BMM_V"
    assert smoke.normalize_selector("support_path") == "support_path_only"
    assert (
        smoke.normalize_selector("bmm_value_frontier")
        == "BMM_V_min_budget_scan_value_frontier"
    )
    assert (
        smoke.normalize_selector("bmm_learned_progress")
        == "BMM_V_min_budget_scan_right_progress"
    )
    assert smoke.parse_str_list("random, geometric, BMM_V") == [
        "random",
        "geometric",
        "BMM_V",
    ]

    frontier = make_policy("bmm_value_frontier")

    def fake_score_bmm_right_budget_scan(*_args, **_kwargs):
        right_budgets = np.asarray([10], dtype=np.int32)
        left = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
        right = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
        return right_budgets, left, right

    frontier.score_bmm_right_budget_scan = fake_score_bmm_right_budget_scan
    frontier_scores = frontier.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(frontier_scores)) == 0

    progress = make_policy("bmm_learned_progress")
    progress.right_budget = 20
    progress.budgets = (10, 20)

    def fake_progress_scan(*_args, **_kwargs):
        right_budgets = np.asarray([10, 20], dtype=np.int32)
        left = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
        right = np.asarray(
            [
                [0.1, 0.2, 0.8],
                [0.9, 0.9, 0.9],
            ],
            dtype=np.float32,
        )
        return right_budgets, left, right

    progress.score_bmm_right_budget_scan = fake_progress_scan
    progress_scores = progress.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(progress_scores)) == 2

    rows = [
        dict(success=0.0, final_goal_distance=1.0, goal_distance_improvement=2.0),
        dict(success=1.0, final_goal_distance=3.0, goal_distance_improvement=4.0),
    ]
    for row in rows:
        row.update(
            steps=5,
            start_goal_distance=5.0,
            start_goal_xy_distance=5.0,
            final_goal_xy_distance=2.0,
            goal_xy_distance_improvement=3.0,
            mean_step_goal_improvement=0.5,
            mean_step_subgoal_improvement=0.25,
            subgoal_reduce_frac=1.0,
            goal_reduce_frac=0.5,
            subgoal_valid_frac=1.0,
            selected_score_mean=0.0,
            selected_source_to_subgoal=10.0,
            selected_subgoal_to_goal=10.0,
            selected_support_path_horizon=80.0,
        )
    agg = smoke.aggregate(rows)
    assert np.isclose(agg["success"], 0.5)
    assert np.isclose(agg["goal_distance_improvement"], 3.0)
    assert_numpy_bc_matches_jax(layer_norm=False)
    assert_numpy_bc_matches_jax(layer_norm=True)

    print("BMM value-subgoal policy-smoke selector checks passed.")


if __name__ == "__main__":
    main()
