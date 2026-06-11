#!/usr/bin/env python
"""Train a state-only BMM critic on fresh PointMaze geodesic labels."""

import ast
import json
from pathlib import Path
import random
import sys

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_metrics, format_metric, rank_metrics
from utils.datasets import Dataset, GCDataset
from utils.pointmaze_graph import (
    adjacency_lists,
    build_dataset_position_graph,
    dataset_xy,
    graph_distance_statistics,
    load_graph_npz,
    median_step_xy,
    parse_xy_dims,
    sample_graph_budget_pairs,
    save_graph_npz,
    source_indices,
)
from utils.pointmaze_grid import (
    free_cell_distance_matrix,
    free_cell_to_state_indices,
    grid_distance_statistics,
    sample_grid_budget_pairs,
    state_to_free_cell_indices,
    unwrap_maze_env,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional directory of OGBench npz files.")
flags.DEFINE_string(
    "reachability_label_type",
    "grid_geodesic",
    "Label type: grid_geodesic or graph.",
)
flags.DEFINE_string("graph_path", "exp/bmm_pointmaze_graph.npz", "Graph npz path.")
flags.DEFINE_bool("rebuild_graph", False, "Rebuild graph labels even if graph exists.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("budgets", "(32, 64, 96, 128)", "Budgets to train/evaluate.")
flags.DEFINE_integer("batch_size", 256, "Training pairs per budget per update.")
flags.DEFINE_integer("eval_pairs", 512, "Heldout pairs per budget.")
flags.DEFINE_integer("steps", 2000, "Training updates.")
flags.DEFINE_integer("eval_interval", 250, "Evaluate every N updates.")
flags.DEFINE_float(
    "pos_boundary_frac",
    0.5,
    "Positive distance lower bound as a fraction of budget.",
)
flags.DEFINE_float(
    "neg_max_factor",
    2.0,
    "Negative distance upper bound as a factor of budget.",
)
flags.DEFINE_float("target_auc", 0.90, "Default passing AUC threshold.")
flags.DEFINE_float("target_gap", 0.20, "Default passing score-gap threshold.")
flags.DEFINE_float("target_auc_128", 0.85, "Passing AUC threshold for H>=128.")
flags.DEFINE_float("target_gap_128", 0.15, "Passing score-gap threshold for H>=128.")
flags.DEFINE_float(
    "lambda_trans",
    0.0,
    "Optional geodesic-valid max-min transitive consistency weight.",
)
flags.DEFINE_float(
    "trans_pos_boundary_frac",
    0.5,
    "Transitive source-goal lower distance bound as a fraction of budget.",
)
flags.DEFINE_integer(
    "num_trans_witnesses",
    1,
    "Number of geodesic-valid witnesses per transitive parent.",
)
flags.DEFINE_bool(
    "fail_on_threshold",
    False,
    "Exit nonzero if final heldout metrics do not pass thresholds.",
)
flags.DEFINE_string("output_json", None, "Optional path to write final metrics.")

config_flags.DEFINE_config_file("agent", "agents/bmm_trl.py", lock_config=False)


def parse_budgets(value):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        parsed = (parsed,)
    budgets = tuple(int(x) for x in parsed)
    if not budgets:
        raise ValueError("--budgets must contain at least one budget.")
    return budgets


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def configure_agent(config, budgets):
    config.budgets = budgets
    config.max_budget = max(budgets)
    config.batch_size = FLAGS.batch_size
    config.diagnostic_critic_mode = "state"
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = FLAGS.lambda_trans
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_sup_pairs = 0
    config.num_rank_pairs = 0
    config.dataset.reachability_label_type = FLAGS.reachability_label_type
    config.dataset.graph_path = FLAGS.graph_path
    config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    config.value_hidden_dims = tuple(config.value_hidden_dims)


def make_grid_context(env, train_dataset, val_dataset, xy_dims):
    maze_env = unwrap_maze_env(env)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    steps_per_cell = float(maze_env._maze_unit) / median_step
    free_cells, cell_distances = free_cell_distance_matrix(maze_env.maze_map)
    train_state_to_cell = state_to_free_cell_indices(
        train_dataset,
        maze_env.maze_map,
        free_cells,
        xy_dims=xy_dims,
        maze_unit=maze_env._maze_unit,
        offset_x=maze_env._offset_x,
        offset_y=maze_env._offset_y,
    )
    val_state_to_cell = state_to_free_cell_indices(
        val_dataset,
        maze_env.maze_map,
        free_cells,
        xy_dims=xy_dims,
        maze_unit=maze_env._maze_unit,
        offset_x=maze_env._offset_x,
        offset_y=maze_env._offset_y,
    )
    train_goal_by_cell = free_cell_to_state_indices(
        train_state_to_cell, len(free_cells)
    )
    val_goal_by_cell = free_cell_to_state_indices(val_state_to_cell, len(free_cells))
    stats = grid_distance_statistics(cell_distances, steps_per_cell)
    return dict(
        kind="grid_geodesic",
        maze_type=maze_env._maze_type,
        maze_unit=float(maze_env._maze_unit),
        median_step_xy=float(median_step),
        steps_per_cell=float(steps_per_cell),
        free_cells=free_cells,
        cell_distances=cell_distances,
        train_state_to_cell=train_state_to_cell,
        val_state_to_cell=val_state_to_cell,
        train_goal_by_cell=train_goal_by_cell,
        val_goal_by_cell=val_goal_by_cell,
        distance_stats=stats,
    )


def make_graph_context(train_dataset, val_dataset, xy_dims):
    graph_path = Path(FLAGS.graph_path)
    if graph_path.exists() and not FLAGS.rebuild_graph:
        graph = load_graph_npz(graph_path)
    else:
        graph = build_dataset_position_graph(
            train_dataset,
            val_dataset,
            xy_dims=xy_dims,
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        save_graph_npz(graph_path, graph)
    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    stats = graph_distance_statistics(adjacency, graph)
    return dict(
        kind="graph",
        graph=graph,
        adjacency=adjacency,
        distance_stats=stats,
    )


def make_label_context(env, train_dataset, val_dataset, xy_dims):
    if FLAGS.reachability_label_type == "grid_geodesic":
        return make_grid_context(env, train_dataset, val_dataset, xy_dims)
    if FLAGS.reachability_label_type == "graph":
        return make_graph_context(train_dataset, val_dataset, xy_dims)
    raise ValueError(
        "train_bmm_geodesic_value.py supports "
        "reachability_label_type='grid_geodesic' or 'graph'."
    )


def sample_context_budget_pairs(dataset, context, split, budget, num_pairs, rng):
    if context["kind"] == "grid_geodesic":
        state_to_cell = context[f"{split}_state_to_cell"]
        goal_by_cell = context[f"{split}_goal_by_cell"]
        return sample_grid_budget_pairs(
            dataset,
            state_to_cell,
            goal_by_cell,
            context["cell_distances"],
            context["steps_per_cell"],
            int(budget),
            int(num_pairs),
            rng,
            pos_boundary_frac=FLAGS.pos_boundary_frac,
            neg_max_factor=FLAGS.neg_max_factor,
        )

    graph = context["graph"]
    state_to_bin = graph[f"{split}_state_to_bin"]
    return sample_graph_budget_pairs(
        dataset,
        state_to_bin,
        graph,
        int(budget),
        int(num_pairs),
        rng,
        pos_boundary_frac=FLAGS.pos_boundary_frac,
        neg_max_factor=FLAGS.neg_max_factor,
        adjacency=context["adjacency"],
    )


def pair_distance(row):
    if "grid_distances" in row:
        return row["grid_distances"]
    return row["graph_distances"]


def make_sup_fields(dataset, context, split, budgets, pairs_per_budget, rng):
    rows = []
    for budget in budgets:
        row = sample_context_budget_pairs(
            dataset,
            context,
            split,
            int(budget),
            int(pairs_per_budget),
            rng,
        )
        if row is None or len(row["labels"]) < pairs_per_budget:
            got = 0 if row is None else len(row["labels"])
            raise ValueError(
                f"Could not sample {pairs_per_budget} {context['kind']} pairs "
                f"for split={split}, H={budget}; got {got}."
            )
        rows.append(row)

    distances = [pair_distance(row) for row in rows]
    return dict(
        value_sup_observations=np.stack([row["observations"] for row in rows], axis=0),
        value_sup_actions=np.stack([row["actions"] for row in rows], axis=0),
        value_sup_goals=np.stack([row["goals"] for row in rows], axis=0),
        value_sup_budgets=np.stack([row["budgets"] for row in rows], axis=0),
        value_sup_offsets=np.stack(
            [np.rint(distance).astype(np.int32) for distance in distances],
            axis=0,
        ),
        value_sup_labels=np.stack([row["labels"] for row in rows], axis=0),
        value_sup_valids=np.ones((len(rows), pairs_per_budget), dtype=np.float32),
        value_sup_distances=np.stack(distances, axis=0).astype(np.float32),
    )


def sample_grid_transitive_v_pairs(dataset, context, split, budgets, batch_size, rng):
    """Sample geodesic-valid state-only transitive tuples for BMM V_H."""
    if context["kind"] != "grid_geodesic":
        raise ValueError("Geodesic transitive V sampling currently supports grid labels.")

    state_to_cell = np.asarray(context[f"{split}_state_to_cell"], dtype=np.int32)
    state_by_cell = context[f"{split}_goal_by_cell"]
    has_state = np.asarray([len(items) > 0 for items in state_by_cell])
    step_distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
        context["steps_per_cell"]
    )
    src_idxs = source_indices(dataset)
    src_idxs = src_idxs[state_to_cell[src_idxs] >= 0]
    budgets = tuple(int(x) for x in budgets)

    observations = []
    actions = []
    goals = []
    value_budgets = []
    value_offsets = []
    witness_observations = []
    witness_actions = []
    witness_goals = []
    witness_offsets = []
    left_budgets = []
    right_budgets = []
    trans_valids = []
    parent_distances = []
    left_distances = []
    right_distances = []
    left_slacks = []
    right_slacks = []
    witness_cell_counts = []
    unique_witness_fracs = []
    trans_parent_oracle_labels = []
    trans_branch_oracle_valids = []
    num_witnesses = int(FLAGS.num_trans_witnesses)
    if num_witnesses < 1:
        raise ValueError("--num_trans_witnesses must be >= 1.")

    attempts = 0
    max_attempts = max(1000, int(batch_size) * 200)
    while len(observations) < int(batch_size) and attempts < max_attempts:
        attempts += 1
        budget = int(rng.choice(budgets))
        if budget <= 1:
            continue
        left_budget = max(1, budget // 2)
        right_budget = max(1, budget - left_budget)
        src_idx = int(rng.choice(src_idxs))
        src_cell = int(state_to_cell[src_idx])
        src_distances = step_distances[src_cell]
        finite_goal = (context["cell_distances"][src_cell] >= 0) & has_state
        goal_lo = max(0.0, float(FLAGS.trans_pos_boundary_frac) * float(budget))
        goal_mask = (
            finite_goal
            & (src_distances >= goal_lo)
            & (src_distances <= float(budget))
        )
        if not goal_mask.any() and goal_lo > 0.0:
            goal_mask = finite_goal & (src_distances <= float(budget))
        goal_cells = np.nonzero(goal_mask)[0]
        if len(goal_cells) == 0:
            continue

        goal_cell = int(rng.choice(goal_cells))
        witness_mask = (
            has_state
            & (context["cell_distances"][src_cell] >= 0)
            & (context["cell_distances"][:, goal_cell] >= 0)
            & (src_distances <= float(left_budget))
            & (step_distances[:, goal_cell] <= float(right_budget))
        )
        witness_cells = np.nonzero(witness_mask)[0]
        if len(witness_cells) == 0:
            continue

        replace = len(witness_cells) < num_witnesses
        sampled_witness_cells = rng.choice(
            witness_cells, size=num_witnesses, replace=replace
        )
        goal_idx = int(rng.choice(state_by_cell[goal_cell]))
        observations.append(np.asarray(dataset["observations"])[src_idx])
        actions.append(np.asarray(dataset["actions"])[src_idx])
        goals.append(np.asarray(dataset["observations"])[goal_idx])
        value_budgets.append(budget)
        parent_distance = float(src_distances[goal_cell])
        value_offsets.append(parent_distance)
        parent_distances.append(parent_distance)
        witness_cell_counts.append(float(len(witness_cells)))
        unique_witness_fracs.append(
            float(len(np.unique(sampled_witness_cells)) / float(num_witnesses))
        )
        trans_parent_oracle_labels.append(float(parent_distance <= float(budget)))

        parent_witness_observations = []
        parent_witness_actions = []
        parent_witness_goals = []
        parent_witness_offsets = []
        parent_left_budgets = []
        parent_right_budgets = []
        parent_valids = []
        parent_left_distances = []
        parent_right_distances = []
        parent_left_slacks = []
        parent_right_slacks = []
        parent_branch_oracle_valids = []
        for witness_cell in sampled_witness_cells:
            witness_cell = int(witness_cell)
            witness_idx = int(rng.choice(state_by_cell[witness_cell]))
            left_distance = float(src_distances[witness_cell])
            right_distance = float(step_distances[witness_cell, goal_cell])
            parent_witness_observations.append(
                np.asarray(dataset["observations"])[witness_idx]
            )
            parent_witness_actions.append(np.asarray(dataset["actions"])[witness_idx])
            parent_witness_goals.append(np.asarray(dataset["observations"])[witness_idx])
            parent_witness_offsets.append(left_distance)
            parent_left_budgets.append(left_budget)
            parent_right_budgets.append(right_budget)
            parent_valids.append(1.0)
            parent_left_distances.append(left_distance)
            parent_right_distances.append(right_distance)
            parent_left_slacks.append(float(left_budget) - left_distance)
            parent_right_slacks.append(float(right_budget) - right_distance)
            parent_branch_oracle_valids.append(
                float(
                    left_distance <= float(left_budget)
                    and right_distance <= float(right_budget)
                )
            )

        witness_observations.append(parent_witness_observations)
        witness_actions.append(parent_witness_actions)
        witness_goals.append(parent_witness_goals)
        witness_offsets.append(parent_witness_offsets)
        left_budgets.append(parent_left_budgets)
        right_budgets.append(parent_right_budgets)
        trans_valids.append(parent_valids)
        left_distances.append(parent_left_distances)
        right_distances.append(parent_right_distances)
        left_slacks.append(parent_left_slacks)
        right_slacks.append(parent_right_slacks)
        trans_branch_oracle_valids.append(parent_branch_oracle_valids)

    if len(observations) < int(batch_size):
        raise ValueError(
            f"Could not sample {batch_size} grid transitive tuples for split={split}; "
            f"got {len(observations)} after {attempts} attempts."
        )

    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        value_goals=np.asarray(goals, dtype=np.float32),
        value_budgets=np.asarray(value_budgets, dtype=np.int32),
        value_offsets=np.rint(value_offsets).astype(np.int32),
        value_midpoint_observations=np.swapaxes(
            np.asarray(witness_observations, dtype=np.float32), 0, 1
        ),
        value_midpoint_actions=np.swapaxes(
            np.asarray(witness_actions, dtype=np.float32), 0, 1
        ),
        value_midpoint_goals=np.swapaxes(
            np.asarray(witness_goals, dtype=np.float32), 0, 1
        ),
        value_midpoint_offsets=np.rint(
            np.swapaxes(np.asarray(witness_offsets, dtype=np.float32), 0, 1)
        ).astype(np.int32),
        value_left_budgets=np.swapaxes(np.asarray(left_budgets, dtype=np.int32), 0, 1),
        value_right_budgets=np.swapaxes(
            np.asarray(right_budgets, dtype=np.int32), 0, 1
        ),
        trans_valids=np.swapaxes(np.asarray(trans_valids, dtype=np.float32), 0, 1),
        trans_parent_distances=np.asarray(parent_distances, dtype=np.float32),
        trans_left_distances=np.swapaxes(
            np.asarray(left_distances, dtype=np.float32), 0, 1
        ),
        trans_right_distances=np.swapaxes(
            np.asarray(right_distances, dtype=np.float32), 0, 1
        ),
        trans_left_slacks=np.swapaxes(np.asarray(left_slacks, dtype=np.float32), 0, 1),
        trans_right_slacks=np.swapaxes(
            np.asarray(right_slacks, dtype=np.float32), 0, 1
        ),
        trans_witness_cell_counts=np.asarray(witness_cell_counts, dtype=np.float32),
        trans_unique_witness_fracs=np.asarray(unique_witness_fracs, dtype=np.float32),
        trans_parent_oracle_labels=np.asarray(
            trans_parent_oracle_labels, dtype=np.float32
        ),
        trans_branch_oracle_valids=np.swapaxes(
            np.asarray(trans_branch_oracle_valids, dtype=np.float32), 0, 1
        ),
        trans_sample_acceptance_rate=np.asarray(
            float(len(observations)) / float(max(attempts, 1)), dtype=np.float32
        ),
        trans_attempts_per_sample=np.asarray(
            float(attempts) / float(max(len(observations), 1)), dtype=np.float32
        ),
    )


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(values[finite].mean())


def coarse_hist(values, bins):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=np.asarray(bins, dtype=np.float64))
    return dict(
        counts=[int(x) for x in counts],
        edges=[float(x) for x in edges],
        total=int(len(finite)),
    )


def summarize_transitive_batch(trans_batch, budgets):
    if trans_batch is None:
        return None
    value_budgets = np.asarray(trans_batch["value_budgets"])
    parent_distances = np.asarray(trans_batch["trans_parent_distances"])
    left_distances = np.asarray(trans_batch["trans_left_distances"])
    right_distances = np.asarray(trans_batch["trans_right_distances"])
    left_slacks = np.asarray(trans_batch["trans_left_slacks"])
    right_slacks = np.asarray(trans_batch["trans_right_slacks"])
    left_budgets = np.asarray(trans_batch["value_left_budgets"])
    right_budgets = np.asarray(trans_batch["value_right_budgets"])
    trans_valids = np.asarray(trans_batch["trans_valids"]) > 0

    rows = []
    histograms = {}
    ratio_bins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf]
    slack_bins = [-np.inf, -1e-6, 0.0, 5.0, 10.0, 20.0, 40.0, np.inf]
    for budget in budgets:
        budget = int(budget)
        parent_mask = value_budgets == budget
        witness_mask = trans_valids & parent_mask[None, :]
        if not parent_mask.any():
            continue
        left_budget = max(1, budget // 2)
        right_budget = max(1, budget - left_budget)
        rows.append(
            dict(
                budget=budget,
                count=int(parent_mask.sum()),
                trans_budget_count=int(parent_mask.sum()),
                parent_distance_mean=finite_mean(parent_distances[parent_mask]),
                left_distance_mean=finite_mean(left_distances[witness_mask]),
                right_distance_mean=finite_mean(right_distances[witness_mask]),
                left_slack_mean=finite_mean(left_slacks[witness_mask]),
                right_slack_mean=finite_mean(right_slacks[witness_mask]),
                witness_cell_count_mean=finite_mean(
                    np.asarray(trans_batch["trans_witness_cell_counts"])[parent_mask]
                ),
                unique_witness_frac=finite_mean(
                    np.asarray(trans_batch["trans_unique_witness_fracs"])[parent_mask]
                ),
                zero_left_frac=finite_mean(
                    (left_distances[witness_mask] <= 1e-6).astype(np.float32)
                ),
                zero_right_frac=finite_mean(
                    (right_distances[witness_mask] <= 1e-6).astype(np.float32)
                ),
                parent_oracle_label_mean=finite_mean(
                    np.asarray(trans_batch["trans_parent_oracle_labels"])[parent_mask]
                ),
                branch_oracle_valid_mean=finite_mean(
                    np.asarray(trans_batch["trans_branch_oracle_valids"])[witness_mask]
                ),
            )
        )
        histograms[str(budget)] = dict(
            parent_distance_over_H=coarse_hist(
                parent_distances[parent_mask] / float(max(budget, 1)), ratio_bins
            ),
            left_distance_over_h=coarse_hist(
                left_distances[witness_mask] / float(max(left_budget, 1)), ratio_bins
            ),
            right_distance_over_H_minus_h=coarse_hist(
                right_distances[witness_mask] / float(max(right_budget, 1)), ratio_bins
            ),
            left_slack=coarse_hist(left_slacks[witness_mask], slack_bins),
            right_slack=coarse_hist(right_slacks[witness_mask], slack_bins),
        )

    return dict(
        trans_sample_acceptance_rate=float(
            np.asarray(trans_batch["trans_sample_acceptance_rate"])
        ),
        trans_attempts_per_sample=float(
            np.asarray(trans_batch["trans_attempts_per_sample"])
        ),
        num_trans_witnesses=int(np.asarray(trans_valids).shape[0]),
        budget_rows=rows,
        histograms=histograms,
    )


def info_to_float_dict(info):
    result = {}
    for key, value in info.items():
        try:
            arr = np.asarray(value)
            if arr.shape == ():
                result[key] = float(arr)
        except (TypeError, ValueError):
            pass
    return result


def score_sup_batch(agent, batch, budgets):
    logits = agent.critic_logits_for_pair_grid(
        batch["value_sup_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    scores = np.asarray(jax.nn.sigmoid(logits))
    mean_scores = scores.mean(axis=0)
    min_scores = scores.min(axis=0)
    labels = np.asarray(batch["value_sup_labels"])
    valids = np.asarray(batch["value_sup_valids"]) > 0
    sup_budgets = np.asarray(batch["value_sup_budgets"])
    distances = np.asarray(batch["value_sup_distances"])
    observations = np.asarray(batch["value_sup_observations"])
    goals = np.asarray(batch["value_sup_goals"])

    report = {
        "mean": binary_metrics(mean_scores[valids], labels[valids]),
        "ensemble_min": binary_metrics(min_scores[valids], labels[valids]),
        "budget_rows": [],
    }
    for budget in budgets:
        mask = valids & (sup_budgets == int(budget))
        if not mask.any():
            continue
        obs = observations[mask]
        goal = goals[mask]
        euclidean = np.linalg.norm(goal[:, :2] - obs[:, :2], axis=-1)
        row_labels = labels[mask]
        row_distances = distances[mask]
        report["budget_rows"].append(
            {
                "budget": int(budget),
                "mean": binary_metrics(mean_scores[mask], row_labels),
                "ensemble_min": binary_metrics(min_scores[mask], row_labels),
                "baselines": {
                    "distance_oracle": rank_metrics(-row_distances, row_labels),
                    "euclidean": rank_metrics(-euclidean, row_labels),
                },
            }
        )
    return report


def monotonicity_violation(agent, batch, budgets, max_pairs=2048):
    observations = np.asarray(batch["value_sup_observations"]).reshape(
        -1, batch["value_sup_observations"].shape[-1]
    )
    actions = np.asarray(batch["value_sup_actions"]).reshape(
        -1, batch["value_sup_actions"].shape[-1]
    )
    goals = np.asarray(batch["value_sup_goals"]).reshape(
        -1, batch["value_sup_goals"].shape[-1]
    )
    count = min(len(observations), int(max_pairs))
    observations = observations[:count]
    actions = actions[:count]
    goals = goals[:count]

    score_rows = []
    for budget in budgets:
        budget_arr = np.full(count, int(budget), dtype=np.int32)
        logits = agent.critic_logits_for(
            observations,
            actions,
            goals,
            budget_arr,
            offsets=budget_arr,
        )
        score_rows.append(np.asarray(jax.nn.sigmoid(logits)).mean(axis=0))
    scores = np.stack(score_rows, axis=0)
    if len(budgets) < 2:
        return 0.0
    return float((scores[:-1] > scores[1:] + 1e-6).mean())


def threshold_for_budget(budget):
    if int(budget) >= 128:
        return FLAGS.target_auc_128, FLAGS.target_gap_128
    return FLAGS.target_auc, FLAGS.target_gap


def passed(report, budgets):
    rows_by_budget = {row["budget"]: row for row in report["budget_rows"]}
    for budget in budgets:
        row = rows_by_budget.get(int(budget))
        if row is None:
            return False
        min_auc, min_gap = threshold_for_budget(budget)
        for key in ("mean", "ensemble_min"):
            metrics = row[key]
            if metrics["pos_count"] == 0 or metrics["neg_count"] == 0:
                return False
            gap = metrics["pos_mean"] - metrics["neg_mean"]
            if metrics["auc"] < min_auc or gap < min_gap:
                return False
    return True


def print_report(title, step, report, mono=None, loss=None):
    loss_text = "" if loss is None else f" loss={format_metric(loss)}"
    mono_text = "" if mono is None else f" mono={format_metric(mono)}"
    print(f"\n{title} step={step}{loss_text}{mono_text}")
    print(
        "H | auc | gap | pos | neg | min_auc | min_gap | dist_auc | euc_auc | pos_n | neg_n"
    )
    print(
        "--|-----|-----|-----|-----|---------|---------|----------|---------|-------|------"
    )
    for row in report["budget_rows"]:
        mean = row["mean"]
        ens_min = row["ensemble_min"]
        gap = mean["pos_mean"] - mean["neg_mean"]
        min_gap = ens_min["pos_mean"] - ens_min["neg_mean"]
        baselines = row["baselines"]
        print(
            f"{row['budget']:4d} | {format_metric(mean['auc'])} | "
            f"{format_metric(gap)} | {format_metric(mean['pos_mean'])} | "
            f"{format_metric(mean['neg_mean'])} | {format_metric(ens_min['auc'])} | "
            f"{format_metric(min_gap)} | "
            f"{format_metric(baselines['distance_oracle']['auc'])} | "
            f"{format_metric(baselines['euclidean']['auc'])} | "
            f"{mean['pos_count']:5d} | {mean['neg_count']:5d}"
        )


def print_transitive_summary(summary, info):
    if summary is None:
        return
    ratio = info.get("critic/loss_trans_over_sup", np.nan)
    print(
        "trans | "
        f"accept={format_metric(summary['trans_sample_acceptance_rate'])} | "
        f"attempts/sample={format_metric(summary['trans_attempts_per_sample'])} | "
        f"K={summary['num_trans_witnesses']} | "
        f"loss_trans/sup={format_metric(ratio)}"
    )
    print(
        "H | parents | parent_d | left_d | right_d | left_slack | right_slack | "
        "w_cells | uniq_w | zero_l | zero_r"
    )
    print(
        "--|---------|----------|--------|---------|------------|-------------|"
        "---------|--------|--------|-------"
    )
    for row in summary["budget_rows"]:
        print(
            f"{row['budget']:4d} | {row['trans_budget_count']:7d} | "
            f"{format_metric(row['parent_distance_mean'])} | "
            f"{format_metric(row['left_distance_mean'])} | "
            f"{format_metric(row['right_distance_mean'])} | "
            f"{format_metric(row['left_slack_mean'])} | "
            f"{format_metric(row['right_slack_mean'])} | "
            f"{format_metric(row['witness_cell_count_mean'])} | "
            f"{format_metric(row['unique_witness_frac'])} | "
            f"{format_metric(row['zero_left_frac'])} | "
            f"{format_metric(row['zero_right_frac'])}"
        )


def context_metadata(context):
    if context["kind"] == "grid_geodesic":
        return dict(
            kind=context["kind"],
            maze_type=context["maze_type"],
            maze_unit=context["maze_unit"],
            median_step_xy=context["median_step_xy"],
            steps_per_cell=context["steps_per_cell"],
            free_cell_count=int(len(context["free_cells"])),
            distance_stats=context["distance_stats"],
        )
    graph = context["graph"]
    return dict(
        kind=context["kind"],
        graph_metadata=graph["metadata"],
        graph_path=FLAGS.graph_path,
        node_count=int(len(graph["bin_centers"])),
        edge_count=int(len(graph["edge_src"])),
        distance_stats=context["distance_stats"],
    )


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed)
    config = FLAGS.agent
    if config["agent_name"] != "bmm_trl":
        raise ValueError("train_bmm_geodesic_value.py requires bmm_trl.")
    budgets = parse_budgets(FLAGS.budgets)
    configure_agent(config, budgets)

    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    context = make_label_context(env, train_dataset, val_dataset, xy_dims)
    print("BMM geodesic value diagnostic")
    print(f"  env_name: {FLAGS.env_name}")
    print(f"  label_type: {context['kind']}")
    print(f"  budgets: {budgets}")
    print(f"  lambda_trans: {FLAGS.lambda_trans}")
    print(f"  num_trans_witnesses: {FLAGS.num_trans_witnesses}")
    print(f"  context: {context_metadata(context)}")

    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    gc_train = dataset_class(Dataset.create(**train_dataset), config)
    example_batch = gc_train.sample(1)
    agent = agents[config["agent_name"]].create(FLAGS.seed, example_batch, config)

    eval_batch = make_sup_fields(
        val_dataset,
        context,
        "val",
        budgets,
        FLAGS.eval_pairs,
        rng,
    )
    final_loss = None
    train_report = None
    eval_report = score_sup_batch(agent, eval_batch, budgets)
    eval_mono = monotonicity_violation(agent, eval_batch, budgets)
    print_report("eval", 0, eval_report, mono=eval_mono)
    transitive_history = []
    last_update_info = {}
    last_transitive_summary = None

    for step in range(1, FLAGS.steps + 1):
        train_batch = gc_train.sample(config.batch_size)
        transitive_fields = None
        if FLAGS.lambda_trans > 0.0:
            transitive_fields = sample_grid_transitive_v_pairs(
                train_dataset,
                context,
                "train",
                budgets,
                config.batch_size,
                rng,
            )
            train_batch.update(transitive_fields)
        train_batch.update(
            make_sup_fields(
                train_dataset,
                context,
                "train",
                budgets,
                config.batch_size,
                rng,
            )
        )
        agent, info = agent.update(train_batch)
        last_update_info = info_to_float_dict(info)
        final_loss = float(info["critic/loss_sup"])
        if step % FLAGS.eval_interval == 0 or step == FLAGS.steps:
            train_report = score_sup_batch(agent, train_batch, budgets)
            eval_report = score_sup_batch(agent, eval_batch, budgets)
            train_mono = monotonicity_violation(agent, train_batch, budgets)
            eval_mono = monotonicity_violation(agent, eval_batch, budgets)
            last_transitive_summary = summarize_transitive_batch(
                transitive_fields, budgets
            )
            if last_transitive_summary is not None:
                transitive_history.append(
                    dict(step=int(step), train=last_transitive_summary)
                )
            print_report("train", step, train_report, mono=train_mono, loss=final_loss)
            print_transitive_summary(last_transitive_summary, last_update_info)
            print_report("eval", step, eval_report, mono=eval_mono)

    final_passed = passed(eval_report, budgets)
    print(f"\nFinal heldout threshold pass: {final_passed}")
    final_report = dict(
        train=train_report,
        eval=eval_report,
        eval_monotonicity_violation=eval_mono,
        last_update_info=last_update_info,
        last_transitive_summary=last_transitive_summary,
        transitive_history=transitive_history,
        passed=bool(final_passed),
        config=dict(
            env_name=FLAGS.env_name,
            reachability_label_type=FLAGS.reachability_label_type,
            budgets=[int(x) for x in budgets],
            batch_size=int(FLAGS.batch_size),
            eval_pairs=int(FLAGS.eval_pairs),
            steps=int(FLAGS.steps),
            eval_interval=int(FLAGS.eval_interval),
            final_loss_sup=final_loss,
            lambda_trans=float(FLAGS.lambda_trans),
            trans_pos_boundary_frac=float(FLAGS.trans_pos_boundary_frac),
            num_trans_witnesses=int(FLAGS.num_trans_witnesses),
            context=context_metadata(context),
        ),
    )
    if FLAGS.output_json is not None:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nWrote geodesic value report to {output_path}")

    if FLAGS.fail_on_threshold and not final_passed:
        raise SystemExit("Final heldout geodesic value metrics did not pass.")


if __name__ == "__main__":
    app.run(main)
