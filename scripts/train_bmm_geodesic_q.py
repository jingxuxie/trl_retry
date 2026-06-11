#!/usr/bin/env python
"""Train an action-conditioned BMM critic on fresh PointMaze geodesic Q labels."""

import ast
import copy
import json
from pathlib import Path
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from agents.bmm_trl import masked_mean
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_metrics, format_metric, rank_metrics
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from utils.pointmaze_graph import (
    adjacency_lists,
    bin_to_state_indices,
    build_dataset_position_graph,
    dataset_xy,
    graph_step_distances,
    graph_distance_statistics,
    load_graph_npz,
    median_step_xy,
    parse_xy_dims,
    save_graph_npz,
    shortest_hop_distances,
    valid_transition_indices,
)
from utils.pointmaze_grid import (
    free_cell_distance_matrix,
    free_cell_to_state_indices,
    grid_distance_statistics,
    sample_grid_budget_q_pairs,
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
flags.DEFINE_enum(
    "geodesic_budget_unit",
    "env_steps",
    ["env_steps", "grid_cells"],
    "Budget units for grid_geodesic labels.",
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("budgets", "(32, 64, 96, 128)", "Budgets to train/evaluate.")
flags.DEFINE_string(
    "trans_budgets",
    None,
    "Optional budgets for Q/V transitive parents; defaults to --budgets.",
)
flags.DEFINE_integer("batch_size", 256, "Training pairs per budget per update.")
flags.DEFINE_integer(
    "sup_pairs_per_budget",
    0,
    "Direct supervised Q pairs per budget per update; <=0 uses --batch_size.",
)
flags.DEFINE_integer(
    "trans_pairs_per_update",
    0,
    "Q/V transitive parent tuples per update; <=0 uses --batch_size.",
)
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
    "lambda_qv_trans",
    0.0,
    "Q/V max-min transitive loss weight using a frozen state-value teacher.",
)
flags.DEFINE_enum(
    "qv_trans_loss_type",
    "bce_equal",
    ["bce_equal", "prob_hinge", "bce_lower_bound"],
    "Q/V transitive loss: equality BCE or lower-bound consistency variants.",
)
flags.DEFINE_float(
    "qv_trans_bce_margin",
    0.0,
    "Probability margin for gated BCE lower-bound Q/V transitive loss.",
)
flags.DEFINE_float(
    "trans_pos_boundary_frac",
    0.5,
    "Q/V transitive parent lower distance bound as a fraction of budget.",
)
flags.DEFINE_integer(
    "num_trans_witnesses",
    1,
    "Number of witnesses per Q/V transitive parent.",
)
flags.DEFINE_enum(
    "trans_witness_mode",
    "avoid_endpoints",
    ["uniform_valid", "avoid_endpoints", "slack_balanced", "boundary_balanced"],
    "How to choose valid Q/V transitive witnesses.",
)
flags.DEFINE_float(
    "trans_endpoint_epsilon",
    1e-6,
    "Distance threshold below which a transitive witness is treated as an endpoint.",
)
flags.DEFINE_float(
    "trans_boundary_beta",
    0.25,
    "Minimum branch-distance fraction for boundary_balanced witness sampling.",
)
flags.DEFINE_string(
    "value_restore_path",
    None,
    "Optional state-value critic checkpoint path for Q-V_next consistency.",
)
flags.DEFINE_integer(
    "value_restore_epoch",
    None,
    "Optional state-value critic checkpoint epoch for Q-V_next consistency.",
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


def positive_or_default(value, default):
    value = int(value)
    if value <= 0:
        return int(default)
    return value


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
    config.diagnostic_critic_mode = "action"
    config.value_only = True
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
    config.dataset.reachability_label_type = FLAGS.reachability_label_type
    config.dataset.graph_path = FLAGS.graph_path
    config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    config.value_hidden_dims = tuple(config.value_hidden_dims)


def make_grid_context(env, train_dataset, val_dataset, xy_dims):
    maze_env = unwrap_maze_env(env)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    steps_per_cell = float(maze_env._maze_unit) / median_step
    label_distance_scale = (
        1.0 if FLAGS.geodesic_budget_unit == "grid_cells" else steps_per_cell
    )
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
    stats = grid_distance_statistics(cell_distances, label_distance_scale)
    return dict(
        kind="grid_geodesic",
        geodesic_budget_unit=str(FLAGS.geodesic_budget_unit),
        maze_type=maze_env._maze_type,
        maze_unit=float(maze_env._maze_unit),
        median_step_xy=float(median_step),
        steps_per_cell=float(steps_per_cell),
        label_distance_scale=float(label_distance_scale),
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
        "train_bmm_geodesic_q.py supports "
        "reachability_label_type='grid_geodesic' or 'graph'."
    )


def sample_graph_budget_q_pairs(
    dataset,
    state_to_bin,
    graph,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
    adjacency=None,
):
    """Sample balanced graph-distance Q pairs labeled from the next state."""
    budget = int(budget)
    rng = np.random.default_rng() if rng is None else rng
    adjacency = (
        adjacency
        if adjacency is not None
        else adjacency_lists(
            len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
        )
    )
    state_to_bin = np.asarray(state_to_bin, dtype=np.int32)
    src_idxs = valid_transition_indices(dataset)
    src_idxs = src_idxs[src_idxs + 1 < len(state_to_bin)]
    src_idxs = src_idxs[
        (state_to_bin[src_idxs] >= 0) & (state_to_bin[src_idxs + 1] >= 0)
    ]
    goal_by_bin = bin_to_state_indices(state_to_bin, len(graph["bin_centers"]))
    has_goal = np.asarray([len(items) > 0 for items in goal_by_bin])
    remaining_budget = max(float(budget - 1), 1.0)
    distance_cache = {}

    observations = []
    actions = []
    next_observations = []
    goals = []
    budgets = []
    remaining_budgets = []
    labels = []
    graph_distances = []
    source_bins = []
    next_bins = []
    goal_bins = []
    source_idxs_out = []

    def distances_for_bin(bin_idx):
        bin_idx = int(bin_idx)
        if bin_idx not in distance_cache:
            hops = shortest_hop_distances(adjacency, bin_idx)
            distance_cache[bin_idx] = graph_step_distances(hops, graph)
        return distance_cache[bin_idx]

    def add_pairs(target_label, target_count):
        attempts = 0
        max_attempts = max(1000, int(target_count) * 100)
        while target_count > 0 and attempts < max_attempts:
            attempts += 1
            src_idx = int(rng.choice(src_idxs))
            src_bin = int(state_to_bin[src_idx])
            next_bin = int(state_to_bin[src_idx + 1])
            distances = distances_for_bin(next_bin)
            finite = np.isfinite(distances) & has_goal
            if target_label == 1.0:
                lo = max(0.0, float(pos_boundary_frac) * remaining_budget)
                hi = remaining_budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any() and lo > 0.0:
                    candidate_mask = finite & (distances <= hi)
            else:
                lo = np.nextafter(remaining_budget, np.inf)
                hi = float(neg_max_factor) * remaining_budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any():
                    candidate_mask = finite & (distances > remaining_budget)

            candidate_bins = np.nonzero(candidate_mask)[0]
            if len(candidate_bins) == 0:
                continue
            goal_bin = int(rng.choice(candidate_bins))
            goal_idx = int(rng.choice(goal_by_bin[goal_bin]))
            distance = float(distances[goal_bin])
            observations.append(np.asarray(dataset["observations"])[src_idx])
            actions.append(np.asarray(dataset["actions"])[src_idx])
            next_observations.append(np.asarray(dataset["observations"])[src_idx + 1])
            goals.append(np.asarray(dataset["observations"])[goal_idx])
            budgets.append(budget)
            remaining_budgets.append(remaining_budget)
            labels.append(float(distance <= remaining_budget))
            graph_distances.append(distance)
            source_bins.append(src_bin)
            next_bins.append(next_bin)
            goal_bins.append(goal_bin)
            source_idxs_out.append(src_idx)
            target_count -= 1

    num_pos = int(num_pairs) // 2
    num_neg = int(num_pairs) - num_pos
    add_pairs(1.0, num_pos)
    add_pairs(0.0, num_neg)

    if len(labels) == 0:
        return None
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        budgets=np.asarray(budgets, dtype=np.int32),
        remaining_budgets=np.asarray(remaining_budgets, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        graph_distances=np.asarray(graph_distances, dtype=np.float32),
        source_bins=np.asarray(source_bins, dtype=np.int32),
        next_bins=np.asarray(next_bins, dtype=np.int32),
        goal_bins=np.asarray(goal_bins, dtype=np.int32),
        source_idxs=np.asarray(source_idxs_out, dtype=np.int32),
    )


def sample_context_budget_pairs(dataset, context, split, budget, num_pairs, rng):
    if context["kind"] == "grid_geodesic":
        state_to_cell = context[f"{split}_state_to_cell"]
        goal_by_cell = context[f"{split}_goal_by_cell"]
        return sample_grid_budget_q_pairs(
            dataset,
            state_to_cell,
            goal_by_cell,
            context["cell_distances"],
            context["label_distance_scale"],
            int(budget),
            int(num_pairs),
            rng,
            pos_boundary_frac=FLAGS.pos_boundary_frac,
            neg_max_factor=FLAGS.neg_max_factor,
        )

    graph = context["graph"]
    state_to_bin = graph[f"{split}_state_to_bin"]
    return sample_graph_budget_q_pairs(
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
        has_both_classes = (
            row is not None
            and np.any(row["labels"] == 1.0)
            and np.any(row["labels"] == 0.0)
        )
        if row is None or len(row["labels"]) < pairs_per_budget or not has_both_classes:
            got = 0 if row is None else len(row["labels"])
            raise ValueError(
                f"Could not sample {pairs_per_budget} {context['kind']} pairs "
                f"with both classes for split={split}, H={budget}; got {got}."
            )
        rows.append(row)

    distances = [pair_distance(row) for row in rows]
    return dict(
        value_sup_observations=np.stack([row["observations"] for row in rows], axis=0),
        value_sup_actions=np.stack([row["actions"] for row in rows], axis=0),
        value_sup_next_observations=np.stack(
            [row["next_observations"] for row in rows], axis=0
        ),
        value_sup_goals=np.stack([row["goals"] for row in rows], axis=0),
        value_sup_budgets=np.stack([row["budgets"] for row in rows], axis=0),
        value_sup_remaining_budgets=np.stack(
            [row["remaining_budgets"] for row in rows], axis=0
        ),
        value_sup_offsets=np.stack(
            [np.rint(distance).astype(np.int32) for distance in distances],
            axis=0,
        ),
        value_sup_labels=np.stack([row["labels"] for row in rows], axis=0),
        value_sup_valids=np.ones((len(rows), pairs_per_budget), dtype=np.float32),
        value_sup_distances=np.stack(distances, axis=0).astype(np.float32),
    )


def bce_loss(pred_logit, target):
    log_pred = jax.nn.log_sigmoid(pred_logit)
    log_not_pred = jax.nn.log_sigmoid(-pred_logit)
    return -(log_pred * target + log_not_pred * (1.0 - target))


def select_witness_cells(
    witness_mask,
    left_distance,
    right_distance,
    left_budget,
    right_budget,
    num_witnesses,
    rng,
):
    witness_cells = np.nonzero(witness_mask)[0]
    if len(witness_cells) == 0:
        return None

    eps = float(FLAGS.trans_endpoint_epsilon)
    non_endpoint = (left_distance > eps) & (right_distance > eps)
    candidate_cells = witness_cells
    fallback_used = 0.0

    if FLAGS.trans_witness_mode == "avoid_endpoints":
        preferred = witness_cells[non_endpoint[witness_cells]]
        if len(preferred) > 0:
            candidate_cells = preferred
        else:
            fallback_used = 1.0
    elif FLAGS.trans_witness_mode == "boundary_balanced":
        beta = float(FLAGS.trans_boundary_beta)
        boundary_mask = (
            non_endpoint
            & (left_distance >= beta * float(left_budget))
            & (right_distance >= beta * float(right_budget))
        )
        preferred = witness_cells[boundary_mask[witness_cells]]
        if len(preferred) > 0:
            candidate_cells = preferred
        else:
            non_endpoint_cells = witness_cells[non_endpoint[witness_cells]]
            if len(non_endpoint_cells) > 0:
                candidate_cells = non_endpoint_cells
            fallback_used = 1.0
    elif FLAGS.trans_witness_mode == "slack_balanced":
        non_endpoint_cells = witness_cells[non_endpoint[witness_cells]]
        if len(non_endpoint_cells) > 0:
            candidate_cells = non_endpoint_cells
        else:
            fallback_used = 1.0
        left_slack = float(left_budget) - left_distance[candidate_cells]
        right_slack = float(right_budget) - right_distance[candidate_cells]
        scores = -np.abs(left_slack - right_slack)
        order = np.argsort(scores)[::-1]
        top_count = min(
            len(candidate_cells),
            max(int(num_witnesses), int(np.ceil(0.25 * len(candidate_cells)))),
        )
        candidate_cells = candidate_cells[order[:top_count]]
    elif FLAGS.trans_witness_mode != "uniform_valid":
        raise ValueError(f"Unsupported trans_witness_mode={FLAGS.trans_witness_mode}")

    replace = len(candidate_cells) < int(num_witnesses)
    sampled = rng.choice(candidate_cells, size=int(num_witnesses), replace=replace)
    unique_count = len(np.unique(sampled))
    return dict(
        witness_cells=witness_cells,
        candidate_cells=candidate_cells,
        sampled_witness_cells=sampled,
        effective_unique_witness_count=float(unique_count),
        unique_witness_frac=float(unique_count / float(num_witnesses)),
        replacement_used=float(replace),
        fallback_used=float(fallback_used),
    )


def sample_grid_qv_transitive_pairs(dataset, context, split, budgets, batch_size, rng):
    """Sample Q/V-valid tuples for y=max_w min(Q_h(s,a,w), V_{H-h}(w,g))."""
    if context["kind"] != "grid_geodesic":
        raise ValueError("Q/V transitive sampling currently supports grid labels.")

    state_to_cell = np.asarray(context[f"{split}_state_to_cell"], dtype=np.int32)
    state_by_cell = context[f"{split}_goal_by_cell"]
    has_state = np.asarray([len(items) > 0 for items in state_by_cell])
    distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
        context["label_distance_scale"]
    )
    src_idxs = valid_transition_indices(dataset)
    src_idxs = src_idxs[src_idxs + 1 < len(state_to_cell)]
    src_idxs = src_idxs[
        (state_to_cell[src_idxs] >= 0) & (state_to_cell[src_idxs + 1] >= 0)
    ]
    budgets = tuple(int(x) for x in budgets)
    num_witnesses = int(FLAGS.num_trans_witnesses)
    if num_witnesses < 1:
        raise ValueError("--num_trans_witnesses must be >= 1.")

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
    witness_candidate_counts = []
    effective_unique_witness_counts = []
    unique_witness_fracs = []
    replacement_used = []
    witness_fallback_used = []

    attempts = 0
    max_attempts = max(1000, int(batch_size) * 200)
    while len(observations) < int(batch_size) and attempts < max_attempts:
        attempts += 1
        budget = int(rng.choice(budgets))
        if budget <= 2:
            continue
        left_budget = max(1, budget // 2)
        right_budget = max(1, budget - left_budget)
        left_remaining = max(float(left_budget) - 1.0, 1.0)

        src_idx = int(rng.choice(src_idxs))
        next_cell = int(state_to_cell[src_idx + 1])
        next_distances = distances[next_cell]
        finite_goal = (context["cell_distances"][next_cell] >= 0) & has_state
        goal_hi = max(float(budget) - 1.0, 1.0)
        goal_lo = max(0.0, float(FLAGS.trans_pos_boundary_frac) * goal_hi)
        goal_mask = (
            finite_goal
            & (next_distances >= goal_lo)
            & (next_distances <= goal_hi)
        )
        if not goal_mask.any() and goal_lo > 0.0:
            goal_mask = finite_goal & (next_distances <= goal_hi)
        goal_cells = np.nonzero(goal_mask)[0]
        if len(goal_cells) == 0:
            continue

        goal_cell = int(rng.choice(goal_cells))
        right_to_goal = distances[:, goal_cell]
        witness_mask = (
            has_state
            & (context["cell_distances"][next_cell] >= 0)
            & (context["cell_distances"][:, goal_cell] >= 0)
            & (next_distances <= left_remaining)
            & (right_to_goal <= float(right_budget))
        )
        selection = select_witness_cells(
            witness_mask,
            next_distances,
            right_to_goal,
            left_remaining,
            right_budget,
            num_witnesses,
            rng,
        )
        if selection is None:
            continue

        sampled_witness_cells = selection["sampled_witness_cells"]
        goal_idx = int(rng.choice(state_by_cell[goal_cell]))
        parent_distance = float(next_distances[goal_cell])
        observations.append(np.asarray(dataset["observations"])[src_idx])
        actions.append(np.asarray(dataset["actions"])[src_idx])
        goals.append(np.asarray(dataset["observations"])[goal_idx])
        value_budgets.append(budget)
        value_offsets.append(parent_distance)
        parent_distances.append(parent_distance)
        witness_cell_counts.append(float(len(selection["witness_cells"])))
        witness_candidate_counts.append(float(len(selection["candidate_cells"])))
        effective_unique_witness_counts.append(
            selection["effective_unique_witness_count"]
        )
        unique_witness_fracs.append(selection["unique_witness_frac"])
        replacement_used.append(selection["replacement_used"])
        witness_fallback_used.append(selection["fallback_used"])

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
        for witness_cell in sampled_witness_cells:
            witness_cell = int(witness_cell)
            witness_idx = int(rng.choice(state_by_cell[witness_cell]))
            left_distance = float(next_distances[witness_cell])
            right_distance = float(right_to_goal[witness_cell])
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
            parent_left_slacks.append(left_remaining - left_distance)
            parent_right_slacks.append(float(right_budget) - right_distance)

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

    if len(observations) < int(batch_size):
        raise ValueError(
            f"Could not sample {batch_size} Q/V transitive tuples for split={split}; "
            f"got {len(observations)} after {attempts} attempts."
        )

    return dict(
        qv_observations=np.asarray(observations, dtype=np.float32),
        qv_actions=np.asarray(actions, dtype=np.float32),
        qv_goals=np.asarray(goals, dtype=np.float32),
        qv_budgets=np.asarray(value_budgets, dtype=np.int32),
        qv_offsets=np.rint(value_offsets).astype(np.int32),
        qv_midpoint_observations=np.swapaxes(
            np.asarray(witness_observations, dtype=np.float32), 0, 1
        ),
        qv_midpoint_actions=np.swapaxes(
            np.asarray(witness_actions, dtype=np.float32), 0, 1
        ),
        qv_midpoint_goals=np.swapaxes(
            np.asarray(witness_goals, dtype=np.float32), 0, 1
        ),
        qv_midpoint_offsets=np.rint(
            np.swapaxes(np.asarray(witness_offsets, dtype=np.float32), 0, 1)
        ).astype(np.int32),
        qv_left_budgets=np.swapaxes(np.asarray(left_budgets, dtype=np.int32), 0, 1),
        qv_right_budgets=np.swapaxes(np.asarray(right_budgets, dtype=np.int32), 0, 1),
        qv_valids=np.swapaxes(np.asarray(trans_valids, dtype=np.float32), 0, 1),
        qv_parent_distances=np.asarray(parent_distances, dtype=np.float32),
        qv_left_distances=np.swapaxes(
            np.asarray(left_distances, dtype=np.float32), 0, 1
        ),
        qv_right_distances=np.swapaxes(
            np.asarray(right_distances, dtype=np.float32), 0, 1
        ),
        qv_left_slacks=np.swapaxes(np.asarray(left_slacks, dtype=np.float32), 0, 1),
        qv_right_slacks=np.swapaxes(np.asarray(right_slacks, dtype=np.float32), 0, 1),
        qv_witness_cell_counts=np.asarray(witness_cell_counts, dtype=np.float32),
        qv_witness_candidate_counts=np.asarray(
            witness_candidate_counts, dtype=np.float32
        ),
        qv_effective_unique_witness_counts=np.asarray(
            effective_unique_witness_counts, dtype=np.float32
        ),
        qv_unique_witness_fracs=np.asarray(unique_witness_fracs, dtype=np.float32),
        qv_replacement_used=np.asarray(replacement_used, dtype=np.float32),
        qv_witness_fallback_used=np.asarray(witness_fallback_used, dtype=np.float32),
        qv_sample_acceptance_rate=np.asarray(
            float(len(observations)) / float(max(attempts, 1)), dtype=np.float32
        ),
        qv_attempts_per_sample=np.asarray(
            float(attempts) / float(max(len(observations), 1)), dtype=np.float32
        ),
    )


def qv_transitive_loss(
    agent,
    batch,
    grad_params,
    value_agent,
    qv_trans_loss_type,
    qv_trans_bce_margin,
    trans_budgets,
):
    parent_logits = agent.critic_logits_for(
        batch["qv_observations"],
        batch["qv_actions"],
        batch["qv_goals"],
        batch["qv_budgets"],
        offsets=batch["qv_offsets"],
        grad_params=grad_params,
    )
    parent_r = jax.nn.sigmoid(parent_logits)

    witness_goals = batch["qv_midpoint_goals"]
    witness_shape = witness_goals.shape[:-1]
    parent_observations = jnp.broadcast_to(
        batch["qv_observations"][None, ...],
        witness_shape + batch["qv_observations"].shape[-1:],
    )
    parent_actions = jnp.broadcast_to(
        batch["qv_actions"][None, ...],
        witness_shape + batch["qv_actions"].shape[-1:],
    )
    parent_goals = jnp.broadcast_to(
        batch["qv_goals"][None, ...], witness_shape + batch["qv_goals"].shape[-1:]
    )

    first_logits = agent.critic_logits_for_pair_grid(
        parent_observations,
        parent_actions,
        witness_goals,
        batch["qv_left_budgets"],
        offsets=batch["qv_midpoint_offsets"],
        target=True,
    )
    second_logits = value_agent.critic_logits_for_pair_grid(
        batch["qv_midpoint_observations"],
        batch["qv_midpoint_actions"],
        parent_goals,
        batch["qv_right_budgets"],
        offsets=batch["qv_right_budgets"],
    )
    first_r = jax.nn.sigmoid(first_logits)
    second_r = jax.nn.sigmoid(second_logits)
    witness_valids = jnp.asarray(batch["qv_valids"], dtype=first_r.dtype)
    y_candidates = jnp.minimum(first_r, second_r)
    y_candidates = jnp.where(witness_valids[None, ...] > 0, y_candidates, -1.0)
    y_trans = jax.lax.stop_gradient(jnp.max(y_candidates, axis=1))
    trans_valids = (witness_valids.max(axis=0) > 0).astype(parent_logits.dtype)
    y_trans = jnp.clip(y_trans, 0.0, 1.0)
    y_trans = jnp.where(trans_valids[None, :] > 0, y_trans, 0.0)
    bce = bce_loss(parent_logits, y_trans)
    target_minus_parent = y_trans - parent_r
    if qv_trans_loss_type == "bce_equal":
        loss_values = bce
        loss_mask = trans_valids
    elif qv_trans_loss_type == "prob_hinge":
        loss_values = jnp.square(jnp.maximum(target_minus_parent, 0.0))
        loss_mask = trans_valids
    elif qv_trans_loss_type == "bce_lower_bound":
        gate = jax.lax.stop_gradient(
            (target_minus_parent > qv_trans_bce_margin).astype(parent_logits.dtype)
        )
        loss_values = bce
        loss_mask = trans_valids * gate
    else:
        raise ValueError(f"Unsupported qv_trans_loss_type={qv_trans_loss_type}")
    loss = masked_mean(loss_values, loss_mask)

    info = dict(
        loss_qv_trans=loss,
        qv_parent_r_mean=masked_mean(parent_r, trans_valids),
        qv_y_trans_mean=masked_mean(y_trans, trans_valids),
        qv_target_minus_parent_mean=masked_mean(target_minus_parent, trans_valids),
        qv_frac_y_trans_gt_parent=masked_mean(
            (target_minus_parent > 0.0).astype(parent_logits.dtype), trans_valids
        ),
        qv_frac_y_trans_lt_parent=masked_mean(
            (target_minus_parent < 0.0).astype(parent_logits.dtype), trans_valids
        ),
        qv_first_q_mean=masked_mean(first_r, witness_valids),
        qv_second_v_mean=masked_mean(second_r, witness_valids),
        qv_valid_frac=trans_valids.mean(),
    )
    qv_budgets = jnp.asarray(batch["qv_budgets"], dtype=parent_logits.dtype)
    for budget in tuple(int(x) for x in trans_budgets):
        budget_key = f"H{budget}"
        parent_budget_mask = trans_valids * (qv_budgets == float(budget)).astype(
            parent_logits.dtype
        )
        witness_budget_mask = witness_valids * (
            qv_budgets[None, :] == float(budget)
        ).astype(first_r.dtype)
        info[f"loss_qv_trans_by_budget/{budget_key}"] = masked_mean(
            loss_values, parent_budget_mask
        )
        info[f"qv_y_trans_mean_by_budget/{budget_key}"] = masked_mean(
            y_trans, parent_budget_mask
        )
        info[f"qv_parent_r_mean_by_budget/{budget_key}"] = masked_mean(
            parent_r, parent_budget_mask
        )
        info[f"qv_first_q_mean_by_budget/{budget_key}"] = masked_mean(
            first_r, witness_budget_mask
        )
        info[f"qv_second_v_mean_by_budget/{budget_key}"] = masked_mean(
            second_r, witness_budget_mask
        )
    return loss, info


def update_with_qv_trans(
    agent,
    batch,
    value_agent,
    lambda_qv_trans,
    qv_trans_loss_type="bce_equal",
    qv_trans_bce_margin=0.0,
    trans_budgets=(),
):
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        critic_loss, critic_info = agent.critic_loss(batch, grad_params)
        qv_loss, qv_info = qv_transitive_loss(
            agent,
            batch,
            grad_params,
            value_agent,
            qv_trans_loss_type,
            qv_trans_bce_margin,
            trans_budgets,
        )
        loss = critic_loss + lambda_qv_trans * qv_loss
        info = {f"critic/{key}": value for key, value in critic_info.items()}
        for key, value in qv_info.items():
            info[f"critic/{key}"] = value
        info["critic/total_loss_with_qv"] = loss
        return loss, info

    new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    agent.target_update(new_network, "critic")
    return agent.replace(network=new_network, rng=new_rng), info


update_with_qv_trans = jax.jit(
    update_with_qv_trans,
    static_argnames=("qv_trans_loss_type", "trans_budgets"),
)


def rank_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = np.sqrt((x_rank**2).sum() * (y_rank**2).sum())
    if denom <= 0.0:
        return np.nan
    return float((x_rank * y_rank).sum() / denom)


def q_v_next_consistency(agent, batch, value_agent=None):
    q_logits = agent.critic_logits_for_pair_grid(
        batch["value_sup_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    q_scores = np.asarray(jax.nn.sigmoid(q_logits)).mean(axis=0)
    labels = np.asarray(batch["value_sup_labels"])
    valids = np.asarray(batch["value_sup_valids"]) > 0
    budgets = np.asarray(batch["value_sup_budgets"])
    if value_agent is None:
        return dict(value_checkpoint_available=False)

    v_logits = value_agent.critic_logits_for_pair_grid(
        batch["value_sup_next_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_remaining_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    v_scores = np.asarray(jax.nn.sigmoid(v_logits)).mean(axis=0)
    v_next_budget_rows = []
    for budget in sorted(np.unique(budgets[valids]).astype(np.int32)):
        mask = valids & (budgets == int(budget))
        if not mask.any():
            continue
        v_next_budget_rows.append(
            dict(
                budget=int(budget),
                metrics=binary_metrics(v_scores[mask], labels[mask]),
            )
        )
    v_next_metrics = binary_metrics(v_scores[valids], labels[valids])
    return dict(
        value_checkpoint_available=True,
        mean_abs_prob_diff=float(np.abs(q_scores[valids] - v_scores[valids]).mean()),
        rank_correlation=rank_correlation(q_scores[valids], v_scores[valids]),
        v_next_auc=rank_metrics(v_scores[valids], labels[valids])["auc"],
        v_next_metrics=v_next_metrics,
        v_next_budget_rows=v_next_budget_rows,
    )


def score_sup_batch(agent, batch, budgets, value_agent=None):
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
    next_observations = np.asarray(batch["value_sup_next_observations"])
    goals = np.asarray(batch["value_sup_goals"])

    report = {
        "mean": binary_metrics(mean_scores[valids], labels[valids]),
        "ensemble_min": binary_metrics(min_scores[valids], labels[valids]),
        "q_v_next_consistency": q_v_next_consistency(
            agent, batch, value_agent=value_agent
        ),
        "budget_rows": [],
    }
    for budget in budgets:
        mask = valids & (sup_budgets == int(budget))
        if not mask.any():
            continue
        next_obs = next_observations[mask]
        goal = goals[mask]
        euclidean = np.linalg.norm(goal[:, :2] - next_obs[:, :2], axis=-1)
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
    consistency = report.get("q_v_next_consistency", {})
    if consistency.get("value_checkpoint_available"):
        print(
            "Q-V_next | "
            f"abs_diff={format_metric(consistency['mean_abs_prob_diff'])} | "
            f"rank_corr={format_metric(consistency['rank_correlation'])} | "
            f"v_next_auc={format_metric(consistency['v_next_auc'])}"
        )
    else:
        print("Q-V_next | skipped (no value checkpoint supplied)")


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(values[finite].mean())


def summarize_qv_transitive_batch(batch, budgets):
    if batch is None:
        return None
    qv_budgets = np.asarray(batch["qv_budgets"])
    qv_valids = np.asarray(batch["qv_valids"]) > 0
    parent_distances = np.asarray(batch["qv_parent_distances"])
    left_distances = np.asarray(batch["qv_left_distances"])
    right_distances = np.asarray(batch["qv_right_distances"])
    left_slacks = np.asarray(batch["qv_left_slacks"])
    right_slacks = np.asarray(batch["qv_right_slacks"])
    rows = []
    for budget in budgets:
        budget = int(budget)
        parent_mask = qv_budgets == budget
        witness_mask = qv_valids & parent_mask[None, :]
        if not parent_mask.any():
            continue
        rows.append(
            dict(
                budget=budget,
                count=int(parent_mask.sum()),
                parent_distance_mean=finite_mean(parent_distances[parent_mask]),
                left_distance_mean=finite_mean(left_distances[witness_mask]),
                right_distance_mean=finite_mean(right_distances[witness_mask]),
                left_slack_mean=finite_mean(left_slacks[witness_mask]),
                right_slack_mean=finite_mean(right_slacks[witness_mask]),
                witness_cell_count_mean=finite_mean(
                    np.asarray(batch["qv_witness_cell_counts"])[parent_mask]
                ),
                witness_candidate_count_mean=finite_mean(
                    np.asarray(batch["qv_witness_candidate_counts"])[parent_mask]
                ),
                effective_unique_witness_count_mean=finite_mean(
                    np.asarray(batch["qv_effective_unique_witness_counts"])[
                        parent_mask
                    ]
                ),
                replacement_used_frac=finite_mean(
                    np.asarray(batch["qv_replacement_used"])[parent_mask]
                ),
                unique_witness_frac=finite_mean(
                    np.asarray(batch["qv_unique_witness_fracs"])[parent_mask]
                ),
                zero_left_frac=finite_mean(
                    (left_distances[witness_mask] <= 1e-6).astype(np.float32)
                ),
                zero_right_frac=finite_mean(
                    (right_distances[witness_mask] <= 1e-6).astype(np.float32)
                ),
            )
        )
    return dict(
        qv_sample_acceptance_rate=float(np.asarray(batch["qv_sample_acceptance_rate"])),
        qv_attempts_per_sample=float(np.asarray(batch["qv_attempts_per_sample"])),
        num_trans_witnesses=int(np.asarray(qv_valids).shape[0]),
        trans_witness_mode=str(FLAGS.trans_witness_mode),
        budget_rows=rows,
    )


def print_qv_summary(summary, info):
    if summary is None:
        return
    print(
        "qv_trans | "
        f"accept={format_metric(summary['qv_sample_acceptance_rate'])} | "
        f"attempts/sample={format_metric(summary['qv_attempts_per_sample'])} | "
        f"K={summary['num_trans_witnesses']} | "
        f"mode={summary['trans_witness_mode']} | "
        f"loss={format_metric(info.get('critic/loss_qv_trans', np.nan))}"
    )
    print(
        "qv_target | "
        f"parent={format_metric(info.get('critic/qv_parent_r_mean', np.nan))} | "
        f"target={format_metric(info.get('critic/qv_y_trans_mean', np.nan))} | "
        f"target-parent={format_metric(info.get('critic/qv_target_minus_parent_mean', np.nan))} | "
        f"gt_parent={format_metric(info.get('critic/qv_frac_y_trans_gt_parent', np.nan))} | "
        f"lt_parent={format_metric(info.get('critic/qv_frac_y_trans_lt_parent', np.nan))}"
    )
    print("H | parents | parent_d | left_d | right_d | eff_K | repl | zero_l | zero_r")
    print("--|---------|----------|--------|---------|-------|------|--------|-------")
    for row in summary["budget_rows"]:
        print(
            f"{row['budget']:4d} | {row['count']:7d} | "
            f"{format_metric(row['parent_distance_mean'])} | "
            f"{format_metric(row['left_distance_mean'])} | "
            f"{format_metric(row['right_distance_mean'])} | "
            f"{format_metric(row['effective_unique_witness_count_mean'])} | "
            f"{format_metric(row['replacement_used_frac'])} | "
            f"{format_metric(row['zero_left_frac'])} | "
            f"{format_metric(row['zero_right_frac'])}"
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


def context_metadata(context):
    if context["kind"] == "grid_geodesic":
        return dict(
            kind=context["kind"],
            geodesic_budget_unit=context["geodesic_budget_unit"],
            maze_type=context["maze_type"],
            maze_unit=context["maze_unit"],
            median_step_xy=context["median_step_xy"],
            steps_per_cell=context["steps_per_cell"],
            label_distance_scale=context["label_distance_scale"],
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
        raise ValueError("train_bmm_geodesic_q.py requires bmm_trl.")
    budgets = parse_budgets(FLAGS.budgets)
    trans_budgets = (
        budgets if FLAGS.trans_budgets is None else parse_budgets(FLAGS.trans_budgets)
    )
    sup_pairs_per_budget = positive_or_default(
        FLAGS.sup_pairs_per_budget, FLAGS.batch_size
    )
    trans_pairs_per_update = positive_or_default(
        FLAGS.trans_pairs_per_update, FLAGS.batch_size
    )
    configure_agent(config, budgets)

    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    context = make_label_context(env, train_dataset, val_dataset, xy_dims)
    print("BMM geodesic Q diagnostic")
    print(f"  env_name: {FLAGS.env_name}")
    print(f"  label_type: {context['kind']}")
    print(f"  geodesic_budget_unit: {FLAGS.geodesic_budget_unit}")
    print(f"  budgets: {budgets}")
    print(f"  trans_budgets: {trans_budgets}")
    print(f"  lambda_qv_trans: {FLAGS.lambda_qv_trans}")
    print(f"  qv_trans_loss_type: {FLAGS.qv_trans_loss_type}")
    print(f"  qv_trans_bce_margin: {FLAGS.qv_trans_bce_margin}")
    print(f"  num_trans_witnesses: {FLAGS.num_trans_witnesses}")
    print(f"  trans_witness_mode: {FLAGS.trans_witness_mode}")
    print(f"  sup_pairs_per_budget: {sup_pairs_per_budget}")
    print(f"  trans_pairs_per_update: {trans_pairs_per_update}")
    print(f"  context: {context_metadata(context)}")

    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    gc_train = dataset_class(Dataset.create(**train_dataset), config)
    example_batch = gc_train.sample(1)
    agent = agents[config["agent_name"]].create(FLAGS.seed, example_batch, config)
    value_agent = None
    if FLAGS.value_restore_path is not None or FLAGS.value_restore_epoch is not None:
        if FLAGS.value_restore_path is None or FLAGS.value_restore_epoch is None:
            raise ValueError(
                "--value_restore_path and --value_restore_epoch must be set together."
            )
        value_config = copy.deepcopy(config)
        value_config.diagnostic_critic_mode = "state"
        value_config.value_only = True
        value_agent = agents[value_config["agent_name"]].create(
            FLAGS.seed, example_batch, value_config
        )
        value_agent = restore_agent(
            value_agent, FLAGS.value_restore_path, FLAGS.value_restore_epoch
        )
    if FLAGS.lambda_qv_trans > 0.0 and value_agent is None:
        raise ValueError(
            "--lambda_qv_trans > 0 requires --value_restore_path and "
            "--value_restore_epoch for the frozen V teacher."
        )

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
    eval_report = score_sup_batch(agent, eval_batch, budgets, value_agent=value_agent)
    eval_mono = monotonicity_violation(agent, eval_batch, budgets)
    print_report("eval", 0, eval_report, mono=eval_mono)
    last_update_info = {}
    qv_history = []
    last_qv_summary = None

    for step in range(1, FLAGS.steps + 1):
        base_batch_size = (
            trans_pairs_per_update if FLAGS.lambda_qv_trans > 0.0 else FLAGS.batch_size
        )
        train_batch = gc_train.sample(base_batch_size)
        qv_fields = None
        if FLAGS.lambda_qv_trans > 0.0:
            qv_fields = sample_grid_qv_transitive_pairs(
                train_dataset,
                context,
                "train",
                trans_budgets,
                trans_pairs_per_update,
                rng,
            )
            train_batch.update(qv_fields)
        train_batch.update(
            make_sup_fields(
                train_dataset,
                context,
                "train",
                budgets,
                sup_pairs_per_budget,
                rng,
            )
        )
        if FLAGS.lambda_qv_trans > 0.0:
            agent, info = update_with_qv_trans(
                agent,
                train_batch,
                value_agent,
                FLAGS.lambda_qv_trans,
                qv_trans_loss_type=FLAGS.qv_trans_loss_type,
                qv_trans_bce_margin=FLAGS.qv_trans_bce_margin,
                trans_budgets=trans_budgets,
            )
        else:
            agent, info = agent.update(train_batch)
        last_update_info = info_to_float_dict(info)
        final_loss = float(info["critic/loss_sup"])
        if step % FLAGS.eval_interval == 0 or step == FLAGS.steps:
            train_report = score_sup_batch(
                agent, train_batch, budgets, value_agent=value_agent
            )
            eval_report = score_sup_batch(
                agent, eval_batch, budgets, value_agent=value_agent
            )
            train_mono = monotonicity_violation(agent, train_batch, budgets)
            eval_mono = monotonicity_violation(agent, eval_batch, budgets)
            last_qv_summary = summarize_qv_transitive_batch(qv_fields, trans_budgets)
            if last_qv_summary is not None:
                qv_history.append(dict(step=int(step), train=last_qv_summary))
            print_report("train", step, train_report, mono=train_mono, loss=final_loss)
            print_qv_summary(last_qv_summary, last_update_info)
            print_report("eval", step, eval_report, mono=eval_mono)

    final_passed = passed(eval_report, budgets)
    print(f"\nFinal heldout threshold pass: {final_passed}")
    final_report = dict(
        train=train_report,
        eval=eval_report,
        eval_monotonicity_violation=eval_mono,
        last_update_info=last_update_info,
        last_qv_summary=last_qv_summary,
        qv_history=qv_history,
        passed=bool(final_passed),
        config=dict(
            env_name=FLAGS.env_name,
            reachability_label_type=FLAGS.reachability_label_type,
            budgets=[int(x) for x in budgets],
            trans_budgets=[int(x) for x in trans_budgets],
            batch_size=int(FLAGS.batch_size),
            sup_pairs_per_budget=int(sup_pairs_per_budget),
            trans_pairs_per_update=int(trans_pairs_per_update),
            eval_pairs=int(FLAGS.eval_pairs),
            steps=int(FLAGS.steps),
            eval_interval=int(FLAGS.eval_interval),
            final_loss_sup=final_loss,
            lambda_qv_trans=float(FLAGS.lambda_qv_trans),
            qv_trans_loss_type=str(FLAGS.qv_trans_loss_type),
            qv_trans_bce_margin=float(FLAGS.qv_trans_bce_margin),
            trans_pos_boundary_frac=float(FLAGS.trans_pos_boundary_frac),
            num_trans_witnesses=int(FLAGS.num_trans_witnesses),
            trans_witness_mode=str(FLAGS.trans_witness_mode),
            trans_endpoint_epsilon=float(FLAGS.trans_endpoint_epsilon),
            trans_boundary_beta=float(FLAGS.trans_boundary_beta),
            value_restore_path=FLAGS.value_restore_path,
            value_restore_epoch=FLAGS.value_restore_epoch,
            context=context_metadata(context),
        ),
    )
    if FLAGS.output_json is not None:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nWrote geodesic Q report to {output_path}")

    if FLAGS.fail_on_threshold and not final_passed:
        raise SystemExit("Final heldout geodesic Q metrics did not pass.")


if __name__ == "__main__":
    app.run(main)
