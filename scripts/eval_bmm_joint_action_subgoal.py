#!/usr/bin/env python
"""Joint action-subgoal diagnostic for BMM Q/V critics."""

import argparse
import copy
import json
from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts.bmm_reachability_utils import format_metric
from utils.flax_utils import restore_agent


SUPPORTED_CANDIDATE_ACTION_MODES = (
    "same_cell_cached",
    "neighbor_cell",
    "directional",
    "oracle_diverse",
    "dataset_global_oracle",
)


def score_flat(agent, observations, actions, goals, budgets, batch_size):
    mean_scores = []
    min_scores = []
    count = len(observations)
    for start in range(0, count, int(batch_size)):
        end = min(start + int(batch_size), count)
        logits = agent.critic_logits_for(
            observations[start:end],
            actions[start:end],
            goals[start:end],
            budgets[start:end],
            offsets=budgets[start:end],
        )
        scores = np.asarray(jax.nn.sigmoid(logits))
        mean_scores.append(scores.mean(axis=0))
        min_scores.append(scores.min(axis=0))
    return np.concatenate(mean_scores), np.concatenate(min_scores)


def unique_in_order(values):
    seen = set()
    kept = []
    for value in np.asarray(values, dtype=np.int32).reshape(-1):
        value = int(value)
        if value not in seen:
            kept.append(value)
            seen.add(value)
    return np.asarray(kept, dtype=np.int32)


def fill_candidate_indices(chosen, pool, count, rng):
    chosen = unique_in_order(chosen)
    pool = unique_in_order(pool)
    count = int(count)
    if len(chosen) >= count:
        return chosen[:count]
    if len(pool) == 0:
        if len(chosen) == 0:
            raise ValueError("Cannot fill candidate actions from an empty pool.")
        pool = chosen
    remaining = count - len(chosen)
    chosen_set = set(int(x) for x in chosen)
    available = np.asarray([int(x) for x in pool if int(x) not in chosen_set], dtype=np.int32)
    if len(available) == 0:
        available = pool
    extra = rng.choice(available, size=remaining, replace=len(available) < remaining)
    return np.concatenate([chosen, np.asarray(extra, dtype=np.int32)])[:count]


def select_diverse_by_distance(pool, distances, count, rng):
    pool = unique_in_order(pool)
    distances = np.asarray(distances, dtype=np.float32)
    if len(pool) == 0:
        return pool
    order = np.argsort(distances)
    ordered = pool[order]
    if len(ordered) <= int(count):
        return ordered
    ranks = np.linspace(0, len(ordered) - 1, int(count))
    selected = ordered[np.unique(np.round(ranks).astype(np.int32))]
    return fill_candidate_indices(selected, ordered, count, rng)


def local_transition_pool(by_cell, cells):
    chunks = []
    for cell in np.asarray(cells, dtype=np.int32).reshape(-1):
        cell = int(cell)
        if 0 <= cell < len(by_cell) and len(by_cell[cell]) > 0:
            chunks.append(np.asarray(by_cell[cell], dtype=np.int32))
    if not chunks:
        return np.asarray([], dtype=np.int32)
    return unique_in_order(np.concatenate(chunks))


def finite_pool_to_goal(pool, state_to_cell, cell_distances_raw, step_distances, goal_cell):
    pool = unique_in_order(pool)
    if len(pool) == 0:
        return pool, np.asarray([], dtype=np.float32)
    next_cells = np.asarray(state_to_cell, dtype=np.int32)[pool + 1]
    finite = (next_cells >= 0) & (cell_distances_raw[next_cells, int(goal_cell)] >= 0)
    pool = pool[finite]
    next_cells = next_cells[finite]
    distances = step_distances[next_cells, int(goal_cell)].astype(np.float32)
    return pool, distances


def direction_key(delta):
    row, col = int(delta[0]), int(delta[1])
    return (
        0 if row == 0 else int(np.sign(row)),
        0 if col == 0 else int(np.sign(col)),
    )


def select_directional_candidates(
    pool,
    distances,
    source_cell,
    state_to_cell,
    free_cells,
    count,
    rng,
    preferred=(),
):
    pool = unique_in_order(pool)
    distances = np.asarray(distances, dtype=np.float32)
    if len(pool) == 0:
        return fill_candidate_indices(preferred, pool, count, rng)
    next_cells = np.asarray(state_to_cell, dtype=np.int32)[pool + 1]
    groups = {}
    source_xy = np.asarray(free_cells[int(source_cell)], dtype=np.int32)
    for idx, transition_idx in enumerate(pool):
        key = direction_key(np.asarray(free_cells[int(next_cells[idx])]) - source_xy)
        current = groups.get(key)
        if current is None or distances[idx] < current[1]:
            groups[key] = (int(transition_idx), float(distances[idx]))
    directional = np.asarray(
        [value[0] for key, value in sorted(groups.items())], dtype=np.int32
    )
    chosen = fill_candidate_indices(preferred, directional, min(len(directional), count), rng)
    if len(chosen) < int(count):
        diverse = select_diverse_by_distance(pool, distances, count, rng)
        chosen = fill_candidate_indices(chosen, diverse, count, rng)
    return chosen[: int(count)]


def resize_cached_candidates(queries, candidate_count, rng):
    current_count = int(queries["actions"].shape[1])
    candidate_count = int(candidate_count)
    if current_count == candidate_count:
        return queries
    idx_rows = []
    for row in np.asarray(queries["candidate_source_idxs"], dtype=np.int32):
        if current_count > candidate_count:
            idx_rows.append(row[:candidate_count])
        else:
            idx_rows.append(fill_candidate_indices(row, row, candidate_count, rng))
    return replace_candidate_action_indices(
        queries,
        queries["_candidate_dataset"],
        queries["_candidate_context"],
        np.asarray(idx_rows, dtype=np.int32),
    )


def replace_candidate_action_indices(queries, dataset, context, candidate_idxs):
    queries = copy.copy(queries)
    candidate_idxs = np.asarray(candidate_idxs, dtype=np.int32)
    state_to_cell = np.asarray(context["val_state_to_cell"], dtype=np.int32)
    step_distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
        context["distance_scale"]
    )
    goal_cells = np.asarray(queries["goal_cells"], dtype=np.int32)
    next_cells = state_to_cell[candidate_idxs + 1]
    distances = step_distances[next_cells, goal_cells[:, None]].astype(np.float32)
    goals = np.repeat(
        np.asarray(queries["goals"], dtype=np.float32)[:, :1, :],
        candidate_idxs.shape[1],
        axis=1,
    )
    observations = np.repeat(
        np.asarray(queries["observations"], dtype=np.float32)[:, :1, :],
        candidate_idxs.shape[1],
        axis=1,
    )
    queries["observations"] = observations
    queries["candidate_observations"] = np.asarray(dataset["observations"])[
        candidate_idxs
    ].astype(np.float32)
    queries["actions"] = np.asarray(dataset["actions"])[candidate_idxs].astype(
        np.float32
    )
    queries["next_observations"] = np.asarray(dataset["observations"])[
        candidate_idxs + 1
    ].astype(np.float32)
    queries["goals"] = goals
    queries["candidate_source_idxs"] = candidate_idxs
    queries["candidate_next_cells"] = next_cells.astype(np.int32)
    queries["distances"] = distances
    queries["labels"] = (
        distances <= float(queries.get("remaining_budget", queries["budget"] - 1))
    ).astype(np.float32)
    return queries


def apply_candidate_action_mode(
    queries,
    dataset,
    context,
    mode,
    candidate_count,
    neighbor_hops,
    rng,
):
    if mode not in SUPPORTED_CANDIDATE_ACTION_MODES:
        raise ValueError(
            f"Unsupported candidate action mode {mode!r}. "
            f"Expected one of {SUPPORTED_CANDIDATE_ACTION_MODES}."
        )

    queries = ar.hydrate_query_candidate_fields(queries, dataset, context, split="val")
    queries = copy.copy(queries)
    queries["_candidate_dataset"] = dataset
    queries["_candidate_context"] = context
    candidate_count = int(candidate_count)

    if mode == "same_cell_cached":
        result = resize_cached_candidates(queries, candidate_count, rng)
        result.pop("_candidate_dataset", None)
        result.pop("_candidate_context", None)
        return result

    state_to_cell = np.asarray(context["val_state_to_cell"], dtype=np.int32)
    cell_distances_raw = np.asarray(context["cell_distances"], dtype=np.int32)
    step_distances = cell_distances_raw.astype(np.float32) * float(
        context["distance_scale"]
    )
    by_cell = ar.transition_indices_by_cell(dataset, state_to_cell)
    all_pool = local_transition_pool(by_cell, np.arange(len(by_cell), dtype=np.int32))
    free_cells = np.asarray(context["free_cells"], dtype=np.int32)

    candidate_rows = []
    for row, (source_cell, goal_cell) in enumerate(
        zip(np.asarray(queries["source_cells"], dtype=np.int32), np.asarray(queries["goal_cells"], dtype=np.int32))
    ):
        source_cell = int(source_cell)
        goal_cell = int(goal_cell)
        cached_row = np.asarray(queries["candidate_source_idxs"][row], dtype=np.int32)
        logged = np.asarray([int(cached_row[0])], dtype=np.int32)

        if mode in ("neighbor_cell", "directional"):
            neighbor_cells = np.nonzero(
                (cell_distances_raw[source_cell] >= 0)
                & (cell_distances_raw[source_cell] <= int(neighbor_hops))
            )[0]
            pool = local_transition_pool(by_cell, neighbor_cells)
            preferred = logged
        else:
            pool = all_pool
            preferred = np.asarray([], dtype=np.int32)

        finite_pool, distances = finite_pool_to_goal(
            pool, state_to_cell, cell_distances_raw, step_distances, goal_cell
        )
        if len(finite_pool) == 0:
            finite_pool = cached_row
            distances = np.asarray(queries["distances"][row], dtype=np.float32)

        if mode == "directional":
            selected = select_directional_candidates(
                finite_pool,
                distances,
                source_cell,
                state_to_cell,
                free_cells,
                candidate_count,
                rng,
                preferred=preferred,
            )
        elif mode == "dataset_global_oracle":
            selected = finite_pool[np.argsort(distances)[:candidate_count]]
            selected = fill_candidate_indices(selected, finite_pool, candidate_count, rng)
        else:
            selected = select_diverse_by_distance(
                finite_pool, distances, candidate_count, rng
            )
            selected = fill_candidate_indices(preferred, selected, candidate_count, rng)
        candidate_rows.append(selected.astype(np.int32))

    result = replace_candidate_action_indices(
        queries,
        dataset,
        context,
        np.asarray(candidate_rows, dtype=np.int32),
    )
    result.pop("_candidate_dataset", None)
    result.pop("_candidate_context", None)
    return result


def candidate_action_diagnostics(queries, xy_dims=(0, 1)):
    distances = np.asarray(queries["distances"], dtype=np.float32)
    candidate_next_cells = np.asarray(queries["candidate_next_cells"], dtype=np.int32)
    diag = dict(
        oracle_best_selected_distance_mean=float(distances.min(axis=1).mean()),
        logged_selected_distance_mean=float(distances[:, 0].mean()),
        random_selected_distance_mean=float(distances.mean()),
        next_distance_spread_mean=float(
            (distances.max(axis=1) - distances.min(axis=1)).mean()
        ),
        unique_next_cell_count_mean=float(
            np.asarray(
                [len(np.unique(row)) for row in candidate_next_cells],
                dtype=np.float32,
            ).mean()
        ),
    )
    if "candidate_observations" in queries:
        xy = np.asarray(queries["candidate_observations"], dtype=np.float32)[
            ..., tuple(xy_dims)
        ]
        deltas = xy[:, :, None, :] - xy[:, None, :, :]
        pairwise = np.linalg.norm(deltas, axis=-1)
        diag["source_position_spread_mean"] = float(pairwise.max(axis=(1, 2)).mean())
    else:
        diag["source_position_spread_mean"] = float("nan")
    return diag


def sample_joint_candidates(
    val_dataset,
    context,
    queries,
    left_budget,
    right_budget,
    num_subgoals,
    rng,
):
    """Sample one shared subgoal set for every cached action-ranking query."""
    state_to_cell = np.asarray(context["val_state_to_cell"], dtype=np.int32)
    goal_by_cell = context["val_goal_by_cell"]
    has_state = np.asarray([len(items) > 0 for items in goal_by_cell])
    cell_distances_raw = np.asarray(context["cell_distances"], dtype=np.int32)
    step_distances = cell_distances_raw.astype(np.float32) * float(
        context["distance_scale"]
    )

    source_cells = np.asarray(queries["source_cells"], dtype=np.int32)
    goal_cells = np.asarray(queries["goal_cells"], dtype=np.int32)
    candidate_next_cells = np.asarray(queries["candidate_next_cells"], dtype=np.int32)

    subgoal_observations = []
    subgoal_cells = []
    source_distances = []
    next_distances = []
    right_distances = []
    state_valids = []
    action_valids = []
    action_any_valids = []
    state_any_valids = []

    for row, (source_cell, goal_cell) in enumerate(zip(source_cells, goal_cells)):
        source_cell = int(source_cell)
        goal_cell = int(goal_cell)
        next_cells = candidate_next_cells[row].astype(np.int32)

        source_finite = (cell_distances_raw[source_cell] >= 0) & has_state
        right_finite = (cell_distances_raw[:, goal_cell] >= 0) & has_state
        next_finite = (cell_distances_raw[next_cells] >= 0) & right_finite[None, :]
        finite = (source_finite & right_finite) | next_finite.any(axis=0)
        fallback = np.nonzero(finite)[0]
        if len(fallback) == 0:
            raise ValueError("No finite subgoal candidates for a cached query.")

        d_source = step_distances[source_cell]
        d_next = step_distances[next_cells]
        d_right = step_distances[:, goal_cell]
        state_valid = (source_finite & right_finite) & (
            d_source <= float(left_budget)
        ) & (d_right <= float(right_budget))
        action_valid = next_finite & (
            d_next <= max(float(left_budget) - 1.0, 1.0)
        ) & (d_right[None, :] <= float(right_budget))
        preferred = np.nonzero(state_valid | action_valid.any(axis=0))[0]

        chosen = []
        preferred_count = min(len(preferred), int(num_subgoals) // 2)
        if preferred_count > 0:
            chosen.extend(
                rng.choice(preferred, size=preferred_count, replace=False).tolist()
            )
        remaining = int(num_subgoals) - len(chosen)
        chosen_set = set(chosen)
        pool = np.asarray([cell for cell in fallback if cell not in chosen_set])
        if len(pool) == 0:
            pool = fallback
        chosen.extend(
            rng.choice(pool, size=remaining, replace=len(pool) < remaining).tolist()
        )
        chosen = np.asarray(chosen[: int(num_subgoals)], dtype=np.int32)

        subgoal_idxs = [int(rng.choice(goal_by_cell[int(cell)])) for cell in chosen]
        subgoal_observations.append(np.asarray(val_dataset["observations"])[subgoal_idxs])
        subgoal_cells.append(chosen)
        source_distances.append(d_source[chosen])
        next_distances.append(d_next[:, chosen])
        right_distances.append(d_right[chosen])
        state_valids.append(state_valid[chosen].astype(np.float32))
        action_valids.append(action_valid[:, chosen].astype(np.float32))
        action_any_valids.append(float(action_valid.any()))
        state_any_valids.append(float(state_valid.any()))

    return dict(
        source_observations=np.asarray(queries["observations"][:, 0], dtype=np.float32),
        candidate_observations=np.asarray(
            queries["candidate_observations"], dtype=np.float32
        ),
        actions=np.asarray(queries["actions"], dtype=np.float32),
        goals=np.asarray(queries["goals"][:, 0], dtype=np.float32),
        subgoal_observations=np.asarray(subgoal_observations, dtype=np.float32),
        subgoal_cells=np.asarray(subgoal_cells, dtype=np.int32),
        source_distances=np.asarray(source_distances, dtype=np.float32),
        next_distances=np.asarray(next_distances, dtype=np.float32),
        right_distances=np.asarray(right_distances, dtype=np.float32),
        direct_source_distances=np.asarray(
            queries["source_distances"], dtype=np.float32
        ),
        direct_next_distances=np.asarray(queries["distances"], dtype=np.float32),
        state_valids=np.asarray(state_valids, dtype=np.float32),
        action_valids=np.asarray(action_valids, dtype=np.float32),
        budget=int(left_budget) + int(right_budget),
        left_budget=int(left_budget),
        right_budget=int(right_budget),
        candidate_action_count=int(queries["actions"].shape[1]),
        num_subgoals=int(num_subgoals),
        oracle_any_action_valid_frac=float(np.mean(action_any_valids)),
        oracle_any_state_valid_frac=float(np.mean(state_any_valids)),
    )


def pair_shape(batch):
    return (
        batch["source_observations"].shape[0],
        batch["candidate_action_count"],
        batch["num_subgoals"],
    )


def source_mode_inputs(batch):
    num_queries, num_actions, num_subgoals = pair_shape(batch)
    observations = np.repeat(
        batch["source_observations"][:, None, None, :],
        num_actions,
        axis=1,
    )
    observations = np.repeat(observations, num_subgoals, axis=2)
    return observations


def own_state_inputs(batch):
    num_queries, _, num_subgoals = pair_shape(batch)
    return np.repeat(batch["candidate_observations"][:, :, None, :], num_subgoals, axis=2)


def pair_actions(batch):
    _, _, num_subgoals = pair_shape(batch)
    return np.repeat(batch["actions"][:, :, None, :], num_subgoals, axis=2)


def pair_subgoals(batch):
    num_queries, num_actions, _ = pair_shape(batch)
    subgoals = np.repeat(
        batch["subgoal_observations"][:, None, :, :], num_actions, axis=1
    )
    assert subgoals.shape[:3] == (num_queries, num_actions, batch["num_subgoals"])
    return subgoals


def value_subgoal_scores(value_agent, batch, branch, batch_size):
    num_queries, num_actions, num_subgoals = pair_shape(batch)
    subgoals = batch["subgoal_observations"]
    actions = np.repeat(batch["actions"][:, :1, :], num_subgoals, axis=1)
    if branch == "left":
        observations = np.repeat(
            batch["source_observations"][:, None, :], num_subgoals, axis=1
        )
        goals = subgoals
        budget = int(batch["left_budget"])
    elif branch == "right":
        observations = subgoals
        goals = np.repeat(batch["goals"][:, None, :], num_subgoals, axis=1)
        budget = int(batch["right_budget"])
    else:
        raise ValueError(f"Unsupported value branch: {branch}")
    budgets = np.full(num_queries * num_subgoals, budget, dtype=np.int32)
    mean, min_score = score_flat(
        value_agent,
        observations.reshape((-1, observations.shape[-1])),
        actions.reshape((-1, actions.shape[-1])),
        goals.reshape((-1, goals.shape[-1])),
        budgets,
        batch_size,
    )
    mean = mean.reshape((num_queries, num_subgoals))
    min_score = min_score.reshape((num_queries, num_subgoals))
    mean = np.repeat(mean[:, None, :], num_actions, axis=1)
    min_score = np.repeat(min_score[:, None, :], num_actions, axis=1)
    return mean, min_score


def score_vv_joint(value_agent, batch, batch_size):
    left_mean, left_min = value_subgoal_scores(value_agent, batch, "left", batch_size)
    right_mean, right_min = value_subgoal_scores(value_agent, batch, "right", batch_size)
    return {
        "mean": np.minimum(left_mean, right_mean),
        "ensemble_min": np.minimum(left_min, right_min),
    }


def score_qv_joint(agent, value_agent, batch, q_state_mode, batch_size):
    if q_state_mode == "source_state":
        observations = source_mode_inputs(batch)
    elif q_state_mode == "own_state":
        observations = own_state_inputs(batch)
    else:
        raise ValueError(f"Unsupported q_state_mode: {q_state_mode}")
    actions = pair_actions(batch)
    goals = pair_subgoals(batch)
    shape = pair_shape(batch)
    budgets = np.full(np.prod(shape), int(batch["left_budget"]), dtype=np.int32)
    q_mean, q_min = score_flat(
        agent,
        observations.reshape((-1, observations.shape[-1])),
        actions.reshape((-1, actions.shape[-1])),
        goals.reshape((-1, goals.shape[-1])),
        budgets,
        batch_size,
    )
    q_mean = q_mean.reshape(shape)
    q_min = q_min.reshape(shape)
    right_mean, right_min = value_subgoal_scores(value_agent, batch, "right", batch_size)
    return {
        "mean": np.minimum(q_mean, right_mean),
        "ensemble_min": np.minimum(q_min, right_min),
    }


def selection_metrics(scores, batch):
    scores = np.asarray(scores, dtype=np.float64)
    num_queries, num_actions, num_subgoals = scores.shape
    flat_selected = np.argmax(scores.reshape((num_queries, -1)), axis=1)
    selected_actions = flat_selected // num_subgoals
    selected_subgoals = flat_selected % num_subgoals
    rows = np.arange(num_queries)

    source_d = batch["source_distances"][rows, selected_subgoals]
    next_d = batch["next_distances"][rows, selected_actions, selected_subgoals]
    right_d = batch["right_distances"][rows, selected_subgoals]
    state_valid = batch["state_valids"][rows, selected_subgoals]
    action_valid = batch["action_valids"][rows, selected_actions, selected_subgoals]
    direct_source = batch["direct_source_distances"]
    direct_next = batch["direct_next_distances"][rows, selected_actions]
    left_budget = float(batch["left_budget"])
    right_budget = float(batch["right_budget"])
    action_left_budget = max(left_budget - 1.0, 1.0)
    midpoint_error = np.abs(source_d - left_budget) + np.abs(right_d - right_budget)
    action_midpoint_error = np.abs(next_d - action_left_budget) + np.abs(
        right_d - right_budget
    )
    return dict(
        state_valid_frac=float(state_valid.mean()),
        action_valid_frac=float(action_valid.mean()),
        selected_source_distance=float(source_d.mean()),
        selected_next_distance=float(next_d.mean()),
        selected_right_distance=float(right_d.mean()),
        source_path_stretch=float((source_d + right_d - direct_source).mean()),
        next_path_stretch=float((next_d + right_d - direct_next).mean()),
        midpoint_error=float(midpoint_error.mean()),
        action_midpoint_error=float(action_midpoint_error.mean()),
        selected_unique_subgoal_cells=int(
            len(np.unique(batch["subgoal_cells"][rows, selected_subgoals]))
        ),
        selected_unique_action_slots=int(len(np.unique(selected_actions))),
        selected_nonlogged_action_frac=float((selected_actions != 0).mean()),
        selected_action_slot_mean=float(selected_actions.mean()),
    )


def oracle_baselines(batch, rng):
    num_queries, num_actions, num_subgoals = pair_shape(batch)
    source_error = np.abs(batch["source_distances"] - float(batch["left_budget"])) + np.abs(
        batch["right_distances"] - float(batch["right_budget"])
    )
    state_scores = -np.repeat(source_error[:, None, :], num_actions, axis=1)
    action_error = np.abs(
        batch["next_distances"] - max(float(batch["left_budget"]) - 1.0, 1.0)
    ) + np.abs(batch["right_distances"][:, None, :] - float(batch["right_budget"]))
    action_scores = 1e3 * batch["action_valids"] - action_error
    return {
        "oracle_action_valid_midpoint": action_scores,
        "oracle_state_midpoint": state_scores,
        "random": rng.random((num_queries, num_actions, num_subgoals)),
    }


def joint_candidate_coverage(batch, action_diag):
    action_left_budget = max(float(batch["left_budget"]) - 1.0, 1.0)
    action_error = np.abs(batch["next_distances"] - action_left_budget) + np.abs(
        batch["right_distances"][:, None, :] - float(batch["right_budget"])
    )
    state_error = np.abs(batch["source_distances"] - float(batch["left_budget"])) + np.abs(
        batch["right_distances"] - float(batch["right_budget"])
    )
    valid_action_error = np.where(batch["action_valids"] > 0.0, action_error, np.inf)
    has_valid_action = np.isfinite(valid_action_error).any(axis=(1, 2))
    valid_state_error = np.where(batch["state_valids"] > 0.0, state_error, np.inf)
    has_valid_state = np.isfinite(valid_state_error).any(axis=1)

    coverage = dict(action_diag)
    coverage.update(
        oracle_any_action_valid_frac=float(batch["action_valids"].any(axis=(1, 2)).mean()),
        oracle_any_state_valid_frac=float(batch["state_valids"].any(axis=1).mean()),
        oracle_action_valid_pair_frac=float(batch["action_valids"].mean()),
        oracle_state_valid_subgoal_frac=float(batch["state_valids"].mean()),
        oracle_best_action_midpoint_error=float(action_error.min(axis=(1, 2)).mean()),
        oracle_best_state_midpoint_error=float(state_error.min(axis=1).mean()),
        oracle_best_valid_action_midpoint_error=float(
            valid_action_error[has_valid_action].min(axis=(1, 2)).mean()
        )
        if has_valid_action.any()
        else float("nan"),
        oracle_best_valid_state_midpoint_error=float(
            valid_state_error[has_valid_state].min(axis=1).mean()
        )
        if has_valid_state.any()
        else float("nan"),
    )
    return coverage


def markdown(result):
    lines = [
        "# BMM joint action-subgoal diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"budget: `{result['budget']}` split `{result['left_budget']}/{result['right_budget']}`",
        f"candidate action mode: `{result['candidate_action_mode']}`",
        f"queries: `{result['num_queries']}`, actions/query: `{result['candidate_action_count']}`, subgoals/query: `{result['num_subgoals']}`",
        f"query cache: `{result['query_cache_path']}`",
        "",
        "## Candidate Set",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in result["candidate_set"].items():
        lines.append(f"| {key} | {format_metric(value)} |")
    lines.extend(
        [
            "",
            "## Scores",
            "",
            "| scorer | score | state_valid | action_valid | source_d | next_d | right_d | source_stretch | next_stretch | midpoint_err | action_mid_err | unique_subgoals | unique_actions | nonlogged_action |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        m = row["metrics"]
        lines.append(
            "| {name} | {score} | {sv} | {av} | {sd} | {nd} | {rd} | {ss} | {ns} | {me} | {ame} | {uc} | {ua} | {nl} |".format(
                name=row["name"],
                score=row["score"],
                sv=format_metric(m["state_valid_frac"]),
                av=format_metric(m["action_valid_frac"]),
                sd=format_metric(m["selected_source_distance"]),
                nd=format_metric(m["selected_next_distance"]),
                rd=format_metric(m["selected_right_distance"]),
                ss=format_metric(m["source_path_stretch"]),
                ns=format_metric(m["next_path_stretch"]),
                me=format_metric(m["midpoint_error"]),
                ame=format_metric(m["action_midpoint_error"]),
                uc=m["selected_unique_subgoal_cells"],
                ua=m["selected_unique_action_slots"],
                nl=format_metric(m["selected_nonlogged_action_frac"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--geodesic_budget_unit", default="env_steps")
    parser.add_argument("--xy_dims", default="0,1")
    parser.add_argument("--budgets", default="40,80,160")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--left_budget", type=int, default=80)
    parser.add_argument("--right_budget", type=int, default=80)
    parser.add_argument("--query_cache_path", required=True)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_subgoals", type=int, default=64)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument(
        "--candidate_action_mode",
        default="same_cell_cached",
        choices=SUPPORTED_CANDIDATE_ACTION_MODES,
    )
    parser.add_argument("--neighbor_hops", type=int, default=2)
    parser.add_argument("--coverage_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critics", nargs="*", default=[])
    parser.add_argument("--value_restore_path", default=None)
    parser.add_argument("--value_restore_epoch", type=int, default=None)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    budgets = ar.parse_int_list(args.budgets)
    rng = np.random.default_rng(args.seed)
    dataset_path = ar.dataset_path_from_dir(args.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    context = ar.make_grid_context(
        env,
        train_dataset,
        val_dataset,
        ar.parse_xy_dims(args.xy_dims),
        args.geodesic_budget_unit,
    )
    queries = ar.load_query_cache(args.query_cache_path)
    queries = ar.hydrate_query_candidate_fields(queries, val_dataset, context, split="val")
    if args.num_queries > 0:
        for key in ar.QUERY_CACHE_KEYS:
            if key in queries:
                queries[key] = queries[key][: args.num_queries]
    queries = apply_candidate_action_mode(
        queries,
        val_dataset,
        context,
        args.candidate_action_mode,
        args.candidate_count,
        args.neighbor_hops,
        rng,
    )
    action_diag = candidate_action_diagnostics(
        queries, xy_dims=ar.parse_xy_dims(args.xy_dims)
    )
    batch = sample_joint_candidates(
        val_dataset,
        context,
        queries,
        args.left_budget,
        args.right_budget,
        args.num_subgoals,
        rng,
    )

    rows = []
    for name, scores in oracle_baselines(batch, rng).items():
        rows.append(
            dict(
                name=name,
                score="oracle",
                metrics=selection_metrics(scores, batch),
            )
        )

    if not args.coverage_only:
        if args.value_restore_path is None or args.value_restore_epoch is None:
            raise ValueError(
                "--value_restore_path and --value_restore_epoch are required unless "
                "--coverage_only is set."
            )
        if not args.critics:
            raise ValueError("--critics is required unless --coverage_only is set.")

        value_agent = ar.configure_restore_agent(
            args, train_dataset, budgets, critic_mode="state"
        )
        value_agent = restore_agent(
            value_agent, args.value_restore_path, args.value_restore_epoch
        )
        vv_scores = score_vv_joint(value_agent, batch, args.score_batch_size)
        for score_name, scores in vv_scores.items():
            rows.append(
                dict(
                    name="V/V_teacher",
                    score=score_name,
                    metrics=selection_metrics(scores, batch),
                )
            )

        for spec in args.critics:
            name, restore_path, restore_epoch = ar.parse_critic_spec(spec)
            agent = ar.configure_restore_agent(
                args, train_dataset, budgets, critic_mode="action"
            )
            agent = restore_agent(agent, restore_path, restore_epoch)
            for q_state_mode in ("source_state", "own_state"):
                scores_by_name = score_qv_joint(
                    agent,
                    value_agent,
                    batch,
                    q_state_mode,
                    args.score_batch_size,
                )
                for score_name, scores in scores_by_name.items():
                    rows.append(
                        dict(
                            name=f"{name}_Q/V_{q_state_mode}",
                            score=score_name,
                            metrics=selection_metrics(scores, batch),
                        )
                    )

    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        budget=int(args.budget),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        candidate_action_mode=args.candidate_action_mode,
        neighbor_hops=int(args.neighbor_hops),
        coverage_only=bool(args.coverage_only),
        num_queries=int(batch["source_observations"].shape[0]),
        candidate_action_count=int(batch["candidate_action_count"]),
        num_subgoals=int(batch["num_subgoals"]),
        query_cache_path=args.query_cache_path,
        value_restore_path=args.value_restore_path,
        value_restore_epoch=None
        if args.value_restore_epoch is None
        else int(args.value_restore_epoch),
        candidate_set=joint_candidate_coverage(batch, action_diag),
        rows=rows,
    )
    text = markdown(result)
    print(text)
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
