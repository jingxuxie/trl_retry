#!/usr/bin/env python
"""Value-only subgoal and nearest-neighbor controller diagnostic."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts import eval_bmm_subgoal_selection as subgoal
from scripts.bmm_reachability_utils import format_metric
from utils.flax_utils import restore_agent


def local_transition_pool(by_cell, cells):
    chunks = []
    for cell in np.asarray(cells, dtype=np.int32).reshape(-1):
        cell = int(cell)
        if 0 <= cell < len(by_cell) and len(by_cell[cell]) > 0:
            chunks.append(np.asarray(by_cell[cell], dtype=np.int32))
    if not chunks:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def make_nn_controller_context(train_dataset, context, controller_hops):
    state_to_cell = np.asarray(context["train_state_to_cell"], dtype=np.int32)
    by_cell = ar.transition_indices_by_cell(train_dataset, state_to_cell)
    cell_distances = np.asarray(context["cell_distances"], dtype=np.int32)
    step_distances = cell_distances.astype(np.float32) * float(context["distance_scale"])
    local_pools = []
    neighbor_pools = []
    for source_cell in range(cell_distances.shape[0]):
        local_pools.append(local_transition_pool(by_cell, [source_cell]))
        neighbor_cells = np.nonzero(
            (cell_distances[source_cell] >= 0)
            & (cell_distances[source_cell] <= int(controller_hops))
        )[0]
        neighbor_pools.append(local_transition_pool(by_cell, neighbor_cells))
    return dict(
        state_to_cell=state_to_cell,
        cell_distances=cell_distances,
        step_distances=step_distances,
        local_pools=local_pools,
        neighbor_pools=neighbor_pools,
        controller_hops=int(controller_hops),
    )


def selected_indices(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.argmax(scores, axis=1)


def nn_controller_metrics(scores, batch, controller):
    selected = selected_indices(scores)
    rows = np.arange(len(selected))
    source_cells = np.asarray(batch["source_cells"], dtype=np.int32)
    subgoal_cells = np.asarray(batch["subgoal_cells"], dtype=np.int32)[rows, selected]
    step_distances = np.asarray(controller["step_distances"], dtype=np.float32)
    state_to_cell = np.asarray(controller["state_to_cell"], dtype=np.int32)

    query_source_d = []
    nn_source_d = []
    nn_source_to_query_d = []
    next_d = []
    valid = []
    selected_transition_idxs = []

    for source_cell, subgoal_cell in zip(source_cells, subgoal_cells):
        source_cell = int(source_cell)
        subgoal_cell = int(subgoal_cell)
        query_d = float(step_distances[source_cell, subgoal_cell])
        pool = np.asarray(controller["neighbor_pools"][source_cell], dtype=np.int32)
        if len(pool) == 0:
            query_source_d.append(query_d)
            nn_source_d.append(np.nan)
            nn_source_to_query_d.append(np.nan)
            next_d.append(np.nan)
            valid.append(0.0)
            selected_transition_idxs.append(-1)
            continue

        transition_cells = state_to_cell[pool]
        transition_next_cells = state_to_cell[pool + 1]
        finite = (
            (transition_cells >= 0)
            & (transition_next_cells >= 0)
            & (step_distances[transition_cells, subgoal_cell] >= 0)
            & (step_distances[transition_next_cells, subgoal_cell] >= 0)
            & (step_distances[source_cell, transition_cells] >= 0)
        )
        if not finite.any():
            query_source_d.append(query_d)
            nn_source_d.append(np.nan)
            nn_source_to_query_d.append(np.nan)
            next_d.append(np.nan)
            valid.append(0.0)
            selected_transition_idxs.append(-1)
            continue

        valid_pool = pool[finite]
        valid_sources = transition_cells[finite]
        valid_next = transition_next_cells[finite]
        next_distances = step_distances[valid_next, subgoal_cell]
        best = int(np.argmin(next_distances))
        selected_idx = int(valid_pool[best])
        source_cell_for_action = int(valid_sources[best])
        next_cell_for_action = int(valid_next[best])

        query_source_d.append(query_d)
        nn_source_d.append(float(step_distances[source_cell_for_action, subgoal_cell]))
        nn_source_to_query_d.append(
            float(step_distances[source_cell, source_cell_for_action])
        )
        next_d.append(float(step_distances[next_cell_for_action, subgoal_cell]))
        valid.append(1.0)
        selected_transition_idxs.append(selected_idx)

    query_source_d = np.asarray(query_source_d, dtype=np.float32)
    nn_source_d = np.asarray(nn_source_d, dtype=np.float32)
    nn_source_to_query_d = np.asarray(nn_source_to_query_d, dtype=np.float32)
    next_d = np.asarray(next_d, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    valid_mask = valid > 0.0

    def masked_mean(values):
        if not valid_mask.any():
            return float("nan")
        return float(np.asarray(values, dtype=np.float32)[valid_mask].mean())

    query_improvement = query_source_d - next_d
    transition_improvement = nn_source_d - next_d
    return dict(
        nn_valid_frac=float(valid.mean()),
        nn_query_source_distance=masked_mean(query_source_d),
        nn_action_source_distance=masked_mean(nn_source_d),
        nn_next_distance=masked_mean(next_d),
        nn_source_to_query_distance=masked_mean(nn_source_to_query_d),
        nn_query_improvement=masked_mean(query_improvement),
        nn_transition_improvement=masked_mean(transition_improvement),
        nn_query_reduces_frac=float(((query_improvement > 0.0) & valid_mask).sum() / max(valid_mask.sum(), 1)),
        nn_transition_reduces_frac=float(
            ((transition_improvement > 0.0) & valid_mask).sum() / max(valid_mask.sum(), 1)
        ),
        nn_selected_unique_actions=int(
            len(np.unique([idx for idx in selected_transition_idxs if idx >= 0]))
        ),
    )


def _choice_metrics_for_target(
    selected,
    batch,
    controller,
    *,
    target_cells,
    choice_mode,
):
    rows = np.arange(len(selected))
    source_cells = np.asarray(batch["source_cells"], dtype=np.int32)
    goal_cells = np.asarray(batch["goal_cells"], dtype=np.int32)
    subgoal_cells = np.asarray(batch["subgoal_cells"], dtype=np.int32)[rows, selected]
    step_distances = np.asarray(controller["step_distances"], dtype=np.float32)
    state_to_cell = np.asarray(controller["state_to_cell"], dtype=np.int32)
    pools = controller.get("local_pools", controller["neighbor_pools"])

    valid = []
    source_to_query = []
    subgoal_before = []
    subgoal_after = []
    goal_before = []
    goal_after = []
    selected_transition_idxs = []

    for source_cell, subgoal_cell, goal_cell, target_cell in zip(
        source_cells, subgoal_cells, goal_cells, target_cells
    ):
        source_cell = int(source_cell)
        subgoal_cell = int(subgoal_cell)
        goal_cell = int(goal_cell)
        target_cell = int(target_cell)
        pool = np.asarray(pools[source_cell], dtype=np.int32)
        before_subgoal = float(step_distances[source_cell, subgoal_cell])
        before_goal = float(step_distances[source_cell, goal_cell])
        subgoal_before.append(before_subgoal)
        goal_before.append(before_goal)

        if len(pool) == 0:
            valid.append(0.0)
            source_to_query.append(np.nan)
            subgoal_after.append(np.nan)
            goal_after.append(np.nan)
            selected_transition_idxs.append(-1)
            continue

        transition_cells = state_to_cell[pool]
        transition_next_cells = state_to_cell[pool + 1]
        finite = (
            (transition_cells >= 0)
            & (transition_next_cells >= 0)
            & (step_distances[source_cell, transition_cells] >= 0)
            & (step_distances[transition_next_cells, target_cell] >= 0)
            & (step_distances[transition_next_cells, subgoal_cell] >= 0)
            & (step_distances[transition_next_cells, goal_cell] >= 0)
        )
        if not finite.any():
            valid.append(0.0)
            source_to_query.append(np.nan)
            subgoal_after.append(np.nan)
            goal_after.append(np.nan)
            selected_transition_idxs.append(-1)
            continue

        valid_pool = pool[finite]
        valid_sources = transition_cells[finite]
        valid_next = transition_next_cells[finite]
        target_next_distances = step_distances[valid_next, target_cell]
        subgoal_next_distances = step_distances[valid_next, subgoal_cell]
        goal_next_distances = step_distances[valid_next, goal_cell]

        if choice_mode == "min_target":
            chosen = int(np.argmin(target_next_distances))
            selected_transition_idxs.append(int(valid_pool[chosen]))
            source_to_query.append(float(step_distances[source_cell, valid_sources[chosen]]))
            subgoal_after.append(float(subgoal_next_distances[chosen]))
            goal_after.append(float(goal_next_distances[chosen]))
        elif choice_mode == "random_mean":
            selected_transition_idxs.extend(int(idx) for idx in valid_pool)
            source_to_query.append(
                float(step_distances[source_cell, valid_sources].mean())
            )
            subgoal_after.append(float(subgoal_next_distances.mean()))
            goal_after.append(float(goal_next_distances.mean()))
        else:
            raise ValueError(f"Unsupported choice_mode={choice_mode}")
        valid.append(1.0)

    valid = np.asarray(valid, dtype=np.float32)
    valid_mask = valid > 0.0
    subgoal_before = np.asarray(subgoal_before, dtype=np.float32)
    subgoal_after = np.asarray(subgoal_after, dtype=np.float32)
    goal_before = np.asarray(goal_before, dtype=np.float32)
    goal_after = np.asarray(goal_after, dtype=np.float32)
    source_to_query = np.asarray(source_to_query, dtype=np.float32)

    def masked_mean(values):
        if not valid_mask.any():
            return float("nan")
        return float(np.asarray(values, dtype=np.float32)[valid_mask].mean())

    subgoal_improvement = subgoal_before - subgoal_after
    goal_improvement = goal_before - goal_after
    return dict(
        valid_frac=float(valid.mean()),
        subgoal_before=masked_mean(subgoal_before),
        subgoal_after=masked_mean(subgoal_after),
        subgoal_improvement=masked_mean(subgoal_improvement),
        subgoal_reduces_frac=float(
            ((subgoal_improvement > 0.0) & valid_mask).sum() / max(valid_mask.sum(), 1)
        ),
        goal_before=masked_mean(goal_before),
        goal_after=masked_mean(goal_after),
        goal_improvement=masked_mean(goal_improvement),
        goal_reduces_frac=float(
            ((goal_improvement > 0.0) & valid_mask).sum() / max(valid_mask.sum(), 1)
        ),
        source_to_query=masked_mean(source_to_query),
        selected_unique_actions=int(
            len(np.unique([idx for idx in selected_transition_idxs if idx >= 0]))
        ),
    )


def add_prefixed(metrics, prefix, values):
    for key, value in values.items():
        metrics[f"{prefix}_{key}"] = value


def low_level_controller_metrics(scores, batch, controller):
    selected = selected_indices(scores)
    rows = np.arange(len(selected))
    subgoal_cells = np.asarray(batch["subgoal_cells"], dtype=np.int32)[rows, selected]
    goal_cells = np.asarray(batch["goal_cells"], dtype=np.int32)
    metrics = {}
    add_prefixed(
        metrics,
        "local_progress_max",
        _choice_metrics_for_target(
            selected,
            batch,
            controller,
            target_cells=subgoal_cells,
            choice_mode="min_target",
        ),
    )
    add_prefixed(
        metrics,
        "direct_goal_same_cell",
        _choice_metrics_for_target(
            selected,
            batch,
            controller,
            target_cells=goal_cells,
            choice_mode="min_target",
        ),
    )
    add_prefixed(
        metrics,
        "random_same_cell",
        _choice_metrics_for_target(
            selected,
            batch,
            controller,
            target_cells=subgoal_cells,
            choice_mode="random_mean",
        ),
    )
    return metrics


def geometric_midpoint_scores(batch):
    midpoint = 0.5 * (
        np.asarray(batch["source_observations"], dtype=np.float32)[:, None, :2]
        + np.asarray(batch["goals"], dtype=np.float32)[:, None, :2]
    )
    subgoals = np.asarray(batch["subgoal_observations"], dtype=np.float32)
    return -np.linalg.norm(subgoals[:, :, :2] - midpoint, axis=-1)


def score_rows(value_agent, batch, batch_size, rng):
    rows = []
    for name, scores in subgoal.oracle_baselines(batch, rng).items():
        rows.append((name, scores))
    rows.append(("geometric_midpoint", geometric_midpoint_scores(batch)))
    rows.append(("BMM_V_value", subgoal.score_vv(value_agent, batch, batch_size)))
    return rows


def row_metrics(scores, batch, controller):
    metrics = dict(subgoal.selection_metrics(scores, batch))
    metrics.update(nn_controller_metrics(scores, batch, controller))
    metrics.update(low_level_controller_metrics(scores, batch, controller))
    return metrics


def markdown(result):
    lines = [
        "# BMM value-subgoal controller diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"budget: `{result['budget']}` split `{result['left_budget']}/{result['right_budget']}`",
        f"queries: `{result['num_queries']}`, candidates/query: `{result['num_candidates']}`",
        f"controller hops: `{result['controller_hops']}`",
        f"value checkpoint: `{result['value_restore_path']}:{result['value_restore_epoch']}`",
        "",
        "| scorer | state_valid | source_stretch | midpoint_err | source_d | right_d | local_subgoal_improve | local_goal_improve | local_goal_reduce | direct_goal_improve | random_subgoal_improve | unique_cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        m = row["metrics"]
        lines.append(
            "| {name} | {sv} | {ss} | {me} | {sd} | {rd} | {lsi} | {lgi} | {lgr} | {dgi} | {rsi} | {uc} |".format(
                name=row["name"],
                sv=format_metric(m["state_valid_frac"]),
                ss=format_metric(m["source_path_stretch"]),
                me=format_metric(m["midpoint_error"]),
                sd=format_metric(m["selected_source_distance"]),
                rd=format_metric(m["selected_right_distance"]),
                lsi=format_metric(m["local_progress_max_subgoal_improvement"]),
                lgi=format_metric(m["local_progress_max_goal_improvement"]),
                lgr=format_metric(m["local_progress_max_goal_reduces_frac"]),
                dgi=format_metric(m["direct_goal_same_cell_goal_improvement"]),
                rsi=format_metric(m["random_same_cell_subgoal_improvement"]),
                uc=m["selected_unique_cells"],
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
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--controller_hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
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
    batch = subgoal.sample_subgoal_candidates(
        val_dataset,
        context,
        queries,
        args.budget,
        args.left_budget,
        args.right_budget,
        args.num_candidates,
        rng,
    )
    batch["source_cells"] = np.asarray(queries["source_cells"], dtype=np.int32)[
        : batch["subgoal_cells"].shape[0]
    ]
    batch["goal_cells"] = np.asarray(queries["goal_cells"], dtype=np.int32)[
        : batch["subgoal_cells"].shape[0]
    ]

    value_agent = ar.configure_restore_agent(
        args, train_dataset, budgets, critic_mode="state"
    )
    value_agent = restore_agent(
        value_agent, args.value_restore_path, args.value_restore_epoch
    )
    controller = make_nn_controller_context(train_dataset, context, args.controller_hops)

    rows = [
        dict(name=name, metrics=row_metrics(scores, batch, controller))
        for name, scores in score_rows(value_agent, batch, args.score_batch_size, rng)
    ]
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        budget=int(args.budget),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_queries=int(batch["subgoal_cells"].shape[0]),
        num_candidates=int(batch["subgoal_cells"].shape[1]),
        controller_hops=int(args.controller_hops),
        query_cache_path=args.query_cache_path,
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
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
