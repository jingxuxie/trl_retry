#!/usr/bin/env python
"""Inspect value-subgoal choices at OGBench eval-task reset states."""

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
from scripts import eval_bmm_value_subgoal_policy_smoke as smoke
from utils.flax_utils import restore_agent
from utils.pointmaze_grid import unwrap_maze_env


def inspect_selector(policy, observation, goal, top_k):
    source_cell = policy.obs_to_cell(observation)
    goal_cell = policy.obs_to_cell(goal)
    cells = policy.candidate_cells(source_cell, goal_cell)
    if len(cells) == 0:
        return dict(source_cell=source_cell, goal_cell=goal_cell, rows=[])
    support_horizon = policy.support_path_horizon(source_cell, goal_cell, cells)
    subgoals = policy.subgoals_for_cells(cells)
    scores = policy.selector_scores(
        observation, goal, cells, subgoals, source_cell, goal_cell
    )
    source_to_goal = float(policy.step_distances[source_cell, goal_cell])
    order = np.argsort(scores)[::-1][: int(top_k)]
    rows = []
    for rank, idx in enumerate(order, start=1):
        cell = int(cells[int(idx)])
        source_d = float(policy.step_distances[source_cell, cell])
        right_d = float(policy.step_distances[cell, goal_cell])
        progress = source_to_goal - right_d
        path_slack = source_d + right_d - source_to_goal
        rows.append(
            dict(
                rank=int(rank),
                cell=cell,
                score=float(scores[int(idx)]),
                source_to_subgoal=source_d,
                subgoal_to_goal=right_d,
                source_to_goal=source_to_goal,
                goal_progress=float(progress),
                path_slack=float(path_slack),
                valid=bool(
                    source_d <= float(policy.left_budget)
                    and right_d <= float(support_horizon)
                ),
                support_path_horizon=int(support_horizon),
                xy=np.asarray(subgoals[int(idx), :2], dtype=float).tolist(),
            )
        )
    return dict(
        source_cell=int(source_cell),
        goal_cell=int(goal_cell),
        source_to_goal=source_to_goal,
        rows=rows,
    )


def markdown(result):
    lines = [
        "# BMM eval-task subgoal inspection",
        "",
        f"env: `{result['env_name']}`",
        f"task ids: `{result['task_ids']}`",
        f"selectors: `{result['selectors']}`",
        f"budgets: `{result['left_budget']}/{result['right_budget']}`",
        f"support path horizon mode: `{result.get('support_path_horizon_mode', 'fixed')}`",
        f"candidates: `{result['num_subgoal_candidates']}`, top_k: `{result['top_k']}`",
        f"support frontier left gate: `{result.get('support_frontier_left_gate', 'support')}`",
        f"support frontier min progress frac: `{result.get('support_frontier_min_progress_frac', 0.0)}`",
        f"support frontier max xy factor: `{result.get('support_frontier_max_xy_factor', 0.0)}`",
        f"subgoal sample mode: `{result.get('subgoal_sample_mode', 'random')}`",
        f"candidate sample mode: `{result.get('candidate_sample_mode', 'random')}`",
        f"require goal progress: `{result.get('require_goal_progress', False)}`",
        "",
    ]
    for task in result["tasks"]:
        lines.extend(
            [
                f"## Task {task['task_id']}",
                "",
                f"source cell: `{task['source_cell']}`, goal cell: `{task['goal_cell']}`, source_to_goal: `{task['source_to_goal']:.4f}`",
                "",
            ]
        )
        for selector in task["selectors"]:
            lines.extend(
                [
                    f"### {selector['name']}",
                    "",
                    "| rank | cell | score | source_d | right_d | progress | path_slack | support_h | valid | xy |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in selector["rows"]:
                lines.append(
                    "| {rank} | {cell} | {score:.6f} | {source_d:.4f} | {right_d:.4f} | {progress:.4f} | {path_slack:.4f} | {support_h} | {valid} | {xy} |".format(
                        rank=row["rank"],
                        cell=row["cell"],
                        score=row["score"],
                        source_d=row["source_to_subgoal"],
                        right_d=row["subgoal_to_goal"],
                        progress=row["goal_progress"],
                        path_slack=row["path_slack"],
                        support_h=row.get("support_path_horizon", result["right_budget"]),
                        valid=int(row["valid"]),
                        xy=[round(float(x), 3) for x in row["xy"]],
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
    parser.add_argument("--left_budget", type=int, default=80)
    parser.add_argument("--right_budget", type=int, default=80)
    parser.add_argument("--controller_hops", type=int, default=0)
    parser.add_argument("--num_subgoal_candidates", type=int, default=64)
    parser.add_argument("--selectors", default="geometric_midpoint,BMM_V_min")
    parser.add_argument("--task_ids", default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--value_gate_threshold", type=float, default=0.5)
    parser.add_argument("--support_gate_left_frac", type=float, default=1.0)
    parser.add_argument(
        "--support_frontier_left_gate",
        choices=("support", "xy", "grid", "grid_xy", "support_grid_xy"),
        default="support",
    )
    parser.add_argument("--support_frontier_min_progress_frac", type=float, default=0.0)
    parser.add_argument("--support_frontier_max_xy_factor", type=float, default=0.0)
    parser.add_argument(
        "--subgoal_sample_mode",
        choices=("random", "center", "first"),
        default="random",
        help="How to choose a representative observation for each candidate cell.",
    )
    parser.add_argument(
        "--candidate_sample_mode",
        choices=("random", "topk", "stratified"),
        default="random",
        help="How to choose candidate cells when there are more than the cap.",
    )
    parser.add_argument(
        "--require_goal_progress",
        action="store_true",
        help=(
            "Diagnostic extraction gate. If set, candidate subgoals must reduce "
            "grid/geodesic distance to the goal when any such candidate exists."
        ),
    )
    parser.add_argument(
        "--support_path_horizon_mode",
        choices=("fixed", "source_goal_grid", "local_grid_min_right"),
        default="fixed",
        help=(
            "For support-frontier/path selectors, choose the support-distance "
            "cache horizon. 'fixed' uses --right_budget; 'source_goal_grid' "
            "uses max(--right_budget, ceil(grid source-goal distance)); "
            "'local_grid_min_right' uses the smallest grid right distance "
            "among local-progress candidates."
        ),
    )
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    budgets = ar.parse_int_list(args.budgets)
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
    value_agent = ar.configure_restore_agent(
        args, train_dataset, budgets, critic_mode="state"
    )
    value_agent = restore_agent(
        value_agent, args.value_restore_path, args.value_restore_epoch
    )

    selectors = [smoke.normalize_selector(x) for x in smoke.parse_str_list(args.selectors)]
    task_ids = smoke.parse_int_list(args.task_ids)
    tasks = []
    for task_id in task_ids:
        observation, info = env.reset(options=dict(task_id=int(task_id)))
        goal = info["goal"]
        selector_rows = []
        source_cell = None
        goal_cell = None
        source_to_goal = None
        for selector_idx, selector in enumerate(selectors):
            policy = smoke.ValueSubgoalNNPolicy(
                value_agent,
                train_dataset,
                context,
                unwrap_maze_env(env),
                args.left_budget,
                args.right_budget,
                args.controller_hops,
                args.num_subgoal_candidates,
                np.random.default_rng(int(args.seed) + 1009 * selector_idx),
                args.score_batch_size,
                selector=selector,
                budgets=budgets,
                value_gate_threshold=args.value_gate_threshold,
                support_gate_left_frac=args.support_gate_left_frac,
                support_frontier_left_gate=args.support_frontier_left_gate,
                support_frontier_min_progress_frac=args.support_frontier_min_progress_frac,
                support_frontier_max_xy_factor=args.support_frontier_max_xy_factor,
                support_path_horizon_mode=args.support_path_horizon_mode,
                subgoal_sample_mode=args.subgoal_sample_mode,
                candidate_sample_mode=args.candidate_sample_mode,
                require_goal_progress=args.require_goal_progress,
            )
            inspected = inspect_selector(policy, observation, goal, args.top_k)
            source_cell = inspected["source_cell"]
            goal_cell = inspected["goal_cell"]
            source_to_goal = inspected["source_to_goal"]
            selector_rows.append(dict(name=selector, rows=inspected["rows"]))
        tasks.append(
            dict(
                task_id=int(task_id),
                source_cell=int(source_cell),
                goal_cell=int(goal_cell),
                source_to_goal=float(source_to_goal),
                selectors=selector_rows,
            )
        )

    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        task_ids=task_ids,
        selectors=selectors,
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        top_k=int(args.top_k),
        value_gate_threshold=float(args.value_gate_threshold),
        support_gate_left_frac=float(args.support_gate_left_frac),
        support_frontier_left_gate=args.support_frontier_left_gate,
        support_frontier_min_progress_frac=float(args.support_frontier_min_progress_frac),
        support_frontier_max_xy_factor=float(args.support_frontier_max_xy_factor),
        subgoal_sample_mode=args.subgoal_sample_mode,
        candidate_sample_mode=args.candidate_sample_mode,
        require_goal_progress=bool(args.require_goal_progress),
        support_path_horizon_mode=args.support_path_horizon_mode,
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        tasks=tasks,
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
