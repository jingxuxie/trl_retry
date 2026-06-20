#!/usr/bin/env python
"""Trace replanned subgoals during one BMM subgoal-controller rollout."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts import eval_bmm_subgoal_bc_controller as bc_eval
from scripts import eval_bmm_value_subgoal_policy_smoke as smoke
from utils.flax_utils import restore_agent
from utils.pointmaze_grid import unwrap_maze_env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--geodesic_budget_unit", default="env_steps")
    parser.add_argument("--xy_dims", default="0,1")
    parser.add_argument("--budgets", default="20,40,80,160,320")
    parser.add_argument("--left_budget", type=int, default=20)
    parser.add_argument("--right_budget", type=int, default=400)
    parser.add_argument("--controller_hops", type=int, default=0)
    parser.add_argument("--num_subgoal_candidates", type=int, default=64)
    parser.add_argument("--selector", default="BMM_V_min_budget_scan_value_frontier")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--subgoal_commit_steps", type=int, default=5)
    parser.add_argument("--subgoal_replan_distance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset_seed_base", type=int, default=-1)
    parser.add_argument("--value_gate_threshold", type=float, default=0.5)
    parser.add_argument("--support_gate_left_frac", type=float, default=1.0)
    parser.add_argument(
        "--support_frontier_left_gate",
        choices=("support", "xy", "grid", "grid_xy", "support_grid_xy"),
        default="grid",
    )
    parser.add_argument("--support_frontier_min_progress_frac", type=float, default=0.0)
    parser.add_argument("--support_frontier_max_xy_factor", type=float, default=0.0)
    parser.add_argument(
        "--support_path_horizon_mode",
        choices=("fixed", "source_goal_grid", "local_grid_min_right"),
        default="fixed",
    )
    parser.add_argument(
        "--subgoal_sample_mode",
        choices=("random", "center", "first"),
        default="center",
    )
    parser.add_argument(
        "--candidate_sample_mode",
        choices=("random", "topk", "stratified"),
        default="stratified",
    )
    parser.add_argument(
        "--require_goal_progress",
        action="store_true",
        help=(
            "Diagnostic extraction gate. If set, candidate subgoals must reduce "
            "grid/geodesic distance to the goal when any such candidate exists."
        ),
    )
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--actor_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--value_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--layer_norm", default="True")
    parser.add_argument("--bc_restore_path", required=True)
    parser.add_argument("--bc_offsets", default="1,2,4,8,16,32,64,80")
    parser.add_argument("--bc_steps", type=int, default=2000)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_hidden_dims", default="(256,256)")
    parser.add_argument("--bc_lr", type=float, default=3e-4)
    parser.add_argument("--bc_layer_norm", action="store_true")
    parser.add_argument("--bc_goal_rep", choices=("observation", "oracle"), default="observation")
    parser.add_argument("--bc_inference", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--final_goal_switch_distance", type=float, default=20.0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    return parser.parse_args(argv)


def trace_rollout(env, policy, args):
    reset_kwargs = dict(options=dict(task_id=int(args.task_id)))
    if int(args.reset_seed_base) >= 0:
        reset_kwargs["seed"] = int(args.reset_seed_base) + 1009 * int(args.task_id)
    observation, info = env.reset(**reset_kwargs)
    goal = info["goal"]
    goal_cell = policy.obs_to_cell(goal)
    start_cell = policy.obs_to_cell(observation)
    start_goal_d = float(policy.step_distances[start_cell, goal_cell])
    active_choice = None
    active_until_step = 0
    traces = []
    final_info = {}
    final_observation = observation
    select_time = 0.0
    action_time = 0.0
    env_time = 0.0
    t0 = time.perf_counter()
    for step in range(int(args.max_steps)):
        if active_choice is not None:
            refreshed = smoke.refresh_choice_source(policy, active_choice, observation)
            if (
                float(args.subgoal_replan_distance) >= 0.0
                and refreshed is not None
                and refreshed["source_to_subgoal"] <= float(args.subgoal_replan_distance)
            ):
                active_choice = None
        if active_choice is None or step >= active_until_step:
            select_start = time.perf_counter()
            active_choice = policy.select_subgoal(observation, goal)
            select_time += time.perf_counter() - select_start
            active_until_step = step + max(1, int(args.subgoal_commit_steps))
            choice = smoke.refresh_choice_source(policy, active_choice, observation)
            if choice is not None:
                progress = choice["source_to_goal"] - choice["subgoal_to_goal"]
                path_slack = (
                    choice["source_to_subgoal"]
                    + choice["subgoal_to_goal"]
                    - choice["source_to_goal"]
                )
                traces.append(
                    dict(
                        step=int(step),
                        source_cell=int(choice["source_cell"]),
                        goal_cell=int(choice["goal_cell"]),
                        subgoal_cell=int(choice["subgoal_cell"]),
                        source_to_goal=float(choice["source_to_goal"]),
                        source_to_subgoal=float(choice["source_to_subgoal"]),
                        subgoal_to_goal=float(choice["subgoal_to_goal"]),
                        goal_progress=float(progress),
                        path_slack=float(path_slack),
                        subgoal_score=float(choice["subgoal_score"]),
                        subgoal_xy=np.asarray(
                            choice["subgoal_observation"], dtype=float
                        )[:2].tolist(),
                    )
                )
        choice = smoke.refresh_choice_source(policy, active_choice, observation)
        action_start = time.perf_counter()
        if choice is None:
            action = np.zeros(policy.action_dim, dtype=np.float32)
        else:
            action = policy.action_for_choice(observation, goal, choice)
        action_time += time.perf_counter() - action_start
        env_start = time.perf_counter()
        next_observation, _reward, terminated, truncated, info = env.step(
            np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        )
        env_time += time.perf_counter() - env_start
        final_info = info
        final_observation = next_observation
        observation = next_observation
        if terminated or truncated:
            break
    final_cell = policy.obs_to_cell(final_observation)
    final_goal_d = float(policy.step_distances[final_cell, goal_cell])
    return dict(
        task_id=int(args.task_id),
        selector=smoke.normalize_selector(args.selector),
        start_cell=int(start_cell),
        goal_cell=int(goal_cell),
        start_goal_distance=float(start_goal_d),
        final_goal_distance=float(final_goal_d),
        goal_distance_improvement=float(start_goal_d - final_goal_d),
        success=float(final_info.get("success", final_info.get("episode", {}).get("success", 0.0))),
        steps=int(step + 1),
        traces=traces,
        wall_time_sec=float(time.perf_counter() - t0),
        select_time_sec=float(select_time),
        action_time_sec=float(action_time),
        env_step_time_sec=float(env_time),
    )


def markdown(result):
    lines = [
        "# BMM subgoal rollout trace",
        "",
        f"env: `{result['env_name']}`",
        f"task: `{result['task_id']}`, selector: `{result['selector']}`",
        f"success: `{result['success']}`, final_d: `{result['final_goal_distance']:.4f}`, improve: `{result['goal_distance_improvement']:.4f}`",
        f"steps: `{result['steps']}`, replans: `{len(result['traces'])}`",
        f"timing wall/select/action/env: `{result['wall_time_sec']:.3f}` / `{result['select_time_sec']:.3f}` / `{result['action_time_sec']:.3f}` / `{result['env_step_time_sec']:.3f}`",
        f"require goal progress: `{result.get('require_goal_progress', False)}`",
        "",
        "| replan | step | source | subgoal | source_goal | source_subgoal | subgoal_goal | progress | path_slack | score | xy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(result["traces"]):
        lines.append(
            "| {idx} | {step} | {source} | {subgoal} | {source_goal:.4f} | {source_subgoal:.4f} | {subgoal_goal:.4f} | {progress:.4f} | {slack:.4f} | {score:.6f} | {xy} |".format(
                idx=idx,
                step=row["step"],
                source=row["source_cell"],
                subgoal=row["subgoal_cell"],
                source_goal=row["source_to_goal"],
                source_subgoal=row["source_to_subgoal"],
                subgoal_goal=row["subgoal_to_goal"],
                progress=row["goal_progress"],
                slack=row["path_slack"],
                score=row["subgoal_score"],
                xy=[round(float(x), 3) for x in row["subgoal_xy"]],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
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
    bc_model, bc_params, bc_info = bc_eval.load_bc(
        args.bc_restore_path, train_dataset, args
    )
    policy = bc_eval.ValueSubgoalBCPolicy(
        value_agent,
        train_dataset,
        context,
        unwrap_maze_env(env),
        args.left_budget,
        args.right_budget,
        args.controller_hops,
        args.num_subgoal_candidates,
        np.random.default_rng(int(args.seed)),
        args.score_batch_size,
        selector=smoke.normalize_selector(args.selector),
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
        bc_model=bc_model,
        bc_params=bc_params,
        bc_goal_rep=args.bc_goal_rep,
        bc_layer_norm=args.bc_layer_norm,
        bc_inference=args.bc_inference,
        final_goal_switch_distance=args.final_goal_switch_distance,
    )
    rollout = trace_rollout(env, policy, args)
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        max_steps=int(args.max_steps),
        subgoal_commit_steps=int(args.subgoal_commit_steps),
        subgoal_replan_distance=float(args.subgoal_replan_distance),
        subgoal_sample_mode=args.subgoal_sample_mode,
        candidate_sample_mode=args.candidate_sample_mode,
        require_goal_progress=bool(args.require_goal_progress),
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        bc_restore_path=args.bc_restore_path,
        bc_final_info=bc_info,
        **rollout,
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
