#!/usr/bin/env python
"""Analyze initial BMM-vs-support route choices for scene-graph rollouts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_scene_graph_bc_controller as scene_eval
from utils.pointmaze_graph import load_graph_distance_matrix_npz, load_graph_npz


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def load_successes(path):
    if path is None:
        return {}, None
    result = json.loads(Path(path).read_text())
    name = result["selectors"][0]["name"]
    return {
        (int(row["task"]), int(row["episode"])): float(row["success"])
        for row in result["selectors"][0]["episodes"]
    }, name


def summarize_choices(policy, observation, source_rep, goal_rep):
    source_rep = np.asarray(source_rep, dtype=np.float32)
    goal_rep = np.asarray(goal_rep, dtype=np.float32)
    source_bin = policy.rep_to_bin(source_rep)
    goal_bin = policy.rep_to_bin(goal_rep)
    source_goal_d = policy.graph_distance(source_bin, goal_bin)
    bins, path_cost = policy.candidate_bins(source_bin, goal_bin)
    out = dict(
        source_bin=int(source_bin),
        goal_bin=int(goal_bin),
        source_to_goal=finite_float(source_goal_d),
        source_x=finite_float(source_rep[0]),
        source_y=finite_float(source_rep[1]),
        goal_x=finite_float(goal_rep[0]),
        goal_y=finite_float(goal_rep[1]),
        delta_x=finite_float(goal_rep[0] - source_rep[0]),
        delta_y=finite_float(goal_rep[1] - source_rep[1]),
        num_candidates=int(len(bins)),
    )
    if len(bins) == 0:
        return out

    bmm = policy.bmm_scores(
        observation,
        source_rep,
        goal_rep,
        bins,
        policy.left_budget,
        policy.right_budget,
    )
    combined = -path_cost + float(policy.args.bmm_tiebreak_weight) * bmm
    support_idx = int(np.argmax(-path_cost))
    bmm_idx = int(np.argmax(combined))
    support_rank_in_bmm = int(np.where(np.argsort(combined)[::-1] == support_idx)[0][0]) + 1
    bmm_rank_in_support = int(np.where(np.argsort(-path_cost)[::-1] == bmm_idx)[0][0]) + 1

    support_bin = int(bins[support_idx])
    bmm_bin = int(bins[bmm_idx])
    support_left = policy.graph_distance(source_bin, support_bin)
    support_right = policy.graph_distance(support_bin, goal_bin)
    bmm_left = policy.graph_distance(source_bin, bmm_bin)
    bmm_right = policy.graph_distance(bmm_bin, goal_bin)

    top_two = np.sort(combined)[-2:]
    score_margin = float(top_two[-1] - top_two[-2]) if len(top_two) == 2 else float("nan")

    out.update(
        support_bin=support_bin,
        bmm_bin=bmm_bin,
        same_bin=bool(support_bin == bmm_bin),
        support_path_cost=finite_float(path_cost[support_idx]),
        bmm_path_cost=finite_float(path_cost[bmm_idx]),
        path_cost_delta=finite_float(path_cost[bmm_idx] - path_cost[support_idx]),
        support_bmm_score=finite_float(bmm[support_idx]),
        bmm_bmm_score=finite_float(bmm[bmm_idx]),
        bmm_score_delta=finite_float(bmm[bmm_idx] - bmm[support_idx]),
        support_combined_score=finite_float(combined[support_idx]),
        bmm_combined_score=finite_float(combined[bmm_idx]),
        bmm_combined_margin=finite_float(score_margin),
        support_rank_in_bmm=support_rank_in_bmm,
        bmm_rank_in_support=bmm_rank_in_support,
        support_left=finite_float(support_left),
        support_right=finite_float(support_right),
        bmm_left=finite_float(bmm_left),
        bmm_right=finite_float(bmm_right),
        right_delta=finite_float(bmm_right - support_right),
        left_delta=finite_float(bmm_left - support_left),
    )
    return out


def threshold_diagnostics(rows):
    labeled = [
        row
        for row in rows
        if row.get("bmm_success") is not None and row.get("support_success") is not None
    ]
    if not labeled:
        return []
    features = [
        "source_to_goal",
        "path_cost_delta",
        "bmm_score_delta",
        "bmm_bmm_score",
        "support_bmm_score",
        "bmm_path_cost",
        "support_path_cost",
        "right_delta",
        "left_delta",
        "source_x",
        "source_y",
        "goal_x",
        "goal_y",
        "delta_x",
        "delta_y",
        "support_rank_in_bmm",
        "bmm_rank_in_support",
    ]
    diagnostics = []
    for feature in features:
        values = sorted(
            {
                float(row[feature])
                for row in labeled
                if isinstance(row.get(feature), (int, float))
                and np.isfinite(float(row[feature]))
            }
        )
        if not values:
            continue
        best = None
        for threshold in values:
            for op in ("lt", "ge"):
                successes = 0
                choose_bmm = 0
                for row in labeled:
                    value = float(row[feature])
                    use_bmm = value < threshold if op == "lt" else value >= threshold
                    choose_bmm += int(use_bmm)
                    successes += (
                        row["bmm_success"] if use_bmm else row["support_success"]
                    ) > 0.5
                candidate = dict(
                    feature=feature,
                    op=op,
                    threshold=float(threshold),
                    success=float(successes / len(labeled)),
                    successes=int(successes),
                    episodes=int(len(labeled)),
                    choose_bmm=int(choose_bmm),
                )
                if best is None or candidate["success"] > best["success"]:
                    best = candidate
        if best is not None:
            diagnostics.append(best)
    diagnostics.sort(key=lambda row: (row["success"], row["successes"]), reverse=True)
    return diagnostics


def markdown(result):
    lines = [
        "# Scene-graph route-choice diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"tasks: `{result['task_ids']}`, episodes/task: `{result['episodes_per_task']}`, offset: `{result['episode_offset']}`",
        f"value: `{result['value_restore_path']}:{result['value_restore_epoch']}`",
        "",
    ]
    if result["threshold_diagnostics"]:
        lines.extend(
            [
                "Best one-feature threshold rules, choosing BMM when the condition holds:",
                "",
                "| feature | op | threshold | success | choose_bmm |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in result["threshold_diagnostics"][:10]:
            lines.append(
                "| {feature} | {op} | {threshold:.6g} | {successes}/{episodes} ({success:.4f}) | {choose_bmm} |".format(
                    **row
                )
            )
        lines.append("")
    lines.extend(
        [
            "| task | ep | bmm_success | support_success | same_bin | source_goal | sx | sy | gx | gy | path_delta | bmm_delta | right_delta | bmm_score | support_score |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {task} | {episode} | {bmm_success} | {support_success} | {same_bin} | {source_to_goal:.1f} | {source_x:.2f} | {source_y:.2f} | {goal_x:.2f} | {goal_y:.2f} | {path_cost_delta:.4f} | {bmm_score_delta:.4f} | {right_delta:.1f} | {bmm_bmm_score:.4f} | {support_bmm_score:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="humanoidmaze-giant-navigate-oraclerep-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--distance_matrix_path", default=None)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--budgets", default="256,512,1024,1536")
    parser.add_argument("--budget_feature", default="log_scalar")
    parser.add_argument("--critic_obs_rep_key", default=None)
    parser.add_argument("--critic_absdiff_goal_feature", action="store_true")
    parser.add_argument("--actor_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--value_hidden_dims", default="(1024,1024,1024,1024)")
    parser.add_argument("--layer_norm", default="True")
    parser.add_argument("--left_budget", type=int, default=64)
    parser.add_argument("--right_budget", type=int, default=1536)
    parser.add_argument("--num_subgoal_candidates", type=int, default=128)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--bmm_tiebreak_weight", type=float, default=0.05)
    parser.add_argument("--final_goal_switch_distance", type=float, default=128.0)
    parser.add_argument("--final_goal_switch_mode", choices=("distance", "value_confidence"), default="distance")
    parser.add_argument("--confidence_min_direct_distance", type=float, default=128.0)
    parser.add_argument("--direct_confidence_budget", type=int, default=256)
    parser.add_argument("--direct_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--online_bin_mode", choices=("strict", "nearest"), default="nearest")
    parser.add_argument("--nearest_bin_max_dist", type=float, default=0.0)
    parser.add_argument("--subgoal_rep_mode", choices=("first", "center"), default="center")
    parser.add_argument("--task_ids", default="1,2,3,4,5")
    parser.add_argument("--episodes_per_task", type=int, default=15)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--seed_global_reset_noise", action="store_true")
    parser.add_argument("--bmm_result_json", default=None)
    parser.add_argument("--support_result_json", default=None)
    parser.add_argument("--bc_layer_norm", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)
    args.selector = "BMM_support_path"

    env, train_dataset, _ = make_env_and_datasets(
        args.env_name,
        dataset_path=scene_eval.dataset_path_from_dir(args.dataset_dir),
    )
    graph = load_graph_npz(args.graph_path)
    distance_path = args.distance_matrix_path or str(
        Path(args.graph_path).with_name(f"{Path(args.graph_path).stem}_distance_matrix.npz")
    )
    distances = load_graph_distance_matrix_npz(distance_path, graph)
    if distances is None:
        raise FileNotFoundError(f"Missing or incompatible graph distance matrix: {distance_path}")
    value_agent = scene_eval.configure_value_agent(args, train_dataset)
    policy = scene_eval.SceneGraphPolicy(
        value_agent,
        None,
        None,
        train_dataset,
        graph,
        distances,
        args,
        controller_agent=None,
    )

    bmm_successes, bmm_name = load_successes(args.bmm_result_json)
    support_successes, support_name = load_successes(args.support_result_json)
    task_ids = scene_eval.parse_int_list(args.task_ids)
    rows = []
    for task_id in task_ids:
        for ep in range(int(args.episodes_per_task)):
            episode_id = int(args.episode_offset) + int(ep)
            reset_seed = int(args.seed) + 1009 * int(task_id) + episode_id
            if bool(args.seed_global_reset_noise):
                np.random.seed(reset_seed)
            obs, info = env.reset(seed=reset_seed, options={"task_id": int(task_id)})
            goal_rep = np.asarray(info["goal"], dtype=np.float32)
            source_rep = policy.current_rep(env, obs)
            row = summarize_choices(policy, obs, source_rep, goal_rep)
            key = (int(task_id), int(episode_id))
            row.update(
                task=int(task_id),
                episode=int(episode_id),
                reset_seed=int(reset_seed),
                bmm_success=bmm_successes.get(key),
                support_success=support_successes.get(key),
            )
            rows.append(row)
            print(
                "task={task} ep={episode} bmm_success={bmm_success} "
                "support_success={support_success} same_bin={same_bin} "
                "path_delta={path_cost_delta:.4f} bmm_delta={bmm_score_delta:.4f}".format(
                    **row
                ),
                flush=True,
            )

    result = dict(
        env_name=args.env_name,
        graph_path=str(args.graph_path),
        distance_matrix_path=str(distance_path),
        value_restore_path=str(args.value_restore_path),
        value_restore_epoch=int(args.value_restore_epoch),
        budgets=scene_eval.parse_int_list(args.budgets),
        budget_feature=str(args.budget_feature),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        bmm_tiebreak_weight=float(args.bmm_tiebreak_weight),
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        episode_offset=int(args.episode_offset),
        seed=int(args.seed),
        seed_global_reset_noise=bool(args.seed_global_reset_noise),
        bmm_label_name=bmm_name,
        support_label_name=support_name,
        rows=rows,
        threshold_diagnostics=threshold_diagnostics(rows),
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    out.with_suffix(".md").write_text(markdown(result))
    print(f"Wrote route-choice diagnostic to {out}", flush=True)
    print(f"Wrote markdown summary to {out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
