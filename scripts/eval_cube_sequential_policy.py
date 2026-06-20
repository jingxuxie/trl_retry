#!/usr/bin/env python
"""Evaluate a cube policy with sequential oracle-representation subgoals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.eval_policy_checkpoint import load_config, parse_int_list
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent


XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALER = 10.0


def oracle_to_xyz(oracle: np.ndarray) -> np.ndarray:
    return oracle.reshape(-1, 3) / XYZ_SCALER + XYZ_CENTER


def xyz_to_oracle(xyz: np.ndarray) -> np.ndarray:
    return ((xyz - XYZ_CENTER) * XYZ_SCALER).reshape(-1)


def current_oracle(env) -> np.ndarray:
    return np.asarray(env.unwrapped.compute_oracle_observation(), dtype=np.float32)


def block_order(current: np.ndarray, final_goal: np.ndarray, mode: str) -> list[int]:
    current_xyz = oracle_to_xyz(current)
    goal_xyz = oracle_to_xyz(final_goal)
    dists = np.linalg.norm(current_xyz - goal_xyz, axis=1)
    ids = list(range(len(dists)))
    if mode == "fixed":
        return ids
    if mode == "nearest":
        return sorted(ids, key=lambda i: dists[i])
    if mode == "farthest":
        return sorted(ids, key=lambda i: -dists[i])
    if mode == "final_z_then_nearest":
        return sorted(ids, key=lambda i: (goal_xyz[i, 2], dists[i]))
    if mode == "final_z_then_farthest":
        return sorted(ids, key=lambda i: (goal_xyz[i, 2], -dists[i]))
    raise ValueError(f"Unknown order mode {mode!r}")


def make_block_subgoal(current: np.ndarray, final_goal_xyz: np.ndarray, block_idx: int) -> np.ndarray:
    subgoal_xyz = oracle_to_xyz(current).copy()
    subgoal_xyz[int(block_idx)] = final_goal_xyz[int(block_idx)]
    return xyz_to_oracle(subgoal_xyz).astype(np.float32)


def make_custom_block_subgoal(current: np.ndarray, block_idx: int, target_xyz: np.ndarray) -> np.ndarray:
    subgoal_xyz = oracle_to_xyz(current).copy()
    subgoal_xyz[int(block_idx)] = np.asarray(target_xyz, dtype=np.float32)
    return xyz_to_oracle(subgoal_xyz).astype(np.float32)


def is_swap_like(current_xyz: np.ndarray, final_goal_xyz: np.ndarray, tolerance: float) -> bool:
    if current_xyz.shape[0] != 2:
        return False
    return bool(
        np.linalg.norm(current_xyz[0] - final_goal_xyz[1]) <= tolerance
        and np.linalg.norm(current_xyz[1] - final_goal_xyz[0]) <= tolerance
    )


def parking_target(current_xyz: np.ndarray, final_goal_xyz: np.ndarray, block_idx: int, args) -> np.ndarray:
    target = current_xyz[int(block_idx)].copy()
    mode = str(args.parking_mode)
    if mode == "beyond_goal_y":
        sign = 1.0 if final_goal_xyz[int(block_idx), 1] >= 0.0 else -1.0
        target[1] = sign * float(args.parking_abs_y)
    elif mode == "behind_start_y":
        sign = 1.0 if current_xyz[int(block_idx), 1] >= 0.0 else -1.0
        target[1] = sign * float(args.parking_abs_y)
    elif mode == "positive_y":
        target[1] = float(args.parking_abs_y)
    elif mode == "negative_y":
        target[1] = -float(args.parking_abs_y)
    elif mode == "x_high":
        target[0] = float(XYZ_CENTER[0] + args.parking_x_delta)
    elif mode == "x_low":
        target[0] = float(XYZ_CENTER[0] - args.parking_x_delta)
    else:
        raise ValueError(f"Unknown parking mode {mode!r}")
    return target


@partial(jax.jit, static_argnames=("action_dim", "flow_steps", "zero_init"))
def sample_gcfbc_flow_action(
    network,
    observation,
    goal,
    seed,
    temperature: float,
    action_dim: int,
    flow_steps: int,
    zero_init: bool,
):
    if zero_init:
        actions = jnp.zeros((*observation.shape[:-1], action_dim), dtype=jnp.float32)
    else:
        actions = temperature * jax.random.normal(
            seed,
            (
                *observation.shape[:-1],
                action_dim,
            ),
        )
    for i in range(flow_steps):
        t = jnp.full((*observation.shape[:-1], 1), i / flow_steps)
        vels = network.select("actor_flow")(observation, goal, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1, 1)


def sample_action(agent, observation, goal, rng_holder, temperature: float, flow_sample_mode: str):
    if flow_sample_mode != "agent":
        if str(agent.config.get("agent_name", "")) != "gcfbc":
            raise ValueError("--flow_sample_mode is only supported for GCFBC checkpoints")
        zero_init = flow_sample_mode == "zero"
        if zero_init:
            key = jax.random.PRNGKey(0)
        else:
            rng_holder["rng"], key = jax.random.split(rng_holder["rng"])
        action = sample_gcfbc_flow_action(
            agent.network,
            jnp.asarray(observation),
            jnp.asarray(goal),
            key,
            float(temperature),
            int(agent.config["action_dim"]),
            int(agent.config["flow_steps"]),
            bool(zero_init),
        )
        return np.asarray(action)

    rng_holder["rng"], key = jax.random.split(rng_holder["rng"])
    action = agent.sample_actions(
        observations=np.asarray(observation),
        goals=np.asarray(goal),
        seed=key,
        temperature=float(temperature),
    )
    return np.asarray(action)


def run_goal_segment(
    *,
    agent,
    env,
    observation,
    goal,
    rng_holder,
    max_steps: int,
    temperature: float,
    flow_sample_mode: str,
):
    rows = []
    info = {}
    done = False
    for _ in range(int(max_steps)):
        action = sample_action(agent, observation, goal, rng_holder, temperature, flow_sample_mode)
        observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
        rows.append(
            {
                "reward": float(reward),
                "success": float(info.get("success", 0.0)),
                "oracle": current_oracle(env).tolist(),
            }
        )
        done = bool(terminated or truncated)
        if done:
            break
    return observation, info, done, rows


def evaluate_episode(agent, env, task_id: int, args, rng_holder):
    observation, reset_info = env.reset(options=dict(task_id=int(task_id)))
    final_goal = np.asarray(reset_info["goal"], dtype=np.float32)
    order = block_order(current_oracle(env), final_goal, args.order)
    final_goal_xyz = oracle_to_xyz(final_goal)
    max_steps = int(args.max_steps or getattr(env, "_max_episode_steps", 500))
    steps_used = 0
    done = False
    info = {"success": False}
    segment_summaries = []

    if args.strategy == "one_pass":
        for block_idx in order:
            cur = current_oracle(env)
            cur_xyz = oracle_to_xyz(cur)
            block_dist = float(np.linalg.norm(cur_xyz[block_idx] - final_goal_xyz[block_idx]))
            if block_dist <= float(args.skip_block_tol):
                continue
            subgoal = make_block_subgoal(cur, final_goal_xyz, int(block_idx))
            subgoal_xyz = oracle_to_xyz(subgoal)
            steps = min(int(args.steps_per_block), max_steps - steps_used)
            if steps <= 0:
                break
            observation, info, done, rows = run_goal_segment(
                agent=agent,
                env=env,
                observation=observation,
                goal=subgoal,
                rng_holder=rng_holder,
                max_steps=steps,
                temperature=float(args.eval_temperature),
                flow_sample_mode=str(args.flow_sample_mode),
            )
            steps_used += len(rows)
            cur_after = current_oracle(env)
            cur_after_xyz = oracle_to_xyz(cur_after)
            segment_summaries.append(
                {
                    "block": int(block_idx),
                    "steps": len(rows),
                    "target_xyz": subgoal_xyz[block_idx].tolist(),
                    "block_dist_before": block_dist,
                    "block_dist_after": float(
                        np.linalg.norm(cur_after_xyz[block_idx] - final_goal_xyz[block_idx])
                    ),
                    "success_after": float(info.get("success", 0.0)),
                }
            )
            if done:
                break
    elif args.strategy in {"dynamic_retry", "park_then_dynamic"}:
        segment_steps = int(args.retry_steps_per_block or args.steps_per_block)
        if args.strategy == "park_then_dynamic":
            cur = current_oracle(env)
            cur_xyz = oracle_to_xyz(cur)
            if is_swap_like(cur_xyz, final_goal_xyz, float(args.parking_swap_tol)):
                park_order = block_order(cur, final_goal, args.order)
                block_idx = (
                    int(args.parking_block_index)
                    if int(args.parking_block_index) >= 0
                    else int(park_order[0])
                )
                target_xyz = parking_target(cur_xyz, final_goal_xyz, block_idx, args)
                subgoal = make_custom_block_subgoal(cur, block_idx, target_xyz)
                steps = min(int(args.parking_steps or segment_steps), max_steps - steps_used)
                if steps > 0:
                    observation, info, done, rows = run_goal_segment(
                        agent=agent,
                        env=env,
                        observation=observation,
                        goal=subgoal,
                        rng_holder=rng_holder,
                        max_steps=steps,
                        temperature=float(args.eval_temperature),
                        flow_sample_mode=str(args.flow_sample_mode),
                    )
                    steps_used += len(rows)
                    cur_after = current_oracle(env)
                    cur_after_xyz = oracle_to_xyz(cur_after)
                    segment_summaries.append(
                        {
                            "block": block_idx,
                            "mode": "park",
                            "steps": len(rows),
                            "target_xyz": target_xyz.tolist(),
                            "block_dist_before": float(
                                np.linalg.norm(cur_xyz[block_idx] - target_xyz)
                            ),
                            "block_dist_after": float(
                                np.linalg.norm(cur_after_xyz[block_idx] - target_xyz)
                            ),
                            "success_after": float(info.get("success", 0.0)),
                        }
                    )
        for pass_id in range(int(args.max_segment_passes)):
            if done:
                break
            cur = current_oracle(env)
            cur_xyz = oracle_to_xyz(cur)
            dists = np.linalg.norm(cur_xyz - final_goal_xyz, axis=1)
            remaining = [
                block_idx
                for block_idx in block_order(cur, final_goal, args.order)
                if dists[block_idx] > float(args.skip_block_tol)
            ]
            if not remaining:
                break
            block_idx = int(remaining[0])
            block_dist = float(dists[block_idx])
            subgoal = make_block_subgoal(cur, final_goal_xyz, block_idx)
            subgoal_xyz = oracle_to_xyz(subgoal)
            steps = min(segment_steps, max_steps - steps_used)
            if steps <= 0:
                break
            observation, info, done, rows = run_goal_segment(
                agent=agent,
                env=env,
                observation=observation,
                goal=subgoal,
                rng_holder=rng_holder,
                max_steps=steps,
                temperature=float(args.eval_temperature),
                flow_sample_mode=str(args.flow_sample_mode),
            )
            steps_used += len(rows)
            cur_after = current_oracle(env)
            cur_after_xyz = oracle_to_xyz(cur_after)
            segment_summaries.append(
                {
                    "block": block_idx,
                    "pass": pass_id + 1,
                    "steps": len(rows),
                    "target_xyz": subgoal_xyz[block_idx].tolist(),
                    "block_dist_before": block_dist,
                    "block_dist_after": float(
                        np.linalg.norm(cur_after_xyz[block_idx] - final_goal_xyz[block_idx])
                    ),
                    "success_after": float(info.get("success", 0.0)),
                }
            )
            if done or bool(info.get("success", 0.0)):
                break
    else:
        raise ValueError(f"Unknown strategy {args.strategy!r}")

    if not done and steps_used < max_steps and int(args.final_steps) > 0:
        steps = min(int(args.final_steps), max_steps - steps_used)
        observation, info, done, rows = run_goal_segment(
            agent=agent,
            env=env,
            observation=observation,
            goal=final_goal,
            rng_holder=rng_holder,
            max_steps=steps,
            temperature=float(args.eval_temperature),
            flow_sample_mode=str(args.flow_sample_mode),
        )
        steps_used += len(rows)
        segment_summaries.append(
            {
                "block": "final",
                "steps": len(rows),
                "success_after": float(info.get("success", 0.0)),
            }
        )

    final_oracle = current_oracle(env)
    final_xyz = oracle_to_xyz(final_oracle)
    final_dists = np.linalg.norm(final_xyz - final_goal_xyz, axis=1)
    return {
        "task": int(task_id),
        "success": float(info.get("success", 0.0)),
        "steps": int(steps_used),
        "truncated": bool(steps_used >= max_steps and not bool(info.get("success", 0.0))),
        "order": [int(item) for item in order],
        "final_block_dists": final_dists.tolist(),
        "max_final_block_dist": float(np.max(final_dists)),
        "mean_final_block_dist": float(np.mean(final_dists)),
        "segments": segment_summaries,
    }


def aggregate(rows):
    successes = sum(float(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "success": successes / max(1, len(rows)),
        "successes": int(round(successes)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "max_final_block_dist": float(np.mean([row["max_final_block_dist"] for row in rows])),
        "mean_final_block_dist": float(np.mean([row["mean_final_block_dist"] for row in rows])),
    }


def markdown(result):
    lines = [
        "# Cube sequential policy evaluation",
        "",
        f"checkpoint: `{result['restore_path']}:{result['restore_epoch']}`",
        f"env: `{result['env_name']}`",
        f"strategy: `{result['strategy']}`",
        f"flow sample mode: `{result['flow_sample_mode']}`",
        f"order: `{result['order']}`",
        f"steps per block: `{result['steps_per_block']}`",
        f"retry steps per block: `{result['retry_steps_per_block']}`",
        f"max segment passes: `{result['max_segment_passes']}`",
        f"final steps: `{result['final_steps']}`",
        "",
        "| task | success | episodes | mean steps | mean max block dist |",
        "|---:|---:|---:|---:|---:|",
    ]
    for task_id, row in result["per_task"].items():
        lines.append(
            "| {task} | {success:.4f} | {episodes} | {steps:.1f} | {dist:.4f} |".format(
                task=task_id,
                success=row["success"],
                episodes=row["episodes"],
                steps=row["mean_steps"],
                dist=row["max_final_block_dist"],
            )
        )
    overall = result["overall"]
    lines.extend(
        [
            "",
            "| overall success | successes | episodes | mean steps |",
            "|---:|---:|---:|---:|",
            "| {success:.4f} | {succ} | {eps} | {steps:.1f} |".format(
                success=overall["success"],
                succ=overall["successes"],
                eps=overall["episodes"],
                steps=overall["mean_steps"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_path", required=True)
    parser.add_argument("--restore_epoch", type=int, required=True)
    parser.add_argument("--env_name", default="cube-double-play-oraclerep-v0")
    parser.add_argument("--task_ids", default="1,2,3,4,5")
    parser.add_argument("--eval_episodes", type=int, default=15)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        default="one_pass",
        choices=["one_pass", "dynamic_retry", "park_then_dynamic"],
    )
    parser.add_argument(
        "--flow_sample_mode",
        default="agent",
        choices=["agent", "zero", "temperature_scaled"],
        help=(
            "GCFBC sampling mode. 'agent' preserves the checkpoint API; 'zero' "
            "uses deterministic zero-noise flow extraction; 'temperature_scaled' "
            "scales the flow prior by eval_temperature."
        ),
    )
    parser.add_argument("--order", default="final_z_then_nearest", choices=[
        "fixed",
        "nearest",
        "farthest",
        "final_z_then_nearest",
        "final_z_then_farthest",
    ])
    parser.add_argument("--steps_per_block", type=int, default=180)
    parser.add_argument("--retry_steps_per_block", type=int, default=None)
    parser.add_argument("--max_segment_passes", type=int, default=8)
    parser.add_argument("--parking_steps", type=int, default=None)
    parser.add_argument("--parking_block_index", type=int, default=-1)
    parser.add_argument("--parking_mode", default="behind_start_y", choices=[
        "beyond_goal_y",
        "behind_start_y",
        "positive_y",
        "negative_y",
        "x_high",
        "x_low",
    ])
    parser.add_argument("--parking_abs_y", type=float, default=0.2)
    parser.add_argument("--parking_x_delta", type=float, default=0.15)
    parser.add_argument("--parking_swap_tol", type=float, default=0.04)
    parser.add_argument(
        "--reset_controller_rng_each_episode",
        action="store_true",
        help="Reset the stochastic controller sampling key from task and episode ids.",
    )
    parser.add_argument("--final_steps", type=int, default=140)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--skip_block_tol", type=float, default=0.04)
    parser.add_argument("--block_retry_tol", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    flags, config = load_config(args.restore_path)
    env_name = args.env_name or flags["env_name"]
    np.random.seed(int(args.seed))
    env, train_dataset, _ = make_env_and_datasets(env_name, dataset_path=None)
    train_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_dataset.sample(1)
    agent = agents[config.agent_name].create(int(args.seed), example_batch, config)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    rng_holder = {"rng": jax.random.PRNGKey(int(args.seed))}

    task_rows = defaultdict(list)
    for task_id in parse_int_list(args.task_ids):
        for episode_idx in range(int(args.eval_episodes)):
            if bool(args.reset_controller_rng_each_episode):
                episode_seed = int(args.seed) + 10000 * int(task_id) + int(episode_idx)
                rng_holder["rng"] = jax.random.PRNGKey(episode_seed)
            row = evaluate_episode(agent, env, int(task_id), args, rng_holder)
            task_rows[str(task_id)].append(row)

    all_rows = [row for rows in task_rows.values() for row in rows]
    result = {
        "restore_path": str(args.restore_path),
        "restore_epoch": int(args.restore_epoch),
        "env_name": env_name,
        "agent_name": str(config.agent_name),
        "task_ids": parse_int_list(args.task_ids),
        "eval_episodes": int(args.eval_episodes),
        "strategy": str(args.strategy),
        "flow_sample_mode": str(args.flow_sample_mode),
        "order": str(args.order),
        "steps_per_block": int(args.steps_per_block),
        "retry_steps_per_block": int(args.retry_steps_per_block or args.steps_per_block),
        "max_segment_passes": int(args.max_segment_passes),
        "parking_steps": int(args.parking_steps or (args.retry_steps_per_block or args.steps_per_block)),
        "parking_block_index": int(args.parking_block_index),
        "parking_mode": str(args.parking_mode),
        "reset_controller_rng_each_episode": bool(args.reset_controller_rng_each_episode),
        "final_steps": int(args.final_steps),
        "max_steps": int(args.max_steps or getattr(env, "_max_episode_steps", 500)),
        "per_task": {task_id: aggregate(rows) for task_id, rows in task_rows.items()},
        "overall": aggregate(all_rows),
        "episodes": all_rows,
    }
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
