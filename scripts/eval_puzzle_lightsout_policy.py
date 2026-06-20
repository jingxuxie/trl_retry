#!/usr/bin/env python
"""Evaluate a policy with an online Lights Out subgoal planner."""

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.eval_policy_checkpoint import load_config, parse_int_list
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent


def unwrap_env(env):
    cur = env
    for _ in range(16):
        if hasattr(cur, "compute_oracle_observation"):
            return cur
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            cur = cur.unwrapped
            continue
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        break
    raise ValueError("Could not find underlying puzzle environment.")


def lightsout_matrix(rows, cols):
    n = int(rows) * int(cols)
    mat = np.zeros((n, n), dtype=np.uint8)
    for row in range(int(rows)):
        for col in range(int(cols)):
            press = row * int(cols) + col
            for drow, dcol in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                nrow = row + drow
                ncol = col + dcol
                if 0 <= nrow < int(rows) and 0 <= ncol < int(cols):
                    mat[nrow * int(cols) + ncol, press] = 1
    return mat


def gf2_inverse(mat):
    mat = np.asarray(mat, dtype=np.uint8) & 1
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError(f"Expected square matrix, got {mat.shape}.")
    aug = np.concatenate([mat.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for cand in range(row, n):
            if aug[cand, col]:
                pivot = cand
                break
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for other in range(n):
            if other != row and aug[other, col]:
                aug[other] ^= aug[row]
        row += 1
    if row != n:
        raise ValueError("Lights Out transition matrix is not invertible.")
    return aug[:, n:]


def gf2_solve(mat, rhs):
    """Solve mat @ x = rhs over GF(2), returning one solution or None."""
    mat = np.asarray(mat, dtype=np.uint8) & 1
    rhs = np.asarray(rhs, dtype=np.uint8).copy() & 1
    n_rows, n_cols = mat.shape
    aug = np.concatenate([mat.copy(), rhs[:, None]], axis=1)
    pivots = []
    row = 0
    for col in range(n_cols):
        pivot = None
        for cand in range(row, n_rows):
            if aug[cand, col]:
                pivot = cand
                break
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for other in range(n_rows):
            if other != row and aug[other, col]:
                aug[other] ^= aug[row]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break

    for rest in range(row, n_rows):
        if not aug[rest, :n_cols].any() and aug[rest, n_cols]:
            return None

    out = np.zeros(n_cols, dtype=np.uint8)
    for pivot_row, pivot_col in enumerate(pivots):
        out[pivot_col] = aug[pivot_row, n_cols]
    return out


def solve_presses(press_mat, inv_mat, current_bits, target_bits):
    diff = (np.asarray(current_bits, dtype=np.uint8) ^ np.asarray(target_bits, dtype=np.uint8)) & 1
    if inv_mat is not None:
        return (np.asarray(inv_mat, dtype=np.uint8) @ diff) & 1
    return gf2_solve(press_mat, diff)


def bitvec(value):
    return (np.rint(np.asarray(value, dtype=np.float32)).astype(np.uint8) & 1).astype(np.uint8)


class PuzzleLightsOutPolicy:
    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.core = unwrap_env(env)
        self.args = args
        self.rows = int(self.core._num_rows)
        self.cols = int(self.core._num_cols)
        self.n = self.rows * self.cols
        self.press_mat = lightsout_matrix(self.rows, self.cols)
        try:
            self.inv_mat = gf2_inverse(self.press_mat)
            self.solver_mode = "inverse"
        except ValueError:
            self.inv_mat = None
            self.solver_mode = "linear_solve"
        self.rng = jax.random.PRNGKey(int(args.seed) + 2029)

    def current_bits(self):
        return bitvec(self.core.compute_oracle_observation())

    def remaining_presses(self, current_bits, target_bits):
        return solve_presses(self.press_mat, self.inv_mat, current_bits, target_bits)

    def remaining_count(self, current_bits, target_bits):
        presses = self.remaining_presses(current_bits, target_bits)
        if presses is None:
            return int(np.count_nonzero(bitvec(current_bits) ^ bitvec(target_bits)))
        return int(presses.sum())

    def choose_press(self, current_bits, target_bits):
        presses = self.remaining_presses(current_bits, target_bits)
        if presses is None:
            after = np.asarray(current_bits, dtype=np.uint8)[None, :] ^ self.press_mat.T
            distances = np.count_nonzero(after ^ np.asarray(target_bits, dtype=np.uint8)[None, :], axis=1)
            return int(np.argmin(distances))
        candidates = np.nonzero(presses)[0]
        if len(candidates) == 0:
            return None
        if self.args.press_order == "first":
            return int(candidates[0])
        if self.args.press_order == "random":
            return int(candidates[np.random.randint(len(candidates))])
        if self.args.press_order == "nearest":
            effector = np.asarray(self.core._data.site_xpos[self.core._pinch_site_id])
            button_sites = np.asarray(
                [self.core._data.site_xpos[self.core._button_site_ids[int(i)]] for i in candidates]
            )
            return int(candidates[int(np.argmin(np.linalg.norm(button_sites - effector[None, :], axis=1)))])
        raise ValueError(f"Unknown press order {self.args.press_order!r}.")

    def select_goal(self, current_bits, target_bits):
        remaining = self.remaining_count(current_bits, target_bits)
        if remaining <= int(self.args.direct_when_presses_leq):
            return dict(
                goal_bits=np.asarray(target_bits, dtype=np.float32),
                press=None,
                remaining=remaining,
                direct=True,
            )
        goal_bits = np.array(current_bits, copy=True)
        presses = []
        virtual_bits = np.array(current_bits, copy=True)
        for _ in range(max(1, int(self.args.presses_per_subgoal))):
            press = self.choose_press(virtual_bits, target_bits)
            if press is None:
                break
            presses.append(int(press))
            virtual_bits ^= self.press_mat[:, int(press)]
        if len(presses) == 0:
            goal_bits = np.asarray(target_bits, dtype=np.uint8)
        else:
            goal_bits = virtual_bits
        return dict(
            goal_bits=goal_bits.astype(np.float32),
            press=int(presses[0]) if presses else None,
            presses=presses,
            remaining=remaining,
            direct=False,
        )

    def action(self, observation, goal_bits):
        self.rng, key = jax.random.split(self.rng)
        action = self.agent.sample_actions(
            np.asarray(observation, dtype=np.float32),
            goals=np.asarray(goal_bits, dtype=np.float32),
            seed=key,
            temperature=float(self.args.eval_temperature),
        )
        return np.asarray(action).astype(np.float32)


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def evaluate(policy, task_ids, episodes_per_task, max_steps, args):
    rows = []
    for task_id in task_ids:
        for episode in range(int(episodes_per_task)):
            t0 = time.time()
            reset_seed = int(args.seed) + 1009 * int(task_id) + int(episode)
            obs, info = policy.env.reset(seed=reset_seed, options={"task_id": int(task_id)})
            target_bits = bitvec(info["goal"])
            start_bits = policy.current_bits()
            start_presses = policy.remaining_count(start_bits, target_bits)
            active = None
            active_start_bits = None
            active_until = -1
            selected_remaining = []
            selected_direct = []
            intended = 0
            state_changes = 0
            success = 0.0
            terminated = truncated = False
            for step in range(int(max_steps)):
                current_bits = policy.current_bits()
                if policy.remaining_count(current_bits, target_bits) == 0:
                    active = dict(goal_bits=target_bits.astype(np.float32), direct=True, remaining=0)
                need_new = active is None or step >= active_until
                if active is not None and np.array_equal(current_bits, bitvec(active["goal_bits"])):
                    need_new = True
                if active_start_bits is not None and not np.array_equal(current_bits, active_start_bits):
                    state_changes += 1
                    if active is not None and np.array_equal(current_bits, bitvec(active["goal_bits"])):
                        intended += 1
                    need_new = True
                if need_new:
                    active = policy.select_goal(current_bits, target_bits)
                    active_start_bits = np.array(current_bits, copy=True)
                    active_until = step + max(1, int(args.subgoal_commit_steps))
                    selected_remaining.append(float(active["remaining"]))
                    selected_direct.append(float(active["direct"]))
                action = policy.action(obs, active["goal_bits"])
                if policy.agent.config.get("pe_type") != "discrete":
                    action = np.clip(action, -1.0, 1.0)
                obs, reward, terminated, truncated, step_info = policy.env.step(action)
                success = float(step_info.get("success", False))
                if success or terminated or truncated:
                    break
            final_bits = policy.current_bits()
            final_presses = policy.remaining_count(final_bits, target_bits)
            row = dict(
                task=int(task_id),
                episode=int(episode),
                success=float(success),
                steps=int(step + 1),
                start_presses=int(start_presses),
                final_presses=int(final_presses),
                press_improve=float(start_presses - final_presses),
                state_changes=int(state_changes),
                intended_changes=int(intended),
                intended_change_frac=float(intended / max(state_changes, 1)),
                selected_remaining=finite_mean(selected_remaining),
                selected_direct=finite_mean(selected_direct),
                wall_s=float(time.time() - t0),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            rows.append(row)
            print(
                "task={task} ep={episode} success={success:.1f} steps={steps} "
                "presses {start_presses}->{final_presses} changes={state_changes} "
                "intended={intended_change_frac:.2f}".format(**row),
                flush=True,
            )
    return rows


def aggregate(rows):
    out = {}
    for key in [
        "success",
        "steps",
        "start_presses",
        "final_presses",
        "press_improve",
        "state_changes",
        "intended_changes",
        "intended_change_frac",
        "selected_remaining",
        "selected_direct",
        "wall_s",
    ]:
        out[key] = finite_mean([row.get(key, float("nan")) for row in rows])
    out["episodes"] = len(rows)
    out["steps_per_s"] = float(
        np.sum([row.get("steps", 0) for row in rows])
        / max(np.sum([row.get("wall_s", 0.0) for row in rows]), 1e-9)
    )
    return out


def markdown(result):
    lines = [
        "# Puzzle Lights Out policy evaluation",
        "",
        f"checkpoint: `{result['restore_path']}:{result['restore_epoch']}`",
        f"env: `{result['env_name']}`",
        f"episodes/task: `{result['episodes_per_task']}`",
        f"press order: `{result['press_order']}`",
        f"solver: `{result['solver_mode']}`",
        f"commit steps: `{result['subgoal_commit_steps']}`",
        f"presses/subgoal: `{result['presses_per_subgoal']}`",
        "",
        "| task | success | start presses | final presses | improve | state changes | intended frac | wall s/ep |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id, agg in result["per_task"].items():
        lines.append(
            "| {task} | {success:.4f} | {start_presses:.2f} | {final_presses:.2f} | "
            "{press_improve:.2f} | {state_changes:.2f} | {intended_change_frac:.4f} | {wall_s:.2f} |".format(
                task=task_id, **agg
            )
        )
    overall = result["overall"]
    lines.extend(
        [
            "",
            "| overall success | final presses | press improve | intended frac | wall s/ep |",
            "|---:|---:|---:|---:|---:|",
            "| {success:.4f} | {final_presses:.2f} | {press_improve:.2f} | {intended_change_frac:.4f} | {wall_s:.2f} |".format(
                **overall
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_path", required=True)
    parser.add_argument("--restore_epoch", type=int, required=True)
    parser.add_argument("--env_name", default=None)
    parser.add_argument("--task_ids", default="1,2,3,4,5")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--subgoal_commit_steps", type=int, default=50)
    parser.add_argument("--presses_per_subgoal", type=int, default=1)
    parser.add_argument("--direct_when_presses_leq", type=int, default=0)
    parser.add_argument("--press_order", default="nearest", choices=("nearest", "first", "random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    flags, config = load_config(args.restore_path)
    env_name = args.env_name or flags["env_name"]

    np.random.seed(int(args.seed))
    env, train_dataset, _ = make_env_and_datasets(env_name, dataset_path=None)
    gc_train = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = gc_train.sample(1)
    agent = agents[config.agent_name].create(int(args.seed), example_batch, config)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    policy = PuzzleLightsOutPolicy(agent, env, args)
    task_ids = parse_int_list(args.task_ids)
    rows = evaluate(policy, task_ids, args.eval_episodes, args.max_steps, args)
    per_task = {
        str(task_id): aggregate([row for row in rows if row["task"] == int(task_id)])
        for task_id in task_ids
    }
    result = dict(
        restore_path=str(args.restore_path),
        restore_epoch=int(args.restore_epoch),
        env_name=env_name,
        eval_episodes=int(args.eval_episodes),
        episodes_per_task=int(args.eval_episodes),
        task_ids=task_ids,
        max_steps=int(args.max_steps),
        subgoal_commit_steps=int(args.subgoal_commit_steps),
        presses_per_subgoal=int(args.presses_per_subgoal),
        direct_when_presses_leq=int(args.direct_when_presses_leq),
        press_order=str(args.press_order),
        solver_mode=str(policy.solver_mode),
        rows=rows,
        per_task=per_task,
        overall=aggregate(rows),
    )
    text = markdown(result)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    if args.output_markdown:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
