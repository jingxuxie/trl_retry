#!/usr/bin/env python
"""Tiny value-subgoal policy smoke with a nearest-neighbor low-level controller."""

import argparse
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
from scripts import eval_bmm_value_subgoal_controller as ctl
from utils.flax_utils import restore_agent
from utils.pointmaze_grid import unwrap_maze_env, xy_to_ij


def parse_int_list(value):
    value = str(value).strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value):
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_selector(name):
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "random": "random",
        "geometric": "geometric_midpoint",
        "geometric_midpoint": "geometric_midpoint",
        "bmm": "BMM_V",
        "bmm_v": "BMM_V",
        "bmm_v_value": "BMM_V",
        "value": "BMM_V",
        "oracle": "oracle_midpoint",
        "oracle_midpoint": "oracle_midpoint",
        "oracle_state_midpoint": "oracle_midpoint",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown selector '{name}'. Expected one of: "
            "random, geometric_midpoint, BMM_V, oracle_midpoint."
        )
    return aliases[key]


def finite_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


class ValueSubgoalNNPolicy:
    def __init__(
        self,
        value_agent,
        train_dataset,
        context,
        maze_env,
        left_budget,
        right_budget,
        controller_hops,
        num_subgoal_candidates,
        rng,
        score_batch_size,
        selector="BMM_V",
    ):
        self.value_agent = value_agent
        self.train_dataset = train_dataset
        self.context = context
        self.left_budget = int(left_budget)
        self.right_budget = int(right_budget)
        self.num_subgoal_candidates = int(num_subgoal_candidates)
        self.rng = rng
        self.score_batch_size = int(score_batch_size)
        self.free_cells = np.asarray(context["free_cells"], dtype=np.int32)
        self.cell_to_idx = {
            tuple(int(x) for x in cell): idx for idx, cell in enumerate(self.free_cells)
        }
        self.step_distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
            context["distance_scale"]
        )
        self.train_goal_by_cell = context["train_goal_by_cell"]
        self.has_train_state = np.asarray([len(items) > 0 for items in self.train_goal_by_cell])
        self.action_dim = np.asarray(train_dataset["actions"])[0].shape[-1]
        self.controller = ctl.make_nn_controller_context(
            train_dataset, context, controller_hops
        )
        self.selector = normalize_selector(selector)
        self.maze_unit = float(maze_env._maze_unit)
        self.offset_x = float(maze_env._offset_x)
        self.offset_y = float(maze_env._offset_y)

    def obs_to_cell(self, observation):
        ij = xy_to_ij(
            np.asarray(observation, dtype=np.float32)[None, :2],
            maze_unit=self.maze_unit,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )[0]
        return int(self.cell_to_idx.get(tuple(int(x) for x in ij), -1))

    def candidate_cells(self, source_cell, goal_cell):
        if source_cell < 0 or goal_cell < 0:
            return np.asarray([], dtype=np.int32)
        finite = (
            (self.context["cell_distances"][source_cell] >= 0)
            & (self.context["cell_distances"][:, goal_cell] >= 0)
            & self.has_train_state
        )
        cells = np.nonzero(finite)[0]
        if len(cells) == 0:
            return np.asarray([], dtype=np.int32)
        if len(cells) <= self.num_subgoal_candidates:
            return cells.astype(np.int32)
        # Include a small set of midpoint-like cells, then fill randomly.
        midpoint_error = np.abs(self.step_distances[source_cell, cells] - self.left_budget) + np.abs(
            self.step_distances[cells, goal_cell] - self.right_budget
        )
        midpoint_count = min(len(cells), max(4, self.num_subgoal_candidates // 2))
        chosen = cells[np.argsort(midpoint_error)[:midpoint_count]].tolist()
        remaining = self.num_subgoal_candidates - len(chosen)
        pool = np.asarray([cell for cell in cells if int(cell) not in set(chosen)])
        if remaining > 0 and len(pool) > 0:
            chosen.extend(
                self.rng.choice(pool, size=remaining, replace=len(pool) < remaining).tolist()
            )
        return np.asarray(chosen[: self.num_subgoal_candidates], dtype=np.int32)

    def subgoals_for_cells(self, candidate_cells):
        subgoal_idxs = [
            int(self.rng.choice(self.train_goal_by_cell[int(cell)]))
            for cell in candidate_cells
        ]
        return np.asarray(self.train_dataset["observations"])[subgoal_idxs].astype(
            np.float32
        )

    def score_bmm_subgoals(self, observation, goal, subgoals):
        zeros = np.zeros((len(subgoals), self.action_dim), dtype=np.float32)
        source_obs = np.repeat(
            np.asarray(observation, dtype=np.float32)[None, :], len(subgoals), axis=0
        )
        goal_obs = np.repeat(
            np.asarray(goal, dtype=np.float32)[None, :], len(subgoals), axis=0
        )
        left = np.full(len(subgoals), self.left_budget, dtype=np.int32)
        right = np.full(len(subgoals), self.right_budget, dtype=np.int32)
        left_scores = []
        right_scores = []
        for start in range(0, len(subgoals), self.score_batch_size):
            end = min(start + self.score_batch_size, len(subgoals))
            left_logits = self.value_agent.critic_logits_for(
                source_obs[start:end],
                zeros[start:end],
                subgoals[start:end],
                left[start:end],
                offsets=left[start:end],
            )
            right_logits = self.value_agent.critic_logits_for(
                subgoals[start:end],
                zeros[start:end],
                goal_obs[start:end],
                right[start:end],
                offsets=right[start:end],
            )
            left_scores.append(np.asarray(jax.nn.sigmoid(left_logits)).mean(axis=0))
            right_scores.append(np.asarray(jax.nn.sigmoid(right_logits)).mean(axis=0))
        scores = np.minimum(np.concatenate(left_scores), np.concatenate(right_scores))
        return scores

    def selector_scores(self, observation, goal, candidate_cells, subgoals, source_cell, goal_cell):
        if self.selector == "random":
            return self.rng.random(len(candidate_cells)).astype(np.float32)
        if self.selector == "geometric_midpoint":
            midpoint = 0.5 * (
                np.asarray(observation, dtype=np.float32)[:2]
                + np.asarray(goal, dtype=np.float32)[:2]
            )
            return -np.linalg.norm(subgoals[:, :2] - midpoint[None, :], axis=1)
        if self.selector == "oracle_midpoint":
            source_d = self.step_distances[int(source_cell), candidate_cells]
            right_d = self.step_distances[candidate_cells, int(goal_cell)]
            return -(
                np.abs(source_d - float(self.left_budget))
                + np.abs(right_d - float(self.right_budget))
            )
        if self.selector == "BMM_V":
            return self.score_bmm_subgoals(observation, goal, subgoals)
        raise AssertionError(f"Unhandled selector {self.selector}")

    def select_subgoal(self, observation, goal):
        source_cell = self.obs_to_cell(observation)
        goal_cell = self.obs_to_cell(goal)
        if source_cell < 0 or goal_cell < 0:
            return None
        cells = self.candidate_cells(source_cell, goal_cell)
        if len(cells) == 0:
            return None
        subgoals = self.subgoals_for_cells(cells)
        scores = self.selector_scores(
            observation, goal, cells, subgoals, source_cell, goal_cell
        )
        selected = int(np.argmax(scores))
        subgoal_cell = int(cells[selected])
        return dict(
            selector=self.selector,
            subgoal_observation=subgoals[selected],
            subgoal_cell=subgoal_cell,
            subgoal_score=float(scores[selected]),
            source_cell=int(source_cell),
            goal_cell=int(goal_cell),
            source_to_subgoal=float(self.step_distances[source_cell, subgoal_cell]),
            subgoal_to_goal=float(self.step_distances[subgoal_cell, goal_cell]),
            source_to_goal=float(self.step_distances[source_cell, goal_cell]),
        )

    def action_toward(self, source_cell, subgoal_cell):
        pool = np.asarray(self.controller["neighbor_pools"][int(source_cell)], dtype=np.int32)
        if len(pool) == 0:
            return np.zeros(self.action_dim, dtype=np.float32), None
        state_to_cell = np.asarray(self.controller["state_to_cell"], dtype=np.int32)
        next_cells = state_to_cell[pool + 1]
        valid = (next_cells >= 0) & (self.step_distances[next_cells, int(subgoal_cell)] >= 0)
        if not valid.any():
            return np.zeros(self.action_dim, dtype=np.float32), None
        valid_pool = pool[valid]
        valid_next_cells = next_cells[valid]
        next_d = self.step_distances[valid_next_cells, int(subgoal_cell)]
        selected = int(valid_pool[int(np.argmin(next_d))])
        return np.asarray(self.train_dataset["actions"])[selected].astype(np.float32), selected


def run_policy_smoke(env, policy, task_ids, episodes_per_task, max_steps):
    rows = []
    for task_id in task_ids:
        for episode in range(int(episodes_per_task)):
            observation, info = env.reset(options=dict(task_id=int(task_id)))
            goal = info.get("goal")
            if goal is None:
                raise ValueError("Goal-conditioned env reset did not return info['goal'].")
            start_cell = policy.obs_to_cell(observation)
            goal_cell = policy.obs_to_cell(goal)
            start_goal_d = (
                float(policy.step_distances[start_cell, goal_cell])
                if start_cell >= 0 and goal_cell >= 0
                else float("nan")
            )
            subgoal_improvements = []
            goal_improvements = []
            subgoal_valids = []
            selected_scores = []
            source_to_subgoals = []
            subgoal_to_goals = []
            action_count = 0
            done = False
            final_info = {}
            final_observation = observation
            for step in range(int(max_steps)):
                choice = policy.select_subgoal(observation, goal)
                if choice is None:
                    action = np.zeros(policy.action_dim, dtype=np.float32)
                else:
                    if hasattr(policy, "action_for_choice"):
                        action = policy.action_for_choice(observation, goal, choice)
                    else:
                        action, _ = policy.action_toward(
                            choice["source_cell"], choice["subgoal_cell"]
                        )
                    subgoal_valids.append(
                        float(
                            choice["source_to_subgoal"] <= policy.left_budget
                            and choice["subgoal_to_goal"] <= policy.right_budget
                        )
                    )
                    selected_scores.append(choice["subgoal_score"])
                    source_to_subgoals.append(choice["source_to_subgoal"])
                    subgoal_to_goals.append(choice["subgoal_to_goal"])
                    before_subgoal = choice["source_to_subgoal"]
                    before_goal = choice["source_to_goal"]
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                next_observation, reward, terminated, truncated, info = env.step(action)
                final_info = info
                final_observation = next_observation
                if choice is not None:
                    next_cell = policy.obs_to_cell(next_observation)
                    if next_cell >= 0:
                        after_subgoal = float(
                            policy.step_distances[next_cell, choice["subgoal_cell"]]
                        )
                        after_goal = float(policy.step_distances[next_cell, choice["goal_cell"]])
                        subgoal_improvements.append(before_subgoal - after_subgoal)
                        goal_improvements.append(before_goal - after_goal)
                action_count += 1
                observation = next_observation
                done = bool(terminated or truncated)
                if done:
                    break
            final_cell = policy.obs_to_cell(final_observation)
            final_goal_d = (
                float(policy.step_distances[final_cell, goal_cell])
                if final_cell >= 0 and goal_cell >= 0
                else float("nan")
            )
            rows.append(
                dict(
                    task_id=int(task_id),
                    episode=int(episode),
                    steps=int(action_count),
                    done=bool(done),
                    success=float(final_info.get("success", final_info.get("episode", {}).get("success", 0.0))),
                    start_goal_distance=start_goal_d,
                    final_goal_distance=final_goal_d,
                    goal_distance_improvement=start_goal_d - final_goal_d,
                    mean_step_goal_improvement=finite_mean(goal_improvements),
                    mean_step_subgoal_improvement=finite_mean(subgoal_improvements),
                    subgoal_reduce_frac=float(
                        np.mean(np.asarray(subgoal_improvements) > 0.0)
                    )
                    if len(subgoal_improvements)
                    else float("nan"),
                    goal_reduce_frac=float(np.mean(np.asarray(goal_improvements) > 0.0))
                    if len(goal_improvements)
                    else float("nan"),
                    subgoal_valid_frac=finite_mean(subgoal_valids),
                    selected_score_mean=finite_mean(selected_scores),
                    selected_source_to_subgoal=finite_mean(source_to_subgoals),
                    selected_subgoal_to_goal=finite_mean(subgoal_to_goals),
                )
            )
    return rows


def aggregate(rows):
    keys = [
        "success",
        "steps",
        "start_goal_distance",
        "final_goal_distance",
        "goal_distance_improvement",
        "mean_step_goal_improvement",
        "mean_step_subgoal_improvement",
        "subgoal_reduce_frac",
        "goal_reduce_frac",
        "subgoal_valid_frac",
        "selected_score_mean",
        "selected_source_to_subgoal",
        "selected_subgoal_to_goal",
    ]
    return {key: finite_mean([row[key] for row in rows]) for key in keys}


def markdown(result):
    lines = [
        "# BMM value-subgoal policy smoke",
        "",
        f"env: `{result['env_name']}`",
        f"task ids: `{result['task_ids']}`",
        f"episodes/task: `{result['episodes_per_task']}`, max steps: `{result['max_steps']}`",
        f"controller hops: `{result['controller_hops']}`",
        "",
        "| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce | selected_src_d | selected_right_d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {success:.4f} | {final:.4f} | {improve:.4f} | {mean_goal:.4f} | {valid:.4f} | {subgoal_reduce:.4f} | {goal_reduce:.4f} | {src_d:.4f} | {right_d:.4f} |".format(
                name=row["name"],
                success=m["success"],
                final=m["final_goal_distance"],
                improve=m["goal_distance_improvement"],
                mean_goal=m["mean_step_goal_improvement"],
                valid=m["subgoal_valid_frac"],
                subgoal_reduce=m["subgoal_reduce_frac"],
                goal_reduce=m["goal_reduce_frac"],
                src_d=m["selected_source_to_subgoal"],
                right_d=m["selected_subgoal_to_goal"],
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
    parser.add_argument(
        "--selectors",
        default="BMM_V",
        help="Comma-separated selectors: random,geometric_midpoint,BMM_V,oracle_midpoint.",
    )
    parser.add_argument("--task_ids", default="1")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
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
    task_ids = parse_int_list(args.task_ids)
    selectors = [normalize_selector(name) for name in parse_str_list(args.selectors)]
    selector_results = []
    for selector_idx, selector in enumerate(selectors):
        policy = ValueSubgoalNNPolicy(
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
        )
        rows = run_policy_smoke(
            env,
            policy,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
        )
        for row in rows:
            row["selector"] = selector
        selector_results.append(
            dict(name=selector, aggregate=aggregate(rows), episodes=rows)
        )
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        max_steps=int(args.max_steps),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        controller_hops=int(args.controller_hops),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        selector_names=[row["name"] for row in selector_results],
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        selectors=selector_results,
        # Keep the original single-selector JSON keys for existing consumers.
        aggregate=selector_results[0]["aggregate"] if selector_results else {},
        episodes=selector_results[0]["episodes"] if selector_results else [],
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
