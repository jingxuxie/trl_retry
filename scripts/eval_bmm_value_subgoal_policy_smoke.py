#!/usr/bin/env python
"""Tiny value-subgoal policy smoke with a nearest-neighbor low-level controller."""

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

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts import eval_bmm_value_subgoal_controller as ctl
from utils.flax_utils import restore_agent
from utils.pointmaze_graph import valid_transition_indices
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
        "bmm_v_min": "BMM_V_min",
        "bmm_min": "BMM_V_min",
        "value_min": "BMM_V_min",
        "bmm_v_left_gate": "BMM_V_left_gate",
        "bmm_left_gate": "BMM_V_left_gate",
        "bmm_v_min_left_gate": "BMM_V_min_left_gate",
        "bmm_min_left_gate": "BMM_V_min_left_gate",
        "bmm_v_min_budget_scan_left_gate": "BMM_V_min_budget_scan_left_gate",
        "bmm_budget_scan_left_gate": "BMM_V_min_budget_scan_left_gate",
        "bmm_min_budget_scan_left_gate": "BMM_V_min_budget_scan_left_gate",
        "bmm_v_min_budget_scan_value_gate": "BMM_V_min_budget_scan_value_gate",
        "bmm_budget_scan_value_gate": "BMM_V_min_budget_scan_value_gate",
        "bmm_min_budget_scan_value_gate": "BMM_V_min_budget_scan_value_gate",
        "bmm_learned_gate": "BMM_V_min_budget_scan_value_gate",
        "bmm_v_min_budget_scan_value_frontier": "BMM_V_min_budget_scan_value_frontier",
        "bmm_budget_scan_value_frontier": "BMM_V_min_budget_scan_value_frontier",
        "bmm_min_budget_scan_value_frontier": "BMM_V_min_budget_scan_value_frontier",
        "bmm_value_frontier": "BMM_V_min_budget_scan_value_frontier",
        "bmm_frontier": "BMM_V_min_budget_scan_value_frontier",
        "bmm_v_min_budget_scan_right_progress": "BMM_V_min_budget_scan_right_progress",
        "bmm_budget_scan_right_progress": "BMM_V_min_budget_scan_right_progress",
        "bmm_min_budget_scan_right_progress": "BMM_V_min_budget_scan_right_progress",
        "bmm_right_progress": "BMM_V_min_budget_scan_right_progress",
        "bmm_learned_progress": "BMM_V_min_budget_scan_right_progress",
        "bmm_v_min_budget_scan_support_gate": "BMM_V_min_budget_scan_support_gate",
        "bmm_budget_scan_support_gate": "BMM_V_min_budget_scan_support_gate",
        "bmm_min_budget_scan_support_gate": "BMM_V_min_budget_scan_support_gate",
        "bmm_support_gate": "BMM_V_min_budget_scan_support_gate",
        "bmm_v_min_budget_scan_support_frontier": "BMM_V_min_budget_scan_support_frontier",
        "bmm_budget_scan_support_frontier": "BMM_V_min_budget_scan_support_frontier",
        "bmm_min_budget_scan_support_frontier": "BMM_V_min_budget_scan_support_frontier",
        "bmm_support_frontier": "BMM_V_min_budget_scan_support_frontier",
        "bmm_v_min_budget_scan_support_path": "BMM_V_min_budget_scan_support_path",
        "bmm_budget_scan_support_path": "BMM_V_min_budget_scan_support_path",
        "bmm_min_budget_scan_support_path": "BMM_V_min_budget_scan_support_path",
        "bmm_support_path": "BMM_V_min_budget_scan_support_path",
        "bmm_support_path_progress": "BMM_V_min_budget_scan_support_path",
        "support_path": "support_path_only",
        "support_path_only": "support_path_only",
        "support_path_progress": "support_path_only",
        "value": "BMM_V",
        "oracle": "oracle_midpoint",
        "oracle_midpoint": "oracle_midpoint",
        "oracle_state_midpoint": "oracle_midpoint",
        "oracle_path": "oracle_path_progress",
        "oracle_path_progress": "oracle_path_progress",
        "oracle_progress": "oracle_path_progress",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown selector '{name}'. Expected one of: "
            "random, geometric_midpoint, BMM_V, BMM_V_min, "
            "BMM_V_left_gate, BMM_V_min_left_gate, "
            "BMM_V_min_budget_scan_left_gate, "
            "BMM_V_min_budget_scan_value_gate, "
            "BMM_V_min_budget_scan_value_frontier, "
            "BMM_V_min_budget_scan_right_progress, "
            "BMM_V_min_budget_scan_support_gate, "
            "BMM_V_min_budget_scan_support_frontier, "
            "BMM_V_min_budget_scan_support_path, support_path_only, "
            "oracle_midpoint, "
            "oracle_path_progress."
        )
    return aliases[key]


def finite_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def xy_distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float32)[:2] - np.asarray(b, dtype=np.float32)[:2]
        )
    )


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
        final_goal_switch_distance=-1.0,
        budgets=None,
        value_gate_threshold=0.5,
        support_gate_left_frac=1.0,
        support_frontier_left_gate="support",
        support_frontier_min_progress_frac=0.0,
        support_frontier_max_xy_factor=0.0,
        support_path_horizon_mode="fixed",
        subgoal_sample_mode="random",
        candidate_sample_mode="random",
        choice_cache_mode="none",
        require_goal_progress=False,
    ):
        self.value_agent = value_agent
        self.train_dataset = train_dataset
        self.context = context
        self.left_budget = int(left_budget)
        self.right_budget = int(right_budget)
        self.num_subgoal_candidates = int(num_subgoal_candidates)
        self.rng = rng
        self.score_batch_size = int(score_batch_size)
        self.budgets = tuple(
            sorted({int(x) for x in (budgets if budgets is not None else ())})
        )
        self.free_cells = np.asarray(context["free_cells"], dtype=np.int32)
        self.cell_to_idx = {
            tuple(int(x) for x in cell): idx for idx, cell in enumerate(self.free_cells)
        }
        self.step_distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
            context["distance_scale"]
        )
        finite_steps = self.step_distances[
            np.isfinite(self.step_distances) & (self.step_distances > 0.0)
        ]
        self.min_step_distance = (
            float(finite_steps.min()) if len(finite_steps) else 0.0
        )
        self.train_goal_by_cell = context["train_goal_by_cell"]
        self.train_observations = np.asarray(
            train_dataset["observations"], dtype=np.float32
        )
        self.has_train_state = np.asarray([len(items) > 0 for items in self.train_goal_by_cell])
        self.subgoal_sample_mode = str(subgoal_sample_mode)
        if self.subgoal_sample_mode not in {"random", "center", "first"}:
            raise ValueError(
                "--subgoal_sample_mode must be one of random, center, first."
            )
        self.candidate_sample_mode = str(candidate_sample_mode)
        if self.candidate_sample_mode not in {"random", "topk", "stratified"}:
            raise ValueError(
                "--candidate_sample_mode must be one of random, topk, stratified."
            )
        self.choice_cache_mode = str(choice_cache_mode)
        if self.choice_cache_mode not in {"none", "cell"}:
            raise ValueError("--choice_cache_mode must be one of none, cell.")
        self.require_goal_progress = bool(require_goal_progress)
        self.choice_cache = {}
        self.choice_cache_hits = 0
        self.choice_cache_misses = 0
        self.representative_goal_by_cell = self.make_representative_goal_by_cell()
        self.action_dim = np.asarray(train_dataset["actions"])[0].shape[-1]
        self.oracle_rep_dim = None
        if "oracle_reps" in train_dataset:
            self.oracle_rep_dim = int(np.asarray(train_dataset["oracle_reps"]).shape[-1])
        self.controller = ctl.make_nn_controller_context(
            train_dataset, context, controller_hops
        )
        self.selector = normalize_selector(selector)
        self.final_goal_switch_distance = float(final_goal_switch_distance)
        self.value_gate_threshold = float(value_gate_threshold)
        self.support_gate_left_frac = float(support_gate_left_frac)
        self.support_frontier_left_gate = str(support_frontier_left_gate)
        self.support_frontier_min_progress_frac = float(
            support_frontier_min_progress_frac
        )
        self.support_frontier_max_xy_factor = float(support_frontier_max_xy_factor)
        self.support_path_horizon_mode = str(support_path_horizon_mode)
        if self.support_path_horizon_mode not in {
            "fixed",
            "source_goal_grid",
            "local_grid_min_right",
        }:
            raise ValueError(
                "--support_path_horizon_mode must be 'fixed', "
                "'source_goal_grid', or 'local_grid_min_right'."
            )
        self.median_step_xy = float(context.get("median_step_xy", 1.0))
        self.maze_unit = float(maze_env._maze_unit)
        self.offset_x = float(maze_env._offset_x)
        self.offset_y = float(maze_env._offset_y)

    def make_representative_goal_by_cell(self):
        representatives = np.full(len(self.train_goal_by_cell), -1, dtype=np.int32)
        if self.subgoal_sample_mode == "random":
            return representatives
        for cell, idxs in enumerate(self.train_goal_by_cell):
            if not len(idxs):
                continue
            idxs = np.asarray(idxs, dtype=np.int32)
            if self.subgoal_sample_mode == "first":
                representatives[cell] = int(idxs[0])
                continue
            xy = self.train_observations[idxs, :2]
            center = xy.mean(axis=0)
            representatives[cell] = int(
                idxs[int(np.argmin(np.linalg.norm(xy - center[None, :], axis=1)))]
            )
        return representatives

    def dataset_support_distances(self, max_h):
        max_h = int(max_h)
        key = f"train_support_distances_h{max_h}"
        if key in self.context:
            return self.context[key]
        state_to_cell = np.asarray(self.context["train_state_to_cell"], dtype=np.int32)
        num_cells = len(self.free_cells)
        n = len(state_to_cell)
        support = np.full((num_cells, num_cells), np.inf, dtype=np.float32)
        np.fill_diagonal(support, 0.0)
        valid = np.zeros(n, dtype=bool)
        valid[valid_transition_indices(self.train_dataset)] = True
        alive = np.ones(n, dtype=bool)
        flat = support.reshape(-1)
        for h in range(1, max_h + 1):
            alive = alive[:-1] & valid[h - 1 : n - 1]
            src = state_to_cell[: n - h][alive]
            dst = state_to_cell[h:n][alive]
            mask = (src >= 0) & (dst >= 0)
            if not np.any(mask):
                continue
            pairs = np.unique(src[mask] * num_cells + dst[mask])
            flat[pairs] = np.minimum(flat[pairs], float(h))
        self.context[key] = support
        return support

    def support_path_horizon(self, source_cell, goal_cell, candidate_cells=None):
        horizon = int(self.right_budget)
        if int(source_cell) < 0 or int(goal_cell) < 0:
            return horizon
        if (
            self.support_path_horizon_mode == "local_grid_min_right"
            and candidate_cells is not None
        ):
            cells = np.asarray(candidate_cells, dtype=np.int32)
            source_d = self.step_distances[int(source_cell), cells]
            right_d = self.step_distances[cells, int(goal_cell)]
            source_goal_d = float(self.step_distances[int(source_cell), int(goal_cell)])
            progress = source_goal_d - right_d
            mask = (
                (source_d >= 0.0)
                & (right_d >= 0.0)
                & (source_d <= float(self.left_budget))
                & (progress > 0.0)
            )
            if np.any(mask):
                return max(horizon, int(np.ceil(float(np.min(right_d[mask])))))
        if self.support_path_horizon_mode == "source_goal_grid":
            source_goal_d = float(self.step_distances[int(source_cell), int(goal_cell)])
            if np.isfinite(source_goal_d):
                horizon = max(horizon, int(np.ceil(source_goal_d)))
        if self.support_path_horizon_mode == "local_grid_min_right":
            source_goal_d = float(self.step_distances[int(source_cell), int(goal_cell)])
            if np.isfinite(source_goal_d):
                horizon = max(horizon, int(np.ceil(source_goal_d)))
        return horizon

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
        if self.candidate_sample_mode == "topk":
            order = np.argsort(midpoint_error, kind="stable")
            return cells[order[: self.num_subgoal_candidates]].astype(np.int32)
        midpoint_count = min(len(cells), max(4, self.num_subgoal_candidates // 2))
        chosen = cells[np.argsort(midpoint_error)[:midpoint_count]].tolist()
        remaining = self.num_subgoal_candidates - len(chosen)
        pool = np.asarray([cell for cell in cells if int(cell) not in set(chosen)])
        if self.candidate_sample_mode == "stratified":
            if remaining > 0 and len(pool) > 0:
                source_d = self.step_distances[source_cell, pool]
                order = np.argsort(source_d, kind="stable")
                ordered_pool = pool[order]
                if len(ordered_pool) <= remaining:
                    chosen.extend(ordered_pool.tolist())
                else:
                    picks = np.linspace(
                        0, len(ordered_pool) - 1, num=remaining, dtype=np.int32
                    )
                    chosen.extend(ordered_pool[picks].tolist())
            return np.asarray(chosen[: self.num_subgoal_candidates], dtype=np.int32)
        if remaining > 0 and len(pool) > 0:
            chosen.extend(
                self.rng.choice(pool, size=remaining, replace=len(pool) < remaining).tolist()
            )
        return np.asarray(chosen[: self.num_subgoal_candidates], dtype=np.int32)

    def subgoals_for_cells(self, candidate_cells):
        if self.subgoal_sample_mode == "random":
            subgoal_idxs = [
                int(self.rng.choice(self.train_goal_by_cell[int(cell)]))
                for cell in candidate_cells
            ]
        else:
            subgoal_idxs = [
                int(self.representative_goal_by_cell[int(cell)])
                for cell in candidate_cells
            ]
        return self.train_observations[subgoal_idxs].astype(np.float32)

    def cache_key(self, selector, source_cell, goal_cell):
        if self.choice_cache_mode == "none":
            return None
        return (str(selector), int(source_cell), int(goal_cell))

    def refresh_choice_for_source(self, choice, source_cell):
        out = dict(choice)
        out["source_cell"] = int(source_cell)
        if source_cell >= 0:
            out["source_to_subgoal"] = float(
                self.step_distances[source_cell, out["subgoal_cell"]]
            )
            out["source_to_goal"] = float(
                self.step_distances[source_cell, out["goal_cell"]]
            )
        else:
            out["source_to_subgoal"] = float("nan")
            out["source_to_goal"] = float("nan")
        return out

    def value_goal_vectors(self, observations):
        """Convert full observations to the value critic goal representation."""
        observations = np.asarray(observations, dtype=np.float32)
        if self.oracle_rep_dim is None:
            return observations
        return observations[..., : self.oracle_rep_dim].astype(np.float32)

    def score_bmm_branches(
        self,
        observation,
        goal,
        subgoals,
        ensemble_reduce="mean",
        left_budget=None,
        right_budget=None,
    ):
        left_budget = int(self.left_budget if left_budget is None else left_budget)
        right_budget = int(self.right_budget if right_budget is None else right_budget)
        zeros = np.zeros((len(subgoals), self.action_dim), dtype=np.float32)
        source_obs = np.repeat(
            np.asarray(observation, dtype=np.float32)[None, :], len(subgoals), axis=0
        )
        goal_obs = np.repeat(
            np.asarray(goal, dtype=np.float32)[None, :], len(subgoals), axis=0
        )
        subgoal_reps = self.value_goal_vectors(subgoals)
        goal_reps = self.value_goal_vectors(goal_obs)
        left = np.full(len(subgoals), left_budget, dtype=np.int32)
        right = np.full(len(subgoals), right_budget, dtype=np.int32)
        left_scores = []
        right_scores = []
        for start in range(0, len(subgoals), self.score_batch_size):
            end = min(start + self.score_batch_size, len(subgoals))
            left_logits = self.value_agent.critic_logits_for(
                source_obs[start:end],
                zeros[start:end],
                subgoal_reps[start:end],
                left[start:end],
                offsets=left[start:end],
            )
            right_logits = self.value_agent.critic_logits_for(
                subgoals[start:end],
                zeros[start:end],
                goal_reps[start:end],
                right[start:end],
                offsets=right[start:end],
            )
            left_probs = np.asarray(jax.nn.sigmoid(left_logits))
            right_probs = np.asarray(jax.nn.sigmoid(right_logits))
            if ensemble_reduce == "min":
                left_scores.append(left_probs.min(axis=0))
                right_scores.append(right_probs.min(axis=0))
            elif ensemble_reduce == "mean":
                left_scores.append(left_probs.mean(axis=0))
                right_scores.append(right_probs.mean(axis=0))
            else:
                raise ValueError(f"Unknown BMM ensemble reduction: {ensemble_reduce}")
        return np.concatenate(left_scores), np.concatenate(right_scores)

    def score_bmm_subgoals(
        self,
        observation,
        goal,
        subgoals,
        ensemble_reduce="mean",
        left_budget=None,
        right_budget=None,
    ):
        left_scores, right_scores = self.score_bmm_branches(
            observation,
            goal,
            subgoals,
            ensemble_reduce=ensemble_reduce,
            left_budget=left_budget,
            right_budget=right_budget,
        )
        return np.minimum(left_scores, right_scores)

    def score_bmm_left_branch(
        self,
        observation,
        subgoals,
        ensemble_reduce="mean",
        left_budget=None,
    ):
        left_budget = int(self.left_budget if left_budget is None else left_budget)
        zeros = np.zeros((len(subgoals), self.action_dim), dtype=np.float32)
        source_obs = np.repeat(
            np.asarray(observation, dtype=np.float32)[None, :], len(subgoals), axis=0
        )
        subgoal_reps = self.value_goal_vectors(subgoals)
        left = np.full(len(subgoals), left_budget, dtype=np.int32)
        left_scores = []
        for start in range(0, len(subgoals), self.score_batch_size):
            end = min(start + self.score_batch_size, len(subgoals))
            left_logits = self.value_agent.critic_logits_for(
                source_obs[start:end],
                zeros[start:end],
                subgoal_reps[start:end],
                left[start:end],
                offsets=left[start:end],
            )
            left_probs = np.asarray(jax.nn.sigmoid(left_logits))
            if ensemble_reduce == "min":
                left_scores.append(left_probs.min(axis=0))
            elif ensemble_reduce == "mean":
                left_scores.append(left_probs.mean(axis=0))
            else:
                raise ValueError(f"Unknown BMM ensemble reduction: {ensemble_reduce}")
        return np.concatenate(left_scores)

    def score_bmm_right_budget_scan(
        self,
        observation,
        goal,
        subgoals,
        ensemble_reduce="mean",
    ):
        """Score one left branch and all right-budget branches for scan selectors."""
        right_budgets = np.asarray(self.right_budget_scan_values(), dtype=np.int32)
        left_scores = self.score_bmm_left_branch(
            observation,
            subgoals,
            ensemble_reduce=ensemble_reduce,
            left_budget=self.left_budget,
        )
        if len(right_budgets) == 0 or len(subgoals) == 0:
            empty = np.zeros((0, len(subgoals)), dtype=np.float32)
            return right_budgets, left_scores, empty

        num_candidates = len(subgoals)
        flat_count = len(right_budgets) * num_candidates
        flat_subgoals = np.repeat(
            np.asarray(subgoals, dtype=np.float32)[None, :, :],
            len(right_budgets),
            axis=0,
        ).reshape(flat_count, -1)
        goal_obs = np.repeat(
            np.asarray(goal, dtype=np.float32)[None, :], flat_count, axis=0
        )
        goal_reps = self.value_goal_vectors(goal_obs)
        zeros = np.zeros((flat_count, self.action_dim), dtype=np.float32)
        offsets = np.repeat(right_budgets, num_candidates).astype(np.int32)

        right_scores = []
        for start in range(0, flat_count, self.score_batch_size):
            end = min(start + self.score_batch_size, flat_count)
            right_logits = self.value_agent.critic_logits_for(
                flat_subgoals[start:end],
                zeros[start:end],
                goal_reps[start:end],
                offsets[start:end],
                offsets=offsets[start:end],
            )
            right_probs = np.asarray(jax.nn.sigmoid(right_logits))
            if ensemble_reduce == "min":
                right_scores.append(right_probs.min(axis=0))
            elif ensemble_reduce == "mean":
                right_scores.append(right_probs.mean(axis=0))
            else:
                raise ValueError(f"Unknown BMM ensemble reduction: {ensemble_reduce}")
        right_scores = np.concatenate(right_scores).reshape(
            len(right_budgets), num_candidates
        )
        return right_budgets, left_scores, right_scores

    def right_budget_scan_values(self):
        right_budgets = [
            int(x)
            for x in self.budgets
            if int(x) >= int(self.left_budget) and int(x) <= int(self.right_budget)
        ]
        if int(self.right_budget) not in right_budgets:
            right_budgets.append(int(self.right_budget))
        if not right_budgets:
            right_budgets = [int(self.right_budget)]
        return sorted(set(right_budgets))

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
        if self.selector == "oracle_path_progress":
            source_d = self.step_distances[int(source_cell), candidate_cells]
            right_d = self.step_distances[candidate_cells, int(goal_cell)]
            source_goal_d = float(self.step_distances[int(source_cell), int(goal_cell)])
            target_left = min(float(self.left_budget), source_goal_d)
            path_slack = np.abs(source_d + right_d - source_goal_d)
            return -(np.abs(source_d - target_left) + path_slack)
        if self.selector == "BMM_V":
            return self.score_bmm_subgoals(observation, goal, subgoals)
        if self.selector == "BMM_V_min":
            return self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
        if self.selector == "BMM_V_left_gate":
            scores = self.score_bmm_subgoals(observation, goal, subgoals)
            source_d = self.step_distances[int(source_cell), candidate_cells]
            gated = np.where(source_d <= float(self.left_budget), scores, -np.inf)
            return gated if np.any(np.isfinite(gated)) else scores
        if self.selector == "BMM_V_min_left_gate":
            scores = self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
            source_d = self.step_distances[int(source_cell), candidate_cells]
            gated = np.where(source_d <= float(self.left_budget), scores, -np.inf)
            return gated if np.any(np.isfinite(gated)) else scores
        if self.selector == "BMM_V_min_budget_scan_left_gate":
            source_d = self.step_distances[int(source_cell), candidate_cells]
            right_budgets, left_scores, right_scores_by_budget = (
                self.score_bmm_right_budget_scan(
                    observation,
                    goal,
                    subgoals,
                    ensemble_reduce="min",
                )
            )
            best = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            for budget_idx, right_budget in enumerate(right_budgets):
                raw_scores = np.minimum(left_scores, right_scores_by_budget[budget_idx])
                normalized_scores = raw_scores / max(float(right_budget), 1.0)
                gated = np.where(
                    source_d <= float(self.left_budget),
                    normalized_scores,
                    -np.inf,
                )
                best = np.maximum(best, gated.astype(np.float32))
            if np.any(np.isfinite(best)):
                return best
            return self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
        if self.selector == "BMM_V_min_budget_scan_value_gate":
            right_budgets, left_scores, right_scores_by_budget = (
                self.score_bmm_right_budget_scan(
                    observation,
                    goal,
                    subgoals,
                    ensemble_reduce="min",
                )
            )
            best = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            best_ungated = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            for budget_idx, right_budget in enumerate(right_budgets):
                right_scores = right_scores_by_budget[budget_idx]
                raw_scores = np.minimum(left_scores, right_scores)
                normalized_scores = raw_scores / max(float(right_budget), 1.0)
                best_ungated = np.maximum(
                    best_ungated, normalized_scores.astype(np.float32)
                )
                gated = np.where(
                    left_scores >= self.value_gate_threshold,
                    normalized_scores,
                    -np.inf,
                )
                best = np.maximum(best, gated.astype(np.float32))
            if np.any(np.isfinite(best)):
                return best
            return best_ungated
        if self.selector == "BMM_V_min_budget_scan_value_frontier":
            source_d = self.step_distances[int(source_cell), candidate_cells]
            right_budgets, left_scores, right_scores_by_budget = (
                self.score_bmm_right_budget_scan(
                    observation,
                    goal,
                    subgoals,
                    ensemble_reduce="min",
                )
            )
            left_slack = float(self.min_step_distance)
            local_gate = (
                np.isfinite(source_d)
                & (source_d <= float(self.left_budget) + left_slack)
            )
            frontier_scores = -(
                np.abs(source_d - float(self.left_budget))
                / max(float(self.left_budget), 1.0)
            )
            best = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            best_ungated = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            for budget_idx, _right_budget in enumerate(right_budgets):
                right_scores = right_scores_by_budget[budget_idx]
                raw_scores = np.minimum(left_scores, right_scores)
                # Frontier distance is the primary conservative constraint; value
                # breaks ties among near-frontier cells without admitting far states.
                value_scores = (
                    frontier_scores
                    + 0.05 * raw_scores
                    + 0.05 * right_scores
                )
                best_ungated = np.maximum(
                    best_ungated,
                    np.where(local_gate, value_scores, -np.inf).astype(np.float32),
                )
                gated = np.where(
                    local_gate & (left_scores >= self.value_gate_threshold),
                    value_scores,
                    -np.inf,
                )
                best = np.maximum(best, gated.astype(np.float32))
            if np.any(np.isfinite(best)):
                return best
            if np.any(np.isfinite(best_ungated)):
                return best_ungated
            return self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
        if self.selector == "BMM_V_min_budget_scan_right_progress":
            source_d = self.step_distances[int(source_cell), candidate_cells]
            right_budgets, left_scores, right_scores_by_budget = (
                self.score_bmm_right_budget_scan(
                    observation,
                    goal,
                    subgoals,
                    ensemble_reduce="min",
                )
            )
            left_slack = float(self.min_step_distance)
            local_gate = (
                np.isfinite(source_d)
                & (source_d <= float(self.left_budget) + left_slack)
            )
            first_right_budget = np.full(
                len(candidate_cells),
                float(self.right_budget) * 2.0,
                dtype=np.float32,
            )
            best_left_score = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            best_right_score = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            best_left_score = np.maximum(
                best_left_score, left_scores.astype(np.float32)
            )
            for budget_idx, right_budget in enumerate(right_budgets):
                right_scores = right_scores_by_budget[budget_idx]
                best_right_score = np.maximum(
                    best_right_score, right_scores.astype(np.float32)
                )
                feasible = right_scores >= self.value_gate_threshold
                first_right_budget = np.minimum(
                    first_right_budget,
                    np.where(feasible, float(right_budget), first_right_budget),
                ).astype(np.float32)
            frontier_penalty = (
                np.abs(source_d - float(self.left_budget))
                / max(float(self.left_budget), 1.0)
            )
            scores = (
                -first_right_budget / max(float(self.right_budget), 1.0)
                + 0.05 * best_right_score
                + 0.02 * best_left_score
                - 0.05 * frontier_penalty
            )
            learned_local_gate = (
                local_gate & (best_left_score >= self.value_gate_threshold)
            )
            gated = np.where(learned_local_gate, scores, -np.inf)
            if np.any(np.isfinite(gated)):
                return gated.astype(np.float32)
            gated = np.where(local_gate, scores, -np.inf)
            if np.any(np.isfinite(gated)):
                return gated.astype(np.float32)
            return scores.astype(np.float32)
        if self.selector == "BMM_V_min_budget_scan_support_gate":
            support = self.dataset_support_distances(
                max(self.left_budget, self.right_budget)
            )
            left_support_d = support[int(source_cell), candidate_cells]
            right_budgets, left_scores, right_scores_by_budget = (
                self.score_bmm_right_budget_scan(
                    observation,
                    goal,
                    subgoals,
                    ensemble_reduce="min",
                )
            )
            best = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            best_ungated = np.full(len(candidate_cells), -np.inf, dtype=np.float32)
            for budget_idx, right_budget in enumerate(right_budgets):
                right_support_d = support[candidate_cells, int(goal_cell)]
                raw_scores = np.minimum(
                    left_scores, right_scores_by_budget[budget_idx]
                )
                normalized_scores = raw_scores / max(float(right_budget), 1.0)
                best_ungated = np.maximum(
                    best_ungated, normalized_scores.astype(np.float32)
                )
                gated = np.where(
                    (
                        left_support_d
                        <= self.support_gate_left_frac * float(self.left_budget)
                    )
                    & (right_support_d <= float(right_budget)),
                    normalized_scores,
                    -np.inf,
                )
                best = np.maximum(best, gated.astype(np.float32))
            if np.any(np.isfinite(best)):
                return best
            return best_ungated
        if self.selector == "BMM_V_min_budget_scan_support_frontier":
            support_horizon = self.support_path_horizon(
                source_cell, goal_cell, candidate_cells
            )
            support = self.dataset_support_distances(
                max(self.left_budget, support_horizon)
            )
            left_support_d = support[int(source_cell), candidate_cells]
            right_support_d = support[candidate_cells, int(goal_cell)]
            grid_left_d = self.step_distances[int(source_cell), candidate_cells]
            left_scores = self.score_bmm_left_branch(
                observation,
                subgoals,
                ensemble_reduce="min",
            )
            frontier_scores = left_scores - (
                right_support_d / max(float(support_horizon), 1.0)
            )
            xy_gate = np.ones(len(candidate_cells), dtype=bool)
            if self.support_frontier_max_xy_factor > 0.0:
                max_xy = (
                    self.support_frontier_max_xy_factor
                    * float(self.left_budget)
                    * self.median_step_xy
                )
                xy_gate = (
                    np.linalg.norm(
                        subgoals[:, :2]
                        - np.asarray(observation, dtype=np.float32)[None, :2],
                        axis=1,
                    )
                    <= max_xy
                )
            support_gate = (
                left_support_d
                <= self.support_gate_left_frac * float(self.left_budget)
            )
            grid_gate = grid_left_d <= float(self.left_budget)
            if self.support_frontier_left_gate == "support":
                left_gate = support_gate & xy_gate
            elif self.support_frontier_left_gate == "xy":
                left_gate = xy_gate
            elif self.support_frontier_left_gate == "grid":
                left_gate = grid_gate
            elif self.support_frontier_left_gate == "grid_xy":
                left_gate = grid_gate & xy_gate
            elif self.support_frontier_left_gate == "support_grid_xy":
                left_gate = support_gate & grid_gate & xy_gate
            else:
                raise ValueError(
                    "--support_frontier_left_gate must be one of "
                    "support, xy, grid, grid_xy, support_grid_xy."
                )
            base_gate = left_gate & np.isfinite(right_support_d)
            if self.support_frontier_min_progress_frac > 0.0:
                progress_gate = grid_left_d >= (
                    self.support_frontier_min_progress_frac
                    * float(self.left_budget)
                )
                if np.any(base_gate & progress_gate):
                    base_gate = base_gate & progress_gate
            gated = np.where(
                base_gate,
                frontier_scores,
                -np.inf,
            )
            if np.any(np.isfinite(gated)):
                return gated.astype(np.float32)
            return self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
        if self.selector == "BMM_V_min_budget_scan_support_path":
            support_horizon = self.support_path_horizon(
                source_cell, goal_cell, candidate_cells
            )
            support = self.dataset_support_distances(
                max(self.left_budget, support_horizon)
            )
            left_support_d = support[int(source_cell), candidate_cells]
            right_support_d = support[candidate_cells, int(goal_cell)]
            source_goal_support_d = float(support[int(source_cell), int(goal_cell)])
            grid_left_d = self.step_distances[int(source_cell), candidate_cells]
            xy_gate = np.ones(len(candidate_cells), dtype=bool)
            if self.support_frontier_max_xy_factor > 0.0:
                max_xy = (
                    self.support_frontier_max_xy_factor
                    * float(self.left_budget)
                    * self.median_step_xy
                )
                xy_gate = (
                    np.linalg.norm(
                        subgoals[:, :2]
                        - np.asarray(observation, dtype=np.float32)[None, :2],
                        axis=1,
                    )
                    <= max_xy
                )
            left_scores = self.score_bmm_left_branch(
                observation,
                subgoals,
                ensemble_reduce="min",
            )
            finite_right = np.isfinite(right_support_d)
            finite_left = np.isfinite(left_support_d)
            support_gate = (
                finite_left
                & (left_support_d <= float(self.left_budget))
            )
            grid_gate = grid_left_d <= float(self.left_budget)
            if self.support_frontier_left_gate == "support":
                left_gate = support_gate
            elif self.support_frontier_left_gate == "xy":
                left_gate = support_gate & xy_gate
            elif self.support_frontier_left_gate == "grid":
                left_gate = support_gate & grid_gate
            elif self.support_frontier_left_gate == "grid_xy":
                left_gate = support_gate & grid_gate & xy_gate
            elif self.support_frontier_left_gate == "support_grid_xy":
                left_gate = support_gate & grid_gate & xy_gate
            else:
                raise ValueError(
                    "--support_frontier_left_gate must be one of "
                    "support, xy, grid, grid_xy, support_grid_xy."
                )
            if np.isfinite(source_goal_support_d):
                target_left = min(float(self.left_budget), source_goal_support_d)
                path_slack = np.abs(
                    left_support_d + right_support_d - source_goal_support_d
                )
                progress = source_goal_support_d - right_support_d
                base_gate = left_gate & finite_right & (progress > 0.0)
                if self.support_frontier_min_progress_frac > 0.0:
                    progress_gate = left_support_d >= (
                        self.support_frontier_min_progress_frac
                        * float(self.left_budget)
                    )
                    if np.any(base_gate & progress_gate):
                        base_gate = base_gate & progress_gate
                path_scores = -(
                    np.abs(left_support_d - target_left)
                    / max(float(self.left_budget), 1.0)
                    + path_slack / max(source_goal_support_d, 1.0)
                )
                scores = path_scores + 0.01 * left_scores
                gated = np.where(base_gate, scores, -np.inf)
                if np.any(np.isfinite(gated)):
                    return gated.astype(np.float32)
            frontier_scores = left_scores - (
                right_support_d / max(float(support_horizon), 1.0)
            )
            gated = np.where(
                left_gate & finite_right,
                frontier_scores,
                -np.inf,
            )
            if np.any(np.isfinite(gated)):
                return gated.astype(np.float32)
            return self.score_bmm_subgoals(
                observation, goal, subgoals, ensemble_reduce="min"
            )
        if self.selector == "support_path_only":
            support_horizon = self.support_path_horizon(
                source_cell, goal_cell, candidate_cells
            )
            support = self.dataset_support_distances(
                max(self.left_budget, support_horizon)
            )
            left_support_d = support[int(source_cell), candidate_cells]
            right_support_d = support[candidate_cells, int(goal_cell)]
            source_goal_support_d = float(support[int(source_cell), int(goal_cell)])
            grid_left_d = self.step_distances[int(source_cell), candidate_cells]
            xy_gate = np.ones(len(candidate_cells), dtype=bool)
            if self.support_frontier_max_xy_factor > 0.0:
                max_xy = (
                    self.support_frontier_max_xy_factor
                    * float(self.left_budget)
                    * self.median_step_xy
                )
                xy_gate = (
                    np.linalg.norm(
                        subgoals[:, :2]
                        - np.asarray(observation, dtype=np.float32)[None, :2],
                        axis=1,
                    )
                    <= max_xy
                )
            finite_right = np.isfinite(right_support_d)
            finite_left = np.isfinite(left_support_d)
            support_gate = finite_left & (left_support_d <= float(self.left_budget))
            grid_gate = grid_left_d <= float(self.left_budget)
            if self.support_frontier_left_gate == "support":
                left_gate = support_gate
            elif self.support_frontier_left_gate == "xy":
                left_gate = support_gate & xy_gate
            elif self.support_frontier_left_gate == "grid":
                left_gate = support_gate & grid_gate
            elif self.support_frontier_left_gate == "grid_xy":
                left_gate = support_gate & grid_gate & xy_gate
            elif self.support_frontier_left_gate == "support_grid_xy":
                left_gate = support_gate & grid_gate & xy_gate
            else:
                raise ValueError(
                    "--support_frontier_left_gate must be one of "
                    "support, xy, grid, grid_xy, support_grid_xy."
                )
            if np.isfinite(source_goal_support_d):
                target_left = min(float(self.left_budget), source_goal_support_d)
                path_slack = np.abs(
                    left_support_d + right_support_d - source_goal_support_d
                )
                progress = source_goal_support_d - right_support_d
                base_gate = left_gate & finite_right & (progress > 0.0)
                if self.support_frontier_min_progress_frac > 0.0:
                    progress_gate = left_support_d >= (
                        self.support_frontier_min_progress_frac
                        * float(self.left_budget)
                    )
                    if np.any(base_gate & progress_gate):
                        base_gate = base_gate & progress_gate
                path_scores = -(
                    np.abs(left_support_d - target_left)
                    / max(float(self.left_budget), 1.0)
                    + path_slack / max(source_goal_support_d, 1.0)
                )
                gated = np.where(base_gate, path_scores, -np.inf)
                if np.any(np.isfinite(gated)):
                    return gated.astype(np.float32)
            frontier_scores = -(right_support_d / max(float(support_horizon), 1.0))
            gated = np.where(left_gate & finite_right, frontier_scores, -np.inf)
            if np.any(np.isfinite(gated)):
                return gated.astype(np.float32)
            return -np.linalg.norm(
                subgoals[:, :2] - np.asarray(goal, dtype=np.float32)[None, :2],
                axis=1,
            ).astype(np.float32)
        raise AssertionError(f"Unhandled selector {self.selector}")

    def apply_goal_progress_gate(self, scores, source_cell, goal_cell, candidate_cells):
        if not self.require_goal_progress or int(source_cell) < 0 or int(goal_cell) < 0:
            return scores
        source_goal_d = float(self.step_distances[int(source_cell), int(goal_cell)])
        if not np.isfinite(source_goal_d):
            return scores
        right_d = self.step_distances[candidate_cells, int(goal_cell)]
        progress_gate = np.isfinite(right_d) & (right_d < source_goal_d - 1e-6)
        gated = np.where(progress_gate, scores, -np.inf)
        if np.any(np.isfinite(gated)):
            return gated
        # Diagnostic fallback: if the learned selector masks all forward-progress
        # candidates, do not allow an explicit backtracking choice. Prefer a
        # locally reachable progress step, otherwise fall back to any progress.
        left_d = self.step_distances[int(source_cell), candidate_cells]
        local_progress_gate = progress_gate & np.isfinite(left_d) & (
            left_d <= float(self.left_budget) + 1e-6
        )
        if np.any(local_progress_gate):
            fallback_gate = local_progress_gate
        else:
            fallback_gate = progress_gate
        fallback = np.where(fallback_gate, -right_d, -np.inf)
        return fallback if np.any(np.isfinite(fallback)) else scores

    def select_subgoal(self, observation, goal):
        source_cell = self.obs_to_cell(observation)
        goal_cell = self.obs_to_cell(goal)
        if source_cell < 0 or goal_cell < 0:
            return None
        cache_key = self.cache_key(self.selector, source_cell, goal_cell)
        if cache_key is not None:
            cached_choice = self.choice_cache.get(cache_key)
            if cached_choice is not None:
                self.choice_cache_hits += 1
                return self.refresh_choice_for_source(cached_choice, source_cell)
            self.choice_cache_misses += 1
        cells = self.candidate_cells(source_cell, goal_cell)
        if len(cells) == 0:
            return None
        subgoals = self.subgoals_for_cells(cells)
        scores = self.selector_scores(
            observation, goal, cells, subgoals, source_cell, goal_cell
        )
        scores = self.apply_goal_progress_gate(scores, source_cell, goal_cell, cells)
        selected = int(np.argmax(scores))
        subgoal_cell = int(cells[selected])
        subgoal_observation = subgoals[selected]
        if (
            self.final_goal_switch_distance >= 0.0
            and self.step_distances[source_cell, goal_cell]
            <= self.final_goal_switch_distance
        ):
            subgoal_cell = int(goal_cell)
            subgoal_observation = np.asarray(goal, dtype=np.float32)
        choice = dict(
            selector=self.selector,
            subgoal_observation=subgoal_observation,
            subgoal_cell=subgoal_cell,
            subgoal_score=float(scores[selected]),
            support_path_horizon=int(
                self.support_path_horizon(source_cell, goal_cell, cells)
            ),
            source_cell=int(source_cell),
            goal_cell=int(goal_cell),
            source_to_subgoal=float(self.step_distances[source_cell, subgoal_cell]),
            subgoal_to_goal=float(self.step_distances[subgoal_cell, goal_cell]),
            source_to_goal=float(self.step_distances[source_cell, goal_cell]),
        )
        if cache_key is not None:
            self.choice_cache[cache_key] = dict(choice)
        return choice

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


def refresh_choice_source(policy, choice, observation):
    """Refresh source-dependent distances for a held subgoal choice."""
    if choice is None:
        return None
    source_cell = policy.obs_to_cell(observation)
    return policy.refresh_choice_for_source(choice, source_cell)


def select_with_optional_selector(policy, observation, goal, selector=None):
    if selector is None:
        return policy.select_subgoal(observation, goal)
    old_selector = policy.selector
    policy.selector = selector
    try:
        return policy.select_subgoal(observation, goal)
    finally:
        policy.selector = old_selector


def run_policy_smoke(
    env,
    policy,
    task_ids,
    episodes_per_task,
    max_steps,
    subgoal_commit_steps=1,
    subgoal_replan_distance=-1.0,
    early_stop_patience=0,
    early_stop_min_steps=0,
    early_stop_min_delta=0.0,
    fallback_selector=None,
    fallback_patience=0,
    fallback_min_steps=0,
    fallback_min_delta=0.0,
    fallback_max_action_frac=0.0,
    fallback_burst_steps=0,
    fallback_cooldown_steps=0,
    fallback_max_goal_distance=0.0,
    fallback_burst_min_delta=0.0,
    fallback_min_active_subgoal_to_goal=0.0,
    reset_seed_base=-1,
    stop_on_grid_goal_distance=-1.0,
    progress_prefix=None,
):
    rows = []
    subgoal_commit_steps = max(1, int(subgoal_commit_steps))
    subgoal_replan_distance = float(subgoal_replan_distance)
    early_stop_patience = int(early_stop_patience)
    early_stop_min_steps = int(early_stop_min_steps)
    early_stop_min_delta = float(early_stop_min_delta)
    fallback_selector = (
        None if fallback_selector in (None, "") else normalize_selector(fallback_selector)
    )
    fallback_patience = int(fallback_patience)
    fallback_min_steps = int(fallback_min_steps)
    fallback_min_delta = float(fallback_min_delta)
    fallback_max_action_frac = float(fallback_max_action_frac)
    fallback_burst_steps = int(fallback_burst_steps)
    fallback_cooldown_steps = int(fallback_cooldown_steps)
    fallback_max_goal_distance = float(fallback_max_goal_distance)
    fallback_burst_min_delta = float(fallback_burst_min_delta)
    fallback_min_active_subgoal_to_goal = float(fallback_min_active_subgoal_to_goal)
    reset_seed_base = int(reset_seed_base)
    stop_on_grid_goal_distance = float(stop_on_grid_goal_distance)
    for task_id in task_ids:
        for episode in range(int(episodes_per_task)):
            reset_kwargs = dict(options=dict(task_id=int(task_id)))
            if reset_seed_base >= 0:
                reset_kwargs["seed"] = (
                    reset_seed_base + 1009 * int(task_id) + int(episode)
                )
            observation, info = env.reset(**reset_kwargs)
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
            start_goal_xy_d = xy_distance(observation, goal)
            subgoal_improvements = []
            goal_improvements = []
            subgoal_valids = []
            selected_scores = []
            source_to_subgoals = []
            subgoal_to_goals = []
            support_path_horizons = []
            action_count = 0
            done = False
            final_info = {}
            final_observation = observation
            active_choice = None
            active_until_step = 0
            best_goal_d = start_goal_d
            best_goal_step = 0
            current_goal_d_for_fallback = start_goal_d
            fallback_best_goal_d = start_goal_d
            fallback_best_goal_step = 0
            fallback_action_count = 0
            fallback_replan_count = 0
            fallback_burst_count = 0
            fallback_burst_extend_count = 0
            fallback_cooldown_count = 0
            fallback_burst_start_step = None
            fallback_burst_start_goal_d = float("nan")
            fallback_cooldown_until_step = 0
            early_stopped = False
            grid_goal_stopped = False
            episode_start_time = time.perf_counter()
            select_time = 0.0
            action_time = 0.0
            env_step_time = 0.0
            cache_hits_start = int(getattr(policy, "choice_cache_hits", 0))
            cache_misses_start = int(getattr(policy, "choice_cache_misses", 0))
            for step in range(int(max_steps)):
                if active_choice is not None:
                    refreshed = refresh_choice_source(policy, active_choice, observation)
                    if (
                        subgoal_replan_distance >= 0.0
                        and refreshed is not None
                        and refreshed["source_to_subgoal"] <= subgoal_replan_distance
                    ):
                        active_choice = None
                use_fallback = (
                    fallback_selector is not None
                    and fallback_patience > 0
                    and action_count >= fallback_min_steps
                    and action_count - fallback_best_goal_step >= fallback_patience
                    and action_count >= fallback_cooldown_until_step
                )
                if (
                    use_fallback
                    and fallback_max_goal_distance > 0.0
                    and (
                        not np.isfinite(current_goal_d_for_fallback)
                        or current_goal_d_for_fallback > fallback_max_goal_distance
                    )
                ):
                    use_fallback = False
                if (
                    use_fallback
                    and fallback_max_action_frac > 0.0
                    and action_count > 0
                    and fallback_action_count / float(action_count)
                    >= fallback_max_action_frac
                ):
                    use_fallback = False
                if use_fallback and fallback_min_active_subgoal_to_goal > 0.0:
                    active_right_d = (
                        active_choice.get("subgoal_to_goal", float("nan"))
                        if active_choice is not None
                        else float("nan")
                    )
                    if (
                        not np.isfinite(active_right_d)
                        or active_right_d < fallback_min_active_subgoal_to_goal
                    ):
                        use_fallback = False
                if use_fallback and fallback_burst_steps > 0:
                    if fallback_burst_start_step is None:
                        fallback_burst_start_step = action_count
                        fallback_burst_start_goal_d = current_goal_d_for_fallback
                        fallback_burst_count += 1
                    elif action_count - fallback_burst_start_step >= fallback_burst_steps:
                        burst_improved = (
                            fallback_burst_min_delta > 0.0
                            and np.isfinite(fallback_burst_start_goal_d)
                            and np.isfinite(current_goal_d_for_fallback)
                            and current_goal_d_for_fallback
                            < fallback_burst_start_goal_d - fallback_burst_min_delta
                        )
                        if burst_improved:
                            fallback_burst_start_step = action_count
                            fallback_burst_start_goal_d = current_goal_d_for_fallback
                            fallback_burst_extend_count += 1
                        else:
                            use_fallback = False
                            fallback_burst_start_step = None
                            fallback_burst_start_goal_d = float("nan")
                            fallback_best_goal_step = action_count
                            fallback_cooldown_until_step = (
                                action_count + max(fallback_cooldown_steps, 0)
                            )
                            if fallback_cooldown_steps > 0:
                                fallback_cooldown_count += 1
                if not use_fallback:
                    fallback_burst_start_step = None
                    fallback_burst_start_goal_d = float("nan")
                if (
                    use_fallback
                    and active_choice is not None
                    and active_choice.get("selector") != fallback_selector
                ):
                    active_choice = None
                if (
                    not use_fallback
                    and active_choice is not None
                    and active_choice.get("selector") == fallback_selector
                ):
                    active_choice = None
                if active_choice is None or step >= active_until_step:
                    select_start_time = time.perf_counter()
                    active_choice = select_with_optional_selector(
                        policy,
                        observation,
                        goal,
                        fallback_selector if use_fallback else None,
                    )
                    select_time += time.perf_counter() - select_start_time
                    if use_fallback:
                        fallback_replan_count += 1
                    active_until_step = step + subgoal_commit_steps
                choice = refresh_choice_source(policy, active_choice, observation)
                if choice is None:
                    action = np.zeros(policy.action_dim, dtype=np.float32)
                else:
                    action_start_time = time.perf_counter()
                    if hasattr(policy, "action_for_choice"):
                        action = policy.action_for_choice(observation, goal, choice)
                    else:
                        action, _ = policy.action_toward(
                            choice["source_cell"], choice["subgoal_cell"]
                        )
                    action_time += time.perf_counter() - action_start_time
                    subgoal_valids.append(
                        float(
                            choice["source_to_subgoal"] <= policy.left_budget
                            and choice["subgoal_to_goal"]
                            <= choice.get("support_path_horizon", policy.right_budget)
                        )
                    )
                    selected_scores.append(choice["subgoal_score"])
                    source_to_subgoals.append(choice["source_to_subgoal"])
                    subgoal_to_goals.append(choice["subgoal_to_goal"])
                    support_path_horizons.append(
                        choice.get("support_path_horizon", policy.right_budget)
                    )
                    if fallback_selector is not None and choice.get("selector") == fallback_selector:
                        fallback_action_count += 1
                    before_subgoal = choice["source_to_subgoal"]
                    before_goal = choice["source_to_goal"]
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                env_start_time = time.perf_counter()
                next_observation, reward, terminated, truncated, info = env.step(action)
                env_step_time += time.perf_counter() - env_start_time
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
                else:
                    next_cell = policy.obs_to_cell(next_observation)
                if next_cell >= 0 and goal_cell >= 0:
                    current_goal_d = float(policy.step_distances[next_cell, goal_cell])
                    current_goal_d_for_fallback = current_goal_d
                    if (
                        np.isfinite(current_goal_d)
                        and (
                            not np.isfinite(best_goal_d)
                            or current_goal_d < best_goal_d - early_stop_min_delta
                        )
                    ):
                        best_goal_d = current_goal_d
                        best_goal_step = step + 1
                    if (
                        np.isfinite(current_goal_d)
                        and (
                            not np.isfinite(fallback_best_goal_d)
                            or current_goal_d
                            < fallback_best_goal_d - fallback_min_delta
                        )
                    ):
                        fallback_best_goal_d = current_goal_d
                        fallback_best_goal_step = step + 1
                action_count += 1
                observation = next_observation
                done = bool(terminated or truncated)
                if done:
                    break
                if (
                    stop_on_grid_goal_distance >= 0.0
                    and np.isfinite(current_goal_d_for_fallback)
                    and current_goal_d_for_fallback <= stop_on_grid_goal_distance
                ):
                    grid_goal_stopped = True
                    break
                near_goal = (
                    policy.final_goal_switch_distance >= 0.0
                    and np.isfinite(best_goal_d)
                    and best_goal_d <= policy.final_goal_switch_distance
                )
                if (
                    early_stop_patience > 0
                    and action_count >= early_stop_min_steps
                    and action_count - best_goal_step >= early_stop_patience
                    and not near_goal
                ):
                    early_stopped = True
                    break
            final_cell = policy.obs_to_cell(final_observation)
            final_goal_d = (
                float(policy.step_distances[final_cell, goal_cell])
                if final_cell >= 0 and goal_cell >= 0
                else float("nan")
            )
            final_goal_xy_d = xy_distance(final_observation, goal)
            wall_time_sec = time.perf_counter() - episode_start_time
            policy_time_sec = select_time + action_time
            choice_cache_hits = (
                int(getattr(policy, "choice_cache_hits", 0)) - cache_hits_start
            )
            choice_cache_misses = (
                int(getattr(policy, "choice_cache_misses", 0)) - cache_misses_start
            )
            choice_cache_total = choice_cache_hits + choice_cache_misses
            row = dict(
                task_id=int(task_id),
                episode=int(episode),
                steps=int(action_count),
                done=bool(done),
                early_stopped=bool(early_stopped),
                grid_goal_stopped=bool(grid_goal_stopped),
                success=float(
                    final_info.get(
                        "success", final_info.get("episode", {}).get("success", 0.0)
                    )
                ),
                grid_success=float(
                    np.isfinite(final_goal_d) and final_goal_d <= 1e-6
                ),
                start_goal_distance=start_goal_d,
                final_goal_distance=final_goal_d,
                goal_distance_improvement=start_goal_d - final_goal_d,
                start_goal_xy_distance=start_goal_xy_d,
                final_goal_xy_distance=final_goal_xy_d,
                goal_xy_distance_improvement=start_goal_xy_d - final_goal_xy_d,
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
                selected_support_path_horizon=finite_mean(support_path_horizons),
                fallback_action_frac=(
                    float(fallback_action_count) / float(action_count)
                    if action_count > 0
                    else 0.0
                ),
                fallback_replan_count=float(fallback_replan_count),
                fallback_burst_count=float(fallback_burst_count),
                fallback_burst_extend_count=float(fallback_burst_extend_count),
                fallback_cooldown_count=float(fallback_cooldown_count),
                choice_cache_hits=float(choice_cache_hits),
                choice_cache_misses=float(choice_cache_misses),
                choice_cache_hit_frac=(
                    float(choice_cache_hits) / float(choice_cache_total)
                    if choice_cache_total > 0
                    else float("nan")
                ),
                wall_time_sec=float(wall_time_sec),
                steps_per_sec=(
                    float(action_count) / float(wall_time_sec)
                    if wall_time_sec > 0.0
                    else float("nan")
                ),
                policy_time_sec=float(policy_time_sec),
                select_time_sec=float(select_time),
                action_time_sec=float(action_time),
                env_step_time_sec=float(env_step_time),
                other_time_sec=float(
                    max(wall_time_sec - policy_time_sec - env_step_time, 0.0)
                ),
            )
            rows.append(row)
            if progress_prefix is not None:
                print(
                    "{prefix} task={task} episode={episode} success={success:.1f} "
                    "final_d={final_d:.4f} improve={improve:.4f} steps={steps} "
                    "wall_s={wall:.1f} step_s={step_s:.2f}".format(
                        prefix=progress_prefix,
                        task=row["task_id"],
                        episode=row["episode"],
                        success=row["success"],
                        final_d=row["final_goal_distance"],
                        improve=row["goal_distance_improvement"],
                        steps=row["steps"],
                        wall=row["wall_time_sec"],
                        step_s=row["steps_per_sec"],
                    ),
                    flush=True,
                )
    return rows


def aggregate(rows):
    keys = [
        "success",
        "grid_success",
        "steps",
        "start_goal_distance",
        "final_goal_distance",
        "goal_distance_improvement",
        "start_goal_xy_distance",
        "final_goal_xy_distance",
        "goal_xy_distance_improvement",
        "mean_step_goal_improvement",
        "mean_step_subgoal_improvement",
        "subgoal_reduce_frac",
        "goal_reduce_frac",
        "subgoal_valid_frac",
        "selected_score_mean",
        "selected_source_to_subgoal",
        "selected_subgoal_to_goal",
        "selected_support_path_horizon",
        "fallback_action_frac",
        "fallback_replan_count",
        "fallback_burst_count",
        "fallback_burst_extend_count",
        "fallback_cooldown_count",
        "grid_goal_stopped",
        "choice_cache_hits",
        "choice_cache_misses",
        "choice_cache_hit_frac",
        "wall_time_sec",
        "steps_per_sec",
        "policy_time_sec",
        "select_time_sec",
        "action_time_sec",
        "env_step_time_sec",
        "other_time_sec",
    ]
    return {key: finite_mean([row.get(key, float("nan")) for row in rows]) for key in keys}


def per_task_aggregate(rows):
    task_ids = sorted({int(row["task_id"]) for row in rows})
    out = []
    for task_id in task_ids:
        task_rows = [row for row in rows if int(row["task_id"]) == task_id]
        task_mean = aggregate(task_rows)
        task_mean["task_id"] = task_id
        out.append(task_mean)
    return out


def markdown(result):
    lines = [
        "# BMM value-subgoal policy smoke",
        "",
        f"env: `{result['env_name']}`",
        f"task ids: `{result['task_ids']}`",
        f"episodes/task: `{result['episodes_per_task']}`, max steps: `{result['max_steps']}`",
        f"subgoal commit steps: `{result.get('subgoal_commit_steps', 1)}`",
        f"subgoal replan distance: `{result.get('subgoal_replan_distance', -1.0)}`",
        f"early stop patience: `{result.get('early_stop_patience', 0)}`",
        f"fallback selector: `{result.get('fallback_selector')}`",
        f"fallback patience: `{result.get('fallback_patience', 0)}`",
        f"fallback max action frac: `{result.get('fallback_max_action_frac', 0.0)}`",
        f"fallback burst steps: `{result.get('fallback_burst_steps', 0)}`",
        f"fallback cooldown steps: `{result.get('fallback_cooldown_steps', 0)}`",
        f"fallback max goal distance: `{result.get('fallback_max_goal_distance', 0.0)}`",
        f"fallback burst min delta: `{result.get('fallback_burst_min_delta', 0.0)}`",
        f"fallback min active subgoal-to-goal: `{result.get('fallback_min_active_subgoal_to_goal', 0.0)}`",
        f"reset seed base: `{result.get('reset_seed_base', -1)}`",
        f"stop on grid goal distance: `{result.get('stop_on_grid_goal_distance', -1.0)}`",
        f"controller hops: `{result['controller_hops']}`",
        f"final goal switch distance: `{result['final_goal_switch_distance']}`",
        f"support path horizon mode: `{result.get('support_path_horizon_mode', 'fixed')}`",
        f"subgoal sample mode: `{result.get('subgoal_sample_mode', 'random')}`",
        f"candidate sample mode: `{result.get('candidate_sample_mode', 'random')}`",
        f"choice cache mode: `{result.get('choice_cache_mode', 'none')}`",
        f"require goal progress: `{result.get('require_goal_progress', False)}`",
        "",
        "| selector | success | grid_success | final_d | final_xy | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce | selected_src_d | selected_right_d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {success:.4f} | {grid_success:.4f} | {final:.4f} | {final_xy:.4f} | {improve:.4f} | {mean_goal:.4f} | {valid:.4f} | {subgoal_reduce:.4f} | {goal_reduce:.4f} | {src_d:.4f} | {right_d:.4f} |".format(
                name=row["name"],
                success=m["success"],
                grid_success=m.get("grid_success", float("nan")),
                final=m["final_goal_distance"],
                final_xy=m["final_goal_xy_distance"],
                improve=m["goal_distance_improvement"],
                mean_goal=m["mean_step_goal_improvement"],
                valid=m["subgoal_valid_frac"],
                subgoal_reduce=m["subgoal_reduce_frac"],
                goal_reduce=m["goal_reduce_frac"],
                src_d=m["selected_source_to_subgoal"],
                right_d=m["selected_subgoal_to_goal"],
            )
        )
    lines.extend(
        [
            "",
            "Timing:",
            "",
            "| selector | wall_s/ep | steps/s | policy_s/ep | select_s/ep | action_s/ep | env_s/ep | other_s/ep | cache_hit | grid_stop |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {wall:.4f} | {steps_s:.4f} | {policy:.4f} | {select:.4f} | {action:.4f} | {env:.4f} | {other:.4f} | {cache_hit:.4f} | {grid_stop:.4f} |".format(
                name=row["name"],
                wall=m.get("wall_time_sec", float("nan")),
                steps_s=m.get("steps_per_sec", float("nan")),
                policy=m.get("policy_time_sec", float("nan")),
                select=m.get("select_time_sec", float("nan")),
                action=m.get("action_time_sec", float("nan")),
                env=m.get("env_step_time_sec", float("nan")),
                other=m.get("other_time_sec", float("nan")),
                cache_hit=m.get("choice_cache_hit_frac", float("nan")),
                grid_stop=m.get("grid_goal_stopped", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Per-task:",
            "",
            "| selector | task | success | grid_success | final_d | final_xy | improve | subgoal_valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["selectors"]:
        for task_row in per_task_aggregate(row["episodes"]):
            lines.append(
                "| {name} | {task} | {success:.4f} | {grid_success:.4f} | {final:.4f} | {final_xy:.4f} | {improve:.4f} | {valid:.4f} |".format(
                    name=row["name"],
                    task=int(task_row["task_id"]),
                    success=task_row["success"],
                    grid_success=task_row.get("grid_success", float("nan")),
                    final=task_row["final_goal_distance"],
                    final_xy=task_row["final_goal_xy_distance"],
                    improve=task_row["goal_distance_improvement"],
                    valid=task_row["subgoal_valid_frac"],
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
        help=(
            "Comma-separated selectors: random, geometric_midpoint, BMM_V, "
            "BMM_V_min, BMM_V_left_gate, BMM_V_min_left_gate, "
            "BMM_V_min_budget_scan_left_gate, "
            "BMM_V_min_budget_scan_value_gate, "
            "BMM_V_min_budget_scan_value_frontier, "
            "BMM_V_min_budget_scan_support_gate, "
            "BMM_V_min_budget_scan_support_frontier, "
            "BMM_V_min_budget_scan_support_path, support_path_only, oracle_midpoint, "
            "oracle_path_progress."
        ),
    )
    parser.add_argument("--task_ids", default="1")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--subgoal_commit_steps", type=int, default=1)
    parser.add_argument("--subgoal_replan_distance", type=float, default=-1.0)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Diagnostic speedup. If positive, stop a rollout after this many "
            "environment steps without improving the best geodesic goal "
            "distance. Disabled by default."
        ),
    )
    parser.add_argument(
        "--early_stop_min_steps",
        type=int,
        default=0,
        help="Minimum rollout length before --early_stop_patience can trigger.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum geodesic-distance improvement counted by early stopping.",
    )
    parser.add_argument(
        "--fallback_selector",
        default=None,
        help=(
            "Optional selector to use after --fallback_patience steps without "
            "geodesic goal-distance improvement. Disabled by default."
        ),
    )
    parser.add_argument(
        "--fallback_patience",
        type=int,
        default=0,
        help="Steps without improvement before switching to --fallback_selector.",
    )
    parser.add_argument(
        "--fallback_min_steps",
        type=int,
        default=0,
        help="Minimum rollout length before selector fallback can trigger.",
    )
    parser.add_argument(
        "--fallback_min_delta",
        type=float,
        default=0.0,
        help="Minimum geodesic-distance improvement counted by selector fallback.",
    )
    parser.add_argument(
        "--fallback_max_action_frac",
        type=float,
        default=0.0,
        help=(
            "If positive, disable fallback after fallback actions reach this "
            "fraction of elapsed episode actions."
        ),
    )
    parser.add_argument(
        "--fallback_burst_steps",
        type=int,
        default=0,
        help=(
            "If positive, use fallback only in bounded bursts of this many "
            "environment steps, then reset the fallback trigger timer. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--fallback_cooldown_steps",
        type=int,
        default=0,
        help=(
            "Optional minimum cooldown after a fallback burst expires. The "
            "patience timer is also reset at burst expiry."
        ),
    )
    parser.add_argument(
        "--fallback_max_goal_distance",
        type=float,
        default=0.0,
        help=(
            "If positive, only allow fallback when the current geodesic "
            "source-goal distance is at most this value."
        ),
    )
    parser.add_argument(
        "--fallback_burst_min_delta",
        type=float,
        default=0.0,
        help=(
            "If positive, extend a fallback burst for another "
            "--fallback_burst_steps only when geodesic goal distance improved "
            "by at least this amount during the previous burst."
        ),
    )
    parser.add_argument(
        "--fallback_min_active_subgoal_to_goal",
        type=float,
        default=0.0,
        help=(
            "If positive, only allow selector fallback when the current "
            "base subgoal's grid/geodesic subgoal-to-goal distance is at "
            "least this value. Disabled by default."
        ),
    )
    parser.add_argument("--final_goal_switch_distance", type=float, default=-1.0)
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
        help=(
            "How to choose a representative observation for each candidate "
            "subgoal cell. 'random' matches the original stochastic evaluator; "
            "'center' and 'first' make policy extraction deterministic."
        ),
    )
    parser.add_argument(
        "--candidate_sample_mode",
        choices=("random", "topk", "stratified"),
        default="random",
        help=(
            "How to choose candidate subgoal cells when there are more cells "
            "than --num_subgoal_candidates. 'random' matches the original "
            "evaluator; 'topk' and 'stratified' make the candidate set "
            "deterministic."
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
    parser.add_argument(
        "--choice_cache_mode",
        choices=("none", "cell"),
        default="none",
        help=(
            "Diagnostic speedup. 'cell' caches selected subgoals by "
            "(selector, source cell, goal cell), avoiding repeated value "
            "rescoring from the same maze cell. Disabled by default."
        ),
    )
    parser.add_argument(
        "--require_goal_progress",
        action="store_true",
        help=(
            "Diagnostic extraction gate. If set, candidate subgoals must reduce "
            "grid/geodesic distance to the goal when any such candidate exists."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset_seed_base",
        type=int,
        default=-1,
        help=(
            "If nonnegative, seed each env reset with "
            "base + 1009 * task_id + episode so task resets are independent "
            "of evaluation order."
        ),
    )
    parser.add_argument(
        "--stop_on_grid_goal_distance",
        type=float,
        default=-1.0,
        help=(
            "Diagnostic speedup. If nonnegative, stop a rollout once the "
            "grid/geodesic goal distance is at most this threshold. Disabled "
            "by default; official env success remains the env's success flag."
        ),
    )
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
        print(f"Starting selector={selector}", flush=True)
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
            final_goal_switch_distance=args.final_goal_switch_distance,
            budgets=budgets,
            value_gate_threshold=args.value_gate_threshold,
            support_gate_left_frac=args.support_gate_left_frac,
            support_frontier_left_gate=args.support_frontier_left_gate,
            support_frontier_min_progress_frac=args.support_frontier_min_progress_frac,
            support_frontier_max_xy_factor=args.support_frontier_max_xy_factor,
            support_path_horizon_mode=args.support_path_horizon_mode,
            subgoal_sample_mode=args.subgoal_sample_mode,
            candidate_sample_mode=args.candidate_sample_mode,
            choice_cache_mode=args.choice_cache_mode,
            require_goal_progress=args.require_goal_progress,
        )
        rows = run_policy_smoke(
            env,
            policy,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            subgoal_commit_steps=args.subgoal_commit_steps,
            subgoal_replan_distance=args.subgoal_replan_distance,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_min_delta=args.early_stop_min_delta,
            fallback_selector=args.fallback_selector,
            fallback_patience=args.fallback_patience,
            fallback_min_steps=args.fallback_min_steps,
            fallback_min_delta=args.fallback_min_delta,
            fallback_max_action_frac=args.fallback_max_action_frac,
            fallback_burst_steps=args.fallback_burst_steps,
            fallback_cooldown_steps=args.fallback_cooldown_steps,
            fallback_max_goal_distance=args.fallback_max_goal_distance,
            fallback_burst_min_delta=args.fallback_burst_min_delta,
            fallback_min_active_subgoal_to_goal=args.fallback_min_active_subgoal_to_goal,
            reset_seed_base=args.reset_seed_base,
            stop_on_grid_goal_distance=args.stop_on_grid_goal_distance,
            progress_prefix=f"selector={selector}",
        )
        for row in rows:
            row["selector"] = selector
        selector_aggregate = aggregate(rows)
        print(
            "Finished selector={selector} success={success:.4f} final_d={final_d:.4f} "
            "improve={improve:.4f}".format(
                selector=selector,
                success=selector_aggregate["success"],
                final_d=selector_aggregate["final_goal_distance"],
                improve=selector_aggregate["goal_distance_improvement"],
            ),
            flush=True,
        )
        selector_results.append(
            dict(name=selector, aggregate=selector_aggregate, episodes=rows)
        )
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        max_steps=int(args.max_steps),
        subgoal_commit_steps=int(args.subgoal_commit_steps),
        subgoal_replan_distance=float(args.subgoal_replan_distance),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_steps=int(args.early_stop_min_steps),
        early_stop_min_delta=float(args.early_stop_min_delta),
        fallback_selector=(
            None
            if args.fallback_selector in (None, "")
            else normalize_selector(args.fallback_selector)
        ),
        fallback_patience=int(args.fallback_patience),
        fallback_min_steps=int(args.fallback_min_steps),
        fallback_min_delta=float(args.fallback_min_delta),
        fallback_max_action_frac=float(args.fallback_max_action_frac),
        fallback_burst_steps=int(args.fallback_burst_steps),
        fallback_cooldown_steps=int(args.fallback_cooldown_steps),
        fallback_max_goal_distance=float(args.fallback_max_goal_distance),
        fallback_burst_min_delta=float(args.fallback_burst_min_delta),
        fallback_min_active_subgoal_to_goal=float(args.fallback_min_active_subgoal_to_goal),
        reset_seed_base=int(args.reset_seed_base),
        stop_on_grid_goal_distance=float(args.stop_on_grid_goal_distance),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        controller_hops=int(args.controller_hops),
        final_goal_switch_distance=float(args.final_goal_switch_distance),
        value_gate_threshold=float(args.value_gate_threshold),
        support_gate_left_frac=float(args.support_gate_left_frac),
        support_frontier_left_gate=args.support_frontier_left_gate,
        support_frontier_min_progress_frac=float(args.support_frontier_min_progress_frac),
        support_frontier_max_xy_factor=float(args.support_frontier_max_xy_factor),
        support_path_horizon_mode=args.support_path_horizon_mode,
        subgoal_sample_mode=args.subgoal_sample_mode,
        candidate_sample_mode=args.candidate_sample_mode,
        choice_cache_mode=args.choice_cache_mode,
        require_goal_progress=bool(args.require_goal_progress),
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
