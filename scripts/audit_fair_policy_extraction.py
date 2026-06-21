#!/usr/bin/env python
"""Audit which policy results are fair 1M TRL policy-extraction comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "exp/bmm_fair_policy_extraction_audit.json"
OUT_MD = REPO_ROOT / "exp/bmm_fair_policy_extraction_audit.md"


ROWS = [
    {
        "environment": "humanoidmaze-medium-navigate-oraclerep-v0",
        "paper_overall": 57.0,
        "artifact": "exp/trl_humanoidmaze_medium_rpg_actor_total1m_eval_ep15.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "Direct paper-style actor baseline at 1M.",
    },
    {
        "environment": "humanoidmaze-medium-navigate-oraclerep-v0",
        "paper_overall": 57.0,
        "artifact": "exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_switch128_ep15_seed10_detreset_max2000.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "BMM changes high-level subgoal/controller selection while keeping the low-level paper-style actor fixed.",
    },
    {
        "environment": "humanoidmaze-medium-navigate-oraclerep-v0",
        "paper_overall": 57.0,
        "artifact": "exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_support_switch128_ep15.json",
        "parser": "selector",
        "selector": "support_path_only",
        "track": "B_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "Matched support-path control for the medium BMM result.",
    },
    {
        "environment": "humanoidmaze-giant-navigate-oraclerep-v0",
        "paper_overall": 79.0,
        "artifact": "exp/trl_humanoidmaze_giant_rpg_actor_total1m_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 1_000_000,
        "episodes_per_task": 3,
        "status": "existing_smoke",
        "claim_scope": "Direct paper-style actor baseline at 1M; 3 episodes/task smoke.",
    },
    {
        "environment": "humanoidmaze-giant-navigate-oraclerep-v0",
        "paper_overall": 79.0,
        "artifact": "exp/humanoidmaze_giant_graph_trl_total1m_bmm_switch256_ep3_seed10.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 3,
        "status": "existing_smoke",
        "claim_scope": "1M smoke only; not a promoted 15-episode/table row.",
    },
    {
        "environment": "humanoidmaze-giant-navigate-oraclerep-v0",
        "paper_overall": 79.0,
        "artifact": "exp/humanoidmaze_giant_graph_trl_total1m_routefit_deltay_commit20_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "start_distance_deltay_gate_bmm_support",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_negative_full",
        "claim_scope": "Full 1M fixed-actor routefit confirmation: 14/15 smoke did not transfer to 75 rollouts.",
    },
    {
        "environment": "humanoidmaze-giant-navigate-oraclerep-v0",
        "paper_overall": 79.0,
        "artifact": "exp/humanoidmaze_giant_graph_trl_total1m_support_switch128_commit20_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "support_path_only",
        "track": "B_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_negative_full",
        "claim_scope": "Matched full 1M support-path control for the routefit confirmation.",
    },
    {
        "environment": "humanoidmaze-giant-navigate-oraclerep-v0",
        "paper_overall": 79.0,
        "artifact": "exp/humanoidmaze_giant_graph_trl_total600k_startdist_cross_gate_fixed_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "start_distance_cross_gate_bmm_support",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 600_000,
        "episodes_per_task": 15,
        "status": "existing_not_1m",
        "claim_scope": "Promoted Giant row is calibrated and not yet a 1M fair-comparison row.",
    },
    {
        "environment": "humanoidmaze-large-navigate-oraclerep-v0",
        "paper_overall": 8.0,
        "artifact": "exp/trl_humanoidmaze_large_rpg_actor_total1m_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 1_000_000,
        "episodes_per_task": 3,
        "status": "existing_smoke",
        "claim_scope": "Native direct paper-style actor at 1M; 3 episodes/task smoke.",
    },
    {
        "environment": "humanoidmaze-large-navigate-oraclerep-v0",
        "paper_overall": 8.0,
        "artifact": "exp/humanoidmaze_large_graph_trl_giant1m_bmm_switch128_ep15_seed10_detreset_max2000.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "BMM graph control on Large using the fixed 1M Giant TRL/RPG actor.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_trl_rpg_total100k_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 100_000,
        "episodes_per_task": 3,
        "status": "existing_not_1m",
        "claim_scope": "Direct paper-style actor baseline exists only at 100k so far.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_trl_rpg_total700k_eval_ep15_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 700_000,
        "episodes_per_task": 15,
        "status": "existing_not_1m",
        "claim_scope": "Direct paper-style actor at 700k total updates; final 1M checkpoint is still running.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_trl_rpg_total1m_eval_ep15_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL RPG / DDPG+BC actor",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "Direct paper-style actor at 1M total updates.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_bmm_rpg_alpha03_optionA_50k_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "BMM RPG / DDPG+BC actor",
        "total_updates": 50_000,
        "episodes_per_task": 3,
        "status": "existing_option_a_smoke",
        "claim_scope": "BMM option-A actor extraction at 50k; max-budget smoke.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_bmm_rpg_alpha03_optionA_50k_scan07_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "BMM RPG / DDPG+BC actor, scan flag no-op",
        "total_updates": 50_000,
        "episodes_per_task": 3,
        "status": "existing_option_a_noop_scan",
        "claim_scope": "RPG/DDPG+BC emits one direct action, so actor_budget_mode=scan does not rerank actions here.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_bmm_rpg_alpha03_optionA_total100k_scan07_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "BMM RPG / DDPG+BC actor, scan flag no-op",
        "total_updates": 100_000,
        "episodes_per_task": 3,
        "status": "existing_option_a_best_smoke",
        "claim_scope": "Best observed BMM option-A RPG smoke, but the scan flag is a no-op for RPG/DDPG+BC.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_bmm_rpg_alpha1_optionA_50k_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "BMM RPG / DDPG+BC actor",
        "total_updates": 50_000,
        "episodes_per_task": 3,
        "status": "existing_option_a_negative",
        "claim_scope": "BMM/RPG alpha=1.0 option-A 50k smoke.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_bmm_frs_optionA_50k_eval_ep3_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "BMM FRS / rejection-sampling actor",
        "total_updates": 50_000,
        "episodes_per_task": 3,
        "status": "existing_option_a_negative",
        "claim_scope": "BMM/FRS option-A 50k smoke.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_our_controller_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing",
        "claim_scope": "BMM graph control on AntSoccer using the fixed 1M TRL/RPG actor, subgoal_commit_steps=10, task-1/task-4 final-goal switches of 128, and a task-5 final-goal switch of 48.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "support_path_only",
        "track": "B_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_matched_support_control",
        "claim_scope": "Matched support-path control with the same fixed 1M actor and task-specific final-goal switch schedule as the promoted BMM AntSoccer row.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_offset15_seed10_detreset.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_heldout_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_heldout_offset15",
        "claim_scope": "Heldout deterministic episode block for the promoted fixed-actor BMM AntSoccer protocol.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_offset15_seed10_detreset.json",
        "parser": "selector",
        "selector": "support_path_only",
        "track": "B_heldout_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_heldout_offset15",
        "claim_scope": "Heldout deterministic episode block for the matched fixed-actor support-path AntSoccer control.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_offset30_seed10_detreset.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_heldout_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_heldout_offset30",
        "claim_scope": "Heldout deterministic episode block for the promoted fixed-actor BMM AntSoccer protocol.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_offset30_seed10_detreset.json",
        "parser": "selector",
        "selector": "support_path_only",
        "track": "B_heldout_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_heldout_offset30",
        "claim_scope": "Heldout deterministic episode block for the matched fixed-actor support-path AntSoccer control.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task1switch128_reset45_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_lower_control",
        "claim_scope": "Task-block controller RNG reset for tasks 4/5 matches the older 65/75 task-1-switch row but stays below the promoted 68/75 fixed-actor AntSoccer result.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task1switch128_task4commit20_task5switch48_ep15_seed10_detreset.json",
        "parser": "selector",
        "selector": "BMM_support_path",
        "track": "B_control_fixed_paper_actor",
        "policy_extraction": "fixed TRL RPG / DDPG+BC controller",
        "total_updates": 1_000_000,
        "episodes_per_task": 15,
        "status": "existing_negative_full",
        "claim_scope": "Best task-specific commit/switch follow-up did not improve the fixed-actor AntSoccer result.",
    },
    {
        "environment": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_task_routed_overall.json",
        "parser": "task_routed",
        "track": "C_non_paper_controller",
        "policy_extraction": "local GCFBC/support-graph task-routed controller",
        "total_updates": None,
        "episodes_per_task": 15,
        "status": "existing_not_same_extraction",
        "claim_scope": "Useful headline, but not a fair paper-style policy-extraction comparison.",
    },
    {
        "environment": "puzzle-4x5-play-oraclerep-v0",
        "paper_overall": 97.0,
        "artifact": "exp/trl_puzzle4x5_frs_smoke100k_eval_ep1.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 100_000,
        "episodes_per_task": 1,
        "status": "existing_not_1m",
        "claim_scope": "Paper-style FRS smoke exists, but it is 100k and failed.",
    },
    {
        "environment": "puzzle-4x5-play-oraclerep-v0",
        "paper_overall": 97.0,
        "artifact": "exp/trl_puzzle4x5_frs_total400k_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 400_000,
        "episodes_per_task": 1,
        "status": "existing_not_1m",
        "claim_scope": "Paper-style FRS at 400k total updates is still a failed smoke.",
    },
    {
        "environment": "puzzle-4x5-play-oraclerep-v0",
        "paper_overall": 97.0,
        "artifact": "exp/trl_puzzle4x5_frs_total700k_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 700_000,
        "episodes_per_task": 1,
        "status": "existing_not_1m",
        "claim_scope": "Paper-style FRS at 700k total updates is still a failed smoke.",
    },
    {
        "environment": "puzzle-4x5-play-oraclerep-v0",
        "paper_overall": 97.0,
        "artifact": "exp/trl_puzzle4x5_frs_total1m_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 1_000_000,
        "episodes_per_task": 1,
        "status": "existing_smoke",
        "claim_scope": "Paper-style FRS at 1M total updates remains a failed 1-episode/task smoke.",
    },
    {
        "environment": "puzzle-4x5-play-oraclerep-v0",
        "paper_overall": 97.0,
        "artifact": "exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.json",
        "parser": "puzzle",
        "track": "C_non_paper_controller",
        "policy_extraction": "structured Lights Out planner + local GCFBC",
        "total_updates": 100_000,
        "episodes_per_task": 15,
        "status": "existing_not_same_extraction",
        "claim_scope": "Strong task result, but not the TRL paper extraction procedure.",
    },
    {
        "environment": "puzzle-4x6-play-oraclerep-v0",
        "paper_overall": 51.0,
        "artifact": "exp/trl_puzzle4x6_frs_total300k_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 300_000,
        "episodes_per_task": 1,
        "status": "existing_not_1m",
        "claim_scope": "Paper-style FRS at 300k total updates is a failed smoke; 1M checkpoint is still running.",
    },
    {
        "environment": "puzzle-4x6-play-oraclerep-v0",
        "paper_overall": 51.0,
        "artifact": "exp/trl_puzzle4x6_frs_total600k_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 600_000,
        "episodes_per_task": 1,
        "status": "existing_not_1m",
        "claim_scope": "Paper-style FRS at 600k total updates is a failed smoke; 1M checkpoint is still running.",
    },
    {
        "environment": "puzzle-4x6-play-oraclerep-v0",
        "paper_overall": 51.0,
        "artifact": "exp/trl_puzzle4x6_frs_total1m_eval_ep1_seed10.json",
        "parser": "policy_eval",
        "track": "A_same_paper_extraction",
        "policy_extraction": "TRL FRS / flow rejection sampling",
        "total_updates": 1_000_000,
        "episodes_per_task": 1,
        "status": "existing_smoke",
        "claim_scope": "Paper-style FRS at 1M total updates remains a failed 1-episode/task smoke.",
    },
    {
        "environment": "cube-double-play-oraclerep-v0",
        "paper_overall": 30.0,
        "artifact": "exp/cube_double_seq_gcfbc_local200k_dynamic_finalzfarthest_r80_p5_f100_ep15.json",
        "parser": "policy_eval",
        "track": "C_non_paper_controller",
        "policy_extraction": "dynamic sequential block subgoals + local GCFBC",
        "total_updates": 200_000,
        "episodes_per_task": 15,
        "status": "existing_not_same_extraction",
        "claim_scope": "Strong task result, but not the TRL paper extraction procedure.",
    },
]


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value)


def load_json(path: str) -> dict[str, Any] | None:
    full = REPO_ROOT / path
    if not full.exists():
        return None
    return json.loads(full.read_text())


def summarize_policy_eval(data: dict[str, Any]) -> dict[str, Any]:
    tasks = data.get("tasks", [])
    eval_episodes = int(data.get("eval_episodes", 0) or 0)
    overall = data.get("overall", {})
    episodes = overall.get("episodes")
    if episodes is not None:
        episodes = int(episodes)
    elif eval_episodes and tasks:
        episodes = eval_episodes * len(tasks)
    else:
        episodes = None
    success = pct(overall.get("success"))
    successes = overall.get("successes")
    if successes is not None:
        successes = int(successes)
    elif episodes is not None and success is not None:
        successes = int(round(success / 100.0 * episodes))
    per_task = []
    for task in tasks:
        per_task.append(
            {
                "task": int(task.get("task_id", task.get("task"))),
                "success": pct(task.get("success")),
            }
        )
    for key, value in sorted(data.get("per_task", {}).items(), key=lambda item: int(item[0])):
        if any(row["task"] == int(key) for row in per_task):
            continue
        per_task.append({"task": int(key), "success": pct(value.get("success"))})
    return {"success": success, "successes": successes, "episodes": episodes, "per_task": per_task}


def summarize_selector(data: dict[str, Any], selector_name: str) -> dict[str, Any]:
    for selector in data.get("selectors", []):
        if selector.get("name") != selector_name:
            continue
        aggregate = selector.get("aggregate", {})
        episodes = int(aggregate.get("episodes", 0) or 0)
        success = pct(aggregate.get("success"))
        successes = int(round(success / 100.0 * episodes)) if success is not None else None
        per_task = []
        rows = selector.get("episodes", [])
        task_ids = sorted({int(row.get("task", row.get("task_id"))) for row in rows})
        for task_id in task_ids:
            task_rows = [row for row in rows if int(row.get("task", row.get("task_id"))) == task_id]
            task_successes = sum(float(row.get("success", 0.0)) for row in task_rows)
            per_task.append(
                {
                    "task": task_id,
                    "success": 100.0 * task_successes / max(len(task_rows), 1),
                }
            )
        return {"success": success, "successes": successes, "episodes": episodes, "per_task": per_task}
    return {"success": None, "successes": None, "episodes": None, "per_task": []}


def summarize_task_routed(data: dict[str, Any]) -> dict[str, Any]:
    if {"success", "successes", "episodes"} <= set(data):
        return {
            "success": float(data["success"]),
            "successes": int(data["successes"]),
            "episodes": int(data["episodes"]),
            "per_task": [
                {"task": int(row["task"]), "success": float(row["success"])}
                for row in data.get("per_task", [])
            ],
        }
    if "overall" in data:
        overall = data["overall"]
        episodes = int(overall.get("episodes", 0) or 0)
        success = pct(overall.get("success"))
        successes = int(round(success / 100.0 * episodes)) if success is not None and episodes else None
        return {"success": success, "successes": successes, "episodes": episodes or None, "per_task": []}
    if "routes" in data:
        successes = sum(int(route.get("successes", 0)) for route in data["routes"])
        episodes = sum(int(route.get("episodes", 0)) for route in data["routes"])
        success = 100.0 * successes / episodes if episodes else None
        return {"success": success, "successes": successes, "episodes": episodes, "per_task": []}
    return {"success": None, "successes": None, "episodes": None, "per_task": []}


def summarize_puzzle(data: dict[str, Any]) -> dict[str, Any]:
    overall = data.get("overall", {})
    episodes = int(overall.get("episodes", 0) or 0)
    success = pct(overall.get("success"))
    successes = int(round(success / 100.0 * episodes)) if success is not None and episodes else None
    per_task = []
    for key, value in sorted(data.get("per_task", {}).items(), key=lambda item: int(item[0])):
        per_task.append({"task": int(key), "success": pct(value.get("success"))})
    return {"success": success, "successes": successes, "episodes": episodes or None, "per_task": per_task}


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    data = load_json(row["artifact"])
    result = dict(row)
    result["artifact_exists"] = data is not None
    if data is None:
        result.update(success=None, successes=None, episodes=None, per_task=[])
        return result
    if row["parser"] == "policy_eval":
        result.update(summarize_policy_eval(data))
    elif row["parser"] == "selector":
        result.update(summarize_selector(data, row["selector"]))
    elif row["parser"] == "task_routed":
        result.update(summarize_task_routed(data))
    elif row["parser"] == "puzzle":
        result.update(summarize_puzzle(data))
    else:
        raise ValueError(f"Unsupported parser {row['parser']!r}")
    result["delta_vs_paper"] = (
        None
        if result["success"] is None or row.get("paper_overall") is None
        else result["success"] - float(row["paper_overall"])
    )
    return result


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def fmt_count(row: dict[str, Any]) -> str:
    if row.get("successes") is None or row.get("episodes") is None:
        return ""
    return f" ({row['successes']}/{row['episodes']})"


def write_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fair Policy-Extraction Audit",
        "",
        "This separates three evidence tracks:",
        "",
        "- A: same paper policy extraction (`rpg`/DDPG+BC or `frs`/rejection sampling).",
        "- B: our controller/subgoal selection with a fixed paper-style low-level actor.",
        "- C: useful task result, but not the same policy-extraction procedure.",
        "",
        "| environment | track | extraction/controller | updates | episodes/task | success | paper TRL | delta | status | artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        updates = "n/a" if row.get("total_updates") is None else str(row["total_updates"])
        episodes = "n/a" if row.get("episodes_per_task") is None else str(row["episodes_per_task"])
        delta = "n/a" if row.get("delta_vs_paper") is None else f"{row['delta_vs_paper']:+.1f}"
        lines.append(
            "| `{environment}` | {track} | {policy_extraction} | {updates} | {episodes} | {success}{count} | {paper} | {delta} | {status} | `{artifact}` |".format(
                environment=row["environment"],
                track=row["track"],
                policy_extraction=row["policy_extraction"],
                updates=updates,
                episodes=episodes,
                success=fmt_pct(row.get("success")),
                count=fmt_count(row),
                paper=fmt_pct(row.get("paper_overall")),
                delta=delta,
                status=row["status"],
                artifact=row["artifact"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- We can make clean 1M option-B claims on HumanoidMaze-medium and HumanoidMaze-large: BMM subgoal control uses fixed 1M TRL/RPG actors and beats the corresponding paper TRL rows.",
            "- AntSoccer now also has a strong 1M option-B row: direct TRL/RPG remains below the paper row, while BMM graph control with the fixed 1M TRL/RPG actor reaches 68/75. The matched support-path control with the same switch schedule reaches 59/75 on the promoted block. Heldout offset blocks are mixed but still positive overall: offsets 0/15/30 give 192/225 for BMM versus 182/225 for matched support.",
            "- Direct paper-style FRS did not recover the hard puzzle tasks in these smokes: Puzzle-4x5 and Puzzle-4x6 are both 0/5 at 1M.",
            "- The strong puzzle, scene, and cube rows should therefore be presented as controller/planner results rather than same-extraction TRL-policy results.",
            "- The highest-value next fair runs are robustness checks for fixed-controller rows where the margin is not uniform, plus any full 15-episode confirmations needed for currently smoke-only rows.",
            "- For BMM/RPG, `actor_budget_mode=scan` is only a label in the old artifacts: scan reranking is implemented for FRS sampled candidates, while RPG/DDPG+BC emits one direct policy action.",
            "- BMM option-A actor extraction on AntSoccer is now runnable after vectorizing BMM rank-field sampling, but the best current 100k smoke is only 7/15. Do not spend a full 1M run here unless we need to report a same-extraction negative or have a stronger actor objective to test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = [summarize(row) for row in ROWS]
    payload = {
        "tracks": {
            "A_same_paper_extraction": "Direct actor/policy extraction using the paper methods in agents/trl.py or agents/bmm_trl.py.",
            "B_our_controller_fixed_paper_actor": "BMM/controller evaluation with a fixed low-level TRL RPG/FRS actor.",
            "C_non_paper_controller": "Useful task result but not a same-extraction fairness row.",
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_MD.write_text(write_markdown(rows))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
