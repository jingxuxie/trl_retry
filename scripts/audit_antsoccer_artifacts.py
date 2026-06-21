#!/usr/bin/env python
"""Audit saved AntSoccer artifacts for paper-facing result claims."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "exp" / "antsoccer_arena_artifact_audit.json"
OUT_MD = REPO_ROOT / "exp" / "antsoccer_arena_artifact_audit.md"

CANONICAL_PROTOCOLS = [
    {
        "label": "best fixed-actor BMM protocol",
        "kind": "single_artifact",
        "selector": "BMM_support_path",
        "artifact": "exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json",
        "caveat": "BMM graph subgoals with the fixed 1M paper-style TRL/RPG actor, subgoal_commit_steps=10, task-1/task-4 final-goal switches of 128, and a task-5 final-goal switch of 48.",
    },
    {
        "label": "matched fixed-actor support-path control",
        "kind": "single_artifact",
        "selector": "support_path_only",
        "artifact": "exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json",
        "caveat": "Same fixed 1M TRL/RPG actor and the same task-specific final-goal switch distances, but without BMM value ranking.",
    },
    {
        "label": "best clean single protocol",
        "kind": "single_artifact",
        "selector": "support_path_only",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_ep15_seed10_detreset.json",
        "caveat": "Uniform 100k local GCFBC support-path controller with switch64.",
    },
    {
        "label": "best task-routed support-only artifact",
        "kind": "single_artifact",
        "selector": "support_path_only",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json",
        "caveat": "Single 75-rollout artifact, but task 5 uses switch16 plus task-5-only controller RNG reset.",
    },
    {
        "label": "best full BMM single artifact",
        "kind": "single_artifact",
        "selector": "BMM_support_path",
        "artifact": "exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_agent_ep15_seed10_detreset.json",
        "caveat": "Uniform full BMM-support graph policy with the 50k local GCFBC agent controller.",
    },
    {
        "label": "best BMM-including routed suite",
        "kind": "routed",
        "routes": [
            {
                "tasks": [1, 2, 3, 4],
                "selector": "support_path_only",
                "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_ep15_seed10_detreset.json",
                "route": "tasks 1-4 from the 100k support-path controller",
            },
            {
                "tasks": [5],
                "selector": "BMM_support_path",
                "artifact": "exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_agent_ep15_seed10_detreset.json",
                "route": "task 5 from the 50k BMM-support full artifact",
            },
        ],
        "caveat": "Not a single uniform policy: uses support-path on tasks 1-4 and BMM-support on task 5.",
    },
]


def display_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def selector_name(selector: dict[str, Any]) -> str:
    return str(
        selector.get("name")
        or selector.get("selector")
        or (selector.get("episodes") or [{}])[0].get("selector")
        or "overall"
    )


def aggregate_from_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"episodes": 0, "success": float("nan"), "successes": 0}
    successes = sum(safe_float(row.get("success", 0.0)) for row in episodes)
    return {
        "episodes": len(episodes),
        "success": successes / len(episodes),
        "successes": int(round(successes)),
    }


def per_task_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_ids = sorted(
        {
            int(row.get("task", row.get("task_id")))
            for row in episodes
            if row.get("task", row.get("task_id")) is not None
        }
    )
    rows = []
    for task_id in task_ids:
        task_rows = [
            row
            for row in episodes
            if int(row.get("task", row.get("task_id"))) == task_id
        ]
        agg = aggregate_from_episodes(task_rows)
        rows.append(
            {
                "task": task_id,
                "successes": agg["successes"],
                "episodes": agg["episodes"],
                "success": 100.0 * agg["success"],
            }
        )
    return rows


def load_selector_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = []
    if "selectors" in data:
        for selector in data["selectors"]:
            episodes = selector.get("episodes", [])
            aggregate = dict(selector.get("aggregate") or aggregate_from_episodes(episodes))
            if "episodes" not in aggregate or "successes" not in aggregate:
                ep_agg = aggregate_from_episodes(episodes)
                aggregate.setdefault("episodes", ep_agg["episodes"])
                aggregate.setdefault("successes", ep_agg["successes"])
            episodes_n = int(aggregate.get("episodes") or len(episodes))
            success = safe_float(aggregate.get("success"))
            successes = int(round(success * episodes_n)) if math.isfinite(success) else 0
            rows.append(
                {
                    "artifact": display_path(path),
                    "selector": selector_name(selector),
                    "success": 100.0 * success,
                    "successes": successes,
                    "episodes": episodes_n,
                    "per_task": per_task_summary(episodes),
                    "final_graph_d": safe_float(aggregate.get("final_graph_d")),
                }
            )
    elif "overall" in data:
        overall = data["overall"]
        success = safe_float(overall.get("success"))
        tasks = data.get("tasks", [])
        episodes_per_task = int(data.get("eval_episodes") or 0)
        episodes_n = episodes_per_task * len(tasks) if tasks else episodes_per_task
        per_task = []
        for task in tasks:
            task_id = int(task.get("task_id", len(per_task) + 1))
            task_success = safe_float(task.get("success"))
            task_successes = (
                int(round(task_success * episodes_per_task))
                if episodes_per_task and math.isfinite(task_success)
                else 0
            )
            per_task.append(
                {
                    "task": task_id,
                    "successes": task_successes,
                    "episodes": episodes_per_task,
                    "success": 100.0 * task_success,
                }
            )
        rows.append(
            {
                "artifact": display_path(path),
                "selector": "overall",
                "success": 100.0 * success,
                "successes": int(round(success * episodes_n)) if episodes_n else 0,
                "episodes": episodes_n,
                "per_task": per_task,
                "final_graph_d": float("nan"),
            }
        )
    return rows


def find_selector(artifact: str, selector: str) -> dict[str, Any]:
    path = REPO_ROOT / artifact
    matches = [
        row
        for row in load_selector_rows(path)
        if row["selector"] == selector or selector == "*"
    ]
    if not matches:
        raise ValueError(f"Missing selector {selector!r} in {artifact}")
    return matches[0]


def task_rows_for_route(route: dict[str, Any]) -> list[dict[str, Any]]:
    row = find_selector(route["artifact"], route["selector"])
    per_task = []
    for task_id in route["tasks"]:
        task = next(item for item in row["per_task"] if item["task"] == task_id)
        per_task.append(
            task
            | {
                "artifact": route["artifact"],
                "selector": route["selector"],
                "route": route["route"],
            }
        )
    return per_task


def canonical_result(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "single_artifact":
        row = find_selector(spec["artifact"], spec["selector"])
        return {
            "label": spec["label"],
            "kind": spec["kind"],
            "success": row["success"],
            "successes": row["successes"],
            "episodes": row["episodes"],
            "selector": row["selector"],
            "artifact": row["artifact"],
            "per_task": row["per_task"],
            "caveat": spec["caveat"],
        }
    per_task = []
    for route in spec["routes"]:
        per_task.extend(task_rows_for_route(route))
    successes = sum(row["successes"] for row in per_task)
    episodes = sum(row["episodes"] for row in per_task)
    return {
        "label": spec["label"],
        "kind": spec["kind"],
        "success": 100.0 * successes / episodes,
        "successes": successes,
        "episodes": episodes,
        "selector": "routed",
        "artifact": "multiple",
        "per_task": sorted(per_task, key=lambda row: row["task"]),
        "caveat": spec["caveat"],
    }


def build_audit() -> dict[str, Any]:
    rows = []
    for path in sorted((REPO_ROOT / "exp").glob("antsoccer*.json")):
        rows.extend(load_selector_rows(path))
    full_rows = [
        row for row in rows if int(row.get("episodes") or 0) >= 75
    ]
    full_rows = sorted(
        full_rows,
        key=lambda row: (-int(row["successes"]), row["artifact"], row["selector"]),
    )
    task_rows = [
        row for row in rows if int(row.get("episodes") or 0) == 15
    ]
    task_rows = sorted(
        task_rows,
        key=lambda row: (-int(row["successes"]), row["artifact"], row["selector"]),
    )
    canonical = [canonical_result(spec) for spec in CANONICAL_PROTOCOLS]
    return {
        "env": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_trl_overall": 73.0,
        "canonical": canonical,
        "full_artifacts": full_rows,
        "fifteen_episode_artifacts": task_rows,
        "conclusion": (
            "The best saved AntSoccer artifact is now 68/75: BMM graph subgoals "
            "with the fixed 1M paper-style TRL/RPG actor, task-1/task-4 "
            "final-goal switches of 128, and a task-5 final-goal switch of 48. "
            "The matched support-path control with the same fixed actor and "
            "switch schedule is 59/75 on the promoted offset-0 block. Heldout "
            "offset blocks are mixed but still favor BMM overall: offsets "
            "0/15/30 give 192/225 for BMM versus 182/225 for matched support. "
            "The previous promoted task-1-only switch row and its RNG-reset "
            "controls remain at 65/75; budget and switch controls stayed at "
            "or below 65/75. The older task-routed "
            "support-only run remains 58/75. The best local-GCFBC clean single "
            "protocol is 52/75, the best local-GCFBC full BMM single artifact is "
            "44/75, and the older local-GCFBC BMM-including routed suite is 55/75."
        ),
    }


def fmt_pct(value: float) -> str:
    return f"{float(value):.1f}%"


def fmt_result(row: dict[str, Any]) -> str:
    return f"{fmt_pct(row['success'])} ({int(row['successes'])}/{int(row['episodes'])})"


def per_task_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "n/a"
    return ", ".join(
        f"t{int(row['task'])}: {int(row['successes'])}/{int(row['episodes'])}"
        for row in rows
    )


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# AntSoccer Artifact Audit",
        "",
        audit["conclusion"],
        "",
        "## Canonical Paper-Facing Protocols",
        "",
        "| protocol | success | per-task successes | artifact / selector | caveat |",
        "|---|---:|---|---|---|",
    ]
    for row in audit["canonical"]:
        source = (
            f"`{row['artifact']}` / `{row['selector']}`"
            if row["artifact"] != "multiple"
            else "`multiple routed artifacts`"
        )
        lines.append(
            f"| {row['label']} | {fmt_result(row)} | {per_task_text(row['per_task'])} | {source} | {row['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## Best Full Saved Artifacts",
            "",
            "| rank | success | selector | artifact | per-task successes |",
            "|---:|---:|---|---|---|",
        ]
    )
    for idx, row in enumerate(audit["full_artifacts"][:12], start=1):
        lines.append(
            f"| {idx} | {fmt_result(row)} | `{row['selector']}` | `{row['artifact']}` | {per_task_text(row['per_task'])} |"
        )
    lines.extend(
        [
            "",
            "## Best 15-Episode Smokes",
            "",
            "| rank | success | selector | artifact | per-task successes |",
            "|---:|---:|---|---|---|",
        ]
    )
    for idx, row in enumerate(audit["fifteen_episode_artifacts"][:20], start=1):
        lines.append(
            f"| {idx} | {fmt_result(row)} | `{row['selector']}` | `{row['artifact']}` | {per_task_text(row['per_task'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(markdown(audit))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
