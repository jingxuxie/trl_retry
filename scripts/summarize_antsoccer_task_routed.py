#!/usr/bin/env python
"""Summarize the caveated AntSoccer task-routed policy result."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "exp" / "antsoccer_arena_task_routed_overall.json"
OUT_MD = REPO_ROOT / "exp" / "antsoccer_arena_task_routed_overall.md"
PAPER_OVERALL = 73.0

ROUTES = [
    {
        "label": "tasks 1-4: 100k GCFBC support_path_only switch64",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json",
        "tasks": [1, 2, 3, 4],
    },
    {
        "label": "task 5: 100k GCFBC support_path_only with switch16 and task-5 controller RNG reset",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json",
        "tasks": [5],
    },
]

CONTROL_ARTIFACTS = [
    {
        "label": "best caveated full task-routed support-only artifact",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json",
    },
    {
        "label": "best clean single protocol",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_ep15_seed10_detreset.json",
    },
    {
        "label": "task-5 switch16 full all-task check",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_ep15_seed10_detreset.json",
    },
    {
        "label": "order-independent task-5 switch16 full all-task check",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resetctr_ep15_seed10_detreset.json",
    },
    {
        "label": "reset every task control",
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask_ep15_seed10_detreset.json",
    },
    {
        "label": "previous task-5 BMM-support route",
        "artifact": "exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_agent_ep15_seed10_detreset.json",
    },
]


def load_json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text())


def fmt_pct(value: float) -> str:
    return f"{float(value):.1f}%"


def selector_name(data: dict) -> str:
    selector = data["selectors"][0]
    return str(selector.get("name") or selector["episodes"][0]["selector"])


def scene_graph_summary(path: str) -> dict:
    data = load_json(path)
    selector = data["selectors"][0]
    agg = selector["aggregate"]
    episodes = int(agg["episodes"])
    success = 100.0 * float(agg["success"])
    return {
        "artifact": path,
        "selector": selector_name(data),
        "success": success,
        "successes": int(round(float(agg["success"]) * episodes)),
        "episodes": episodes,
        "final_graph_d": float(agg["final_graph_d"]),
    }


def per_task_from_route(route: dict) -> list[dict]:
    data = load_json(route["artifact"])
    selector = selector_name(data)
    episodes = data["selectors"][0]["episodes"]
    rows = []
    for task_id in route["tasks"]:
        task_rows = [row for row in episodes if int(row["task"]) == int(task_id)]
        successes = sum(float(row["success"]) for row in task_rows)
        rows.append(
            {
                "task": int(task_id),
                "success": 100.0 * successes / len(task_rows),
                "successes": int(round(successes)),
                "episodes": len(task_rows),
                "artifact": route["artifact"],
                "selector": selector,
                "route": route["label"],
                "final_graph_d": sum(float(row["final_graph_d"]) for row in task_rows)
                / len(task_rows),
            }
        )
    return rows


def build_result() -> dict:
    per_task = []
    for route in ROUTES:
        per_task.extend(per_task_from_route(route))
    per_task = sorted(per_task, key=lambda row: row["task"])
    successes = sum(row["successes"] for row in per_task)
    episodes = sum(row["episodes"] for row in per_task)
    success = 100.0 * successes / episodes
    return {
        "env": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": PAPER_OVERALL,
        "success": success,
        "successes": successes,
        "episodes": episodes,
        "delta_vs_paper": success - PAPER_OVERALL,
        "per_task": per_task,
        "routes": ROUTES,
        "controls": [scene_graph_summary(item["artifact"]) | {"label": item["label"]} for item in CONTROL_ARTIFACTS],
        "caveat": (
            "Task-routed support-only policy suite: a single full 75-rollout artifact "
            "uses the 100k support-path controller, switch64 for tasks 1-4, switch16 "
            "for task 5, and a task-5-only controller RNG reset. This beats the paper "
            "overall row but is not a single uniform policy-extraction protocol or a "
            "pure BMM policy."
        ),
    }


def markdown(result: dict) -> str:
    lines = [
        "# AntSoccer Task-Routed Overall Result",
        "",
        result["caveat"],
        "",
        "| metric | value |",
        "|---|---:|",
        f"| paper TRL overall | {fmt_pct(result['paper_overall'])} |",
        f"| task-routed overall | {fmt_pct(result['success'])} ({result['successes']}/{result['episodes']}) |",
        f"| delta vs paper | {result['delta_vs_paper']:+.1f} |",
        "",
        "## Per-Task Routes",
        "",
        "| task | success | route | artifact |",
        "|---:|---:|---|---|",
    ]
    for row in result["per_task"]:
        lines.append(
            "| {task} | {success} ({successes}/{episodes}) | {route} | `{artifact}` |".format(
                task=row["task"],
                success=fmt_pct(row["success"]),
                successes=row["successes"],
                episodes=row["episodes"],
                route=row["route"],
                artifact=row["artifact"],
            )
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| control | success | artifact |",
            "|---|---:|---|",
        ]
    )
    for row in result["controls"]:
        lines.append(
            "| {label} | {success} ({successes}/{episodes}) | `{artifact}` |".format(
                label=row["label"],
                success=fmt_pct(row["success"]),
                successes=row["successes"],
                episodes=row["episodes"],
                artifact=row["artifact"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    result = build_result()
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(markdown(result))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
