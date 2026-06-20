#!/usr/bin/env python
"""Audit which paper-listed rows are beaten overall and per task."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_JSON = REPO_ROOT / "exp" / "bmm_advanced_policy_table.json"
OUT_JSON = REPO_ROOT / "exp" / "bmm_paper_task_coverage_audit.json"
OUT_MD = REPO_ROOT / "exp" / "bmm_paper_task_coverage_audit.md"

NON_PROMOTED_ROWS = [
    {
        "env": "antsoccer-arena-navigate-oraclerep-v0",
        "paper_overall": 73.0,
        "artifact": "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_ep15_seed10_detreset.json",
        "reason": "Best clean single-protocol confirmation remains below the paper row; the promoted row is task-routed support-only.",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_pct(value) -> str:
    return f"{float(value):.1f}%"


def promoted_rows() -> list[dict]:
    return load_json(ADVANCED_JSON)["rows"]


def scene_graph_summary(path: Path) -> dict:
    data = load_json(path)
    selector = data["selectors"][0]
    agg = selector["aggregate"]
    return {
        "success": 100.0 * float(agg["success"]),
        "successes": int(round(float(agg["success"]) * int(agg["episodes"]))),
        "episodes": int(agg["episodes"]),
    }


def per_task_gaps(row: dict) -> list[dict]:
    paper = row.get("paper_per_task")
    if paper is None:
        return []
    gaps = []
    for task, paper_success in zip(row["per_task"], paper):
        ours = float(task["success"])
        paper_success = float(paper_success)
        if ours + 1e-9 < paper_success:
            gaps.append(
                {
                    "task": int(task["task"]),
                    "paper": paper_success,
                    "ours": ours,
                    "gap": ours - paper_success,
                }
            )
    return gaps


def build_audit() -> dict:
    promoted = []
    for row in promoted_rows():
        gaps = per_task_gaps(row)
        promoted.append(
            {
                "env": row["env"],
                "paper_overall": float(row["paper_overall"]),
                "ours_overall": float(row["success"]),
                "overall_delta": float(row["delta_vs_paper"]),
                "successes": int(row["successes"]),
                "episodes": int(row["episodes"]),
                "overall_beats_or_matches": float(row["success"]) + 1e-9
                >= float(row["paper_overall"]),
                "paper_per_task_available": row.get("paper_per_task") is not None,
                "per_task_gaps": gaps,
                "all_available_per_tasks_beat_or_match": (
                    row.get("paper_per_task") is not None and len(gaps) == 0
                ),
                "protocol": row["protocol"],
                "caveat": row["caveat"],
            }
        )

    non_promoted = []
    for item in NON_PROMOTED_ROWS:
        summary = scene_graph_summary(REPO_ROOT / item["artifact"])
        non_promoted.append(
            {
                **item,
                **summary,
                "overall_delta": summary["success"] - float(item["paper_overall"]),
                "overall_beats_or_matches": summary["success"] + 1e-9
                >= float(item["paper_overall"]),
            }
        )

    with_task_refs = [row for row in promoted if row["paper_per_task_available"]]
    return {
        "summary": {
            "promoted_rows": len(promoted),
            "promoted_overall_beat_or_match": sum(
                int(row["overall_beats_or_matches"]) for row in promoted
            ),
            "promoted_rows_with_paper_per_task": len(with_task_refs),
            "promoted_rows_all_per_tasks_beat_or_match": sum(
                int(row["all_available_per_tasks_beat_or_match"])
                for row in with_task_refs
            ),
            "non_promoted_rows": len(non_promoted),
            "non_promoted_overall_beat_or_match": sum(
                int(row["overall_beats_or_matches"]) for row in non_promoted
            ),
        },
        "promoted": promoted,
        "non_promoted": non_promoted,
    }


def markdown(audit: dict) -> str:
    s = audit["summary"]
    lines = [
        "# BMM-TRL Paper Task Coverage Audit",
        "",
        "This audit separates promoted artifact-backed rows from rows that remain below the paper target.",
        "",
        "## Summary",
        "",
        "| metric | count |",
        "|---|---:|",
        f"| promoted rows beating/matching paper overall | {s['promoted_overall_beat_or_match']}/{s['promoted_rows']} |",
        f"| promoted rows with paper per-task references | {s['promoted_rows_with_paper_per_task']} |",
        f"| promoted rows beating/matching every available per-task entry | {s['promoted_rows_all_per_tasks_beat_or_match']}/{s['promoted_rows_with_paper_per_task']} |",
        f"| non-promoted rows beating/matching paper overall | {s['non_promoted_overall_beat_or_match']}/{s['non_promoted_rows']} |",
        "",
        "## Promoted Rows",
        "",
        "| environment | paper overall | ours | overall delta | per-task gaps | caveat |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in audit["promoted"]:
        if row["paper_per_task_available"]:
            gaps = row["per_task_gaps"]
            gap_text = (
                "none"
                if not gaps
                else ", ".join(
                    f"task {gap['task']}: {fmt_pct(gap['ours'])} vs {fmt_pct(gap['paper'])}"
                    for gap in gaps
                )
            )
        else:
            gap_text = "paper per-task n/a"
        lines.append(
            "| `{env}` | {paper} | {ours} ({succ}/{eps}) | {delta:+.1f} | {gaps} | {caveat} |".format(
                env=row["env"],
                paper=fmt_pct(row["paper_overall"]),
                ours=fmt_pct(row["ours_overall"]),
                succ=row["successes"],
                eps=row["episodes"],
                delta=row["overall_delta"],
                gaps=gap_text,
                caveat=row["caveat"],
            )
        )

    lines.extend(
        [
            "",
            "## Not Promoted",
            "",
            "| environment | paper overall | best checked full result | delta | reason |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in audit["non_promoted"]:
        lines.append(
            "| `{env}` | {paper} | {ours} ({succ}/{eps}) | {delta:+.1f} | {reason} |".format(
                env=row["env"],
                paper=fmt_pct(row["paper_overall"]),
                ours=fmt_pct(row["success"]),
                succ=row["successes"],
                eps=row["episodes"],
                delta=row["overall_delta"],
                reason=row["reason"],
            )
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
