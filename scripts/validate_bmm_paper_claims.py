#!/usr/bin/env python
"""Validate artifact-backed headline numbers used by the BMM-TRL paper draft."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_advanced_policy_table as advanced_summary
from scripts import audit_bmm_paper_task_coverage as coverage_audit
from scripts import audit_antsoccer_artifacts as antsoccer_audit
from scripts import validate_bmm_latex_static as latex_static
from scripts import summarize_bmm_paper_tables as paper_summary


PAPER_JSON = REPO_ROOT / "exp" / "bmm_paper_tables_final.json"
PAPER_MD = REPO_ROOT / "exp" / "bmm_paper_tables_final.md"
ADVANCED_JSON = REPO_ROOT / "exp" / "bmm_advanced_policy_table.json"
ADVANCED_MD = REPO_ROOT / "exp" / "bmm_advanced_policy_table.md"
COVERAGE_JSON = REPO_ROOT / "exp" / "bmm_paper_task_coverage_audit.json"
COVERAGE_MD = REPO_ROOT / "exp" / "bmm_paper_task_coverage_audit.md"
ANTSOCCER_JSON = REPO_ROOT / "exp" / "antsoccer_arena_artifact_audit.json"
ANTSOCCER_MD = REPO_ROOT / "exp" / "antsoccer_arena_artifact_audit.md"
REPRO_MD = REPO_ROOT / "BMM_TRL_REPRO_COMMANDS.md"


def refresh_summaries() -> None:
    args = paper_summary.parse_args(
        [
            "--output_markdown",
            str(PAPER_MD),
            "--output_json",
            str(PAPER_JSON),
        ]
    )
    report = paper_summary.build_report(args)
    PAPER_MD.parent.mkdir(parents=True, exist_ok=True)
    PAPER_MD.write_text(report["markdown"])
    serializable = {key: value for key, value in report.items() if key != "markdown"}
    PAPER_JSON.write_text(json.dumps(serializable, indent=2))

    rows = advanced_summary.build_rows()
    ADVANCED_JSON.parent.mkdir(parents=True, exist_ok=True)
    ADVANCED_JSON.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    ADVANCED_MD.write_text(advanced_summary.markdown(rows))

    audit = coverage_audit.build_audit()
    COVERAGE_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    COVERAGE_MD.write_text(coverage_audit.markdown(audit))

    antsoccer = antsoccer_audit.build_audit()
    ANTSOCCER_JSON.write_text(json.dumps(antsoccer, indent=2, sort_keys=True) + "\n")
    ANTSOCCER_MD.write_text(antsoccer_audit.markdown(antsoccer))


def load_json(path: Path):
    return json.loads(path.read_text())


def assert_close(name: str, actual: float, expected: float, tol: float = 5e-4) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def assert_ge(name: str, actual: float, threshold: float) -> None:
    if float(actual) + 1e-9 < float(threshold):
        raise AssertionError(f"{name}: expected >= {threshold}, got {actual}")


def find_holdout(data: dict, label: str, comparison: str) -> dict:
    for holdout in data["holdouts"]:
        if holdout["label"] != label:
            continue
        for row in holdout["aggregate"]:
            if row["comparison"] == comparison:
                return row
    raise AssertionError(f"Missing holdout comparison {label} / {comparison}")


def find_scene_qv(data: dict, label: str, variant: str) -> dict:
    for result in data["scene_qv"]:
        if result["label"] != label:
            continue
        for row in result["rows"]:
            if row["variant"] == variant:
                return row
    raise AssertionError(f"Missing Scene-Play row {label} / {variant}")


def find_advanced(data: dict, env: str) -> dict:
    for row in data["rows"]:
        if row["env"] == env:
            return row
    raise AssertionError(f"Missing advanced policy row {env}")


def scene_graph_success(path: str) -> tuple[int, int, float]:
    data = load_json(REPO_ROOT / path)
    aggregate = data["selectors"][0]["aggregate"]
    episodes = int(aggregate["episodes"])
    successes = int(round(float(aggregate["success"]) * episodes))
    return successes, episodes, 100.0 * float(aggregate["success"])


def normalize_surface(text: str) -> str:
    return (
        text.replace("\\%", "%")
        .replace("\\_", "_")
        .replace("\n", " ")
        .replace("  ", " ")
    )


def require_text(path: Path, snippets: list[str]) -> None:
    text = normalize_surface(path.read_text())
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        joined = "\n".join(f"- {item}" for item in missing)
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} is missing:\n{joined}")


def validate_value_tables(paper_data: dict) -> None:
    checks = [
        ("grid-cell H8", "B-A", 0.0149790446, 0.0751437097, -0.3609477125),
        ("env-step H160", "B-A", 0.0243021647, 0.0646712383, -0.5814733081),
        (
            "product ablation grid-cell H8",
            "B-P",
            0.0010986328,
            0.0049772963,
            -0.0265319224,
        ),
        (
            "product ablation env-step H160",
            "B-P",
            0.0027821859,
            0.0107854165,
            -0.0777664747,
        ),
    ]
    for label, comparison, auc, gap, bce in checks:
        row = find_holdout(paper_data, label, comparison)
        prefix = f"{label} {comparison}"
        assert_close(f"{prefix} delta AUC", row["mean_delta_auc"], auc)
        assert_close(f"{prefix} delta gap", row["mean_delta_gap"], gap)
        assert_close(f"{prefix} delta BCE", row["mean_delta_bce"], bce)

    scene_checks = [
        ("Scene-Play train+val graph H128", "B_no_parent_qv_trans", 0.8177185059, 0.2355064240),
        ("Scene-Play train+val graph H128", "P_no_parent_product_qv", 0.8128967285, 0.1969177391),
        ("Scene-Play train-only graph H128", "B_no_parent_qv_trans", 0.7235107422, 0.1365489668),
        ("Scene-Play train-only graph H128", "P_no_parent_product_qv", 0.7205200195, 0.1149664565),
    ]
    for label, variant, auc, gap in scene_checks:
        row = find_scene_qv(paper_data, label, variant)
        prefix = f"{label} {variant}"
        assert_close(f"{prefix} AUC", row["auc"], auc)
        assert_close(f"{prefix} gap", row["gap"], gap)


def validate_advanced_rows(advanced_data: dict) -> None:
    checks = [
        ("puzzle-4x4-play-oraclerep-v0", 34.0, 97.3333, 73, 75, "singular-board"),
        ("puzzle-4x5-play-oraclerep-v0", 97.0, 100.0, 75, 75, "exact discrete"),
        ("puzzle-4x6-play-oraclerep-v0", 51.0, 92.0, 69, 75, "exact discrete"),
        ("humanoidmaze-medium-navigate-oraclerep-v0", 57.0, 94.6667, 71, 75, "official 2000-step"),
        ("humanoidmaze-large-navigate-oraclerep-v0", 8.0, 89.3333, 67, 75, "official 2000-step"),
        ("humanoidmaze-giant-navigate-oraclerep-v0", 79.0, 80.0, 60, 75, "calibrated"),
        ("scene-play-oraclerep-v0", 77.0, 88.0, 66, 75, "matched support-path-only 63/75"),
        ("cube-single-play-oraclerep-v0", 95.0, 98.6667, 74, 75, "Standard Table-2"),
        ("cube-double-play-oraclerep-v0", 30.0, 73.3333, 55, 75, "dynamic block"),
        ("antsoccer-arena-navigate-oraclerep-v0", 73.0, 77.3333, 58, 75, "task-5-specific"),
    ]
    for env, paper, success, successes, episodes, caveat in checks:
        row = find_advanced(advanced_data, env)
        assert_close(f"{env} paper target", row["paper_overall"], paper)
        assert_close(f"{env} success", row["success"], success, tol=1e-3)
        assert_ge(f"{env} delta vs paper", row["delta_vs_paper"], 0.0)
        if int(row["successes"]) != successes or int(row["episodes"]) != episodes:
            raise AssertionError(
                f"{env}: expected {successes}/{episodes}, got "
                f"{row['successes']}/{row['episodes']}"
            )
        if caveat not in row["caveat"]:
            raise AssertionError(f"{env}: caveat missing {caveat!r}")

    giant = find_advanced(advanced_data, "humanoidmaze-giant-navigate-oraclerep-v0")
    heldout = giant.get("heldout", [])
    heldout_successes = [(int(item["successes"]), int(item["episodes"])) for item in heldout]
    if heldout_successes != [(13, 15), (12, 15), (12, 15), (12, 15)]:
        raise AssertionError(f"Unexpected giant heldout smokes: {heldout_successes}")

    scene_support = scene_graph_success(
        "exp/scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json"
    )
    if scene_support != (63, 75, 83.99999737739563):
        raise AssertionError(f"Unexpected Scene-Play support-only control: {scene_support}")


def validate_coverage_audit(coverage_data: dict) -> None:
    summary = coverage_data["summary"]
    expected_summary = {
        "promoted_rows": 10,
        "promoted_overall_beat_or_match": 10,
        "promoted_rows_with_paper_per_task": 9,
        "promoted_rows_all_per_tasks_beat_or_match": 6,
        "non_promoted_rows": 1,
        "non_promoted_overall_beat_or_match": 0,
    }
    for key, expected in expected_summary.items():
        actual = int(summary[key])
        if actual != expected:
            raise AssertionError(f"coverage audit {key}: expected {expected}, got {actual}")

    gap_rows = {
        row["env"]: row["per_task_gaps"]
        for row in coverage_data["promoted"]
        if row["paper_per_task_available"] and row["per_task_gaps"]
    }
    expected_gap_tasks = {
        "humanoidmaze-giant-navigate-oraclerep-v0": [1, 2, 4],
        "scene-play-oraclerep-v0": [2],
        "antsoccer-arena-navigate-oraclerep-v0": [2, 3],
    }
    if set(gap_rows) != set(expected_gap_tasks):
        raise AssertionError(f"Unexpected per-task gap rows: {sorted(gap_rows)}")
    for env, tasks in expected_gap_tasks.items():
        actual_tasks = [int(gap["task"]) for gap in gap_rows[env]]
        if actual_tasks != tasks:
            raise AssertionError(f"{env}: expected gap tasks {tasks}, got {actual_tasks}")

    antsoccer = coverage_data["non_promoted"][0]
    if antsoccer["env"] != "antsoccer-arena-navigate-oraclerep-v0":
        raise AssertionError(f"Unexpected non-promoted row {antsoccer['env']}")
    assert_close("AntSoccer clean single-protocol success", antsoccer["success"], 69.3333333333)
    assert_close("AntSoccer full confirmation paper target", antsoccer["paper_overall"], 73.0)
    if antsoccer["overall_beats_or_matches"]:
        raise AssertionError("AntSoccer clean single-protocol control should remain below the paper target")


def validate_antsoccer_audit(antsoccer_data: dict) -> None:
    expected = {
        "best clean single protocol": (52, 75, 69.3333333333),
        "best task-routed support-only artifact": (58, 75, 77.3333333333),
        "best full BMM single artifact": (44, 75, 58.6666666667),
        "best BMM-including routed suite": (55, 75, 73.3333333333),
    }
    found = {row["label"]: row for row in antsoccer_data["canonical"]}
    if set(found) != set(expected):
        raise AssertionError(f"Unexpected AntSoccer canonical rows: {sorted(found)}")
    for label, (successes, episodes, success) in expected.items():
        row = found[label]
        if int(row["successes"]) != successes or int(row["episodes"]) != episodes:
            raise AssertionError(
                f"{label}: expected {successes}/{episodes}, got "
                f"{row['successes']}/{row['episodes']}"
            )
        assert_close(f"{label} success", row["success"], success, tol=1e-3)
    top = antsoccer_data["full_artifacts"][0]
    if int(top["successes"]) != 58 or top["selector"] != "support_path_only":
        raise AssertionError(f"Unexpected best full AntSoccer artifact: {top}")
    if "No saved AntSoccer artifact contains a 66/75 result" not in antsoccer_data["conclusion"]:
        raise AssertionError("AntSoccer audit conclusion must explicitly rule out 66/75.")


def validate_generated_markdown() -> None:
    require_text(
        ADVANCED_MD,
        [
            "## Heldout Selector Smokes",
            "86.7% (13/15)",
            "80.0% (12/15)",
            "BMM is 66/75 versus matched support-path-only 63/75",
            "`antsoccer-arena-navigate-oraclerep-v0` | 73.0% | 77.3% (58/75)",
            "start_distance_deltay_gate_bmm_support",
        ],
    )
    require_text(
        COVERAGE_MD,
        [
            "promoted rows beating/matching paper overall | 10/10",
            "promoted rows beating/matching every available per-task entry | 6/9",
            "`antsoccer-arena-navigate-oraclerep-v0` | 73.0% | 77.3% (58/75)",
            "`antsoccer-arena-navigate-oraclerep-v0` | 73.0% | 69.3% (52/75)",
            "task 1: 53.3% vs 71.0%, task 2: 80.0% vs 87.0%, task 4: 93.3% vs 94.0%",
            "task 2: 93.3% vs 95.0%",
            "task 2: 66.7% vs 85.0%, task 3: 80.0% vs 89.0%",
        ],
    )
    require_text(
        ANTSOCCER_MD,
        [
            "No saved AntSoccer artifact contains a 66/75 result",
            "best clean single protocol | 69.3% (52/75)",
            "best task-routed support-only artifact | 77.3% (58/75)",
            "best full BMM single artifact | 58.7% (44/75)",
            "best BMM-including routed suite | 73.3% (55/75)",
        ],
    )


def validate_surface_text() -> None:
    require_text(
        REPRO_MD,
        [
            "conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_paper_claims.py",
            "scripts/summarize_bmm_paper_tables.py",
            "scripts/summarize_advanced_policy_table.py",
            "scripts/audit_bmm_paper_task_coverage.py",
            "scripts/audit_antsoccer_artifacts.py",
            "exp/scene_play_graph_gcfbc50k_bmm_left32_right128_ep15_seed10_detreset.json",
            "exp/scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json",
            "scripts/eval_puzzle_lightsout_policy.py",
            "scripts/eval_cube_sequential_policy.py",
            "exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json",
        ],
    )
    hard_rows = [
        "puzzle-4x4-play-oraclerep | 34.0% | 97.3% (73/75)",
        "puzzle-4x5-play-oraclerep | 97.0% | 100.0% (75/75)",
            "puzzle-4x6-play-oraclerep | 51.0% | 92.0% (69/75)",
            "humanoidmaze-medium-navigate-oraclerep | 57.0% | 94.7% (71/75)",
            "humanoidmaze-large-navigate-oraclerep | 8.0% | 89.3% (67/75)",
            "humanoidmaze-giant-navigate-oraclerep | 79.0% | 80.0% (60/75)",
            "scene-play-oraclerep | 77.0% | 88.0% (66/75)",
            "matched support-path-only 63/75",
            "cube-single-play-oraclerep | 95.0% | 98.7% (74/75)",
            "cube-double-play-oraclerep | 30.0% | 73.3% (55/75)",
            "antsoccer-arena-navigate-oraclerep | 73.0% | 77.3% (58/75)",
            "heldout offset smokes reach 13/15, 12/15, 12/15, and 12/15",
            "calibrated BMM/support route selection",
            "dynamic sequential block subgoals",
            "task-routed support-graph subgoals",
            "exact discrete planner plus learned local controller",
        ]
    require_text(REPO_ROOT / "BMM_TRL_CONFERENCE_DRAFT.md", hard_rows)
    require_text(
        REPO_ROOT / "BMM_TRL_PAPER_CLAIM_PACKAGE.md",
        [
            "Hard paper-listed rows from `exp/bmm_advanced_policy_table.md`",
            "The current task-coverage audit is `exp/bmm_paper_task_coverage_audit.md`",
            "all 10 promoted rows beat or match the paper on overall success",
            "6 of 9 rows with paper per-task references beat or match every individual task entry",
            "AntSoccer beats the paper overall only as a task-routed support-only policy suite",
            "The AntSoccer artifact audit is `exp/antsoccer_arena_artifact_audit.md`",
            "BMM_TRL_REPRO_COMMANDS.md",
            "conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_paper_claims.py",
            "the best clean single-protocol AntSoccer result is 69.3% versus the paper's 73.0% overall row",
            "puzzle-4x4-play-oraclerep | 34.0% | 97.3% (73/75)",
            "humanoidmaze-large-navigate-oraclerep | 8.0% | 89.3% (67/75)",
            "cube-single-play-oraclerep | 95.0% | 98.7% (74/75)",
            "cube-double-play-oraclerep | 30.0% | 73.3% (55/75)",
            "antsoccer-arena-navigate-oraclerep | 73.0% | 77.3% (58/75)",
            "calibrated BMM/support route selection",
            "exact Lights Out planner",
            "dynamic sequential block subgoals",
            "heldout offset smokes 13/15, 12/15, 12/15, and 12/15",
            "HumanoidMaze-giant should be reported as a calibrated hard-row success",
            "matched support-path-only 63/75",
        ],
    )
    require_text(
        REPO_ROOT / "paper" / "bmm_trl" / "main.tex",
        [
            "puzzle-4x4-play-oraclerep & 34.0% & 97.3% (73/75)",
            "puzzle-4x5-play-oraclerep & 97.0% & 100.0% (75/75)",
            "puzzle-4x6-play-oraclerep & 51.0% & 92.0% (69/75)",
            "humanoidmaze-medium-navigate-oraclerep & 57.0% & 94.7% (71/75)",
            "humanoidmaze-large-navigate-oraclerep & 8.0% & 89.3% (67/75)",
            "humanoidmaze-giant-navigate-oraclerep & 79.0% & 80.0% (60/75)",
            "matched support-path-only reaches 63/75",
            "cube-single-play-oraclerep & 95.0% & 98.7% (74/75)",
            "cube-double-play-oraclerep & 30.0% & 73.3% (55/75)",
            "antsoccer-arena-navigate-oraclerep & 73.0% & 77.3% (58/75)",
            "BMM_TRL_REPRO_COMMANDS.md",
            "conda run --no-capture-output -n bmm-trl python",
            "scripts/validate_bmm_paper_claims.py",
            "heldout offset smokes reach 13/15, 12/15, 12/15, and 12/15",
            "calibrated BMM/support route selection",
            "dynamic sequential block subgoals",
            "task-routed support-graph subgoals",
            "not a full multi-seed OGBench benchmark",
        ],
    )


def main() -> None:
    refresh_summaries()
    paper_data = load_json(PAPER_JSON)
    advanced_data = load_json(ADVANCED_JSON)
    coverage_data = load_json(COVERAGE_JSON)
    antsoccer_data = load_json(ANTSOCCER_JSON)
    validate_value_tables(paper_data)
    validate_advanced_rows(advanced_data)
    validate_coverage_audit(coverage_data)
    validate_antsoccer_audit(antsoccer_data)
    validate_generated_markdown()
    validate_surface_text()
    latex_static.validate()
    print("PASS: BMM-TRL paper headline claims match artifact-backed summaries.")
    print(f"  refreshed {PAPER_MD.relative_to(REPO_ROOT)}")
    print(f"  refreshed {ADVANCED_MD.relative_to(REPO_ROOT)}")
    print(f"  refreshed {COVERAGE_MD.relative_to(REPO_ROOT)}")
    print(f"  refreshed {ANTSOCCER_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
