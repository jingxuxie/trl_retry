#!/usr/bin/env python
"""Consolidate BMM paper-focused artifacts into compact markdown tables."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_bmm_budget_holdout as holdout_summary


DEFAULT_HOLDOUTS = (
    (
        "grid-cell H8",
        8,
        REPO_ROOT
        / "exp"
        / "bmm_qv_budget_holdout_20260611_005700"
        / "aggregate_h8_all_seeds.json",
    ),
    (
        "env-step H160",
        160,
        REPO_ROOT
        / "exp"
        / "bmm_qv_budget_holdout_20260611_012102"
        / "aggregate_h160_all_seeds.json",
    ),
    (
        "support-graph H120",
        120,
        REPO_ROOT
        / "exp"
        / "bmm_graph_qv_budget_holdout_paper_h120"
        / "aggregate_h120_all_seeds.json",
    ),
    (
        "product ablation grid-cell H8",
        8,
        REPO_ROOT
        / "exp"
        / "bmm_product_vs_maxmin_grid_h8_seeds12"
        / "aggregate_h8_all_product_seeds.json",
    ),
    (
        "product ablation env-step H160",
        160,
        REPO_ROOT
        / "exp"
        / "bmm_product_vs_maxmin_env_h160_seeds12"
        / "aggregate_h160_all_product_seeds.json",
    ),
)

DEFAULT_SCENE_QV = (
    (
        "Scene-Play train+val graph H128",
        128,
        (
            REPO_ROOT
            / "exp"
            / "bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed0_smoke"
            / "summary.json",
            REPO_ROOT
            / "exp"
            / "bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed1_smoke"
            / "summary.json",
        ),
    ),
    (
        "Scene-Play train-only graph H128",
        128,
        (
            REPO_ROOT
            / "exp"
            / "bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed0_smoke"
            / "summary.json",
        ),
    ),
)

DEFAULT_POLICIES = (
    (
        "navigate fixed BC, support-gated budget scan",
        REPO_ROOT
        / "exp"
        / "policy_retry_fixed_bc_geom_bmm_support075_vs_oracle_gate_commit10_replan20_step300_ep5.json",
    ),
    (
        "stitch fixed BC, support-gated budget scan",
        REPO_ROOT
        / "exp"
        / "policy_retry_stitch_fixed_bc_geom_bmm_support_vs_oracle_gate_step300_ep3.json",
    ),
    (
        "large navigate fixed BC, support-frontier",
        REPO_ROOT
        / "exp"
        / "policy_retry_large_nav_support_frontier_step500_ep3.json",
    ),
    (
        "large stitch fixed BC, local-progress value-frontier",
        REPO_ROOT
        / "exp"
        / "policy_retry_large_stitch_value_frontier_localprogress_step500_ep3.json",
    ),
    (
        "large stitch fixed BC, local-progress controls",
        REPO_ROOT
        / "exp"
        / "policy_retry_large_stitch_controls_localprogress_step500_ep3.json",
    ),
    (
        "large navigate oraclerep fixed BC ep5, support-path vs support-only",
        REPO_ROOT
        / "exp"
        / "policy_retry_large_nav_oraclerep_geom_bmm_support_only_grid_step1000_ep5.json",
    ),
    (
        "antmaze medium fixed BC, support-path",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_medium_nav_support_path_bc_h512_ln_switch80_step1000_tasks1_5_ep3.json",
    ),
    (
        "antmaze medium fixed BC, support-only control",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_medium_nav_support_only_vs_bmm_bc_h512_ln_switch80_step1000_tasks1_5_ep3.json",
    ),
    (
        "antmaze large oraclerep oracle-goal BC ep5, geometric",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.json",
    ),
    (
        "antmaze large oraclerep oracle-goal BC ep5, BMM vs support-only right480",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.json",
    ),
    (
        "antmaze large oraclerep oracle-goal BC20k ep3, geometric",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.json",
    ),
    (
        "antmaze large oraclerep oracle-goal BC20k ep3, BMM vs support-only",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.json",
    ),
    (
        "antmaze large oraclerep BC20k ep3, delayed learned repair",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_tasks1_5_supportonly_primary_rightprogress_fallback_pat500_switch40_cand64_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json",
    ),
    (
        "antmaze large oraclerep BC20k ep3, support-only switch40 control",
        REPO_ROOT
        / "exp"
        / "policy_retry_antmaze_large_oraclerep_tasks1_5_support_only_center_stratified64_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json",
    ),
    (
        "fixed BC, adaptive budget scan commit-1",
        REPO_ROOT
        / "exp"
        / "policy_retry_fixed_bc_geom_vs_bmm_budget_scan_80_160_step300_ep3.json",
    ),
    (
        "fixed BC, BMM-only commit-10 fast smoke",
        REPO_ROOT
        / "exp"
        / "policy_retry_bmm_budget_scan_commit10_replan20_step300_ep1.json",
    ),
)

DEFAULT_SCENE_POLICIES = (
    (
        "Scene-Play graph BC 10k, left32/right128",
        REPO_ROOT
        / "exp"
        / "scene_play_graph_bc_h512_ln_steps10000_ep1_step750_smoke.json",
    ),
    (
        "Scene-Play graph BC 10k, left64/right128",
        REPO_ROOT
        / "exp"
        / "scene_play_graph_bc_h512_ln_steps10000_left64_ep1_step750_smoke.json",
    ),
    (
        "Scene-Play local GCFBC 50k smoke, left32/right128",
        REPO_ROOT
        / "exp"
        / "scene_play_graph_gcfbc50k_direct_support_bmm_left32_right128_ep3_seed10_detreset.json",
    ),
    (
        "Scene-Play local GCFBC 50k support-only, left32/right128",
        REPO_ROOT
        / "exp"
        / "scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json",
    ),
    (
        "Scene-Play local GCFBC 50k BMM, left32/right128",
        REPO_ROOT
        / "exp"
        / "scene_play_graph_gcfbc50k_bmm_left32_right128_ep15_seed10_detreset.json",
    ),
)


def parse_holdout(value):
    parts = str(value).split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--holdout must have form label:budget:path, "
            f"got {value!r}."
        )
    label, budget, path = parts
    return label, int(budget), Path(path)


def parse_policy(value):
    parts = str(value).split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"--policy must have form label:path, got {value!r}.")
    label, path = parts
    return label, Path(path)


def display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def fmt(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def existing_default_holdouts():
    return [item for item in DEFAULT_HOLDOUTS if Path(item[2]).exists()]


def existing_default_policies():
    return [item for item in DEFAULT_POLICIES if Path(item[1]).exists()]


def existing_default_scene_qv():
    out = []
    for label, budget, paths in DEFAULT_SCENE_QV:
        existing = tuple(Path(path) for path in paths if Path(path).exists())
        if existing:
            out.append((label, budget, existing))
    return out


def existing_default_scene_policies():
    return [item for item in DEFAULT_SCENE_POLICIES if Path(item[1]).exists()]


def aggregate_holdout(label, budget, path, comparisons):
    path = Path(path)
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "aggregate" in data and "per_seed" in data:
            return {
                "label": label,
                "budget": int(data.get("budget", budget)),
                "path": display_path(path),
                "aggregate": data["aggregate"],
                "per_seed": data["per_seed"],
            }
    rows = holdout_summary.collect_rows([Path(path)], budget=budget)
    aggregate, per_seed = holdout_summary.aggregate_comparisons(
        rows, holdout_summary.parse_comparisons(comparisons)
    )
    return {
        "label": label,
        "budget": int(budget),
        "path": display_path(path),
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def per_task_summary(episodes):
    task_ids = sorted({int(row.get("task_id", row.get("task"))) for row in episodes})
    out = []
    for task_id in task_ids:
        rows = [
            row
            for row in episodes
            if int(row.get("task_id", row.get("task"))) == task_id
        ]
        out.append(
            {
                "task_id": task_id,
                "success": sum(float(row["success"]) for row in rows) / len(rows),
                "final_goal_distance": sum(
                    float(row["final_goal_distance"]) for row in rows
                )
                / len(rows),
            }
        )
    return out


def load_policy_result(label, path):
    path = Path(path)
    data = json.loads(path.read_text())
    rows = []
    per_task = []
    for selector in data.get("selectors", []):
        aggregate = selector["aggregate"]
        rows.append(
            {
                "label": label,
                "selector": selector["name"],
                "success": aggregate["success"],
                "final_goal_distance": aggregate["final_goal_distance"],
                "final_goal_xy_distance": aggregate["final_goal_xy_distance"],
                "goal_distance_improvement": aggregate["goal_distance_improvement"],
                "subgoal_valid_frac": aggregate["subgoal_valid_frac"],
                "steps": aggregate["steps"],
            }
        )
        per_task.append(
            {
                "label": label,
                "selector": selector["name"],
                "tasks": per_task_summary(selector.get("episodes", [])),
            }
        )
    return {
        "label": label,
        "path": display_path(path),
        "env_name": data.get("env_name"),
        "episodes_per_task": data.get("episodes_per_task"),
        "max_steps": data.get("max_steps"),
        "subgoal_commit_steps": data.get("subgoal_commit_steps"),
        "subgoal_replan_distance": data.get("subgoal_replan_distance"),
        "rows": rows,
        "per_task": per_task,
    }


def mean_metric(rows, key):
    vals = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else float("nan")


def infer_controller_steps(data):
    bc_steps = data.get("bc_info", {}).get("bc_steps")
    if bc_steps is not None:
        return bc_steps
    restore_path = str(data.get("controller_agent_restore_path") or "")
    matches = re.findall(r"(?:local|continue)(\d+)k", restore_path.lower())
    if matches:
        return int(matches[-1]) * 1000
    return data.get("controller_agent_restore_epoch")


def aggregate_scene_qv(label, budget, paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows.extend(
            row
            for row in data
            if int(row.get("budget", -1)) == int(budget)
        )
    variants = []
    for preferred in (
        "A_no_parent_no_trans",
        "B_no_parent_qv_trans",
        "P_no_parent_product_qv",
        "F_no_parent_vnext_distill",
    ):
        if any(row.get("variant") == preferred for row in rows):
            variants.append(preferred)
    for row in rows:
        variant = row.get("variant")
        if variant not in variants:
            variants.append(variant)
    out_rows = []
    for variant in variants:
        vrows = [row for row in rows if row.get("variant") == variant]
        if not vrows:
            continue
        seeds = sorted({int(row.get("seed", -1)) for row in vrows})
        out_rows.append(
            {
                "variant": variant,
                "seeds": ",".join(str(seed) for seed in seeds),
                "num_seeds": len(seeds),
                "auc": mean_metric(vrows, "auc"),
                "gap": mean_metric(vrows, "gap"),
                "ensemble_min_auc": mean_metric(vrows, "ensemble_min_auc"),
                "ensemble_min_gap": mean_metric(vrows, "ensemble_min_gap"),
            }
        )
    return {
        "label": label,
        "budget": int(budget),
        "paths": [display_path(path) for path in paths],
        "rows": out_rows,
    }


def load_scene_policy_result(label, path):
    path = Path(path)
    data = json.loads(path.read_text())
    rows = []
    for selector in data.get("selectors", []):
        aggregate = selector["aggregate"]
        rows.append(
            {
                "label": label,
                "selector": selector["name"],
                "success": aggregate.get("success"),
                "final_graph_d": aggregate.get("final_graph_d"),
                "graph_improve": aggregate.get("graph_improve"),
                "final_rep_d": aggregate.get("final_rep_d"),
                "rep_improve": aggregate.get("rep_improve"),
                "subgoal_valid": aggregate.get("subgoal_valid"),
                "selected_bmm_score": aggregate.get("selected_bmm_score"),
            }
        )
    return {
        "label": label,
        "path": display_path(path),
        "env_name": data.get("env_name"),
        "episodes_per_task": data.get("episodes_per_task"),
        "max_steps": data.get("max_steps"),
        "left_budget": data.get("left_budget"),
        "right_budget": data.get("right_budget"),
        "bc_steps": infer_controller_steps(data),
        "rows": rows,
    }


def tabular_section(tabular_json):
    path = Path(tabular_json)
    if not path.exists():
        return ["## Tabular error scaling", "", f"Missing: `{path}`", ""]
    report = json.loads(path.read_text())
    rows = report.get("error_scaling", [])
    if not rows:
        return ["## Tabular error scaling", "", f"No rows in `{path}`.", ""]
    first = rows[0]
    last = rows[-1]
    return [
        "## Tabular error scaling",
        "",
        "| H | BMM balanced sup error | additive/product sup error |",
        "|---:|---:|---:|",
        (
            f"| {first['horizon']} | {fmt(first['bmm_balanced_sup_error'])} | "
            f"{fmt(first['additive_distance_sup_error'])} |"
        ),
        (
            f"| {last['horizon']} | {fmt(last['bmm_balanced_sup_error'])} | "
            f"{fmt(last['additive_distance_sup_error'])} |"
        ),
        "",
        f"Source: `{display_path(path)}`",
        "",
    ]


def holdout_section(results):
    lines = [
        "## Budget-holdout reachability",
        "",
        "| setting | comparison | seeds | delta AUC | delta gap | delta BCE | delta ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result["aggregate"]:
            if row["num_seeds"] <= 0:
                continue
            lines.append(
                "| {label} | {comparison} | {seeds} | {auc} | {gap} | {bce} | {ece} |".format(
                    label=result["label"],
                    comparison=row["comparison"],
                    seeds=row["seeds"],
                    auc=fmt(row["mean_delta_auc"]),
                    gap=fmt(row["mean_delta_gap"]),
                    bce=fmt(row["mean_delta_bce"]),
                    ece=fmt(row["mean_delta_ece"]),
                )
            )
    lines.append("")
    lines.append("Sources:")
    for result in results:
        lines.append(f"- {result['label']}: `{display_path(result['path'])}`")
    lines.append("")
    return lines


def scene_qv_section(results):
    if not results:
        return []
    lines = [
        "## Scene-Play non-maze Q/V transfer",
        "",
        "| setting | variant | seeds | H | AUC | gap | ensemble-min AUC | ensemble-min gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result["rows"]:
            lines.append(
                "| {setting} | {variant} | {seeds} | {budget} | {auc} | {gap} | {min_auc} | {min_gap} |".format(
                    setting=result["label"],
                    variant=row["variant"],
                    seeds=row["seeds"],
                    budget=result["budget"],
                    auc=fmt(row["auc"]),
                    gap=fmt(row["gap"]),
                    min_auc=fmt(row["ensemble_min_auc"]),
                    min_gap=fmt(row["ensemble_min_gap"]),
                )
            )
    lines.extend(["", "Sources:"])
    for result in results:
        lines.append(f"- {result['label']}:")
        for path in result["paths"]:
            lines.append(f"  - `{path}`")
    lines.extend(
        [
            "",
            "Scene-Play caveat: these are value/Q diagnostics, not policy-control results.",
            "",
        ]
    )
    return lines


def policy_section(results):
    lines = [
        "## Fixed-controller policy smoke",
        "",
        "| setting | selector | episodes/task | max steps | commit | success | final_d | final_xy | improve | subgoal_valid | steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result["rows"]:
            lines.append(
                "| {setting} | {selector} | {episodes} | {max_steps} | {commit} | {success} | {final_d} | {final_xy} | {improve} | {valid} | {steps} |".format(
                    setting=result["label"],
                    selector=row["selector"],
                    episodes=result["episodes_per_task"],
                    max_steps=result["max_steps"],
                    commit=result["subgoal_commit_steps"],
                    success=fmt(row["success"]),
                    final_d=fmt(row["final_goal_distance"]),
                    final_xy=fmt(row["final_goal_xy_distance"]),
                    improve=fmt(row["goal_distance_improvement"]),
                    valid=fmt(row["subgoal_valid_frac"]),
                    steps=fmt(row["steps"], digits=1),
                )
            )
    lines.extend(["", "Sources:"])
    for result in results:
        lines.append(f"- {result['label']}: `{display_path(result['path'])}`")
    lines.extend(
        [
            "",
            "Policy caveat: this uses a fixed local goal-conditioned BC controller.",
            "The policy-facing claim is about BMM subgoal selection under the same controller, not end-to-end actor extraction.",
            "",
        ]
    )
    return lines


def scene_policy_section(results):
    if not results:
        return []
    lines = [
        "## Scene-Play policy controls",
        "",
        "| setting | selector | bc steps | episodes/task | max steps | left | right | success | final graph d | graph improve | final rep d | rep improve | subgoal valid | BMM score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result["rows"]:
            lines.append(
                "| {setting} | {selector} | {bc_steps} | {episodes} | {max_steps} | {left} | {right} | {success} | {final_g} | {improve_g} | {final_r} | {improve_r} | {valid} | {bmm} |".format(
                    setting=result["label"],
                    selector=row["selector"],
                    bc_steps=result["bc_steps"],
                    episodes=result["episodes_per_task"],
                    max_steps=result["max_steps"],
                    left=result["left_budget"],
                    right=result["right_budget"],
                    success=fmt(row["success"]),
                    final_g=fmt(row["final_graph_d"]),
                    improve_g=fmt(row["graph_improve"]),
                    final_r=fmt(row["final_rep_d"]),
                    improve_r=fmt(row["rep_improve"]),
                    valid=fmt(row["subgoal_valid"]),
                    bmm=fmt(row["selected_bmm_score"]),
                )
            )
    lines.extend(["", "Sources:"])
    for result in results:
        lines.append(f"- {result['label']}: `{display_path(result['path'])}`")
    lines.extend(
        [
            "",
            "Scene-Play policy caveat: the 10k oracle-goal BC rows are negative boundary smokes. With the stronger 50k local GCFBC controller, direct local control is 46.7% in the 15-rollout smoke, matched support-path-only is 63/75 (84.0%), and BMM support-path is 66/75 (88.0%) over 75 rollouts. This is fixed-controller graph-subgoal evidence, not direct actor extraction.",
            "",
        ]
    )
    return lines


def policy_limitations_section():
    return [
        "## Remaining policy limitations",
        "",
        "| interface | current signal | paper use |",
        "|---|---|---|",
        "| flat Q/RPG/FRS actor extraction | did not recover the hierarchical BC result | limitation / future work |",
        "| fixed local BC controller | enables a clean subgoal-selection comparison | main policy smoke interface |",
        "| value-only BMM subgoal selector on stitch | 53.3% success vs 20.0% geometric, with lower final distance | deployable signal but incomplete |",
        "| dataset-support-gated adaptive BMM budget-scan selector | 96.0% success / zero final distance on navigate and 100.0% success on stitch | current non-oracle policy positive |",
        "| large-stitch local-progress value-frontier selector | 100.0% success / zero final distance over 15 rollouts, versus 0.0% geometric and 20.0% support-path-only in matched 15-rollout controls | positive long-horizon OGBench stitch signal, with explicit no-backtracking extraction caveat |",
        "| support-frontier/path BMM selector | 100.0% success / zero final distance on large navigate, the paper-listed oraclerep large navigate 25-rollout validation, and AntMaze-medium with a stronger fixed BC controller; large-oraclerep support-only reaches 100.0%, AntMaze-medium support-path-only reaches 93.3%, and large-AntMaze oraclerep ep5 reaches 48.0% vs 52.0% support-only under corrected right horizon 480; with a 20k oracle-goal BC controller, large-AntMaze ep3 reaches 66.7% BMM vs 80.0% support-only under the older switch80 protocol; the current best fixed-reset switch40 protocol reaches 86.7% (13/15) with support-path primary plus delayed learned BMM right-progress repair versus 80.0% support-path-only; focused BMM-primary delayed repair still reaches only 1/3 on task 5 | current non-oracle long-horizon policy positive with support-only caveat; large-AntMaze is good enough to freeze as a learned-repair milestone, but the claim is support-path planning plus delayed BMM repair rather than robust pure BMM-primary extraction |",
        "| Scene-Play graph-subgoal BC | BMM support-path reaches 66/75 versus matched support-path-only 63/75 and direct local GCFBC 46.7% in the 15-rollout smoke; task 5 remains weak | positive fixed-controller graph-subgoal evidence with modest BMM margin; not direct actor extraction |",
        "| grid-geodesic oracle-gated selector | matched or upper-bounded support gating in the smokes | oracle comparator only |",
        "| full left/right gate on large navigate | fails when remaining distance exceeds the largest reliable right-budget classifier | motivates support-frontier interface |",
        "",
    ]


def build_report(args):
    holdouts = (
        [parse_holdout(item) for item in args.holdout]
        if args.holdout
        else existing_default_holdouts()
    )
    results = [
        aggregate_holdout(label, budget, path, args.comparisons)
        for label, budget, path in holdouts
    ]
    policy_specs = (
        [parse_policy(item) for item in args.policy]
        if args.policy
        else existing_default_policies()
    )
    policy_results = [load_policy_result(label, path) for label, path in policy_specs]
    scene_qv_specs = existing_default_scene_qv()
    scene_qv_results = [
        aggregate_scene_qv(label, budget, paths)
        for label, budget, paths in scene_qv_specs
    ]
    scene_policy_specs = existing_default_scene_policies()
    scene_policy_results = [
        load_scene_policy_result(label, path)
        for label, path in scene_policy_specs
    ]
    lines = [
        "# BMM-TRL paper experiment tables",
        "",
        "Generated from local JSON artifacts.",
        "",
    ]
    lines.extend(tabular_section(args.tabular_json))
    if results:
        lines.extend(holdout_section(results))
    else:
        lines.extend(["## Budget-holdout reachability", "", "No holdout artifacts found.", ""])
    lines.extend(scene_qv_section(scene_qv_results))
    if policy_results:
        lines.extend(policy_section(policy_results))
    lines.extend(scene_policy_section(scene_policy_results))
    lines.extend(policy_limitations_section())
    return {
        "holdouts": results,
        "scene_qv": scene_qv_results,
        "policies": policy_results,
        "scene_policies": scene_policy_results,
        "markdown": "\n".join(lines),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular_json",
        default=str(REPO_ROOT / "exp" / "bmm_tabular_error_scaling.json"),
    )
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        help="Holdout artifact spec: label:budget:path. Defaults to known local runs.",
    )
    parser.add_argument("--comparisons", default="B-A,F-A,B-P,P-A")
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Policy artifact spec: label:path. Defaults to known local policy smokes.",
    )
    parser.add_argument("--output_markdown", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    print(report["markdown"])
    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report["markdown"])
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in report.items() if k != "markdown"}
        path.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
