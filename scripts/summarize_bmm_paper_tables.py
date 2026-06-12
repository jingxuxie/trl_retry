#!/usr/bin/env python
"""Consolidate BMM paper-focused artifacts into compact markdown tables."""

import argparse
import json
import math
from pathlib import Path
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
        "product ablation H8",
        8,
        REPO_ROOT
        / "exp"
        / "bmm_product_vs_maxmin_grid_h8_seed0"
        / "aggregate_with_seed0_baselines.json",
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


def aggregate_holdout(label, budget, path, comparisons):
    path = Path(path)
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "aggregate" in data and "per_seed" in data:
            return {
                "label": label,
                "budget": int(data.get("budget", budget)),
                "path": str(path),
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
        "path": str(path),
        "aggregate": aggregate,
        "per_seed": per_seed,
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
        f"Source: `{path}`",
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
        lines.append(f"- {result['label']}: `{result['path']}`")
    lines.append("")
    return lines


def policy_limitations_section():
    return [
        "## Policy extraction limitations",
        "",
        "| interface | current signal | paper use |",
        "|---|---|---|",
        "| flat Q action ranking | BMM did not consistently beat A/F | limitation |",
        "| joint action-subgoal | candidate coverage helped, full objective remained mixed | limitation |",
        "| value-only subgoal + nearest-neighbor controller | positive tiny smoke | weak exploratory positive |",
        "| value-only subgoal + BC controller | BMM tied or lost to geometric midpoint | limitation |",
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
    lines.extend(policy_limitations_section())
    return {"holdouts": results, "markdown": "\n".join(lines)}


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
