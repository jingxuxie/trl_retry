#!/usr/bin/env python
"""Aggregate BMM Q/V budget-holdout runs into comparison deltas."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_bmm_qv_budget_holdout as holdout


RUN_RE = re.compile(
    r"^seed(?P<seed>\d+)_(?P<variant>.+)_qv(?P<qv>[^_]+)_"
    r"vnext(?P<vnext>[^_]+)_parent(?P<parent>\d+)\.json$"
)

VARIANT_NAMES = {
    "A": "A_no_parent_no_trans",
    "B": "B_no_parent_qv_trans",
    "C": "C_few_parent_no_trans",
    "D": "D_few_parent_qv_trans",
    "E": "E_full_supervised_upper",
    "F": "F_no_parent_vnext_distill",
    "G": "G_no_parent_oracle_qv",
}

COMPARISON_LABELS = {
    "B-A": "no-parent BMM effect",
    "D-C": "few-parent BMM effect",
    "F-A": "V-next distill control",
    "G-B": "oracle branch control",
}

DELTA_METRICS = (
    "auc",
    "gap",
    "bce",
    "ece",
    "ensemble_min_auc",
    "ensemble_min_gap",
    "ensemble_min_bce",
    "ensemble_min_ece",
    "q_v_next_abs_diff",
    "q_v_next_rank_corr",
)


def finite_values(values):
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def mean(values):
    values = finite_values(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def standard_error(values):
    values = finite_values(values)
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def fmt(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def variant_letter(variant):
    return str(variant).split("_", 1)[0]


def parse_comparisons(value):
    comparisons = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Comparison must have form LEFT-RIGHT: {item}")
        left, right = [part.strip() for part in item.split("-", 1)]
        comparisons.append((left, right, f"{left}-{right}"))
    return comparisons


def load_flat_rows(path, budget):
    path = Path(path)
    if path.is_dir():
        summary_path = path / "summary.json"
        if summary_path.exists():
            return load_flat_rows(summary_path, budget)
        rows = []
        for report_path in sorted(path.glob("seed*_*.json")):
            rows.extend(load_report_rows(report_path, budget))
        return rows
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if int(row.get("budget", -1)) == int(budget)]
    return load_report_rows(path, budget)


def spec_from_report_path(report_path, report):
    match = RUN_RE.match(Path(report_path).name)
    if not match:
        raise ValueError(f"Cannot infer run spec from report filename: {report_path}")
    config = report.get("config", {})
    return dict(
        variant=match.group("variant"),
        seed=int(match.group("seed")),
        supervised_budgets=tuple(int(x) for x in config.get("supervised_budgets", ())),
        parent_label_pairs=int(config.get("parent_label_pairs_per_budget", match.group("parent"))),
        lambda_qv_trans=float(config.get("lambda_qv_trans", 0.0)),
        lambda_vnext_distill=float(config.get("lambda_vnext_distill", 0.0)),
        qv_branch_mode=str(config.get("qv_branch_mode", "learned_q_frozen_v")),
        qv_trans_loss_type=str(config.get("qv_trans_loss_type", "bce_lower_bound")),
        vnext_distill_loss_type=str(config.get("vnext_distill_loss_type", "bce_lower_bound")),
    )


def load_report_rows(report_path, budget):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    spec = spec_from_report_path(report_path, report)
    trans_budgets = tuple(int(x) for x in report.get("config", {}).get("trans_budgets", ()))
    rows = holdout.summarize_report(report_path, spec, trans_budgets)
    return [row for row in rows if int(row.get("budget", -1)) == int(budget)]


def collect_rows(paths, budget):
    rows = []
    for path in paths:
        rows.extend(load_flat_rows(path, budget))
    return rows


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), variant_letter(row["variant"]))
        indexed[key] = row
    return indexed


def aggregate_comparisons(rows, comparisons):
    indexed = index_rows(rows)
    seeds = sorted({seed for seed, _ in indexed})
    per_seed = []
    aggregate = []
    for left, right, name in comparisons:
        deltas = {metric: [] for metric in DELTA_METRICS}
        used_seeds = []
        for seed in seeds:
            left_row = indexed.get((seed, left))
            right_row = indexed.get((seed, right))
            if left_row is None or right_row is None:
                continue
            used_seeds.append(seed)
            row = {
                "comparison": name,
                "seed": seed,
                "left_variant": left_row["variant"],
                "right_variant": right_row["variant"],
            }
            for metric in DELTA_METRICS:
                delta = float(left_row.get(metric, float("nan"))) - float(
                    right_row.get(metric, float("nan"))
                )
                row[f"delta_{metric}"] = delta
                deltas[metric].append(delta)
            per_seed.append(row)
        aggregate_row = {
            "comparison": name,
            "seeds": ",".join(str(seed) for seed in used_seeds),
            "num_seeds": len(used_seeds),
            "interpretation": COMPARISON_LABELS.get(name, ""),
        }
        for metric in DELTA_METRICS:
            aggregate_row[f"mean_delta_{metric}"] = mean(deltas[metric])
            aggregate_row[f"se_delta_{metric}"] = standard_error(deltas[metric])
        aggregate.append(aggregate_row)
    return aggregate, per_seed


def markdown_table(aggregate, per_seed, budget):
    lines = [
        f"# BMM budget-holdout aggregate, H{budget}",
        "",
        f"| comparison | seeds | delta H{budget} AUC | delta H{budget} gap | delta H{budget} BCE | delta H{budget} ECE | delta Q-V abs | interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            "| {comparison} | {seeds} | {auc} | {gap} | {bce} | {ece} | {qv_abs} | {interp} |".format(
                comparison=row["comparison"],
                seeds=row["seeds"] or "-",
                auc=fmt(row["mean_delta_auc"]),
                gap=fmt(row["mean_delta_gap"]),
                bce=fmt(row["mean_delta_bce"]),
                ece=fmt(row["mean_delta_ece"]),
                qv_abs=fmt(row["mean_delta_q_v_next_abs_diff"]),
                interp=row["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "Per-seed deltas:",
            "",
            "| comparison | seed | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs | delta min AUC | delta min gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        lines.append(
            "| {comparison} | {seed} | {auc} | {gap} | {bce} | {ece} | {qv_abs} | {min_auc} | {min_gap} |".format(
                comparison=row["comparison"],
                seed=row["seed"],
                auc=fmt(row["delta_auc"]),
                gap=fmt(row["delta_gap"]),
                bce=fmt(row["delta_bce"]),
                ece=fmt(row["delta_ece"]),
                qv_abs=fmt(row["delta_q_v_next_abs_diff"]),
                min_auc=fmt(row["delta_ensemble_min_auc"]),
                min_gap=fmt(row["delta_ensemble_min_gap"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run dirs, summary JSON files, or per-run JSON files.")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--comparisons", default="B-A,D-C,F-A,G-B")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_markdown", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = collect_rows(args.paths, args.budget)
    comparisons = parse_comparisons(args.comparisons)
    aggregate, per_seed = aggregate_comparisons(rows, comparisons)
    result = {"budget": args.budget, "aggregate": aggregate, "per_seed": per_seed}
    text = markdown_table(aggregate, per_seed, args.budget)
    print(text)
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    if args.output_csv is not None:
        write_csv(aggregate, args.output_csv)
    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
