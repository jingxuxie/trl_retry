#!/usr/bin/env python
"""Run and summarize BMM Q/V budget-holdout diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_bmm_geodesic_q.py"


def parse_int_list(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def lambda_slug(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def tuple_flag(values):
    return "(" + ", ".join(str(int(x)) for x in values) + ")"


def build_run_specs(args):
    eval_budgets = parse_int_list(args.eval_budgets)
    supervised_budgets = parse_int_list(args.supervised_budgets)
    full_supervised_budgets = eval_budgets
    specs = []
    for seed in parse_int_list(args.seeds):
        specs.extend(
            [
                dict(
                    variant="A_no_parent_no_trans",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=0,
                    lambda_qv_trans=0.0,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="B_no_parent_qv_trans",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=0,
                    lambda_qv_trans=args.qv_lambda,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="C_few_parent_no_trans",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=args.few_parent_pairs,
                    lambda_qv_trans=0.0,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="D_few_parent_qv_trans",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=args.few_parent_pairs,
                    lambda_qv_trans=args.qv_lambda,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="E_full_supervised_upper",
                    seed=seed,
                    supervised_budgets=full_supervised_budgets,
                    parent_label_pairs=0,
                    lambda_qv_trans=0.0,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="F_no_parent_vnext_distill",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=0,
                    lambda_qv_trans=0.0,
                    lambda_vnext_distill=args.vnext_lambda,
                    qv_branch_mode="learned_q_frozen_v",
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
                dict(
                    variant="G_no_parent_oracle_qv",
                    seed=seed,
                    supervised_budgets=supervised_budgets,
                    parent_label_pairs=0,
                    lambda_qv_trans=args.qv_lambda,
                    lambda_vnext_distill=0.0,
                    qv_branch_mode=args.oracle_branch_mode,
                    qv_trans_loss_type=args.qv_trans_loss_type,
                    vnext_distill_loss_type=args.vnext_distill_loss_type,
                ),
            ]
        )
    return specs


def variant_key(variant):
    return str(variant).split("_", 1)[0]


def filter_run_specs(specs, variants):
    requested = [part.strip() for part in str(variants).split(",") if part.strip()]
    if not requested:
        return specs
    requested_keys = {part.split("_", 1)[0] for part in requested}
    requested_names = set(requested)
    return [
        spec
        for spec in specs
        if variant_key(spec["variant"]) in requested_keys
        or spec["variant"] in requested_names
    ]


def run_id(spec):
    return (
        f"seed{spec['seed']}_{spec['variant']}_"
        f"qv{lambda_slug(spec['lambda_qv_trans'])}_"
        f"vnext{lambda_slug(spec['lambda_vnext_distill'])}_"
        f"parent{int(spec['parent_label_pairs'])}"
    )


def train_command(args, spec, output_json):
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        f"--env_name={args.env_name}",
        "--reachability_label_type=grid_geodesic",
        f"--geodesic_budget_unit={args.geodesic_budget_unit}",
        f"--budgets={tuple_flag(parse_int_list(args.budgets))}",
        f"--eval_budgets={tuple_flag(parse_int_list(args.eval_budgets))}",
        f"--supervised_budgets={tuple_flag(spec['supervised_budgets'])}",
        f"--trans_budgets={tuple_flag(parse_int_list(args.trans_budgets))}",
        f"--seed={spec['seed']}",
        f"--batch_size={args.batch_size}",
        f"--sup_pairs_per_budget={args.sup_pairs_per_budget}",
        f"--parent_label_pairs_per_budget={spec['parent_label_pairs']}",
        f"--trans_pairs_per_update={args.trans_pairs_per_update}",
        f"--eval_pairs={args.eval_pairs}",
        f"--steps={args.steps}",
        f"--eval_interval={args.eval_interval}",
        f"--lambda_qv_trans={spec['lambda_qv_trans']}",
        f"--qv_trans_loss_type={spec['qv_trans_loss_type']}",
        f"--qv_trans_bce_margin={args.qv_trans_bce_margin}",
        f"--lambda_vnext_distill={spec['lambda_vnext_distill']}",
        f"--vnext_distill_loss_type={spec['vnext_distill_loss_type']}",
        f"--vnext_distill_bce_margin={args.vnext_distill_bce_margin}",
        f"--qv_branch_mode={spec['qv_branch_mode']}",
        f"--num_trans_witnesses={args.num_trans_witnesses}",
        f"--trans_witness_mode={args.trans_witness_mode}",
        f"--agent.actor_hidden_dims={args.actor_hidden_dims}",
        f"--agent.value_hidden_dims={args.value_hidden_dims}",
        f"--agent.layer_norm={args.layer_norm}",
        f"--output_json={output_json}",
    ]
    if args.value_restore_path:
        cmd.extend(
            [
                f"--value_restore_path={args.value_restore_path}",
                f"--value_restore_epoch={args.value_restore_epoch}",
            ]
        )
    if args.fail_on_threshold:
        cmd.append("--fail_on_threshold")
    return cmd


def metric_value(metrics, key):
    if not metrics:
        return float("nan")
    return metrics.get(key, float("nan"))


def gap(metrics):
    return metric_value(metrics, "pos_mean") - metric_value(metrics, "neg_mean")


def by_budget(rows):
    return {int(row["budget"]): row for row in rows or []}


def summarize_report(report_path, spec, trans_budgets):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    qv_rows = by_budget((report.get("last_qv_summary") or {}).get("budget_rows"))
    eval_rows = report["eval"]["budget_rows"]
    consistency = report["eval"].get("q_v_next_consistency", {})
    vnext_rows = by_budget(consistency.get("v_next_budget_rows"))
    info = report.get("last_update_info", {})
    trans_budget_set = {int(x) for x in trans_budgets}
    rows = []
    for eval_row in eval_rows:
        budget = int(eval_row["budget"])
        mean = eval_row["mean"]
        ens_min = eval_row["ensemble_min"]
        qv_row = qv_rows.get(budget, {})
        vnext = vnext_rows.get(budget, {}).get("metrics", {})
        budget_key = f"H{budget}"
        rows.append(
            dict(
                run_id=run_id(spec),
                variant=str(spec["variant"]),
                report_path=str(report_path),
                seed=int(spec["seed"]),
                supervised_budgets=",".join(str(x) for x in spec["supervised_budgets"]),
                parent_label_pairs=int(spec["parent_label_pairs"]),
                lambda_qv_trans=float(spec["lambda_qv_trans"]),
                lambda_vnext_distill=float(spec["lambda_vnext_distill"]),
                qv_branch_mode=str(spec["qv_branch_mode"]),
                qv_trans_loss_type=str(spec["qv_trans_loss_type"]),
                vnext_distill_loss_type=str(spec["vnext_distill_loss_type"]),
                budget=budget,
                is_trans_parent_budget=budget in trans_budget_set,
                auc=metric_value(mean, "auc"),
                gap=gap(mean),
                bce=metric_value(mean, "bce"),
                ece=metric_value(mean, "ece"),
                accuracy=metric_value(mean, "accuracy"),
                pos_mean=metric_value(mean, "pos_mean"),
                neg_mean=metric_value(mean, "neg_mean"),
                pos_count=int(metric_value(mean, "pos_count")),
                neg_count=int(metric_value(mean, "neg_count")),
                ensemble_min_auc=metric_value(ens_min, "auc"),
                ensemble_min_gap=gap(ens_min),
                ensemble_min_bce=metric_value(ens_min, "bce"),
                ensemble_min_ece=metric_value(ens_min, "ece"),
                q_v_next_abs_diff=consistency.get(
                    "mean_abs_prob_diff", float("nan")
                ),
                q_v_next_rank_corr=consistency.get(
                    "rank_correlation", float("nan")
                ),
                v_next_auc=metric_value(vnext, "auc"),
                v_next_gap=gap(vnext),
                v_next_bce=metric_value(vnext, "bce"),
                v_next_ece=metric_value(vnext, "ece"),
                monotonicity=report.get("eval_monotonicity_violation", float("nan")),
                qv_effective_k=qv_row.get(
                    "effective_unique_witness_count_mean", float("nan")
                ),
                qv_replacement_frac=qv_row.get("replacement_used_frac", float("nan")),
                qv_y_trans_mean=info.get(
                    f"critic/qv_y_trans_mean_by_budget/{budget_key}", float("nan")
                ),
                qv_parent_r_mean=info.get(
                    f"critic/qv_parent_r_mean_by_budget/{budget_key}", float("nan")
                ),
                qv_target_minus_parent_mean=info.get(
                    f"critic/qv_target_minus_parent_mean_by_budget/{budget_key}",
                    float("nan"),
                ),
                qv_frac_target_gt_parent=info.get(
                    f"critic/qv_frac_y_trans_gt_parent_by_budget/{budget_key}",
                    float("nan"),
                ),
                qv_frac_target_lt_parent=info.get(
                    f"critic/qv_frac_y_trans_lt_parent_by_budget/{budget_key}",
                    float("nan"),
                ),
                loss_qv_trans_by_budget=info.get(
                    f"critic/loss_qv_trans_by_budget/{budget_key}", float("nan")
                ),
                loss_vnext_distill_by_budget=info.get(
                    f"critic/loss_vnext_distill_by_budget/{budget_key}", float("nan")
                ),
                vnext_target_minus_parent_mean=info.get(
                    f"critic/vnext_target_minus_parent_mean_by_budget/{budget_key}",
                    float("nan"),
                ),
                passed=bool(report.get("passed", False)),
            )
        )
    return rows


def write_summary(rows, csv_path, json_path):
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)


def default_run_dir():
    return REPO_ROOT / "exp" / f"bmm_qv_budget_holdout_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--geodesic_budget_unit", default="grid_cells")
    parser.add_argument("--budgets", default="2,4,8")
    parser.add_argument("--eval_budgets", default="2,4,8")
    parser.add_argument("--supervised_budgets", default="2,4")
    parser.add_argument("--trans_budgets", default="8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--variants",
        default="A,B,C,D,E,F,G",
        help=(
            "Comma-separated variant letters or names to run. "
            "Use A,B,C,D,F for the minimal budget-holdout replication."
        ),
    )
    parser.add_argument("--sup_pairs_per_budget", type=int, default=256)
    parser.add_argument("--few_parent_pairs", type=int, default=16)
    parser.add_argument("--qv_lambda", type=float, default=0.01)
    parser.add_argument("--vnext_lambda", type=float, default=0.01)
    parser.add_argument("--qv_trans_loss_type", default="bce_lower_bound")
    parser.add_argument("--qv_trans_bce_margin", default="0.0")
    parser.add_argument("--vnext_distill_loss_type", default="bce_lower_bound")
    parser.add_argument("--vnext_distill_bce_margin", default="0.0")
    parser.add_argument("--oracle_branch_mode", default="oracle_q_oracle_v")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--trans_pairs_per_update", type=int, default=256)
    parser.add_argument("--eval_pairs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--num_trans_witnesses", type=int, default=4)
    parser.add_argument("--trans_witness_mode", default="slack_balanced")
    parser.add_argument(
        "--value_restore_path",
        default="exp/bmm_grid_cells_value_teacher_2_4_8",
    )
    parser.add_argument("--value_restore_epoch", type=int, default=1000)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--fail_on_threshold", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_dir = Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = filter_run_specs(build_run_specs(args), args.variants)
    trans_budgets = parse_int_list(args.trans_budgets)
    all_rows = []
    for spec in specs:
        rid = run_id(spec)
        output_json = run_dir / f"{rid}.json"
        cmd = train_command(args, spec, output_json)
        print("\n==", rid, "==")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        if output_json.exists() and args.skip_existing:
            print(f"Skipping existing report: {output_json}")
        else:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        all_rows.extend(summarize_report(output_json, spec, trans_budgets))

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    if all_rows:
        write_summary(all_rows, summary_csv, summary_json)
        print(f"\nWrote budget-holdout summary to {summary_csv}")
        print(f"Wrote budget-holdout JSON to {summary_json}")
    else:
        print("\nDry run complete; no summary written.")


if __name__ == "__main__":
    main()
