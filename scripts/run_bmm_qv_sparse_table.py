#!/usr/bin/env python
"""Run and summarize sparse-Q BMM Q/V transitive diagnostics."""

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


def parse_float_list(value):
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def lambda_slug(value):
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def loss_slug(loss_type, lambda_qv_trans):
    if float(lambda_qv_trans) == 0.0:
        return "sup"
    return str(loss_type)


def build_run_specs(args):
    specs = []
    for seed in parse_int_list(args.seeds):
        for sup_pairs in parse_int_list(args.sup_pairs):
            specs.append(
                dict(
                    seed=seed,
                    sup_pairs_per_budget=sup_pairs,
                    lambda_qv_trans=0.0,
                    qv_trans_loss_type=args.qv_trans_loss_type,
                )
            )
        for sup_pairs in parse_int_list(args.sup_pairs):
            for lambda_qv in parse_float_list(args.qv_lambdas):
                specs.append(
                    dict(
                        seed=seed,
                        sup_pairs_per_budget=sup_pairs,
                        lambda_qv_trans=lambda_qv,
                        qv_trans_loss_type=args.qv_trans_loss_type,
                    )
                )
        for sup_pairs in parse_int_list(args.strong_sup_pairs):
            for lambda_qv in parse_float_list(args.strong_qv_lambdas):
                specs.append(
                    dict(
                        seed=seed,
                        sup_pairs_per_budget=sup_pairs,
                        lambda_qv_trans=lambda_qv,
                        qv_trans_loss_type=args.qv_trans_loss_type,
                    )
                )
        if args.include_prob_hinge_sanity:
            specs.append(
                dict(
                    seed=seed,
                    sup_pairs_per_budget=args.prob_hinge_sup_pairs,
                    lambda_qv_trans=args.prob_hinge_lambda,
                    qv_trans_loss_type="prob_hinge",
                )
            )
    return specs


def run_id(spec):
    return (
        f"seed{spec['seed']}_sup{spec['sup_pairs_per_budget']}_"
        f"lam{lambda_slug(spec['lambda_qv_trans'])}_"
        f"{loss_slug(spec['qv_trans_loss_type'], spec['lambda_qv_trans'])}"
    )


def train_command(args, spec, output_json):
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        f"--env_name={args.env_name}",
        "--reachability_label_type=grid_geodesic",
        f"--geodesic_budget_unit={args.geodesic_budget_unit}",
        f"--budgets={args.budgets_tuple}",
        f"--trans_budgets={args.trans_budgets_tuple}",
        f"--seed={spec['seed']}",
        f"--batch_size={args.batch_size}",
        f"--sup_pairs_per_budget={spec['sup_pairs_per_budget']}",
        f"--trans_pairs_per_update={args.trans_pairs_per_update}",
        f"--eval_pairs={args.eval_pairs}",
        f"--steps={args.steps}",
        f"--eval_interval={args.eval_interval}",
        f"--lambda_qv_trans={spec['lambda_qv_trans']}",
        f"--qv_trans_loss_type={spec['qv_trans_loss_type']}",
        f"--qv_trans_bce_margin={args.qv_trans_bce_margin}",
        f"--num_trans_witnesses={args.num_trans_witnesses}",
        f"--trans_witness_mode={args.trans_witness_mode}",
        f"--value_restore_path={args.value_restore_path}",
        f"--value_restore_epoch={args.value_restore_epoch}",
        f"--agent.actor_hidden_dims={args.actor_hidden_dims}",
        f"--agent.value_hidden_dims={args.value_hidden_dims}",
        f"--agent.layer_norm={args.layer_norm}",
        f"--output_json={output_json}",
    ]
    if args.fail_on_threshold:
        cmd.append("--fail_on_threshold")
    return cmd


def metric_value(metrics, key):
    if not metrics:
        return float("nan")
    return metrics.get(key, float("nan"))


def gap(metrics):
    pos = metric_value(metrics, "pos_mean")
    neg = metric_value(metrics, "neg_mean")
    return pos - neg


def by_budget(rows):
    return {int(row["budget"]): row for row in rows or []}


def summarize_report(report_path, spec):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    qv_summary_rows = by_budget((report.get("last_qv_summary") or {}).get("budget_rows"))
    eval_rows = report["eval"]["budget_rows"]
    consistency = report["eval"].get("q_v_next_consistency", {})
    v_next_rows = by_budget(consistency.get("v_next_budget_rows"))
    info = report.get("last_update_info", {})
    rows = []
    for eval_row in eval_rows:
        budget = int(eval_row["budget"])
        mean = eval_row["mean"]
        ens_min = eval_row["ensemble_min"]
        qv_row = qv_summary_rows.get(budget, {})
        v_next = v_next_rows.get(budget, {}).get("metrics", {})
        rows.append(
            dict(
                run_id=run_id(spec),
                report_path=str(report_path),
                seed=int(spec["seed"]),
                sup_pairs_per_budget=int(spec["sup_pairs_per_budget"]),
                lambda_qv_trans=float(spec["lambda_qv_trans"]),
                qv_trans_loss_type=str(spec["qv_trans_loss_type"]),
                budget=budget,
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
                v_next_auc=metric_value(v_next, "auc"),
                v_next_gap=gap(v_next),
                v_next_bce=metric_value(v_next, "bce"),
                v_next_ece=metric_value(v_next, "ece"),
                monotonicity=report.get("eval_monotonicity_violation", float("nan")),
                qv_effective_k=qv_row.get(
                    "effective_unique_witness_count_mean", float("nan")
                ),
                qv_replacement_frac=qv_row.get("replacement_used_frac", float("nan")),
                qv_frac_target_gt_parent=info.get(
                    "critic/qv_frac_y_trans_gt_parent", float("nan")
                ),
                qv_frac_target_lt_parent=info.get(
                    "critic/qv_frac_y_trans_lt_parent", float("nan")
                ),
                loss_qv_trans_by_budget=info.get(
                    f"critic/loss_qv_trans_by_budget/H{budget}", float("nan")
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
    return REPO_ROOT / "exp" / f"bmm_qv_sparse_table_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--geodesic_budget_unit", default="env_steps")
    parser.add_argument("--budgets", default="40,80,160")
    parser.add_argument("--trans_budgets", default="80,160")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--sup_pairs", default="256,64,32,16")
    parser.add_argument("--qv_lambdas", default="0.01")
    parser.add_argument("--strong_sup_pairs", default="64,32,16")
    parser.add_argument("--strong_qv_lambdas", default="0.025")
    parser.add_argument("--qv_trans_loss_type", default="bce_lower_bound")
    parser.add_argument("--qv_trans_bce_margin", default="0.0")
    parser.add_argument("--include_prob_hinge_sanity", action="store_true", default=True)
    parser.add_argument("--no_prob_hinge_sanity", dest="include_prob_hinge_sanity", action="store_false")
    parser.add_argument("--prob_hinge_sup_pairs", type=int, default=32)
    parser.add_argument("--prob_hinge_lambda", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--trans_pairs_per_update", type=int, default=256)
    parser.add_argument("--eval_pairs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--num_trans_witnesses", type=int, default=4)
    parser.add_argument("--trans_witness_mode", default="slack_balanced")
    parser.add_argument(
        "--value_restore_path",
        default="exp/bmm_grid_value_qv_teacher_40_80_160",
    )
    parser.add_argument("--value_restore_epoch", type=int, default=1000)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--fail_on_threshold", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    args.budgets_tuple = "(" + ", ".join(str(x) for x in parse_int_list(args.budgets)) + ")"
    args.trans_budgets_tuple = (
        "(" + ", ".join(str(x) for x in parse_int_list(args.trans_budgets)) + ")"
    )
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_dir = Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = build_run_specs(args)
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
        all_rows.extend(summarize_report(output_json, spec))

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    if all_rows:
        write_summary(all_rows, summary_csv, summary_json)
        print(f"\nWrote sparse table summary to {summary_csv}")
        print(f"Wrote sparse table JSON to {summary_json}")
    else:
        print("\nDry run complete; no summary written.")


if __name__ == "__main__":
    main()
