#!/usr/bin/env python
"""Synthetic checks for BMM Q/V budget-holdout runner summaries."""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bmm_reachability_utils import binary_metrics
from scripts import run_bmm_qv_budget_holdout as holdout


def make_metrics(pos, neg):
    scores = np.asarray([pos, neg], dtype=np.float64)
    labels = np.asarray([1.0, 0.0], dtype=np.float64)
    return binary_metrics(scores, labels)


def main():
    args = holdout.parse_args(["--dry_run"])
    specs = holdout.build_run_specs(args)
    assert len(specs) == 8
    assert [spec["variant"][0] for spec in specs] == list("ABPCDEFG")
    product_spec = specs[2]
    assert product_spec["variant"] == "P_no_parent_product_qv"
    assert product_spec["qv_trans_target_type"] == "product"
    assert any(spec["qv_branch_mode"] == "oracle_q_oracle_v" for spec in specs)
    assert any(spec["lambda_vnext_distill"] > 0.0 for spec in specs)
    filtered = holdout.filter_run_specs(specs, "A,B,C,D,F")
    assert [spec["variant"][0] for spec in filtered] == list("ABCDF")

    spec = specs[1]
    assert spec["variant"] == "B_no_parent_qv_trans"
    cmd = holdout.train_command(args, spec, Path("out.json"))
    assert "--supervised_budgets=(2, 4)" in cmd
    assert "--eval_budgets=(2, 4, 8)" in cmd
    assert "--trans_budgets=(8)" in cmd
    assert "--qv_trans_target_type=max_min" in cmd
    saved_cmd = holdout.train_command(args, spec, Path("out.json"), save_dir=Path("ckpt"))
    assert "--save_dir=ckpt" in saved_cmd
    assert "--save_epoch=1000" in saved_cmd

    report = {
        "eval": {
            "q_v_next_consistency": {
                "value_checkpoint_available": True,
                "mean_abs_prob_diff": 0.12,
                "rank_correlation": 0.9,
                "v_next_budget_rows": [
                    {"budget": 8, "metrics": make_metrics(0.82, 0.20)}
                ],
            },
            "budget_rows": [
                {
                    "budget": 8,
                    "mean": make_metrics(0.80, 0.25),
                    "ensemble_min": make_metrics(0.78, 0.30),
                }
            ],
        },
        "eval_monotonicity_violation": 0.01,
        "last_qv_summary": {
            "budget_rows": [
                {
                    "budget": 8,
                    "effective_unique_witness_count_mean": 3.5,
                    "replacement_used_frac": 0.0,
                }
            ]
        },
        "last_update_info": {
            "critic/qv_y_trans_mean_by_budget/H8": 0.72,
            "critic/qv_parent_r_mean_by_budget/H8": 0.50,
            "critic/qv_target_minus_parent_mean_by_budget/H8": 0.22,
            "critic/qv_frac_y_trans_gt_parent_by_budget/H8": 0.75,
            "critic/qv_frac_y_trans_lt_parent_by_budget/H8": 0.25,
            "critic/loss_qv_trans_by_budget/H8": 0.13,
            "critic/qv_min_candidate_mean_by_budget/H8": 0.70,
            "critic/qv_product_candidate_mean_by_budget/H8": 0.49,
            "critic/loss_vnext_distill_by_budget/H8": 0.17,
            "critic/vnext_target_minus_parent_mean_by_budget/H8": 0.08,
        },
        "passed": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.json"
        report_path.write_text(json.dumps(report))
        rows = holdout.summarize_report(report_path, spec, trans_budgets=(8,))
        assert len(rows) == 1
        row = rows[0]
        assert row["variant"] == "B_no_parent_qv_trans"
        assert row["budget"] == 8
        assert row["is_trans_parent_budget"] is True
        assert row["qv_effective_k"] == 3.5
        assert row["qv_trans_target_type"] == "max_min"
        assert row["qv_target_minus_parent_mean"] == 0.22
        assert row["qv_min_candidate_mean"] == 0.70
        assert row["qv_product_candidate_mean"] == 0.49
        assert row["vnext_target_minus_parent_mean"] == 0.08
        assert np.isfinite(row["ece"])
        holdout.write_summary(
            rows,
            Path(tmpdir) / "summary.csv",
            Path(tmpdir) / "summary.json",
        )

    print("BMM Q/V budget-holdout summary checks passed.")


if __name__ == "__main__":
    main()
