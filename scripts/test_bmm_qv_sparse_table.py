#!/usr/bin/env python
"""Synthetic checks for sparse-Q table summarization."""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bmm_reachability_utils import binary_metrics
from scripts import run_bmm_qv_sparse_table as sparse


def make_metrics(pos, neg):
    scores = np.asarray([pos, neg], dtype=np.float64)
    labels = np.asarray([1.0, 0.0], dtype=np.float64)
    return binary_metrics(scores, labels)


def main():
    metrics = binary_metrics(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    assert "ece" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0

    args = sparse.parse_args(["--dry_run"])
    specs = sparse.build_run_specs(args)
    assert len(specs) == 12
    assert any(
        spec["sup_pairs_per_budget"] == 32
        and spec["lambda_qv_trans"] == 0.01
        and spec["qv_trans_loss_type"] == "prob_hinge"
        for spec in specs
    )

    report = {
        "eval": {
            "q_v_next_consistency": {
                "value_checkpoint_available": True,
                "mean_abs_prob_diff": 0.12,
                "rank_correlation": 0.93,
                "v_next_budget_rows": [
                    {"budget": 80, "metrics": make_metrics(0.8, 0.3)}
                ],
            },
            "budget_rows": [
                {
                    "budget": 80,
                    "mean": make_metrics(0.85, 0.25),
                    "ensemble_min": make_metrics(0.80, 0.30),
                }
            ],
        },
        "eval_monotonicity_violation": 0.01,
        "last_qv_summary": {
            "budget_rows": [
                {
                    "budget": 80,
                    "effective_unique_witness_count_mean": 1.5,
                    "replacement_used_frac": 0.5,
                }
            ]
        },
        "last_update_info": {
            "critic/qv_frac_y_trans_gt_parent": 0.6,
            "critic/qv_frac_y_trans_lt_parent": 0.4,
            "critic/loss_qv_trans_by_budget/H80": 0.2,
        },
        "passed": True,
    }
    spec = {
        "seed": 0,
        "sup_pairs_per_budget": 32,
        "lambda_qv_trans": 0.01,
        "qv_trans_loss_type": "bce_lower_bound",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        path.write_text(json.dumps(report))
        rows = sparse.summarize_report(path, spec)
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "seed0_sup32_lam0p01_bce_lower_bound"
        assert row["budget"] == 80
        assert row["passed"] is True
        assert row["q_v_next_abs_diff"] == 0.12
        assert row["qv_effective_k"] == 1.5
        assert row["qv_frac_target_gt_parent"] == 0.6
        assert np.isfinite(row["bce"])
        assert np.isfinite(row["ece"])
        sparse.write_summary(rows, Path(tmpdir) / "summary.csv", Path(tmpdir) / "summary.json")

    print("BMM Q/V sparse table summary checks passed.")


if __name__ == "__main__":
    main()
