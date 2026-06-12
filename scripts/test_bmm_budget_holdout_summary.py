#!/usr/bin/env python
"""Synthetic checks for BMM budget-holdout aggregate summaries."""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bmm_reachability_utils import binary_metrics
from scripts import summarize_bmm_budget_holdout as summary


def make_metrics(pos, neg):
    scores = np.asarray([pos, neg], dtype=np.float64)
    labels = np.asarray([1.0, 0.0], dtype=np.float64)
    return binary_metrics(scores, labels)


def make_flat_row(seed, variant, budget, auc, gap, qv_abs):
    return {
        "seed": seed,
        "variant": variant,
        "budget": budget,
        "auc": auc,
        "gap": gap,
        "bce": 0.5 - gap * 0.1,
        "ece": 0.2 - gap * 0.01,
        "ensemble_min_auc": auc - 0.01,
        "ensemble_min_gap": gap - 0.02,
        "ensemble_min_bce": 0.6 - gap * 0.1,
        "ensemble_min_ece": 0.25 - gap * 0.01,
        "q_v_next_abs_diff": qv_abs,
        "q_v_next_rank_corr": 0.8,
    }


def make_report(path, seed, variant, lambda_qv_trans=0.0, lambda_vnext_distill=0.0):
    report = {
        "eval": {
            "q_v_next_consistency": {
                "mean_abs_prob_diff": 0.18,
                "rank_correlation": 0.7,
                "v_next_budget_rows": [],
            },
            "budget_rows": [
                {
                    "budget": 8,
                    "mean": make_metrics(0.70, 0.20),
                    "ensemble_min": make_metrics(0.68, 0.24),
                }
            ],
        },
        "eval_monotonicity_violation": 0.0,
        "last_qv_summary": None,
        "last_update_info": {},
        "passed": False,
        "config": {
            "supervised_budgets": [2, 4],
            "parent_label_pairs_per_budget": 0,
            "trans_budgets": [8],
            "lambda_qv_trans": lambda_qv_trans,
            "lambda_vnext_distill": lambda_vnext_distill,
            "qv_branch_mode": "learned_q_frozen_v",
            "qv_trans_loss_type": "bce_lower_bound",
            "qv_trans_target_type": "max_min",
            "vnext_distill_loss_type": "bce_lower_bound",
        },
    }
    path.write_text(json.dumps(report))


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        finished = tmpdir / "finished"
        finished.mkdir()
        rows = [
            make_flat_row(0, "A_no_parent_no_trans", 8, 0.90, 0.50, 0.20),
            make_flat_row(0, "B_no_parent_qv_trans", 8, 0.93, 0.58, 0.16),
            make_flat_row(0, "P_no_parent_product_qv", 8, 0.91, 0.54, 0.18),
            make_flat_row(0, "F_no_parent_vnext_distill", 8, 0.901, 0.501, 0.199),
        ]
        (finished / "summary.json").write_text(json.dumps(rows))

        interrupted = tmpdir / "interrupted"
        interrupted.mkdir()
        make_report(
            interrupted / "seed1_A_no_parent_no_trans_qv0_vnext0_parent0.json",
            seed=1,
            variant="A_no_parent_no_trans",
        )
        make_report(
            interrupted / "seed1_B_no_parent_qv_trans_qv0p01_vnext0_parent0.json",
            seed=1,
            variant="B_no_parent_qv_trans",
            lambda_qv_trans=0.01,
        )

        collected = summary.collect_rows([finished, interrupted], budget=8)
        assert len(collected) == 6
        aggregate, per_seed = summary.aggregate_comparisons(
            collected, summary.parse_comparisons("B-A,B-P,P-A,F-A")
        )
        assert aggregate[0]["comparison"] == "B-A"
        assert aggregate[0]["num_seeds"] == 2
        assert aggregate[0]["mean_delta_auc"] > 0.0
        assert aggregate[0]["mean_delta_gap"] > 0.0
        assert aggregate[0]["mean_delta_q_v_next_abs_diff"] < 0.0
        assert aggregate[1]["comparison"] == "B-P"
        assert aggregate[1]["num_seeds"] == 1
        assert aggregate[2]["comparison"] == "P-A"
        assert aggregate[2]["num_seeds"] == 1
        assert aggregate[3]["comparison"] == "F-A"
        assert aggregate[3]["num_seeds"] == 1
        assert len(per_seed) == 5
        text = summary.markdown_table(aggregate, per_seed, budget=8)
        assert "no-parent BMM effect" in text
        assert "max-min versus product transitive effect" in text
        assert "V-next distill control" in text
        text_160 = summary.markdown_table(aggregate, per_seed, budget=160)
        assert "delta H160 AUC" in text_160

    print("BMM budget-holdout aggregate summary checks passed.")


if __name__ == "__main__":
    main()
