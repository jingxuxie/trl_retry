# BMM-TRL final paper control results

Date: 2026-06-12

This note follows `BMM_TRL_NEXT_STEPS_AFTER_PAPER_EXPERIMENT_RESULTS.md`.

## Purpose

The previous paper-focused results showed that product-style transitive learning
was a strong control on grid-cell H8 seed 0. The plan recommended only one more
small experiment set:

1. Run product `P` for grid-cell H8 seeds 1 and 2.
2. Regenerate paper tables.
3. Optionally run env-step H160 product seed 0.
4. Stop experimenting and write.

## Runs

### Grid-cell H8 product replication

Ran only the missing product variant:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --run_dir=exp/bmm_product_vs_maxmin_grid_h8_seeds12 \
  --variants=P \
  --seeds=1,2 \
  --budgets=2,4,8 \
  --eval_budgets=2,4,8 \
  --supervised_budgets=2,4 \
  --trans_budgets=8 \
  --geodesic_budget_unit=grid_cells \
  --steps=1000 \
  --eval_interval=500 \
  --qv_lambda=0.01 \
  --qv_trans_loss_type=bce_lower_bound \
  --value_restore_path=exp/bmm_grid_cells_value_teacher_2_4_8 \
  --value_restore_epoch=1000
```

Aggregated with existing seed 0 product and existing A/B/F baselines:

```bash
conda run -n bmm-trl python scripts/summarize_bmm_budget_holdout.py \
  exp/bmm_qv_budget_holdout_20260611_000227 \
  exp/bmm_qv_budget_holdout_20260611_001949 \
  exp/bmm_qv_budget_holdout_20260611_005700 \
  exp/bmm_product_vs_maxmin_grid_h8_seed0 \
  exp/bmm_product_vs_maxmin_grid_h8_seeds12 \
  --budget=8 \
  --comparisons=B-A,P-A,B-P,F-A \
  --output_json=exp/bmm_product_vs_maxmin_grid_h8_seeds12/aggregate_h8_all_product_seeds.json \
  --output_markdown=exp/bmm_product_vs_maxmin_grid_h8_seeds12/aggregate_h8_all_product_seeds.md
```

### Optional env-step H160 product control

Because product was close to BMM on H8, ran one env-step seed:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --run_dir=exp/bmm_product_vs_maxmin_env_h160_seed0 \
  --variants=P \
  --seeds=0 \
  --budgets=40,80,160 \
  --eval_budgets=40,80,160 \
  --supervised_budgets=40,80 \
  --trans_budgets=160 \
  --geodesic_budget_unit=env_steps \
  --steps=1000 \
  --eval_interval=500 \
  --qv_lambda=0.01 \
  --qv_trans_loss_type=bce_lower_bound \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000
```

Aggregated with existing seed 0 env-step A/B/F baselines:

```bash
conda run -n bmm-trl python scripts/summarize_bmm_budget_holdout.py \
  exp/bmm_qv_budget_holdout_20260611_010904 \
  exp/bmm_product_vs_maxmin_env_h160_seed0 \
  --budget=160 \
  --comparisons=B-A,P-A,B-P,F-A \
  --output_json=exp/bmm_product_vs_maxmin_env_h160_seed0/aggregate_h160_seed0_product.json \
  --output_markdown=exp/bmm_product_vs_maxmin_env_h160_seed0/aggregate_h160_seed0_product.md
```

## Results

### Grid-cell H8, seeds 0/1/2

| comparison | seeds | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs |
|---|---:|---:|---:|---:|---:|---:|
| B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 | -0.0208 |
| P-A | 0,1,2 | +0.0139 | +0.0702 | -0.3344 | -0.0518 | -0.0186 |
| B-P | 0,1,2 | +0.0011 | +0.0050 | -0.0265 | -0.0057 | -0.0022 |
| F-A | 0,1,2 | +0.0002 | +0.0017 | -0.0063 | -0.0010 | -0.0005 |

Per-seed B-P:

| seed | delta AUC | delta gap | delta BCE | delta ECE |
|---:|---:|---:|---:|---:|
| 0 | +0.0008 | +0.0054 | -0.0259 | -0.0059 |
| 1 | +0.0007 | +0.0054 | -0.0289 | -0.0062 |
| 2 | +0.0017 | +0.0042 | -0.0248 | -0.0051 |

### Env-step H160, seed 0

| comparison | seed | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs |
|---|---:|---:|---:|---:|---:|---:|
| B-A | 0 | +0.0400 | +0.0640 | -0.5976 | -0.0398 | -0.0245 |
| P-A | 0 | +0.0345 | +0.0517 | -0.5166 | -0.0316 | -0.0201 |
| B-P | 0 | +0.0055 | +0.0122 | -0.0810 | -0.0083 | -0.0044 |
| F-A | 0 | +0.0035 | +0.0067 | -0.0774 | -0.0038 | -0.0027 |

## Interpretation

Product does not dominate max-min. BMM is positive over product on every
reported metric in grid-cell H8 across three seeds, and also positive over
product on env-step H160 seed 0.

However, the margin is small. Product captures most of the transitive
bootstrapping gain:

```text
Grid H8 B-P:
+0.0011 AUC, +0.0050 gap

Env H160 seed 0 B-P:
+0.0055 AUC, +0.0122 gap
```

The final safe paper claim is:

```text
Max-min reachability has the cleanest non-expansive error algebra and is
empirically competitive with, and mildly better than, a product-style transitive
control in the current budget-holdout diagnostics.
```

Do not claim decisive empirical dominance over product.

## Final decision

Proceed with the focused arXiv-style report.

Final generated table artifacts:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
assets/bmm_tabular_error_scaling.png
```

Stop running experiments for this version and focus on writing.
