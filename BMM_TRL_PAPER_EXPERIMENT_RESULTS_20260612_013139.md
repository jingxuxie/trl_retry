# BMM-TRL paper experiment results

Date: 2026-06-12

This note records the paper-focused experiments run after pulling
`BMM_TRL_PAPER_EXPERIMENT_PLAN.md`.

## Code changes

- Extended `scripts/test_bmm_tabular.py` so it still runs exact max-min backup checks and now writes:
  - `exp/bmm_tabular_error_scaling.json`
  - `exp/bmm_tabular_error_scaling.md`
  - `exp/bmm_tabular_error_scaling.png`
- Added product-style Q/V transitive targets to `scripts/train_bmm_geodesic_q.py`.
  - `qv_trans_target_type=max_min` keeps the existing BMM target.
  - `qv_trans_target_type=product` uses the same witnesses with `Q_h(s,a,w) * V_{H-h}(w,g)`.
- Added `P_no_parent_product_qv` to `scripts/run_bmm_qv_budget_holdout.py`.
- Added `scripts/summarize_bmm_paper_tables.py` for compact paper tables from local JSON artifacts.
- Updated `BMM_TRL_ARXIV_REPORT.md` with tabular, product-ablation, and graph-replication results.

## Verification

Passed:

```bash
conda run -n bmm-trl python -m py_compile \
  scripts/train_bmm_geodesic_q.py \
  scripts/run_bmm_qv_budget_holdout.py \
  scripts/summarize_bmm_budget_holdout.py \
  scripts/summarize_bmm_paper_tables.py \
  scripts/test_bmm_tabular.py \
  scripts/test_bmm_qv_transitive_shapes.py \
  scripts/test_bmm_qv_budget_holdout.py \
  scripts/test_bmm_budget_holdout_summary.py

conda run -n bmm-trl python scripts/test_bmm_tabular.py
conda run -n bmm-trl python scripts/test_bmm_qv_budget_holdout.py
conda run -n bmm-trl python scripts/test_bmm_budget_holdout_summary.py
conda run -n bmm-trl python scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/summarize_bmm_paper_tables.py \
  --output_markdown=exp/bmm_paper_tables_20260612_013139.md \
  --output_json=exp/bmm_paper_tables_20260612_013139.json
```

The JAX shape test emitted the known non-escalated CUDA discovery warning, then exited 0.

## Experiment 1: tabular error scaling

The tabular script now gives the intended visual/theory artifact. With
per-level residual `epsilon=0.02`:

| H | BMM balanced sup error | additive/product sup error |
|---:|---:|---:|
| 1 | 0.0200 | 0.0200 |
| 1024 | 0.2200 | 20.4800 |

Exact max-min backups still match shortest-path reachability on the directed
chain and undirected grid smoke cases.

Interpretation: the toy scaling figure supports the core algebraic story.

## Experiment 3: support-graph H120 replication

Ran graph H120 budget-holdout A/B/F for seeds 1 and 2, then aggregated with the
existing seed 0 graph run.

Command shape:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --run_dir=exp/bmm_graph_qv_budget_holdout_paper_h120 \
  --reachability_label_type=graph \
  --graph_path=exp/bmm_pointmaze_graph.npz \
  --geodesic_budget_unit=env_steps \
  --variants=A,B,F \
  --seeds=1,2 \
  --budgets=40,80,120 \
  --eval_budgets=40,80,120 \
  --supervised_budgets=40,80 \
  --trans_budgets=120 \
  --steps=500 \
  --eval_interval=250 \
  --value_restore_path=exp/bmm_graph_value_teacher_40_80_120_500 \
  --value_restore_epoch=500
```

Aggregate H120 result:

| comparison | seeds | delta AUC | delta gap | delta BCE | delta ECE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0043 | -0.0194 | +0.0224 | +0.0099 | small AUC gain, worse gap/BCE/ECE |
| F-A | 0,1,2 | -0.0010 | +0.0035 | -0.0189 | -0.0056 | V-next control small, better BCE/ECE |

Per-seed B-A:

| seed | delta AUC | delta gap | delta BCE | delta ECE |
|---:|---:|---:|---:|---:|
| 0 | +0.0066 | -0.0048 | -0.0147 | +0.0010 |
| 1 | -0.0021 | -0.0309 | +0.0449 | +0.0131 |
| 2 | +0.0083 | -0.0224 | +0.0371 | +0.0155 |

Interpretation: support-graph labels are learnable and the graph target is
feasible, but graph H120 is not a strong BMM win after replication. It should be
reported as mixed/diagnostic, not as a main positive table row.

## Experiment 4: max-min versus product

Ran only the missing product-style `P` seed-0 grid-cell H8 job and compared it
against the existing seed-0 A/B/F runs.

Command shape:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --run_dir=exp/bmm_product_vs_maxmin_grid_h8_seed0 \
  --variants=P \
  --seeds=0 \
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

Aggregate H8 result:

| comparison | seed | delta AUC | delta gap | delta BCE | delta ECE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| B-A | 0 | +0.0175 | +0.0703 | -0.3365 | -0.0527 | max-min improves over supervised |
| P-A | 0 | +0.0167 | +0.0649 | -0.3106 | -0.0467 | product also improves over supervised |
| B-P | 0 | +0.0008 | +0.0054 | -0.0259 | -0.0059 | max-min slightly better than product |
| F-A | 0 | +0.0002 | +0.0014 | -0.0054 | -0.0008 | V-next control tiny |

Interpretation: product does not dominate max-min, so the paper is not blocked.
However, product captures most of the H8 gain in this clean setting. The paper
should emphasize the non-expansive theory and target-diagnostic story, and be
careful not to overclaim empirical superiority over product.

## Consolidated paper table

Generated by:

```bash
conda run -n bmm-trl python scripts/summarize_bmm_paper_tables.py \
  --output_markdown=exp/bmm_paper_tables_20260612_013139.md \
  --output_json=exp/bmm_paper_tables_20260612_013139.json
```

Key rows:

| setting | comparison | seeds | delta AUC | delta gap | delta BCE | delta ECE |
|---|---|---:|---:|---:|---:|---:|
| grid-cell H8 | B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 |
| env-step H160 | B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 | -0.0380 |
| support-graph H120 | B-A | 0,1,2 | +0.0043 | -0.0194 | +0.0224 | +0.0099 |
| product ablation H8 | B-P | 0 | +0.0008 | +0.0054 | -0.0259 | -0.0059 |

## Paper decision

Proceed only with the smaller-scoped paper:

```text
BMM-TRL as a non-expansive transitive reachability objective,
validated by target diagnosis and heldout-budget reachability bootstrapping.
Policy extraction remains open.
```

The paper should not claim robust policy improvement or broad empirical
dominance over product transitive learning. The strongest positive evidence is
still grid/geodesic budget-holdout. The support-graph result is feasible but
mixed, and product is a strong control that nearly ties BMM on H8.
