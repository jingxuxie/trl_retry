# BMM-TRL Sparse-Q Label-Efficiency Results

Date: 2026-06-10

This follows `BMM_TRL_NEXT_STEPS_AFTER_QV_LOSS_ABLATION.md`. The goal was to test whether frozen-teacher Q/V max-min transitive consistency helps when direct action-conditioned Q labels are scarce.

## Implementation

- Added 10-bin ECE to the shared `binary_metrics` diagnostic helper.
- Extended `scripts/train_bmm_geodesic_q.py` so Q reports include V-next teacher calibration on the same heldout Q eval batch.
- Added `scripts/run_bmm_qv_sparse_table.py`.
  - Runs the planned seed-0 sparse table.
  - Writes one JSON report per run.
  - Writes compact `summary.csv` and `summary.json` with one row per `(run, budget)`.
- Added `scripts/test_bmm_qv_sparse_table.py` for synthetic summary parsing and ECE checks.

## Verification

Passed:

```bash
python -m py_compile scripts/bmm_reachability_utils.py scripts/train_bmm_geodesic_q.py scripts/run_bmm_qv_sparse_table.py scripts/test_bmm_qv_sparse_table.py scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_qv_sparse_table.py
conda run -n bmm-trl python scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
conda run -n bmm-trl python scripts/test_bmm_transitive_sampler.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
conda run -n bmm-trl python scripts/test_bmm_tabular.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
```

CPU-only tests still print the known JAX CUDA plugin warning before passing.

## Env-Step Sparse-Q Table

Run directory:

```text
exp/bmm_qv_sparse_table_20260610_2235/
```

Setup:

```text
env=pointmaze-medium-navigate-v0
geodesic_budget_unit=env_steps
budgets=(40,80,160)
trans_budgets=(80,160)
seed=0
steps=1000
batch_size=256
eval_pairs=512
trans_pairs_per_update=256
num_trans_witnesses=4
trans_witness_mode=slack_balanced
teacher=exp/bmm_grid_value_qv_teacher_40_80_160/params_1000.pkl
```

Final metrics:

| run | pass | H40 AUC/gap | H80 AUC/gap | H160 AUC/gap | Q-V abs |
|---|---:|---:|---:|---:|---:|
| sup256 | true | 0.9563/0.5359 | 0.9786/0.5138 | 0.9875/0.6421 | 0.2036 |
| sup64 | true | 0.9453/0.3680 | 0.9469/0.3497 | 0.9829/0.5055 | 0.2007 |
| sup32 | true | 0.9357/0.2310 | 0.9452/0.2793 | 0.9715/0.4347 | 0.1704 |
| sup16 | false | 0.9483/0.1648 | 0.9481/0.2505 | 0.9771/0.3655 | 0.1908 |
| sup256 + Q/V 0.01 | true | 0.9564/0.5058 | 0.9765/0.4677 | 0.9857/0.6728 | 0.1327 |
| sup64 + Q/V 0.01 | true | 0.9569/0.3350 | 0.9513/0.3315 | 0.9758/0.5192 | 0.1671 |
| sup32 + Q/V 0.01 | true | 0.9443/0.2457 | 0.9171/0.2932 | 0.9483/0.4033 | 0.1909 |
| sup16 + Q/V 0.01 | false | 0.8849/0.1795 | 0.8980/0.2314 | 0.9632/0.3397 | 0.2341 |
| sup64 + Q/V 0.025 | true | 0.9571/0.3320 | 0.9537/0.3317 | 0.9763/0.5255 | 0.1623 |
| sup32 + Q/V 0.025 | true | 0.9446/0.2430 | 0.9192/0.2931 | 0.9506/0.4104 | 0.1869 |
| sup16 + Q/V 0.025 | false | 0.8820/0.1784 | 0.8975/0.2306 | 0.9654/0.3425 | 0.2349 |
| sup32 + prob_hinge 0.01 | true | 0.9443/0.2469 | 0.9161/0.2931 | 0.9478/0.3998 | 0.1931 |

Key deltas versus same-supervision baseline:

| run | H40 delta | H80 delta | H160 delta | Q-V abs delta |
|---|---:|---:|---:|---:|
| sup64 + Q/V 0.01 | +0.0116/-0.0330 | +0.0044/-0.0183 | -0.0072/+0.0136 | -0.0337 |
| sup64 + Q/V 0.025 | +0.0118/-0.0360 | +0.0068/-0.0180 | -0.0066/+0.0199 | -0.0385 |
| sup32 + Q/V 0.01 | +0.0086/+0.0146 | -0.0281/+0.0139 | -0.0232/-0.0315 | +0.0205 |
| sup32 + Q/V 0.025 | +0.0089/+0.0120 | -0.0260/+0.0138 | -0.0209/-0.0243 | +0.0165 |
| sup16 + Q/V 0.01 | -0.0634/+0.0146 | -0.0500/-0.0190 | -0.0139/-0.0258 | +0.0434 |
| sup16 + Q/V 0.025 | -0.0663/+0.0136 | -0.0505/-0.0199 | -0.0117/-0.0230 | +0.0441 |

Interpretation:

- Env-step Q/V transitive gives a useful consistency gain at `sup64`: Q-V abs diff improves from `0.2007` to `0.1671` or `0.1623`.
- The same rows modestly improve H40/H80 AUC and H160 gap, but reduce H40/H80 gap.
- At `sup32`, Q/V improves H40 gap and H80 gap but hurts H80/H160 AUC and Q-V consistency.
- At `sup16`, Q/V is clearly harmful on AUC and Q-V consistency.
- `prob_hinge` is not materially better than `bce_lower_bound` at `sup32`.

This is mixed, not a clean sparse-label win.

## Grid-Cell Algebra Check

Because the env-step sparse table was mixed, I ran the plan's clean grid-cell diagnostic.

First trained a grid-cell V teacher:

```text
exp/bmm_grid_cells_value_teacher_2_4_8/params_1000.pkl
```

Teacher final eval:

| H | AUC | gap |
|---:|---:|---:|
| 2 | 0.9743 | 0.7106 |
| 4 | 0.9751 | 0.7255 |
| 8 | 0.9940 | 0.8430 |

Grid-cell Q table directory:

```text
exp/bmm_qv_sparse_gridcells_table_20260610_2305/
```

Final metrics:

| run | pass | H2 AUC/gap | H4 AUC/gap | H8 AUC/gap | Q-V abs |
|---|---:|---:|---:|---:|---:|
| sup256 | true | 0.9000/0.4702 | 0.9734/0.6768 | 0.9913/0.8002 | 0.1564 |
| sup64 | false | 0.8876/0.3615 | 0.9730/0.6228 | 0.9854/0.7468 | 0.1578 |
| sup32 | false | 0.8885/0.2987 | 0.9704/0.6007 | 0.9836/0.6986 | 0.1491 |
| sup16 | false | 0.8624/0.2280 | 0.9620/0.5233 | 0.9823/0.6433 | 0.1813 |
| sup256 + Q/V 0.01 | true | 0.9017/0.4590 | 0.9761/0.6966 | 0.9873/0.7820 | 0.1219 |
| sup64 + Q/V 0.01 | false | 0.8892/0.3626 | 0.9731/0.6399 | 0.9866/0.7362 | 0.1270 |
| sup32 + Q/V 0.01 | false | 0.8744/0.2994 | 0.9657/0.5778 | 0.9846/0.7060 | 0.1736 |
| sup16 + Q/V 0.01 | false | 0.8609/0.2307 | 0.9643/0.5206 | 0.9823/0.6455 | 0.1704 |
| sup64 + Q/V 0.025 | false | 0.8894/0.3634 | 0.9731/0.6402 | 0.9871/0.7382 | 0.1252 |
| sup32 + Q/V 0.025 | false | 0.8744/0.2998 | 0.9653/0.5757 | 0.9858/0.7094 | 0.1726 |
| sup16 + Q/V 0.025 | false | 0.8618/0.2309 | 0.9642/0.5189 | 0.9828/0.6501 | 0.1696 |
| sup32 + prob_hinge 0.01 | false | 0.8742/0.2991 | 0.9659/0.5790 | 0.9838/0.7038 | 0.1745 |

Key deltas versus same-supervision baseline:

| run | H2 delta | H4 delta | H8 delta | Q-V abs delta |
|---|---:|---:|---:|---:|
| sup64 + Q/V 0.01 | +0.0016/+0.0011 | +0.0001/+0.0171 | +0.0013/-0.0106 | -0.0308 |
| sup64 + Q/V 0.025 | +0.0018/+0.0018 | +0.0002/+0.0174 | +0.0017/-0.0086 | -0.0327 |
| sup32 + Q/V 0.01 | -0.0141/+0.0007 | -0.0047/-0.0229 | +0.0010/+0.0073 | +0.0245 |
| sup32 + Q/V 0.025 | -0.0141/+0.0011 | -0.0051/-0.0250 | +0.0022/+0.0107 | +0.0235 |
| sup16 + Q/V 0.01 | -0.0015/+0.0028 | +0.0023/-0.0028 | +0.0000/+0.0022 | -0.0108 |
| sup16 + Q/V 0.025 | -0.0006/+0.0029 | +0.0022/-0.0044 | +0.0005/+0.0068 | -0.0117 |

Interpretation:

- Grid-cell Q/V at `sup64` improves Q-V consistency substantially and slightly improves H2/H4 AUC/gap, but it does not fix the H2 threshold failure.
- At `sup32`, Q/V hurts H2/H4 AUC and Q-V consistency, while slightly improving some gaps and H8 AUC.
- At `sup16`, Q/V is mostly neutral to slightly positive on gaps and Q-V consistency, but not enough to pass.
- Grid-cell witness geometry is cleaner than env-steps: final `target < parent` is high for abundant rows, so the lower-bound gate often avoids updates.

## Current Conclusion

The current Q/V transitive implementation is stable and can improve Q-V consistency, but the single-seed sparse tables do not yet show a clean BMM label-efficiency result.

The strongest positive signal is:

```text
sup64, lambda_qv_trans=0.01 or 0.025
```

for both env-step and grid-cell settings. These improve Q-V consistency and some AUC/gap entries without catastrophic degradation.

The strongest negative signal is:

```text
sup16
```

where Q/V transitive is not rescuing sparse direct Q labels and sometimes hurts.

I would not start policy evaluation yet. The next useful step is to inspect whether the Q/V target is too weak or mismatched at low supervision:

- compare frozen V teacher ensemble mean vs ensemble min for the second branch;
- run seeds `{1,2}` only for the informative `sup64` rows, not the full table;
- inspect whether Q/V parent targets are mostly below parent in grid-cell runs, which makes lower-bound updates sparse;
- consider increasing useful target pressure only when `y_trans > parent`, rather than increasing lambda globally.
