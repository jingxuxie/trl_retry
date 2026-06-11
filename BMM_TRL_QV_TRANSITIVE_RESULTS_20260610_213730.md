# BMM-TRL Q/V transitive results

Date: 2026-06-10 21:37 local

This follows `BMM_TRL_NEXT_STEPS_AFTER_WITNESS_SWEEP.md`.

## What changed

- Pulled `jingxuxie/main` and fast-forwarded to `1c5e537`, which added `BMM_TRL_NEXT_STEPS_AFTER_WITNESS_SWEEP.md`.
- Added `--geodesic_budget_unit=env_steps|grid_cells` to:
  - `scripts/train_bmm_geodesic_value.py`
  - `scripts/train_bmm_geodesic_q.py`
- Added value checkpoint saving to `scripts/train_bmm_geodesic_value.py`:
  - `--save_dir`
  - `--save_interval`
- Implemented frozen-teacher Q/V transitive training in `scripts/train_bmm_geodesic_q.py`:
  - target: `y = max_w min(Q_h(s,a,w), V_{H-h}(w,g))`
  - first branch uses the Q target critic;
  - second branch uses restored frozen state-value teacher;
  - new flags: `--lambda_qv_trans`, `--trans_budgets`, `--num_trans_witnesses`, `--trans_witness_mode`, `--trans_pairs_per_update`, `--sup_pairs_per_budget`.
- Added `scripts/test_bmm_qv_transitive_shapes.py`.

## Verification

Passed:

```bash
python -m py_compile scripts/train_bmm_geodesic_value.py scripts/train_bmm_geodesic_q.py scripts/test_bmm_transitive_sampler.py scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_transitive_sampler.py
conda run -n bmm-trl python scripts/test_bmm_qv_transitive_shapes.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
```

CPU/sandboxed tests may print the known JAX CUDA plugin warning before falling back, but all checks above exited with code `0`.

## PointMaze runs

Common setup:

```text
env=pointmaze-medium-navigate-v0
label=grid_geodesic
geodesic_budget_unit=env_steps
budgets=(40,80,160)
steps=1000
batch_size=256
eval_pairs=512
hidden_dims=(256,256)
layer_norm=False
seed=0
```

### V teacher

Saved checkpoint:

```text
exp/bmm_grid_value_qv_teacher_40_80_160/params_1000.pkl
```

Final eval:

| row | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap | pass |
|---|---:|---:|---:|---:|
| V supervised teacher | 0.9531 / 0.4625 | 0.9749 / 0.4995 | 0.9854 / 0.7048 | yes |

Raw report:

```text
exp/bmm_grid_value_qv_teacher_40_80_160.json
```

### Q no-degradation check

Final eval:

| row | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap | pass |
|---|---:|---:|---:|---:|
| Q supervised | 0.9563 / 0.5359 | 0.9786 / 0.5138 | 0.9875 / 0.6421 | yes |
| Q + Q/V trans, `lambda=0.01`, K=4, slack, trans budgets `(80,160)` | 0.9565 / 0.5063 | 0.9763 / 0.4674 | 0.9857 / 0.6712 | yes |

Q/V transitive witness diagnostics at final update:

| H | effective K | replacement frac | zero-left | zero-right |
|---:|---:|---:|---:|---:|
| 80 | 1.0559 | 1.0000 | 0.0000 | 0.0000 |
| 160 | 2.3805 | 0.6726 | 0.0000 | 0.0000 |

Q-V-next consistency:

| row | abs prob diff | rank corr | V-next AUC |
|---|---:|---:|---:|
| Q supervised | 0.2036 | 0.9362 | 0.9341 |
| Q + Q/V trans | 0.1345 | 0.9308 | 0.9341 |

Raw reports:

```text
exp/bmm_grid_q_cell_aligned_sup_40_80_160.json
exp/bmm_grid_q_cell_aligned_qv_trans_40_80_160_lam001_k4_slack.json
```

## Grid-cell budget smoke

Ran a one-step smoke with:

```text
geodesic_budget_unit=grid_cells
budgets=(2,4,8)
lambda_trans=0.01
K=2
```

This was only a plumbing check, not a training result. It confirmed:

```text
label_distance_scale=1.0
distance_stats max_steps=11.0
grid-cell transitive diagnostics are emitted
```

Raw report:

```text
exp/bmm_grid_cells_value_smoke.json
```

## Interpretation

- Frozen-teacher Q/V transitive is implemented and runs end-to-end.
- The first Q/V no-degradation gate passes: Q + Q/V transitive still clears all supervised reachability thresholds.
- Q/V transitive improves H=160 gap versus Q supervised, slightly improves H=40 AUC, and reduces Q-V-next absolute probability difference.
- Q/V transitive hurts H=80 gap in this one run. This is not yet a label-efficiency win.
- Transitive parent budget H=40 is geometrically degenerate for Q because the half-budget first branch is too small after the Q one-step offset. The Q/V run therefore trains/evaluates all budgets `(40,80,160)` but applies Q/V transitive only to parent budgets `(80,160)`.

## Next recommended step

Run sparse-Q label-efficiency checks before policy evaluation:

```text
sup_pairs_per_budget in {64,32,16}
trans_pairs_per_update=256
lambda_qv_trans in {0.0,0.01,0.025}
num_trans_witnesses=4
trans_budgets=(80,160)
seeds={0,1,2} once the single-seed table looks stable
```

Only move to policy smoke after Q/V transitive is at least no-degradation under sparse Q, or after there is a clear reason to test policy despite mixed critic diagnostics.
