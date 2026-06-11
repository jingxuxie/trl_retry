# BMM-TRL after transitive sweep results

Date: 2026-06-10 21:04 local

This follows `BMM_TRL_NEXT_STEPS_AFTER_TRANSITIVE_SWEEP.md`.

## Implementation changes

- Pulled `jingxuxie/main` and fast-forwarded to `d8579bc`, which added the next-step plan.
- Added target safety in `agents/bmm_trl.py`:
  - clip transitive targets to `[0, 1]`;
  - set invalid parent targets to `0` after masking.
- Extended `scripts/train_bmm_geodesic_value.py`:
  - `--trans_witness_mode=uniform_valid|avoid_endpoints|slack_balanced|boundary_balanced`;
  - `--trans_endpoint_epsilon`;
  - `--trans_boundary_beta`;
  - `--sup_pairs_per_budget`;
  - `--trans_pairs_per_update`.
- Added transitive witness diagnostics:
  - `witness_candidate_count_mean`;
  - `effective_unique_witness_count_mean`;
  - `replacement_used_frac`;
  - `witness_fallback_used_frac`;
  - supervised distance-over-budget histograms for parent-distribution comparison.
- Added `scripts/test_bmm_transitive_sampler.py` for synthetic line-graph checks of witness modes and effective-K reporting.

## Verification

Passed:

```bash
python -m py_compile agents/bmm_trl.py scripts/train_bmm_geodesic_value.py scripts/test_bmm_transitive_sampler.py
conda run -n bmm-trl python scripts/test_bmm_tabular.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_bmm_transitive_sampler.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_eval.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
```

Some CPU/lightweight tests printed JAX CUDA plugin warnings under sandboxed access, but all exited with code `0`.

## Cell-aligned V-transitive runs

Common setup:

```text
env=pointmaze-medium-navigate-v0
label=grid_geodesic
budgets=(40,80,160)
steps=1000
eval_pairs=512
batch_size=256
hidden_dims=(256,256)
layer_norm=False
seed=0
```

Final eval metrics:

| run | pass | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap |
|---|---:|---:|---:|---:|
| supervised only | yes | 0.9531 / 0.4625 | 0.9749 / 0.4995 | 0.9854 / 0.7048 |
| trans `lambda=0.01`, K=4, avoid | yes | 0.9498 / 0.4957 | 0.9694 / 0.5247 | 0.9838 / 0.6703 |
| trans `lambda=0.01`, K=4, uniform | yes | 0.9498 / 0.4957 | 0.9694 / 0.5247 | 0.9838 / 0.6703 |
| trans `lambda=0.01`, K=4, slack | yes | 0.9492 / 0.4904 | 0.9746 / 0.5317 | 0.9857 / 0.6924 |
| trans `lambda=0.025`, K=4, avoid | yes | 0.9496 / 0.4958 | 0.9700 / 0.5283 | 0.9839 / 0.6720 |

Key witness diagnostics at final update:

| run | H=40 effK / repl | H=80 effK / repl | H=160 effK / repl | endpoint zeros |
|---|---:|---:|---:|---:|
| avoid `lambda=0.01` | 1.125 / 1.000 | 1.381 / 1.000 | 3.283 / 0.348 | 0.0 / 0.0 |
| uniform `lambda=0.01` | 1.125 / 1.000 | 1.381 / 1.000 | 3.283 / 0.348 | 0.0 / 0.0 |
| slack `lambda=0.01` | 1.108 / 1.000 | 1.368 / 1.000 | 3.207 / 0.391 | 0.0 / 0.0 |
| avoid `lambda=0.025` | 1.125 / 1.000 | 1.381 / 1.000 | 3.283 / 0.348 | 0.0 / 0.0 |

Raw reports:

```text
exp/bmm_grid_value_cell_aligned_sup_40_80_160.json
exp/bmm_grid_value_cell_aligned_trans_40_80_160_lam001_k4_avoid.json
exp/bmm_grid_value_cell_aligned_trans_40_80_160_lam001_k4_uniform.json
exp/bmm_grid_value_cell_aligned_trans_40_80_160_lam001_k4_slack.json
exp/bmm_grid_value_cell_aligned_trans_40_80_160_lam0025_k4_avoid.json
```

## Sparse-label runs

These keep transitive parents fixed at 256 when transitive is enabled, while reducing direct supervised pairs per budget.

Final eval metrics:

| run | pass | H=40 AUC/gap | H=80 AUC/gap | H=160 AUC/gap |
|---|---:|---:|---:|---:|
| supervised, 32 pairs/budget | yes | 0.9295 / 0.2247 | 0.9620 / 0.3526 | 0.9772 / 0.4776 |
| trans, 32 pairs/budget, `lambda=0.01`, K=4, slack | yes | 0.9310 / 0.2561 | 0.9578 / 0.3569 | 0.9765 / 0.4585 |
| supervised, 16 pairs/budget | no | 0.9001 / 0.1887 | 0.9221 / 0.2880 | 0.9586 / 0.4238 |
| trans, 16 pairs/budget, `lambda=0.01`, K=4, slack | no | 0.9265 / 0.1965 | 0.9406 / 0.2948 | 0.9719 / 0.3926 |

Raw reports:

```text
exp/bmm_grid_value_cell_aligned_sparse_sup32_only.json
exp/bmm_grid_value_cell_aligned_sparse_sup32_trans256_lam001_k4_slack.json
exp/bmm_grid_value_cell_aligned_sparse_sup16_only.json
exp/bmm_grid_value_cell_aligned_sparse_sup16_trans256_lam001_k4_slack.json
```

## Interpretation

- Cell-aligned budgets make the V critic learn cleanly. All abundant-supervision runs pass.
- Transitive consistency is stable and does not cause collapse.
- `uniform_valid` and `avoid_endpoints` are identical for this cell-aligned setup because sampled witnesses already have zero endpoint fraction.
- Multi-witness diversity still depends strongly on budget geometry:
  - H=40 and H=80 have effective K near 1.1 and 1.4, with replacement used every time.
  - H=160 has useful diversity, effective K around 3.2.
- The sparse signal is modest, not decisive:
  - At 32 supervised pairs per budget, transitive improves H=40 gap and roughly ties H=80, but slightly reduces H=160 gap.
  - At 16 supervised pairs per budget, transitive improves H=40/H=80 AUC and gap, but still misses the gate by H=40 gap `0.1965 < 0.20`; H=160 AUC improves while gap drops.

## Current decision

Do not move to policy evaluation yet.

The next implementation milestone should be Q/V transitive with a frozen V teacher, but that is not just a sampler flag:

```text
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

`scripts/train_bmm_geodesic_q.py` currently supports supervised Q training and optional Q-vs-V-next reporting, but not a frozen-teacher Q/V transitive training loss. That should be implemented as a separate, focused commit after deciding how to save/reuse the V teacher checkpoint from the value diagnostic.
