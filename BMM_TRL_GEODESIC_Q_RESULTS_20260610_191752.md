# BMM-TRL Geodesic Q Diagnostic Results

Date: 2026-06-10

## Summary

Implemented and ran the first post-geodesic-value milestone from
`BMM_TRL_NEXT_STEPS_AFTER_GEODESIC_VALUE.md`: an action-conditioned geodesic
Q diagnostic.

The clean target is:

```text
Q_H(s_t, a_t, g) = 1[d_grid(s_{t+1}, g) <= H - 1]
```

This is separate from logged future-offset labels and uses balanced
near-boundary positives/negatives per budget.

## Code Added

- `scripts/train_bmm_geodesic_q.py`
  - Action-conditioned critic mode.
  - Fresh balanced geodesic Q pairs per update.
  - Heldout balanced per-budget evaluation.
  - Ensemble-mean and ensemble-min metrics.
  - Optional Q vs restored `V_{H-1}(s_next,g)` consistency hook.
- `utils/pointmaze_grid.py`
  - `sample_grid_budget_q_pairs`.
- `scripts/test_pointmaze_grid_bfs.py`
  - Synthetic checks for Q sampler next-state indexing and labels.

## Main Q Run

Command:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_q.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=grid_geodesic \
  --budgets="(32, 64, 96, 128)" \
  --batch_size=256 \
  --eval_pairs=512 \
  --steps=1000 \
  --eval_interval=250 \
  --agent.value_hidden_dims="(256, 256)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.layer_norm=False \
  --output_json=exp/bmm_grid_geodesic_q_medium_1k.json
```

Final heldout result: `passed=True`, monotonicity violation `0.0000`.

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.9458 | 0.4494 | 0.7601 | 0.3107 | 0.9462 | 0.4508 |
| 64 | 0.9761 | 0.6184 | 0.8330 | 0.2146 | 0.9750 | 0.6154 |
| 96 | 0.9565 | 0.5899 | 0.8261 | 0.2362 | 0.9563 | 0.5893 |
| 128 | 0.9784 | 0.5787 | 0.6665 | 0.0879 | 0.9782 | 0.5669 |

All rows used 256 positives and 256 negatives. The distance oracle AUC was
`1.0000` for every budget, confirming the label construction is separable.

## Short-Horizon Stress Run

Command:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_q.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=grid_geodesic \
  --budgets="(16, 32, 64, 128)" \
  --batch_size=256 \
  --eval_pairs=512 \
  --steps=1000 \
  --eval_interval=250 \
  --agent.value_hidden_dims="(256, 256)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.layer_norm=False \
  --output_json=exp/bmm_grid_geodesic_q_medium_short_1k.json
```

Final heldout result: `passed=True`, monotonicity violation `0.0000`.

| H | AUC | gap | pos_mean | neg_mean | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.9104 | 0.3515 | 0.7147 | 0.3632 | 0.9102 | 0.3501 |
| 32 | 0.9425 | 0.5278 | 0.8528 | 0.3250 | 0.9426 | 0.5324 |
| 64 | 0.9630 | 0.5345 | 0.8255 | 0.2910 | 0.9621 | 0.5386 |
| 128 | 0.9815 | 0.6337 | 0.7874 | 0.1537 | 0.9814 | 0.6302 |

## Interpretation

The action-conditioned Q bridge passes on PointMaze medium. This supports the
diagnosis that clean geodesic labels are learnable in both state-only and
action-conditioned forms, while the earlier high-budget collapse was tied to
logged-offset targets.

The next BMM-specific milestone is to add geodesic-valid max-min transitive
consistency, first in state-only V mode and then in Q/V mode, and check that it
does not damage AUC, score gap, or monotonicity before testing label scarcity.

## Verification

Commands run:

```bash
python -m py_compile utils/pointmaze_grid.py scripts/test_pointmaze_grid_bfs.py scripts/train_bmm_geodesic_q.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/train_bmm_geodesic_q.py --env_name=pointmaze-medium-navigate-v0 --reachability_label_type=grid_geodesic --budgets="(32, 64, 96, 128)" --batch_size=32 --eval_pairs=32 --steps=1 --eval_interval=1 --agent.value_hidden_dims="(64, 64)" --agent.actor_hidden_dims="(64, 64)" --agent.layer_norm=False
```

The CPU/sandbox checks printed JAX CUDA initialization warnings but exited
successfully. The 1k training diagnostics were run with GPU escalation.
