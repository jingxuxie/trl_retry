# BMM-TRL Instrumented Transitive Sweep Results

Date: 2026-06-10

## Summary

Followed `BMM_TRL_NEXT_STEPS_TRANSITIVE_20260610.md` through the next
state-only V-transitive diagnostic slice:

1. added transitive witness instrumentation;
2. added per-budget transitive target metrics in `BMMTRLAgent`;
3. added `num_trans_witnesses` and K-witness max-min targets;
4. ran a controlled lambda sweep on PointMaze medium;
5. ran one stronger sparse-label check at batch size 32.

This does not implement Q/V transitive yet. The new diagnostics show H=64 has
near-degenerate witness coverage, so the safer next step is to decide whether to
use cell-aligned budgets or better witness sampling before moving Q/V.

## Code Changes

- `agents/bmm_trl.py`
  - `critic_logits_for_pair_grid(..., target=True)` support.
  - Multi-witness transitive target:

    ```text
    y_trans = max_w min(V_h(s,w), V_{H-h}(w,g))
    ```

  - Per-budget metrics:
    - `y_trans_mean_H=*`
    - `first_r_mean_H=*`
    - `second_r_mean_H=*`
    - `parent_r_mean_H=*`
    - `loss_trans_H=*`
    - `loss_sup_H=*`
    - `loss_trans_over_sup_H=*`
    - oracle label/branch-valid means when available.

- `scripts/train_bmm_geodesic_value.py`
  - Added `--num_trans_witnesses`.
  - Added witness diagnostics:
    - acceptance rate and attempts/sample;
    - budget counts;
    - parent/left/right distance means;
    - left/right slack means;
    - candidate witness-cell counts;
    - unique witness fraction;
    - zero-distance witness fractions;
    - coarse histograms in output JSON.

- `scripts/test_bmm_agent_shapes.py`
  - Added direct smoke coverage for `[K,B]` multi-witness transitive tensors.

## Sweep Setup

Common command shape:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=grid_geodesic \
  --budgets="(64, 128)" \
  --batch_size=256 \
  --eval_pairs=512 \
  --steps=1000 \
  --eval_interval=500 \
  --agent.value_hidden_dims="(256, 256)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.layer_norm=False
```

All runs passed the threshold gate and had final monotonicity violation `0.0000`.

## Lambda Sweep, K=1

| lambda_trans | H | AUC | gap | pos_mean | neg_mean |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 64 | 0.9636 | 0.5017 | 0.8683 | 0.3666 |
| 0.000 | 128 | 0.9753 | 0.5479 | 0.6490 | 0.1011 |
| 0.010 | 64 | 0.9649 | 0.4910 | 0.8895 | 0.3985 |
| 0.010 | 128 | 0.9792 | 0.5739 | 0.6861 | 0.1122 |
| 0.025 | 64 | 0.9648 | 0.4914 | 0.8853 | 0.3940 |
| 0.025 | 128 | 0.9793 | 0.5719 | 0.6834 | 0.1115 |
| 0.050 | 64 | 0.9641 | 0.4921 | 0.8763 | 0.3842 |
| 0.050 | 128 | 0.9792 | 0.5668 | 0.6757 | 0.1089 |

Interpretation:

- All lambda values are no-degradation by AUC.
- `0.01` and `0.025` look safest.
- `0.05` still passes, but trims the H=128 gap relative to `0.01/0.025`.

## K=4 Multi-Witness Check

Run:

```text
lambda_trans=0.025
num_trans_witnesses=4
```

| K | H | AUC | gap | pos_mean | neg_mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.9648 | 0.4914 | 0.8853 | 0.3940 |
| 1 | 128 | 0.9793 | 0.5719 | 0.6834 | 0.1115 |
| 4 | 64 | 0.9634 | 0.4846 | 0.8892 | 0.4046 |
| 4 | 128 | 0.9784 | 0.5796 | 0.7016 | 0.1220 |

K=4 is not clearly better overall. It slightly improves the H=128 gap but
slightly hurts H=64.

## Witness Diagnostics

Final K=4 abundant run:

| H | parents | parent_d | left_d | right_d | left_slack | right_slack | witness_cells | unique_witness_frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 83 | 39.5916 | 19.7958 | 19.7958 | 12.2042 | 12.2042 | 1.1084 | 0.2711 |
| 128 | 173 | 96.8048 | 49.4895 | 48.6885 | 14.5105 | 15.3115 | 2.2890 | 0.5275 |

Both `parent_oracle_label_mean` and `branch_oracle_valid_mean` were `1.0`, so
the sampler is producing valid transitive tuples.

The important diagnostic is witness coverage:

- H=64 has only about `1.1` valid witness cells on average.
- K=4 therefore mostly resamples the same witness for H=64.
- H=128 has better but still small witness diversity, about `2.3` valid cells.

This explains why K=4 is not a strong improvement on the current `(64,128)`
setup.

## Sparse Batch-Size 32 Check

This is an 8x direct-label reduction relative to `batch_size=256`.

| setting | H | AUC | gap | pos_mean | neg_mean |
|---|---:|---:|---:|---:|---:|
| supervised, K=1 | 64 | 0.9416 | 0.4038 | 0.8568 | 0.4530 |
| supervised, K=1 | 128 | 0.9662 | 0.4344 | 0.5194 | 0.0850 |
| transitive, K=4 | 64 | 0.9631 | 0.3743 | 0.8784 | 0.5042 |
| transitive, K=4 | 128 | 0.9685 | 0.4602 | 0.5724 | 0.1122 |

Interpretation:

- K=4 transitive improves AUC at both budgets.
- It improves H=128 gap.
- It hurts H=64 gap because the negative scores are higher.
- This is a modest positive sparse result, but not a complete label-efficiency
  claim yet. It is still one seed and one sparse setting.

## Recommended Next Step

Do not move to policy evaluation yet.

The highest-value next diagnostic is one of:

1. run a cell-aligned budget diagnostic, e.g. `(40,80,160)`, to see whether
   witness diversity improves when budgets align better with
   `steps_per_cell ~= 19.8`;
2. add boundary/slack-balanced witness sampling and compare it with
   `uniform_valid`;
3. then implement Q/V transitive with a frozen V teacher once V witness coverage
   is no longer obviously degenerate at the main budgets.

The current evidence says the transitive implementation is sane and safe, but
the `(64,128)` witness geometry limits how much multi-witness max-min can help.

## Verification

Commands run:

```bash
python -m py_compile agents/bmm_trl.py scripts/train_bmm_geodesic_value.py scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
conda run -n bmm-trl python scripts/test_pointmaze_grid_bfs.py
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py --env_name=pointmaze-medium-navigate-v0 --reachability_label_type=grid_geodesic --budgets="(64, 128)" --batch_size=32 --eval_pairs=32 --steps=1 --eval_interval=1 --lambda_trans=0.025 --num_trans_witnesses=4 --agent.value_hidden_dims="(64, 64)" --agent.actor_hidden_dims="(64, 64)" --agent.layer_norm=False --output_json=exp/bmm_grid_geodesic_value_trans_k4_smoke.json
```

The sweep and sparse 1k runs were run with GPU escalation.
