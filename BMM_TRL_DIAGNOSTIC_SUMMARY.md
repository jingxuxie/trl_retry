# BMM-TRL Diagnostic Summary

Date: 2026-06-10

Update as of 2026-06-18: this file is now historical. The project has moved
past the original PointMaze high-budget diagnostic failure into graph/geodesic
targets, Q/V budget-holdout controls, and fixed-controller success-rate
experiments. The current paper-facing success-rate evidence is summarized in
`BMM_TRL_ADVANCED_TASK_RESULTS.md`: BMM graph planning reaches 100.0% on
`puzzle-3x3-play-oraclerep-v0` over 75 rollouts and 72.0% on
`humanoidmaze-medium-navigate-oraclerep-v0` over 75 rollouts with the
switch-128 graph-controller protocol, matching or exceeding the corresponding
paper targets under a fixed paper-style TRL/RPG controller.

This is a compact shareable summary of the current BMM-TRL diagnostic state. The implementation is in this TRL repo and OGBench has not been modified.

## Current Question

We are trying to diagnose why a budget-conditioned reachability critic for offline goal-conditioned RL fails to learn useful high-budget decision boundaries on PointMaze.

The intended diagnostic label is:

```text
label = 1[offset <= H]
```

where `offset` is the same-trajectory future offset and `H` is a discrete/dyadic budget.

The critic is action-conditioned:

```text
R(s, a, g, H)
```

The actor is not the focus yet. Policy evaluation is intentionally deferred until the reachability classifier behaves sensibly.

## Implementation State

Relevant current commit:

```text
d6ad6f2 Add supervised BMM reachability diagnostics
```

Main implementation files:

- `agents/bmm_trl.py`
- `utils/datasets.py`
- `scripts/bmm_reachability_utils.py`
- `scripts/eval_bmm_reachability.py`
- `scripts/check_bmm_reachability_report.py`

Longer context:

- `BMM_TRL_DIAGNOSTIC_HANDOFF.md`
- `BMM_TRL_CODE_REVIEW_NEXT_STEPS.md`

The current code supports:

- Balanced supervised per-budget reachability pairs.
- Scalar budget feature: `concat(g, log2(H) / log2(max_budget))`.
- Scalar plus one-hot budget feature: `concat(g, log_scalar, one_hot(H))`.
- Pairwise budget-ranking fields/loss, although ranking probes are currently too slow to be useful without optimization.
- Balanced diagnostic report rows with 2048 positives and 2048 negatives per budget.
- Gate checks on both ensemble mean and ensemble minimum.

## Checks That Pass

Run from the TRL repo with Conda env `bmm-trl`:

```bash
conda run -n bmm-trl python scripts/test_bmm_tabular.py
conda run -n bmm-trl python scripts/test_bmm_dataset_shapes.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_eval.py
conda run -n bmm-trl python scripts/test_bmm_hard_neg_shapes.py
conda run -n bmm-trl python scripts/test_bmm_reachability_gate.py
conda run -n bmm-trl python scripts/test_bmm_supervised_shapes.py
```

The issue is not currently a missing batch key or simple tensor-shape failure.

## Experiments Tried

All PointMaze runs below use:

```text
env_name=pointmaze-medium-navigate-v0
max_budget=512
budgets=(1,2,4,8,16,32,64,128,256,512)
batch_size=512
actor/value hidden dims=(256,256)
```

### 1. Hard-negative BMM runs

Earlier runs used the original BMM losses with same-trajectory hard negatives. Weighted hard-negative sampling raised the average hard-negative budget from about `64` to about `200`, but high-budget AUC/gap remained weak.

Representative weighted fallback run:

```text
run: exp/mrl/Debug/sd000_20260610_135912
setting: lambda_trans=0.1, lambda_hard_neg=1.0
checkpoint: 20000
```

Random-pair diagnostic:

| H | AUC | pos_mean | neg_mean | gap |
|---:|---:|---:|---:|---:|
| 64 | 0.8681 | 0.3050 | 0.1041 | 0.2009 |
| 128 | 0.7887 | 0.3226 | 0.1802 | 0.1424 |
| 256 | 0.6709 | 0.4757 | 0.4139 | 0.0618 |
| 512 | 0.5943 | 0.7678 | 0.7532 | 0.0146 |

This reduced all-ones collapse somewhat, but `H=512` still had almost no positive-negative separation.

### 2. Balanced supervised BCE, scalar budget

Run:

```text
run: exp/mrl/Debug/sd000_20260610_153023
setting: lambda_sup=1.0, lambda_rank=0.0, lambda_trans=0.0
budget_feature=log_scalar
checkpoint: 10000
```

Balanced diagnostic:

| H | AUC | pos_mean | neg_mean | gap |
|---:|---:|---:|---:|---:|
| 64 | 0.8240 | 0.6361 | 0.3499 | 0.2861 |
| 128 | 0.6664 | 0.5013 | 0.4140 | 0.0873 |
| 256 | 0.4899 | 0.4832 | 0.4858 | -0.0026 |
| 512 | 0.5026 | 0.5061 | 0.5055 | 0.0006 |

Monotonicity violation:

```text
mean=0.2190
ensemble_min=0.2151
```

Training coverage at step 10000 was not the problem:

```text
sup_valid_frac=1.0
sup_pos_frac=0.4954
H=512 train counts: pos=198, neg=199
H=512 train scores: pos=0.5146, neg=0.5125
```

Interpretation: the model saw balanced high-budget labels, but still did not separate `H=256/512`.

### 3. Balanced supervised BCE, scalar plus one-hot budget

Run:

```text
run: exp/mrl/Debug/sd000_20260610_155926
setting: lambda_sup=1.0, lambda_rank=0.0, lambda_trans=0.0
budget_feature=log_scalar_onehot
checkpoints: 5000, 10000
```

Balanced diagnostic at 5000:

| H | AUC | pos_mean | neg_mean | gap |
|---:|---:|---:|---:|---:|
| 64 | 0.8364 | 0.7011 | 0.3500 | 0.3511 |
| 128 | 0.6538 | 0.5589 | 0.4846 | 0.0743 |
| 256 | 0.5234 | 0.5254 | 0.5217 | 0.0037 |
| 512 | 0.5239 | 0.5239 | 0.5186 | 0.0053 |

Balanced diagnostic at 10000:

| H | AUC | pos_mean | neg_mean | gap |
|---:|---:|---:|---:|---:|
| 64 | 0.8441 | 0.6761 | 0.3098 | 0.3663 |
| 128 | 0.6784 | 0.5490 | 0.4526 | 0.0965 |
| 256 | 0.5221 | 0.5148 | 0.5115 | 0.0032 |
| 512 | 0.5191 | 0.5078 | 0.5046 | 0.0031 |

Monotonicity violation at 10000:

```text
mean=0.4275
ensemble_min=0.2535
```

Training coverage at step 10000:

```text
sup_valid_frac=1.0
sup_pos_frac=0.4954
H=256 train scores: pos=0.5140, neg=0.5141
H=512 train scores: pos=0.5115, neg=0.5052
```

Interpretation: one-hot budget features did not fix the high-budget problem and worsened monotonicity.

### 4. Pairwise ranking probes

The review suggested adding pairwise boundary ranking:

```text
logit(s,a,g,H_plus) > logit(s,a,g,H_minus)
where H_minus < offset <= H_plus
```

This is implemented, but the current training path is too slow for practical PointMaze probes:

```text
run: exp/mrl/Debug/sd000_20260610_161201
setting: num_rank_pairs=8, lambda_rank=0.5
result: stopped before first train log after several minutes

run: exp/mrl/Debug/sd000_20260610_161542
setting: num_rank_pairs=2, lambda_rank=0.5
result: stopped before first train log after several minutes
```

Interpretation: ranking may still be conceptually useful, but the current implementation likely needs optimization before it can be used as a small diagnostic experiment.

## Main Empirical Pattern

Across variants:

- `H <= 64` can learn meaningful separation.
- `H=128` is weak and unstable.
- `H=256` and `H=512` remain near random under balanced supervised training.
- This persists even when training batches contain balanced valid positives and negatives for high budgets.
- Better budget representation (`log_scalar_onehot`) did not solve the issue.
- Monotonicity is still poor in supervised-only runs.

This points away from simple sampler coverage or missing high-budget negatives as the sole cause.

## Current Hypotheses

1. The action-conditioned label may be semantically noisy at large horizons.
   - Label is same-trajectory offset, but PointMaze has alternate paths.
   - `R(s,a,g,H)` is trained from one logged action while the label is basically a future-state reachability label.

2. The model may not have enough geometry/distance signal for long-horizon PointMaze from the current inputs/objective.
   - High-budget positives and negatives may be visually/state-wise too similar under raw vector observations.

3. The current supervised objective may be learning local short-horizon correlations, but not a global distance threshold.
   - BCE can classify small offsets well.
   - At larger budgets, scores collapse around 0.5 instead of forming a threshold.

4. Pairwise ranking might be the right next objective, but the current implementation is too expensive.
   - Need optimized/vectorized ranking before running meaningful probes.

5. A state-only diagnostic critic `R(s,g,H)` may be needed to isolate whether action-conditioning is the problem.
   - If state-only works and action-conditioned fails, the action-conditioned label/objective is likely mismatched.
   - If state-only also fails, the issue is more likely representation/objective/geometric.

## Suggested Next Diagnostics

These are intentionally small and diagnostic-only.

1. Add a controlled chain/grid dataset with the same neural training path.
   - Goal: verify the supervised classifier can learn long-budget thresholds when offset equals true graph distance.

2. Add state-only diagnostic critic mode.
   - Train/evaluate `R(s,g,H)` with the same balanced pairs.
   - Compare high-budget AUC/gap against action-conditioned `R(s,a,g,H)`.

3. Optimize ranking implementation.
   - Avoid repeated expensive critic calls and Python-side overhead.
   - Then rerun a small 5k `lambda_rank=0.5` probe.

4. Add budget-scan plots for fixed `(s,g)` pairs.
   - For offsets around 64, 128, 256, 512, plot `R(s,a,g,H)` over all budgets.
   - Desired shape: low before `H ~= offset`, high after.

5. Consider using a geodesic/shortest-path proxy in PointMaze diagnostics.
   - Same-trajectory offset may be a weak label for true reachability in maze geometry.
