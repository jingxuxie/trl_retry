# BMM-TRL next steps after Q/V loss ablation

Date: 2026-06-11

This plan follows `BMM_TRL_QV_LOSS_ABLATION_RESULTS_20260610_221317.md`.

## Executive conclusion

You are still on track, and the results are progressing.

The important outcome of the Q/V loss ablation is:

```text
All Q/V transitive loss modes pass the heldout threshold gate.
All Q/V transitive modes substantially improve Q-V-next probability consistency.
No loss mode gives a decisive abundant-label AUC/gap win.
```

This is a healthy result. It says the frozen-teacher Q/V path is stable enough to test the real BMM question:

```text
Does Q/V max-min transitive consistency help when direct Q labels are scarce?
```

Do not move to policy evaluation yet. The next milestone should be a sparse-Q label-efficiency table.

## What the latest ablation means

### Good signs

All rows passed the current heldout gate and final monotonicity remained `0.0`.

The Q/V transitive rows reduce Q-V-next absolute probability difference from:

```text
supervised only: 0.2036
Q/V transitive: ~0.133 to 0.135
```

This is a meaningful Bellman-consistency improvement.

### Mixed signs

All Q/V modes slightly reduce the `H=80` AUC/gap relative to supervised-only, while improving the `H=160` gap. This is not surprising because the abundant supervised baseline is already strong.

### Loss-mode conclusion

The diagnostics confirm the conceptual concern:

```text
bce_equal applies downward pressure on about 40.8% of valid parent predictions.
```

Lower-bound modes remove or gate that downward pressure, but in the abundant-label setting they produce almost the same heldout AUC/gap as equality BCE.

So the right conclusion is:

```text
Do not over-tune loss mode on abundant labels.
Use bce_lower_bound as the default for sparse-Q, and keep prob_hinge as a sanity alternative.
```

## Recommended default now

Use:

```text
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
trans_budgets = (80,160)
trans_pairs_per_update = 256
```

Why:

- `bce_lower_bound` is aligned with the lower-bound interpretation of sampled max-min targets.
- It gave the best Q-V-next absolute difference in the ablation, although only slightly.
- It preserves the BCE/logit-scale gradient when the sampled target is above the parent.
- It avoids penalizing parents that are already above a sampled lower-bound target.

Keep `prob_hinge` for one small sparse sanity check because it is the cleanest inequality loss.

## Milestone 1: sparse-Q label-efficiency table

This is the highest-value next experiment.

### Single-seed first table

Run seed `0` first:

```text
sup_pairs_per_budget in {256,64,32,16}
lambda_qv_trans in {0.0,0.01,0.025}
qv_trans_loss_type = bce_lower_bound
trans_pairs_per_update = 256
num_trans_witnesses = 4
trans_budgets = (80,160)
budgets = (40,80,160)
steps = 1000
```

Also run one small alternative row:

```text
sup_pairs_per_budget = 32
lambda_qv_trans = 0.01
qv_trans_loss_type = prob_hinge
```

This answers whether the inequality formulation matters under scarcity.

### Multi-seed table

If seed `0` shows a useful signal, repeat with:

```text
seeds = {0,1,2}
```

If seed `0` is mixed, do not immediately abandon it. First inspect:

```text
Q-V-next abs diff
H=80 gap
H=160 gap
witness effective K
frac target > parent
frac target < parent
loss_qv_trans_by_budget
```

### Success criteria

A useful BMM sparse-Q result is any of:

```text
higher AUC/gap under 8x or 16x fewer direct Q labels
fewer steps to threshold
better Q-V-next consistency with no meaningful AUC/gap degradation
better calibration/BCE at similar AUC
```

Do **not** require Q/V transitive to beat abundant supervised Q. That is not the claim.

## Milestone 2: add a sparse table runner

The number of rows is now large enough that manual commands will become error-prone.

Add:

```text
scripts/run_bmm_qv_sparse_table.py
```

or a shell script:

```text
scripts/run_bmm_qv_sparse_table.sh
```

The runner should produce a compact CSV/JSON summary with one row per run:

```text
seed
sup_pairs_per_budget
lambda_qv_trans
qv_trans_loss_type
H
AUC
gap
BCE
pos_mean
neg_mean
Q-V-next abs diff
Q-V-next rank corr
monotonicity
qv effective K
replacement frac
frac target > parent
frac target < parent
passed
```

This table is likely to become the first real figure/table for the project.

## Milestone 3: calibration diagnostics

AUC/gap may hide calibration changes. Q/V transitive is partly a consistency/calibration regularizer, so report calibration explicitly.

Add per-budget:

```text
BCE
ECE or 10-bin calibration error
mean predicted score for positives
mean predicted score for negatives
threshold accuracy at 0.5
```

Use the same heldout eval batch for all runs in a table. That makes differences easier to interpret.

## Milestone 4: decide whether the witness geometry is limiting Q/V

The current witness diversity is still small:

```text
H=80:  effective K ~= 1.06
H=160: effective K ~= 2.38
```

If sparse-Q is mixed, run the clean algebra version before moving to policy:

```text
geodesic_budget_unit = grid_cells
budgets = (2,4,8)
trans_budgets = (4,8)
```

This removes env-step calibration and directly tests max-min composition on the maze topology.

Interpretation:

- If grid-cell Q/V transitive helps under sparse labels, the BMM idea is fine and env-step witness geometry is the issue.
- If grid-cell Q/V transitive also fails, inspect loss/teacher/sampling more deeply before policy.

## Milestone 5: V-teacher quality and teacher variants

The frozen V teacher is strong, but the second branch may still be under- or over-confident.

Add a short teacher report to every Q/V run:

```text
V teacher heldout AUC/gap by budget
V teacher calibration/BCE
V teacher monotonicity
V_next AUC on the Q eval batch
```

If sparse-Q results are sensitive, try:

```text
teacher checkpoint at step 500 vs 1000
ensemble mean vs ensemble min for V branch
```

Start with ensemble mean. Later test ensemble-min if overestimation appears.

## Milestone 6: policy smoke only after sparse-Q decision

Policy evaluation should wait until one of these is true:

```text
A. sparse-Q table shows a useful BMM signal; or
B. Q/V transitive robustly improves Q-V consistency with no AUC/gap degradation, and you want to test whether consistency helps policy even without sparse AUC gains.
```

Policy smoke should use budget scanning:

```text
H_hat(s,g) = smallest H with V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Do not use max-budget action scoring as the main policy path. Max-budget reachability is too coarse for action ranking.

## Milestone 7: main-path integration

After the sparse-Q table, integrate the working mode into the normal data path.

Add config fields:

```python
reachability_label_type = "grid_geodesic"
geodesic_budget_unit = "env_steps"
geodesic_mode = "q_next"
lambda_qv_trans = 0.01
qv_trans_loss_type = "bce_lower_bound"
trans_budgets = (80,160)
num_trans_witnesses = 4
trans_witness_mode = "slack_balanced"
sup_pairs_per_budget = 32  # or chosen sparse setting
trans_pairs_per_update = 256
```

Keep standalone scripts for diagnostics, but ensure the main training path can reproduce the same heldout metrics.

## Milestone 8: PointMaze large after medium sparse-Q

Move to PointMaze large only after the medium sparse-Q story is clear.

Before training:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Pick budgets based on the calibrated diameter and class coverage. Do not assume `256/512` are meaningful without checking.

## What not to do next

Do not yet:

```text
start full policy comparisons
move straight to PointMaze large
add ranking or monotonicity penalties
reintroduce logged-offset hard negatives
keep tuning abundant-label Q/V loss modes
```

The next real question is sparse-Q label efficiency.

## Immediate command plan

### Baseline sparse rows

```text
for sup_pairs in 256 64 32 16:
  lambda_qv_trans=0.0
```

### Q/V rows

```text
for sup_pairs in 256 64 32 16:
  lambda_qv_trans=0.01
  qv_trans_loss_type=bce_lower_bound
```

### Stronger Q/V rows

```text
for sup_pairs in 64 32 16:
  lambda_qv_trans=0.025
  qv_trans_loss_type=bce_lower_bound
```

### Sanity alternative

```text
sup_pairs=32
lambda_qv_trans=0.01
qv_trans_loss_type=prob_hinge
```

If the single-seed table is promising, repeat the most informative rows with seeds `{1,2}`.

## Bottom line

The ablation is progressing the project. It shows that Q/V transitive is stable, improves Q-V consistency, and has no catastrophic downside on clean geodesic labels. The abundant-label setting is saturated, so it is not the right place to look for the payoff.

The most likely next progress comes from the sparse-Q table. That is where BMM's max-min transitive structure should have a chance to show value.
