# BMM-TRL next steps after frozen-teacher Q/V transitive

Date: 2026-06-11

This plan follows `BMM_TRL_QV_TRANSITIVE_RESULTS_20260610_213730.md`.

## Executive conclusion

Yes, this is progressing in the right direction.

You have now implemented the policy-relevant BMM composition:

```text
y = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

with a frozen state-value teacher for the second branch. The first run passes the no-degradation threshold gate and improves one important internal diagnostic: Q/V transitive reduces the Q-vs-V-next probability difference from about `0.2036` to `0.1345`.

That is a meaningful milestone. It means the code path is running end-to-end and the Q critic is being pulled toward a Bellman-consistent reachability structure.

The result is not yet a label-efficiency or policy result. The next step should be a **controlled Q/V transitive ablation**, not policy evaluation.

## Current interpretation

### What looks good

The latest run shows:

```text
Q supervised:       passes all thresholds
Q + Q/V transitive: passes all thresholds
Q/V transitive:     improves H=160 gap
Q/V transitive:     reduces Q-V_next absolute probability difference
```

This is exactly the kind of no-degradation checkpoint needed before sparse-label experiments.

### What is still mixed

The Q/V transitive run hurts the `H=80` gap and slightly lowers `H=80/H=160` AUC versus supervised-only. This does not look alarming yet, but it means the transitive target is not automatically helpful in the abundant-label regime.

The witness diagnostics still show limited diversity:

```text
H=80:  effective K ~= 1.06, replacement frac = 1.00
H=160: effective K ~= 2.38, replacement frac ~= 0.67
```

So even with `K=4`, the effective witness count is small. This is better than the old `H=64` setting but still not a rich multi-witness regime.

## Important conceptual issue: equality BCE may be too strict

The current transitive loss appears to train the parent directly toward the sampled soft target:

```text
loss = BCE(parent_logit, y_trans)
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

This is reasonable as a first implementation, but it can be biased downward.

Why? Because with finite sampled witnesses and imperfect branch critics:

```text
y_trans <= true R_H(s,a,g)
```

The sampled max-min target is a **lower-bound witness target**, not necessarily an equality target. If `y_trans` is lower than the already-correct parent score, BCE will pull the parent down. That can raise negative scores or reduce gaps in unexpected ways, especially when the branch models are under-confident.

This may explain the current pattern:

```text
Q/V transitive improves Q-V consistency and H=160 gap,
but hurts H=80 gap.
```

The next major ablation should compare equality-style BCE against a lower-bound-style transitive loss.

## Highest-probability next step

Run a small ablation around the Q/V transitive loss form:

```text
A. supervised only
B. Q/V transitive with equality BCE
C. Q/V transitive with lower-bound hinge
D. Q/V transitive with gated BCE only when y_trans > parent_score
```

Do this before policy evaluation.

## Milestone 1: add Q/V transitive loss modes

Add a flag:

```text
--qv_trans_loss_type=bce_equal|prob_hinge|bce_lower_bound
```

### Mode 1: current equality BCE

Current behavior:

```python
loss_qv = BCE(parent_logits, stopgrad(y_trans))
```

Keep this as the baseline.

### Mode 2: probability lower-bound hinge

Use the transitive target as a lower bound:

```python
parent_r = sigmoid(parent_logits)
loss_qv = mean(mask * relu(stopgrad(y_trans) - parent_r) ** 2)
```

This enforces:

```text
R_H(s,a,g) >= max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

without penalizing parent scores that are already above the sampled witness target.

### Mode 3: gated BCE lower bound

Only apply BCE when the target is above the parent:

```python
parent_r = sigmoid(parent_logits)
gate = stop_gradient((y_trans > parent_r + margin).astype(float))
loss_qv = masked_mean(gate * BCE(parent_logits, stopgrad(y_trans)), mask)
```

Start with:

```text
margin = 0.0
```

This keeps BCE's logit-scale gradients but avoids downward pressure.

### Recommendation

Try `prob_hinge` first. It is the cleanest match to the inequality interpretation.

## Milestone 2: add target calibration diagnostics

For Q/V transitive batches, report:

```text
parent_r_mean
qv_y_trans_mean
qv_target_minus_parent_mean
frac_y_trans_gt_parent
frac_y_trans_lt_parent
loss_qv_trans_by_budget
qv_y_trans_mean_by_budget
qv_parent_r_mean_by_budget
qv_first_q_mean_by_budget
qv_second_v_mean_by_budget
```

The key diagnostic is:

```text
frac_y_trans_lt_parent
```

If this is high, equality BCE is mostly pulling parents down. In that case the lower-bound loss is likely the right fix.

## Milestone 3: repeat the no-degradation Q/V run

Use the same setup as the current run:

```text
env=pointmaze-medium-navigate-v0
label=grid_geodesic
geodesic_budget_unit=env_steps
budgets=(40,80,160)
trans_budgets=(80,160)
steps=1000
batch_size=256
eval_pairs=512
K=4
witness_mode=slack_balanced
lambda_qv_trans=0.01
```

Run:

```text
supervised only
bce_equal
prob_hinge
bce_lower_bound
```

Expected result:

```text
lower-bound modes should preserve or improve Q-V consistency
without shrinking H=80 gap as much as equality BCE.
```

If lower-bound modes do not improve the tradeoff, keep equality BCE for now but use a smaller lambda.

## Milestone 4: sparse-Q label-efficiency table

Once the loss mode is chosen, run the sparse-Q table.

### Matrix

```text
sup_pairs_per_budget in {256,64,32,16}
trans_pairs_per_update = 256
lambda_qv_trans in {0.0,0.01,0.025}
qv_trans_loss_type in {chosen_best}
num_trans_witnesses = 4
trans_budgets = (80,160)
seeds = {0,1,2}
```

Use the same frozen V teacher for all rows.

### Metrics

Report:

```text
heldout AUC/gap by H
BCE/calibration by H
monotonicity violation
Q-V-next abs prob diff
Q-V-next rank correlation
steps to threshold
witness effective K
frac_y_trans_gt_parent
frac_y_trans_lt_parent
```

### Success criteria

A good BMM result is any of:

```text
higher AUC/gap under 8x or 16x fewer direct Q labels
fewer steps to threshold
better Q-V consistency with no AUC/gap degradation
better calibration at similar AUC
```

Do not require Q/V transitive to beat abundant supervised Q.

## Milestone 5: improve witness diversity only if needed

The latest witness stats still show small effective K. If sparse results are mixed, try one cleaner geometry setting before going to large:

```text
geodesic_budget_unit=grid_cells
budgets=(2,4,8)
trans_budgets=(4,8)
```

This is not policy-facing, but it is a clean algebra diagnostic. It removes env-step calibration artifacts and directly tests BMM on the maze topology.

For policy-facing env-step budgets, keep:

```text
budgets=(40,80,160)
trans_budgets=(80,160)
```

because this is cell-aligned enough for medium.

## Milestone 6: main-path integration after sparse-Q table

If Q/V transitive is no-degradation and gives either sparse-label or Q-V-consistency benefits, then integrate into the main data path.

Add config fields:

```python
reachability_label_type = "grid_geodesic"  # logged_offset, graph, grid_geodesic
geodesic_budget_unit = "env_steps"         # env_steps, grid_cells
geodesic_mode = "q_next"                   # value, q_next
lambda_qv_trans = 0.01
qv_trans_loss_type = "prob_hinge"          # or chosen best
trans_budgets = (80,160)
num_trans_witnesses = 4
trans_witness_mode = "slack_balanced"
```

Keep the standalone scripts for diagnostics, but make the main path capable of producing the same fields.

## Milestone 7: policy smoke after diagnostic gates

Policy smoke should wait until:

```text
1. Q supervised passes.
2. Q/V transitive no-degradation passes.
3. Sparse-Q table has at least one useful BMM signal.
4. Main-path geodesic label fields match standalone script metrics.
```

Then test action selection with budget scan:

```text
H_hat(s,g) = smallest H with V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Do not score every action at max budget. A max-budget critic is often too coarse to rank actions, which was already identified as a known risk in the original prototype plan.

## Is this still on track with the research idea?

Yes. You are now directly testing the central BMM idea:

```text
Can max-min transitive consistency improve sample efficiency or consistency for a clean budgeted reachability target?
```

The project has moved through the right stages:

```text
logged-offset failure diagnosed
clean geodesic V learned
clean geodesic Q learned
state-only transitive no-degradation shown
Q/V transitive implemented and no-degradation shown
```

This is very much on track. The next step is to make the transitive loss behave like a robust lower-bound consistency constraint, then test sparse supervision.

## What not to do next

Do not yet:

```text
start policy evaluation
move to PointMaze large
reintroduce logged-offset hard negatives
add ranking or monotonicity penalties
interpret the single Q/V run as a final BMM label-efficiency result
```

PointMaze large is valuable later, but first finish the medium sparse-Q story.

## Immediate run order

1. Add `--qv_trans_loss_type` with `bce_equal`, `prob_hinge`, and `bce_lower_bound`.
2. Add Q/V target-vs-parent diagnostics.
3. Repeat current Q/V no-degradation run with the three loss modes.
4. Pick the best no-degradation mode.
5. Run sparse-Q table with `sup_pairs_per_budget={256,64,32,16}` and `trans_pairs_per_update=256`.
6. Use seeds `{0,1,2}` for the final sparse table.
7. If sparse-Q shows a useful signal, integrate geodesic Q/V fields into the main dataset path.
8. Then start policy smoke.

## Bottom line

Frozen-teacher Q/V transitive is the right direction. The current result is a successful first no-degradation check, not yet the final payoff.

The most likely way to make progress now is to switch from treating sampled max-min as an equality target to treating it as a lower-bound consistency target, then evaluate sparse-Q label efficiency.
