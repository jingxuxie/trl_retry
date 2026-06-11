# BMM-TRL next steps after budget-holdout replication

Date: 2026-06-11

This plan follows `BMM_TRL_BUDGET_HOLDOUT_REPLICATION_RESULTS_20260611_014354.md`.

## Executive conclusion

You are making real progress.

This is the first robust result that directly supports the core BMM research idea:

```text
Shorter-budget Q/V knowledge helps a longer heldout parent budget.
```

The result now replicated over three seeds in two budget units:

```text
Grid-cell H8 holdout:
B-A mean delta = +0.0150 AUC, +0.0751 gap, -0.3609 BCE, -0.0575 ECE
D-C mean delta = +0.0116 AUC, +0.0507 gap, -0.1503 BCE, -0.0758 ECE
F-A near zero

Env-step H160 holdout:
B-A mean delta = +0.0243 AUC, +0.0647 gap, -0.5815 BCE, -0.0380 ECE
D-C mean delta = +0.0041 AUC, +0.0498 gap, -0.1815 BCE, -0.0547 ECE
F-A much smaller than B-A
```

This is no longer just shape debugging or tuning. This is a credible critic-level proof of concept for BMM-style transitive bootstrapping.

## Why progress felt slow

The slow part was not that BMM was inherently impossible to train. The slow part was that the first few diagnostics were not actually testing the core hypothesis.

The project had to eliminate several misleading targets:

1. Logged same-trajectory offset was behavior time, not high-budget reachability.
2. PointMaze-medium `H=256/512` were above the calibrated geodesic diameter and therefore one-class under true reachability.
3. Uniform sparse-label reduction was not the right stress test for transitive bootstrapping.
4. Action-conditioned Q needed a next-state geodesic label.
5. Sampled max-min should be treated as a lower-bound consistency target, not necessarily an equality target.

The new budget-holdout experiment finally matches the intended theory:

```text
Use shorter reachable branches to improve a longer parent budget.
```

That is why the result is cleaner than the previous broad sparse-Q table.

## Stop running broad combinations

Yes, you should use fewer combinations now.

Freeze this as the default BMM diagnostic setting unless a specific failure appears:

```text
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
trans_pairs_per_update = 256
```

Stop sweeping these for now:

```text
lambda_qv_trans
qv_trans_loss_type
num_trans_witnesses
uniform sparse label fractions
abundant-label loss modes
```

The time-consuming phase should now end. The next steps should be small, targeted, and decision-oriented.

## Immediate next step: package the critic result

Before more training, turn the budget-holdout result into a compact artifact.

### Add/update summary file

Create:

```text
BMM_TRL_BUDGET_HOLDOUT_SUMMARY.md
```

Include only:

```text
1. The motivation: uniform sparse labels are not the right test.
2. The budget-holdout design.
3. Grid-cell H8 3-seed aggregate.
4. Env-step H160 3-seed aggregate.
5. V-next distillation control showing near-zero effect.
6. The conclusion: BMM-specific transitive bootstrapping improves heldout parent budgets.
```

This should be a short research-note style document, not another long debugging log.

### Add one simple non-BMM interpolation baseline

Before policy, add a cheap baseline that requires no training:

```text
score_Hparent_interp = max(score_Hshort1, score_Hshort2)
```

For grid cells:

```text
score_H8_interp = max(score_H2, score_H4)
```

For env steps:

```text
score_H160_interp = max(score_H40, score_H80)
```

If this baseline is below BMM transitive, the story becomes cleaner:

```text
BMM is not merely monotonic extrapolation from smaller budgets.
```

This should be computed from existing saved predictions if possible. Do not launch new training for it.

## Next research question: does the better critic give better action ranking?

Do not jump straight to full policy benchmark. First run a cheap policy-facing diagnostic.

### Add offline action-ranking diagnostic

Create:

```text
scripts/eval_bmm_action_ranking.py
```

Goal:

```text
Does Q_H(s,a,g) rank better actions higher?
```

Use only dataset transitions and grid/geodesic oracle distances.

For each sampled goal and source cell:

1. Collect several logged transitions from the same or nearby source cell.
2. Treat their actions as candidate actions.
3. For each candidate transition, compute oracle next-state distance to goal:

```text
d_next = d_grid(s_next_candidate, g)
```

4. Define the oracle better action as smaller `d_next`, or label success by:

```text
1[d_next <= H_action - 1]
```

5. Score candidates with:

```text
Q_H_action(s, a_candidate, g)
```

6. Report:

```text
pairwise ranking accuracy
AUC over candidate actions
mean oracle distance of selected action
fraction selected action improves distance
comparison to behavior logged action
comparison to random candidate
```

This is much cheaper than environment policy evaluation and tells whether the learned Q is useful for action selection.

### Compare three critics

Use the same heldout action-ranking set:

```text
A. Q supervised only
B. Q + Q/V transitive
F. Q + V-next distillation
```

Focus on the budget-holdout trained critics. If B beats A and F on action ranking, the BMM signal is now policy-relevant.

## Minimal policy smoke after action-ranking

If action-ranking is positive, run a small policy smoke. Do not run a full benchmark yet.

### Policy selection rule

Use budget scan:

```text
H_hat(s,g) = smallest H where V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Do not score actions at max budget only. The prototype spec already identified max-budget action scoring as too coarse for ranking.

### Candidate actions

Use the fastest available candidate-action source:

```text
1. FRS actor candidates if the actor path is already available.
2. Behavior-policy sampled candidates if available.
3. Nearest-neighbor dataset action candidates from the current/nearby cell for a diagnostic-only smoke.
```

### First smoke goal

Do not ask:

```text
Does it beat TRL/GCIQL?
```

Ask:

```text
Does BMM action scoring produce non-random, locally sensible behavior?
```

Report:

```text
success rate
average final distance to goal
average first-step distance improvement
budget selected by H_hat
failure examples
```

Run one seed first.

## Main-path integration, but only the frozen setting

Integrate only the frozen winning configuration into the main path:

```text
reachability_label_type = grid_geodesic
geodesic_budget_unit = env_steps
geodesic_mode = q_next
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
```

Do not expose or tune all experimental flags in the main experiment. Keep broad tuning in standalone diagnostic scripts.

## PointMaze large: wait until action ranking or policy smoke

PointMaze large is now reasonable, but only after one of these is true:

```text
1. action-ranking diagnostic shows BMM Q/V transitive improves action ranking; or
2. policy smoke on medium shows nonzero/sensible behavior.
```

When moving to large, do not repeat all earlier sweeps.

Run only:

```text
budget-holdout A/B/F on one seed
then action-ranking diagnostic
then policy smoke if positive
```

First inspect geometry:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Choose budgets from actual class coverage.

## Go / no-go criteria

### Continue toward policy if

```text
budget-holdout summary remains positive
and action-ranking BMM > supervised-only and V-next distill
```

### Pivot if

```text
action-ranking shows no improvement despite critic holdout gains
```

Then pivot to:

```text
BMM as a high-level graph/subgoal planner
rather than a flat action-ranking critic.
```

This would still be a valid research direction and is aligned with the original longer-term idea of plugging BMM into HIGL-style high-level subgoal selection.

## How to reduce runtime from now on

Use this rule:

```text
Every new question gets one seed and only A/B/F first.
Only expand to three seeds if B beats A and F is near A.
```

Default comparison set:

```text
A: supervised/no transitive
B: Q/V transitive
F: V-next distill control
```

Only add C/D if parent-label scarcity is the specific question.
Only add oracle G if B fails and you need to know whether branch quality is the bottleneck.

Stop full matrices unless preparing a final figure.

## Immediate task list

1. Create `BMM_TRL_BUDGET_HOLDOUT_SUMMARY.md` from the 3-seed aggregates.
2. Add the monotone interpolation baseline from existing predictions if feasible.
3. Implement `scripts/eval_bmm_action_ranking.py`.
4. Evaluate A/B/F critics on the same offline action-ranking set.
5. If B wins, run one medium policy smoke with budget scan.
6. If policy smoke is sensible, then inspect PointMaze large geometry and repeat only A/B/F.

## Bottom line

The replication result is the strongest evidence so far. You are getting closer to the research goal.

The project should now transition from open-ended prototyping to a narrow sequence:

```text
summarize critic result -> action-ranking diagnostic -> one policy smoke -> larger maze only if justified
```

This avoids another long round of expensive combinations while still testing whether the BMM critic improvement matters for control.
