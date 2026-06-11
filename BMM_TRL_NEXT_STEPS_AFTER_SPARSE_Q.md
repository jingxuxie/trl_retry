# BMM-TRL next steps after sparse-Q diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_SPARSE_Q_RESULTS_20260610_233416.md`.

## Executive conclusion

You are still moving in the correct direction, but the latest sparse-Q table is **mixed**, not a clean BMM win.

The current evidence says:

```text
1. Q/V transitive is stable.
2. Q/V transitive often improves Q-V consistency.
3. Q/V transitive does not yet produce a clean sparse-label AUC/gap improvement.
4. Very sparse Q labels, especially sup16, are not rescued by the current transitive setup.
```

This is not a reason to give up. It is a reason to stop running broad sparse tables and switch to a more decisive experiment.

The next decisive question should be:

```text
Can BMM transitive consistency teach or improve a longer-horizon budget when direct labels for that parent budget are withheld or heavily reduced?
```

This is closer to the original research hypothesis than reducing all budgets uniformly.

## Why the latest results are still useful

The env-step sparse table shows the strongest positive signal at `sup64`:

```text
sup64 + Q/V improves Q-V abs diff from 0.2007 to 0.1671 or 0.1623
sup64 + Q/V modestly improves H40/H80 AUC
sup64 + Q/V improves H160 gap
```

But it is not a clean sparse-label win because:

```text
H40/H80 gaps drop
sup32 hurts H80/H160 AUC and Q-V consistency
sup16 is clearly worse on AUC and Q-V consistency
```

The grid-cell table tells a similar story:

```text
sup64 improves Q-V consistency and some low-budget metrics
sup32 is mixed or worse
sup16 is mostly neutral to slightly positive on gaps, but not enough to pass
```

This means the current Q/V transitive target is useful as a consistency regularizer, but not yet strong enough to rescue uniformly sparse direct Q labels.

## Do not overinterpret the negative parts

The current sparse protocol reduces direct labels for **all budgets** at once:

```text
H40 / H80 / H160 all get fewer labels
```

But the BMM hypothesis is more specific. It should help most when:

```text
shorter-budget labels are available,
longer-budget parent labels are scarce or missing,
and valid witnesses compose shorter reachability into longer reachability.
```

So the next experiment should not be another full sparse table. It should be a **budget-holdout / long-horizon bootstrapping** test.

## Highest-priority next experiment: budget-holdout bootstrapping

### Goal

Test whether transitive consistency can learn a parent budget that has few or no direct supervised labels.

### Env-step version

Use PointMaze medium with cell-aligned budgets:

```text
budgets = (40, 80, 160)
supervised_budgets = (40, 80)
heldout_parent_budget = 160
trans_budgets = (160)
```

Training variants:

```text
A. supervised H40,H80 only; no H160 labels; no transitive
B. supervised H40,H80 + Q/V transitive parent H160
C. supervised H40,H80 + few H160 labels, no transitive
D. supervised H40,H80 + few H160 labels + Q/V transitive
E. full supervised H40,H80,H160 upper bound
```

Evaluation:

```text
H40, H80, H160 heldout AUC/gap/BCE/ECE
Q-V-next consistency by budget
monotonicity
```

Main success criterion:

```text
Variant B or D improves H160 over its matching non-transitive baseline.
```

This is a much better BMM test than reducing all labels uniformly.

### Grid-cell version

Use the clean algebra setting:

```text
geodesic_budget_unit = grid_cells
budgets = (2, 4, 8)
supervised_budgets = (2, 4)
heldout_parent_budget = 8
trans_budgets = (8)
```

This removes env-step calibration artifacts and directly tests max-min composition on the maze topology.

If BMM cannot improve `H8` when `H2/H4` are learned, then the current neural transitive setup is not delivering the intended bootstrapping signal.

## Add a simple distillation control

The current Q/V transitive improves Q-V consistency. To show the BMM-specific max-min target matters, compare it to a simpler baseline:

```text
Q_H(s,a,g) ~= V_{H-1}(s_next,g)
```

Add a control loss:

```text
loss_q_vnext_distill = BCE(Q_H(s,a,g), stopgrad(V_{H-1}(s_next,g)))
```

or lower-bound/hinge equivalent.

Run:

```text
A. supervised only
B. supervised + Q/V transitive
C. supervised + direct V-next distillation
D. supervised + both
```

Interpretation:

- If direct V-next distillation matches or beats Q/V transitive, then the current benefit is mostly Bellman teacher consistency, not the max-min witness structure.
- If Q/V transitive beats V-next distillation in budget-holdout or sparse-parent settings, that is a real BMM-specific signal.

This control is important before policy claims.

## Add an oracle-branch upper-bound diagnostic

Before more tuning, estimate the maximum possible value of the transitive path.

Add Q/V transitive branch modes:

```text
qv_branch_mode = learned_q_frozen_v | oracle_q_frozen_v | oracle_q_oracle_v
```

### Oracle first branch

For `Q_h(s,a,w)`, use the geodesic label:

```text
1[d(s_next,w) <= h-1]
```

### Oracle second branch

For `V_{H-h}(w,g)`, use:

```text
1[d(w,g) <= H-h]
```

Then:

```text
y_oracle_trans = max_w min(oracle_left, oracle_right)
```

Use this only as a diagnostic target, not as the final method.

Interpretation:

- If oracle-branch transitive helps heldout parent budgets, the idea is good and learned branch quality is the bottleneck.
- If oracle-branch transitive also does not help, the current parent training objective or architecture is the bottleneck.
- If oracle-branch transitive trivially works but learned-branch transitive fails, focus on better branch pretraining, target networks, or teacher quality.

## Suggested implementation changes

### 1. Budget-holdout flags

Add to `scripts/train_bmm_geodesic_q.py` or a new runner:

```text
--supervised_budgets="(40,80)"
--eval_budgets="(40,80,160)"
--trans_budgets="(160,)"
--parent_label_budget_frac=0.0   # fraction of direct labels for heldout parent budget
```

For the grid-cell version:

```text
--geodesic_budget_unit=grid_cells
--supervised_budgets="(2,4)"
--eval_budgets="(2,4,8)"
--trans_budgets="(8,)"
```

### 2. V-next distillation flag

Add:

```text
--lambda_vnext_distill
--vnext_distill_loss_type=bce_equal|prob_hinge|bce_lower_bound
```

Use the same frozen V teacher already loaded for Q/V transitive.

### 3. Oracle branch diagnostic flag

Add:

```text
--qv_branch_mode=learned_q_frozen_v|oracle_q_frozen_v|oracle_q_oracle_v
```

For oracle branches, compute targets from grid distances already available in the sampler.

### 4. Parent-budget reporting

Always report parent-budget-specific rows:

```text
H_parent AUC/gap/BCE/ECE
Q-V abs diff for H_parent
qv_y_trans_mean_Hparent
qv_parent_r_mean_Hparent
frac_y_trans_gt_parent_Hparent
witness effective K_Hparent
```

Budget-holdout experiments should be judged mainly by the heldout parent budget.

## Minimal next run matrix

Start with grid-cell mode because it is the cleanest algebra test.

### Grid-cell budget-holdout, seed 0

```text
budgets=(2,4,8)
supervised_budgets=(2,4)
eval_budgets=(2,4,8)
trans_budgets=(8)
```

Rows:

```text
A. no H8 labels, no transitive
B. no H8 labels + Q/V transitive
C. 16 H8 labels/update, no transitive
D. 16 H8 labels/update + Q/V transitive
E. full supervised labels upper bound
F. no H8 labels + V-next distillation
G. no H8 labels + oracle-branch transitive
```

If B or D improves H8 relative to A or C, repeat with seeds `{1,2}`.

### Env-step budget-holdout, seed 0

```text
budgets=(40,80,160)
supervised_budgets=(40,80)
eval_budgets=(40,80,160)
trans_budgets=(160)
```

Run the same rows after grid-cell results are understood.

## When to move to policy evaluation

Policy smoke should wait until one of these is true:

```text
1. budget-holdout BMM improves the heldout parent budget; or
2. Q/V transitive consistently improves Q-V consistency without hurting AUC/gap, and you explicitly want to test whether consistency alone improves policy.
```

If neither is true, policy evaluation is likely to be noisy and hard to interpret.

## When to move to PointMaze large

Move to larger mazes after the budget-holdout diagnostic.

Large is useful if:

```text
medium budget-holdout works,
and you want a setting where parent budgets like 256/512 are not one-class.
```

Before training large:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Choose budgets from actual class coverage, not from habit.

## What if budget-holdout fails?

If budget-holdout fails even with oracle branches, then the current neural BMM setup is probably not giving useful extra supervision on this diagnostic. That does not mean the idea is dead, but it means the current version should pivot.

Possible pivots:

```text
1. Use BMM as a graph/planning module rather than a pure neural regularizer.
2. Use BMM targets only for high-level subgoal selection, not flat action scoring.
3. Use the reachability classifier for budget scan and planning, while actor learning remains BC/IQL-style.
4. Move to a tabular/graph neural demonstration of the logarithmic error story before more OGBench policy work.
```

This would still be a coherent research direction.

## How to think about the pessimism

The prototype has not yet shown the final payoff, but it has made real progress:

```text
1. Logged-offset high-budget labels were shown to be flawed.
2. Clean geodesic V labels were learned.
3. Clean geodesic Q labels were learned.
4. Q/V transitive runs end-to-end.
5. Q/V transitive improves Q-V consistency.
6. Sparse uniform-label reduction did not produce a clean win, which narrows the next question.
```

The next experiment is much more decisive than another sparse table. It directly asks whether BMM can propagate shorter-budget knowledge to a longer-budget parent. That is the core idea.

## Immediate next task list

1. Add budget-holdout support to the Q training script.
2. Add V-next distillation baseline.
3. Add oracle-branch transitive diagnostic.
4. Run grid-cell budget-holdout `(2,4)->8`, seed 0.
5. If positive, repeat seeds `{1,2}` and then run env-step `(40,80)->160`.
6. If negative, run oracle-branch diagnostic and decide whether to pivot toward graph/planning BMM.
7. Only after a positive budget-holdout or a deliberate consistency-only policy hypothesis should policy smoke begin.

## Bottom line

You are not wasting time. The current sparse-Q results are mixed, but they are informative. They tell us that uniform sparse-label reduction is not the best place to look for the BMM advantage.

The most likely next progress comes from a budget-holdout bootstrapping test:

```text
Can shorter-budget Q/V knowledge teach a longer-budget parent when direct parent labels are missing or scarce?
```

That is the cleanest next test of the BMM research idea.
