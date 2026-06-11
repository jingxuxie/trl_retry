# BMM-TRL next steps after witness sweep

Date: 2026-06-11

This plan follows `BMM_TRL_TRANSITIVE_SWEEP_RESULTS_20260610_202304.md`.

## Executive conclusion

You are still making progress.

The latest results are not a failure of BMM. They are a useful diagnosis of the current PointMaze-medium transitive setup:

```text
The implementation is mostly sane.
The current (64,128) environment-step budgets have poor witness diversity.
The sparse result is modestly positive but not yet a clean label-efficiency claim.
```

The witness issue is most likely not a major code bug. It is mostly caused by the interaction between calibrated grid distance and the half-split budget:

```text
steps_per_cell ~= 19.8
H=64  -> h=32  -> about 1.6 grid cells per branch
H=128 -> h=64  -> about 3.2 grid cells per branch
```

This naturally creates very few valid witness cells, especially for `H=64`. The reported witness counts confirm this:

```text
H=64:  ~1.1 valid witness cells
H=128: ~2.3 valid witness cells
```

So `K=4` is often not four distinct witnesses. It mostly repeats the same witness at `H=64`.

## What the latest results mean

### Good signs

The lambda sweep shows safe transitive behavior:

```text
lambda_trans in {0.01, 0.025, 0.05} all passed thresholds
AUC was no-degradation or slightly better
monotonicity remained 0.0000
```

The sparse batch-size-32 result is modestly positive:

```text
K=4 transitive improved AUC at both H=64 and H=128
K=4 transitive improved H=128 gap
K=4 transitive worsened H=64 gap by raising negative scores
```

This suggests the transitive target has useful signal, but the current witness geometry is limiting it.

### Not yet proven

You do not yet have a strong BMM label-efficiency result. The current table is one seed, one sparse setting, one environment, and a witness set that is nearly degenerate at `H=64`.

### Main diagnosis

The right question is now:

```text
Can max-min transitive consistency help when witnesses are nondegenerate and direct labels are scarce?
```

The next experiments should be designed around exactly that question.

## Code review: likely bugs or issues

### 1. Multi-witness tensor path looks structurally correct

The agent code path implements:

```text
first_r:      [ensemble, K, B]
second_r:     [ensemble, K, B]
y_candidates: min(first_r, second_r)
y_trans:      max over K witnesses
trans_valids: any valid witness for each parent
```

This matches the intended BMM target:

```text
y_trans = max_w min(R_h(s,w), R_{H-h}(w,g))
```

The original prototype spec explicitly planned to start with one witness and later add a vectorized multi-candidate max. So the implementation direction is aligned with the design.

### 2. Add target safety anyway

The current invalid-witness handling sets invalid candidates to `-1` before the `max`. If every parent has at least one valid witness, this is harmless. Still, add a safety clamp:

```python
y_trans = jnp.clip(y_trans, 0.0, 1.0)
y_trans = jnp.where(trans_valids[None, :] > 0, y_trans, 0.0)
```

This prevents future sampler variants from accidentally feeding BCE targets outside `[0,1]` when a row is all-invalid.

### 3. Degenerate endpoint witnesses are valid but may be unhelpful

Current witness sampling can include:

```text
w = s    -> d(s,w)=0
w = g    -> d(w,g)=0
```

These are mathematically valid. But for learning, they can make transitive consistency act like a weak positive regularizer rather than a real composition constraint.

Add and report:

```text
zero_left_frac
zero_right_frac
effective_unique_witness_count
replacement_used_frac
```

Then add a witness mode that avoids endpoints when possible.

### 4. Parent-distribution mismatch is still unmeasured

The transitive parent distribution may not match the supervised/eval distribution. Add:

```text
supervised_parent_distance_over_H_hist
transitive_parent_distance_over_H_hist
```

If transitive parents are easier or concentrated in a narrow distance band, the transitive loss may mostly alter calibration/score scale rather than ranking.

### 5. Budget calibration may be the core issue

The current budgets are in environment-step units, while grid distances are multiples of roughly `19.8` steps. This makes `H/2` splits awkward.

Add an optional diagnostic mode:

```text
geodesic_budget_unit = env_steps | grid_cells
```

For pure BMM algebra diagnostics, `grid_cells` is cleaner:

```text
budgets=(2,4,8)
```

For policy-facing diagnostics, keep `env_steps`.

## Highest-probability next step

Run a **cell-aligned V-transitive diagnostic** before Q/V transitive.

This is the single most useful next experiment because it tests whether the witness issue is truly budget geometry.

## Milestone 1: cell-aligned V-transitive

### Option A: environment-step cell-aligned budgets

Use:

```text
budgets=(40,80,160)
```

These correspond roughly to:

```text
2, 4, 8 grid cells
```

Important rows:

```text
H=80  -> h=40
H=160 -> h=80
```

`H=40` is mostly a branch/supervised budget; its half split is still small.

### Option B: raw grid-cell budget unit

Add a diagnostic flag:

```text
--geodesic_budget_unit=grid_cells
```

Then run:

```text
budgets=(2,4,8)
```

This is the cleanest test of the BMM algebra on the maze topology because it removes environment-step calibration artifacts.

### Run matrix

For either budget mode:

```text
supervised only
transitive lambda=0.01, K=1
transitive lambda=0.01, K=4
transitive lambda=0.025, K=4
```

Report:

```text
heldout AUC/gap
monotonicity
witness_cell_count_mean
effective_unique_witness_count
zero_left/right frac
left/right slack histograms
loss_trans_over_sup
```

### Expected result

If the witness issue is geometry, then cell-aligned budgets should show:

```text
higher witness_cell_count_mean
higher unique_witness_frac
less H=64-like degeneracy
clearer K=4 behavior
```

If not, the sampler needs deeper debugging.

## Milestone 2: nondegenerate witness sampling

Add:

```text
--trans_witness_mode=uniform_valid|avoid_endpoints|slack_balanced|boundary_balanced
```

### uniform_valid

Current behavior.

### avoid_endpoints

Prefer witnesses satisfying:

```text
d(s,w) > eps
d(w,g) > eps
```

Fallback to endpoints only if no non-endpoint witness exists.

### slack_balanced

Prefer witnesses whose branch slack is balanced:

```text
left_slack  = h - d(s,w)
right_slack = H-h - d(w,g)
score(w) = -abs(left_slack - right_slack)
```

Sample from the top candidates or via softmax over `score`.

### boundary_balanced

Prefer nontrivial branch distances:

```text
d(s,w) >= beta * h
d(w,g) >= beta * (H-h)
```

Start with:

```text
beta=0.25
```

Fallback chain:

```text
boundary_balanced -> avoid_endpoints -> uniform_valid
```

### Success criterion

A witness mode is better if it improves diversity without hurting oracle validity:

```text
branch_oracle_valid_mean stays 1.0
witness_cell_count_mean increases
effective_unique_witness_count increases
AUC/gap no-degradation or better
```

## Milestone 3: sparse-label table with fixed transitive budget

The current sparse test changes `batch_size`, which reduces both supervised and transitive samples. For label-efficiency claims, separate them.

Add flags:

```text
--sup_pairs_per_budget
--trans_pairs_per_update
```

Run:

```text
sup_pairs_per_budget in {256,64,32,16}
trans_pairs_per_update = 256
lambda_trans in {0,0.01,0.025}
num_trans_witnesses in {1,4}
```

Use at least:

```text
seeds = 0,1,2
```

The useful result is not necessarily better abundant-label performance. It is:

```text
better AUC/gap under 8x or 16x fewer supervised labels
fewer steps to threshold
better calibration or monotonicity at similar AUC
```

## Milestone 4: Q/V transitive after one clean V witness run

Do not wait for a perfect V sparse win. Once cell-aligned V transitive is no-degradation and witness diversity is not degenerate, implement Q/V transitive.

### Correct target

```text
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

The first branch consumes the action. The second branch is state-only.

### Use a frozen V teacher first

Start with:

```text
Q supervised labels + Q/V transitive using a frozen passed V checkpoint
```

This isolates Q learning from V learning.

### Run matrix

```text
Q supervised only
Q + Q/V transitive, lambda=0.01
Q + Q/V transitive, lambda=0.025
```

Use the better budget set from Milestone 1:

```text
(40,80,160) or grid-cell (2,4,8)
```

Then repeat on `(64,128)` only as a comparability check.

### Gate

First gate:

```text
no degradation versus Q supervised-only
```

Second gate:

```text
sparse-Q label-efficiency improvement
```

Policy evaluation comes after this.

## Milestone 5: PointMaze large only after Q/V transitive

PointMaze large is useful for horizon scaling, but it should not be used to debug witness sampling.

Move to large after:

```text
cell-aligned V transitive is no-degradation
Q/V transitive is no-degradation
```

Before training, run:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Choose budgets based on the actual calibrated diameter and class coverage.

## Milestone 6: policy smoke after Q/V

Policy evaluation should wait until:

```text
V supervised works
Q supervised works
V transitive is no-degradation
Q/V transitive is no-degradation
```

Then policy smoke should use budget scanning, not max-budget action scoring:

```text
H_hat(s,g) = smallest H where V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

This matches the original warning that a large-budget critic may be too coarse for action ranking.

## Concrete run order

1. Patch target safety in `BMMTRLAgent`.
2. Add `effective_unique_witness_count` and `replacement_used_frac` diagnostics.
3. Add `geodesic_budget_unit=grid_cells` or run env-step cell-aligned budgets `(40,80,160)`.
4. Run supervised-only and `lambda_trans=0.01,K=4` on cell-aligned budgets.
5. Add `avoid_endpoints` witness mode.
6. Add `slack_balanced` witness mode.
7. Run sparse table with fixed transitive pairs and reduced supervised pairs.
8. Implement Q/V transitive with frozen V teacher.
9. Run Q/V no-degradation and sparse-Q checks.
10. Inspect PointMaze large diameter and class coverage.
11. Start policy smoke only after Q/V passes.

## Bottom line

This is progress. The witness issue is not a dead end; it tells us that `(64,128)` on PointMaze medium is not the best setting to demonstrate multi-witness BMM.

The fastest path forward is to make the witness geometry nondegenerate, then ask the real BMM question:

```text
Does max-min transitive consistency improve sample efficiency or budget consistency on a clean reachability target when valid witnesses are diverse enough?
```
