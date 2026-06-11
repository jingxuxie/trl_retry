# BMM-TRL next steps after instrumented transitive sweep

Date: 2026-06-11

This plan follows `BMM_TRL_TRANSITIVE_SWEEP_RESULTS_20260610_202304.md`.

## Executive conclusion

You are still making progress.

The latest results say:

```text
1. The transitive implementation is not obviously broken.
2. All lambda values in the V-transitive sweep pass the threshold gate.
3. Multi-witness K=4 is not very helpful on (64,128) because witness diversity is geometrically limited.
4. Sparse K=4 has a modest positive signal: AUC improves at both H=64 and H=128, although H=64 gap worsens.
```

The witness issue is probably **not a generic code bug**. It is mostly a geometry/budget-alignment problem:

```text
PointMaze medium steps_per_cell ~= 19.8
H=64 -> h=32
h=32 permits only about 1.6 grid cells per branch
```

So valid witnesses for `H=64` are nearly forced. The reported `H=64` witness-cell count of about `1.1` is exactly what this geometry predicts. K=4 mostly resamples the same valid witness, so it cannot give a large multi-witness benefit.

The next best move is:

```text
fix/verify small sampler details -> run cell-aligned V-transitive diagnostics -> then implement Q/V transitive with a frozen V teacher.
```

Do not move to policy evaluation yet.

## Is there a bug?

### No obvious shape or max-min bug

The current multi-witness critic path looks structurally correct:

```text
first_r:  [ensemble, K, B]
second_r: [ensemble, K, B]
y_candidates = min(first_r, second_r)
y_trans = max over K
trans_valids = any valid witness over K
```

This matches the intended prototype direction:

```text
y_trans = max_w min(R_h(s,w), R_{H-h}(w,g))
```

The original spec explicitly anticipated starting with one witness and later adding multi-candidate vectorized max, so the current code is on the intended path.

### The main issue is witness geometry

The latest diagnostic table reports:

```text
H=64:  witness cells ~= 1.1
H=128: witness cells ~= 2.3
```

That means K=4 is not really four independent witnesses. It is often one repeated witness for `H=64`, and only a couple of distinct choices for `H=128`.

This explains:

```text
K=4 slightly hurts H=64
K=4 slightly helps H=128 gap
```

This is not surprising and does not invalidate BMM.

### Small code details to verify/fix

These are not confirmed bugs, but they are worth tightening before more experiments.

#### 1. Endpoint/degenerate witnesses

The current witness condition allows witnesses with:

```text
d(s,w)=0
or
d(w,g)=0
```

Those are valid mathematically, but they can create unhelpful transitive targets. Add a mode:

```text
trans_witness_endpoint_mode = allow | avoid | forbid
```

Recommended default:

```text
avoid
```

Implementation:

```text
try non-endpoint witnesses first:
  d(s,w) > eps and d(w,g) > eps
fallback to endpoints only if needed
```

Keep reporting:

```text
zero_left_frac
zero_right_frac
```

#### 2. Effective K should be reported and used

When `len(witness_cells) < K`, sampling with replacement is fine, but the report should make this explicit:

```text
effective_unique_witness_count
unique_witness_frac
fraction_replacement_used
```

For H=64, effective K is close to 1. This should be treated as a geometry limitation, not a failed K=4 experiment.

#### 3. Invalid target safety

The critic masks invalid witnesses by setting invalid candidates to `-1` before max. Since all sampled parents should have valid witnesses, this should not matter. Still, make the target safer:

```python
y_trans = jnp.clip(y_trans, 0.0, 1.0)
y_trans = jnp.where(trans_valids[None, :] > 0, y_trans, 0.0)
```

This avoids accidental BCE targets outside `[0,1]` if a future sampler produces an all-invalid row.

#### 4. Parent distribution mismatch

The transitive parent distribution may not match the supervised/eval distribution. Add a comparison table:

```text
supervised parent distance / H histogram
transitive parent distance / H histogram
```

If transitive parents are much easier than supervised parents, transitive will mainly change score scale rather than ranking.

## Immediate next experiments

## Milestone 1: cell-aligned V-transitive diagnostic

The current budgets are environment-step budgets, but grid distances are multiples of about `19.8` steps. This makes half splits brittle.

Run a cell-aligned budget diagnostic.

### Recommended budgets

Use:

```text
budgets=(40,80,160)
```

Interpretation:

```text
40  ~= 2 grid cells
80  ~= 4 grid cells
160 ~= 8 grid cells
```

For transitive parents, focus on:

```text
H=80  -> h=40
H=160 -> h=80
```

`H=40` is useful as a branch budget and supervised row, but its transitive half split is small.

### Commands

Run supervised-only:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=grid_geodesic \
  --budgets="(40, 80, 160)" \
  --batch_size=256 \
  --eval_pairs=512 \
  --steps=1000 \
  --eval_interval=500 \
  --lambda_trans=0.0 \
  --agent.value_hidden_dims="(256, 256)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.layer_norm=False \
  --output_json=exp/bmm_grid_value_cell_aligned_sup_40_80_160.json
```

Run transitive:

```bash
conda run -n bmm-trl python scripts/train_bmm_geodesic_value.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --reachability_label_type=grid_geodesic \
  --budgets="(40, 80, 160)" \
  --batch_size=256 \
  --eval_pairs=512 \
  --steps=1000 \
  --eval_interval=500 \
  --lambda_trans=0.01 \
  --num_trans_witnesses=4 \
  --agent.value_hidden_dims="(256, 256)" \
  --agent.actor_hidden_dims="(256, 256)" \
  --agent.layer_norm=False \
  --output_json=exp/bmm_grid_value_cell_aligned_trans_40_80_160_lam001_k4.json
```

Then try:

```text
lambda_trans=0.025
```

if `0.01` is no-degradation.

### What to check

Primary:

```text
heldout AUC/gap by H
monotonicity
```

Witness diagnostics:

```text
witness_cell_count_mean for H=80 and H=160
unique_witness_frac
zero_left_frac / zero_right_frac
left/right slack histograms
```

Expected:

```text
witness diversity should improve over (64,128), especially for H=80/160.
```

If witness diversity does not improve, the issue is not budget alignment; inspect sampler degeneracy.

## Milestone 2: add boundary/slack-balanced witness modes

The current witness sampler is roughly `uniform_valid`. Add:

```text
trans_witness_mode = uniform_valid | avoid_endpoints | slack_balanced | boundary_balanced
```

### avoid_endpoints

Prefer:

```text
d(s,w) > eps
and
d(w,g) > eps
```

Fallback to endpoints only if needed.

### slack_balanced

Prefer witnesses with balanced branch slack:

```text
score(w) = -abs((h - d(s,w)) - ((H-h) - d(w,g)))
```

Sample from the top candidates or with softmax over score.

### boundary_balanced

Prefer witnesses with branch distances not too small:

```text
d(s,w) >= beta * h
d(w,g) >= beta * (H-h)
```

Use:

```text
beta in {0.25, 0.5}
```

Fallback to `avoid_endpoints`, then `uniform_valid`.

### Why this matters

A valid witness can be too easy. If one branch has distance zero or both branches are far from their budget boundary, the target becomes a weak positive regularizer rather than a useful compositional constraint.

## Milestone 3: repeat sparse ablation with stronger scarcity

The current batch-size-32 sparse run is a modest positive signal:

```text
K=4 transitive improves AUC at H=64 and H=128,
improves H=128 gap,
but hurts H=64 gap.
```

This is promising but not enough.

### Better sparse protocol

Keep transitive batch size fixed, reduce direct supervised labels:

```text
sup_pairs_per_budget in {256, 64, 32, 16}
trans_pairs_per_update = 256
```

This is cleaner than only changing the whole batch size, because it directly tests whether transitive consistency supplies extra structure beyond direct labels.

If script changes are needed, add flags:

```text
--sup_pairs_per_budget
--trans_pairs_per_update
```

### Seeds

Use:

```text
seed in {0,1,2}
```

at least for the final sparse table.

### Success criteria

A useful BMM result is any of:

```text
higher AUC/gap under 8x or 16x fewer direct labels
fewer steps to threshold
better monotonicity or calibration at similar AUC
```

Do not require transitive to beat abundant supervised BCE.

## Milestone 4: implement Q/V transitive after the cell-aligned check

Once V-transitive is no-degradation under a less degenerate witness setup, implement Q/V transitive.

Do not wait for a perfect sparse V result. Q/V transitive is the policy-relevant object.

### Correct target

For action-conditioned Q:

```text
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

The first branch consumes the action. The second branch should be a state-only V branch.

### Use a frozen V teacher first

Start with a passed V checkpoint as teacher for the second branch:

```text
Q supervised labels + Q/V transitive with frozen V teacher
```

This isolates Q/V composition from simultaneous V learning.

### Experiments

Run:

```text
Q supervised only
Q supervised + Q/V transitive, lambda_trans=0.01
Q supervised + Q/V transitive, lambda_trans=0.025
```

Budgets:

```text
(64,128)
```

Then cell-aligned:

```text
(40,80,160)
```

if the V-transitive cell-aligned run is cleaner.

### Gate

First gate:

```text
no degradation versus Q supervised-only
```

Second gate:

```text
sparse-Q label efficiency improvement
```

Policy evaluation comes after these gates.

## Milestone 5: only then inspect PointMaze large

Move to PointMaze large only after:

```text
medium V transitive is instrumented and no-degradation
medium Q/V transitive is no-degradation
```

Large is for scale and larger horizons, not for debugging witness sampling.

Before training large:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Use budgets based on actual calibrated diameter and class coverage.

## Code changes most likely to help

### 1. Target safety in `BMMTRLAgent`

After computing multi-witness max:

```python
y_trans = jnp.clip(y_trans, 0.0, 1.0)
y_trans = jnp.where(trans_valids[None, :] > 0, y_trans, 0.0)
```

### 2. Witness sampler flags

Add:

```text
--trans_witness_mode=uniform_valid|avoid_endpoints|slack_balanced|boundary_balanced
--trans_endpoint_epsilon=1e-6
--trans_boundary_beta=0.25
--trans_pairs_per_update=256
--sup_pairs_per_budget=256
```

### 3. Parent-distribution diagnostics

Add to output JSON:

```text
supervised_distance_over_H_hist
transitive_parent_distance_over_H_hist
```

### 4. Effective K diagnostics

Add:

```text
effective_unique_witness_count_mean
replacement_used_frac
```

### 5. Optional grid-cell budget unit diagnostic

For pure algebra diagnostics, add a mode:

```text
geodesic_budget_unit=env_steps|grid_cells
```

In `grid_cells` mode:

```text
budget labels use raw BFS cell distances
budgets=(2,4,8)
```

This removes calibration artifacts and directly tests max-min algebra on the maze topology. Keep `env_steps` for policy-facing diagnostics.

## What not to do next

Do not:

```text
jump directly to policy evaluation
interpret K=4 on H=64 as a failure of multi-witness BMM
move to large maze before Q/V transitive is implemented
spend time on ranking/monotonicity penalties
reintroduce logged-offset hard negatives
```

## Suggested immediate run order

1. Patch target safety and endpoint/effective-K diagnostics.
2. Run cell-aligned supervised-only and transitive on `(40,80,160)`.
3. Add `avoid_endpoints` and `slack_balanced` witness modes.
4. Compare `uniform_valid`, `avoid_endpoints`, and `slack_balanced` on `(40,80,160)`.
5. Run sparse ablation with fixed transitive pairs and reduced supervised pairs.
6. Implement Q/V transitive with frozen V teacher.
7. Run Q/V no-degradation check.
8. Then inspect PointMaze large and choose geometry-appropriate budgets.

## Bottom line

The witness issue is not a reason to be discouraged. It is useful information: the current `(64,128)` medium-maze setup has too little valid witness diversity for multi-witness max-min to show its strengths.

You are making progress because the diagnostics have narrowed the problem to a concrete, fixable question:

```text
Can BMM transitive consistency help when witnesses are nondegenerate and direct labels are scarce?
```

The next experiments should be designed exactly around that question.
