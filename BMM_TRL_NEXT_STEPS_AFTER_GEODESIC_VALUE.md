# BMM-TRL next steps after geodesic value success

Date: 2026-06-11

This plan follows `BMM_TRL_GEODESIC_VALUE_RESULTS_20260610_184508.md`.

## Executive conclusion

Yes, this is real progress.

The current state is no longer:

```text
Can the BMM critic learn PointMaze reachability?
```

It is now:

```text
The BMM critic learns clean geodesic V_H(s,g). How do we turn that into a policy-facing Q_H(s,a,g) and then test max-min consistency?
```

The important milestone you just passed is stronger than fixed-batch overfit. The state-only critic trained on fresh supervised batches and passed heldout layout/grid geodesic labels on PointMaze medium for:

```text
H = 32, 64, 96, 128
```

with final heldout AUCs around `0.93-0.98`, substantial score gaps, and zero monotonicity violation.

This supports the diagnosis that the earlier `H=256/512` failure was a target problem caused by logged behavior-time labels, not a basic BMM implementation problem.

## Priority order

The next priority should be:

```text
1. Action-conditioned geodesic Q labels.
2. Max-min transitive consistency on geodesic-valid witnesses.
3. Medium-maze label-scarcity/sample-efficiency ablations.
4. Larger maze after checking calibrated diameter.
5. Main GCDataset/main.py integration and policy evaluation.
```

Do **not** jump directly to policy evaluation yet. The missing bridge is the action-conditioned finite-horizon Q object.

## Why action-conditioned Q is the next milestone

The state-only diagnostic learned:

```text
V_H(s,g) = 1[d_grid(s,g) <= H]
```

Policy extraction needs an action-conditioned object:

```text
Q_H(s,a,g)
```

The clean Bellman-consistent supervised label is:

```text
Q_H(s_t,a_t,g) = 1[d_grid(s_{t+1}, g) <= H - 1]
```

This makes the action matter through the observed next state and avoids the previous mistake of labeling `Q` by logged future offset.

If this Q diagnostic passes, you have a clean path back to actor extraction:

```text
a = argmax_a Q_H(s,a,g)
```

or candidate-action scoring through FRS.

## Milestone 1: action-conditioned geodesic Q diagnostic

### Add script

Create:

```text
scripts/train_bmm_geodesic_q.py
```

It can mostly copy `scripts/train_bmm_geodesic_value.py`, but change pair sampling and labels.

### Data semantics

For each source transition:

```text
s_t = observations[t]
a_t = actions[t]
s_next = observations[t+1]
g = sampled goal state
H = budget
```

Label:

```text
label_Q = 1[d_grid(s_next, g) <= H - 1]
```

For graph labels, use:

```text
label_Q = 1[d_graph(bin(s_next), bin(g)) <= H - 1]
```

### Pair sampling

Balanced per budget:

```text
positive goals: d(s_next,g) in [0.5*(H-1), H-1]
negative goals: d(s_next,g) in (H-1, 2*(H-1)]
```

Fallbacks:

- if `H-1` is too small, use `max(H-1, 1)` for the sampler;
- skip one-class budgets instead of forcing negatives;
- do not use logged-offset negatives.

### Suggested first command

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

Then run a smaller-budget stress test:

```text
budgets=(16,32,64,128)
```

because action effects matter more at shorter horizons.

### Clean losses

Keep the first Q diagnostic pure:

```text
diagnostic_critic_mode=action
value_only=True
lambda_sup=1.0
lambda_trans=0.0
lambda_rank=0.0
num_rank_pairs=0
lambda_mono=0.0
lambda_pos=0.0
lambda_budget_neg=0.0
lambda_hard_neg=0.0
lambda_rand_hinge=0.0
```

### Report metrics

Report the same metrics as the value script:

```text
AUC
gap
pos_mean
neg_mean
ensemble-min AUC/gap
monotonicity violation
distance-oracle AUC
Euclidean AUC
```

Add one new diagnostic:

```text
Q-V_next consistency
```

For the same `(s_t,a_t,g,H)`, compute:

```text
Q_H(s_t,a_t,g)
V_{H-1}(s_next,g)
```

and report:

```text
mean absolute probability difference
rank correlation
AUC of Q labels using V_next scores if a trained V checkpoint is available
```

This catches off-by-one and transition-indexing bugs.

### Passing threshold

Use the value thresholds as the first gate:

```text
H=32:  AUC >= 0.90, gap >= 0.20
H=64:  AUC >= 0.90, gap >= 0.20
H=96:  AUC >= 0.90, gap >= 0.20
H=128: AUC >= 0.85, gap >= 0.15
monotonicity violation <= 0.02
```

If Q is slightly weaker than V, that is acceptable. If it is near chance, debug next-state indexing and `H-1` calibration before doing anything else.

## Milestone 2: max-min transitive consistency on geodesic labels

After Q passes, add transitive consistency. Do this first in **state-only V mode**, then in action-conditioned Q mode.

### State-only V transitive target

For a pair `(s,g,H)` and split `h`:

```text
y_trans = max_w min(V_h(s,w), V_{H-h}(w,g))
```

Use geodesic-valid witnesses:

```text
d(s,w) <= h
d(w,g) <= H-h
```

### Action-conditioned Q transitive target

For `(s,a,g,H)`:

```text
y_trans = max_w min(Q_h(s,a,w), V_{H-h}(w,g))
```

This is the right decomposition because the first branch consumes the action and the second branch is state-only from the witness.

### Budget choices

Use split-closed budgets first:

```text
16, 32, 64, 128
```

with half splits:

```text
H=32  -> h=16
H=64  -> h=32
H=128 -> h=64
```

If `H=16` is too noisy, start transitive at `H=32` and keep `H=16` only as a branch budget.

### Implementation task

Add a geodesic witness sampler:

```text
sample_geodesic_witnesses(s, g, H, h, K)
```

For grid labels, sample witness cells `w_cell` satisfying:

```text
d_grid(s_cell, w_cell) <= h
d_grid(w_cell, g_cell) <= H-h
```

Then choose a dataset state from that cell as `w`.

For graph labels, use graph distances and bin-to-state lists.

### Loss schedule

First no-degradation test:

```text
lambda_sup=1.0
lambda_trans=0.025
```

Then stronger:

```text
lambda_sup=1.0
lambda_trans=0.05
```

Then label-scarce setting:

```text
num_sup_pairs or pairs_per_budget reduced by 4x or 8x
lambda_trans in {0.025, 0.05, 0.1}
```

### What success should mean

The first transitive test does **not** need to beat supervised BCE when labels are abundant. Success is:

```text
transitive loss does not damage AUC/gap/monotonicity
```

The meaningful BMM test is label scarcity:

```text
with fewer direct labels, transitive consistency improves sample efficiency or budget consistency.
```

## Milestone 3: ablation table for the core BMM claim

Run on PointMaze medium geodesic labels:

```text
A. supervised-only V
B. supervised + V transitive
C. supervised-only Q
D. supervised + Q/V transitive
E. supervised + transitive with 4x fewer labels
F. supervised-only with 4x fewer labels
```

Metrics:

```text
heldout AUC/gap by budget
monotonicity violation
calibration / BCE
budget-threshold accuracy
number of direct labels consumed
training steps to threshold
```

The paper-worthy result is not just high AUC. It is:

```text
BMM transitive consistency improves label efficiency or consistency while preserving clean reachability classification.
```

## Milestone 4: larger maze only after medium Q/transitive passes

After medium state-only and Q transitive diagnostics pass, inspect larger mazes:

```bash
conda run -n bmm-trl python scripts/inspect_pointmaze_grid_bfs.py \
  --env_name=pointmaze-large-navigate-v0 \
  --budgets=64,128,256,512
```

Proceed only after checking class coverage. Choose budgets from the actual calibrated diameter.

If `H=256/512` have both positives and negatives, use them. If not, do not force them. The budget list should be geometry-dependent.

Suggested larger-maze progression:

```text
1. pointmaze-large-navigate-v0, state-only grid V
2. pointmaze-large-navigate-v0, action-conditioned grid Q
3. pointmaze-large-navigate-v0, transitive consistency
4. stitch variant only after navigate works
```

## Milestone 5: integrate into main training path

Right now, `scripts/train_bmm_geodesic_value.py` manually constructs geodesic supervised fields. That is fine for diagnostics. Before policy experiments, integrate the label path into the normal data pipeline.

Add dataset config:

```python
reachability_label_type = "logged_offset"  # logged_offset, grid_geodesic, graph
geodesic_mode = "value"                    # value or q_next
geodesic_graph_path = "exp/bmm_pointmaze_graph.npz"
geodesic_xy_dims = "0,1"
geodesic_budget_set = (16,32,64,128)
```

Modify `GCDataset.add_bmm_supervised_fields` to call:

```text
logged-offset sampler
or grid-geodesic sampler
or graph-geodesic sampler
```

Do this only after the standalone Q diagnostic passes. Scripts are faster to iterate on; main integration is for policy-facing runs.

## Milestone 6: policy-facing experiment

Start only after:

1. state-only grid V passes;
2. action-conditioned grid Q passes;
3. transitive consistency does not harm diagnostics.

Policy extraction should not always use `max_budget`. Use a budget scan:

```text
H_hat(s,g) = smallest H such that V_H(s,g) >= tau
H_action = clamp(H_hat(s,g), min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

For FRS, sample candidate actions and score them with `Q_H_action`.

First policy goal should be smoke-level:

```text
Does BMM action scoring choose locally sensible actions and get nonzero success?
```

Do not compare final performance against TRL until the policy path is stable.

## Code-review notes from the current state

### Good

- `train_bmm_geodesic_value.py` trains on fresh geodesic pairs each update and evaluates on a fixed heldout validation pair set.
- The current value run uses clean losses: no transitive, no rank, no monotonicity, no hard negatives.
- Grid context reports maze diameter and calibration, which prevents repeating the `H=256/512` mistake.
- Monotonicity is already perfect in the successful state-only run.

### Watch-outs

- For Q labels, verify valid non-terminal source indices. `s_next` must stay in the same trajectory.
- Be explicit whether budget `H` includes the first action. This plan assumes yes, so the remaining state-only budget is `H-1`.
- For grid calibration, `H-1` is tiny compared with `steps_per_cell ~= 19.8`; this is fine because budgets are much larger than one step, but avoid `H=1/2/4` grid-geodesic diagnostics on PointMaze medium.
- Do not re-enable `lambda_rank`, `lambda_mono`, or logged-offset hard negatives during geodesic Q debugging.
- Keep one-class budgets as skipped, not failed.

## Immediate next run order

1. Implement `scripts/train_bmm_geodesic_q.py` by adapting the value script.
2. Run grid-geodesic Q on PointMaze medium with budgets `(32,64,96,128)`.
3. Run a shorter-horizon Q stress test with `(16,32,64,128)` if sampling supports it.
4. Add geodesic-valid witness sampling for state-only V transitive consistency.
5. Run supervised-only vs supervised+transitive under abundant labels to verify no degradation.
6. Run the same comparison under label scarcity to test whether BMM helps.
7. Repeat on action-conditioned Q/V transitive consistency.
8. Inspect PointMaze large diameter and class coverage.
9. Integrate geodesic label types into the main dataset path.
10. Start policy-facing smoke only after Q and transitive diagnostics pass.

## Bottom line

You are making progress. The result you just got is exactly the kind of checkpoint needed before returning to BMM max-min. The next decisive experiment is action-conditioned geodesic Q, not another state-only value run and not policy evaluation.

Once Q passes, the core BMM question becomes testable:

```text
Does max-min transitive consistency improve sample efficiency or budget consistency on a clean reachability target?
```
