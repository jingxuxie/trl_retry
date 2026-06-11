# BMM-TRL next steps after joint action-subgoal diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_JOINT_ACTION_SUBGOAL_RESULTS_20260611_133210.md`.

## Executive decision

The latest result gives a clearer decision:

```text
Continue flat Q extraction: no.
Stop the whole project: no.
Continue only as high-level BMM reachability / subgoal planning: yes, conditionally.
```

The joint `(a,w)` diagnostic is the first policy-facing diagnostic where BMM Q/V beats A/F on the metrics it should affect:

```text
B > A/F on state-valid fraction.
B > A/F on action-valid fraction.
B > A/F on source/next path stretch.
B > A/F on midpoint/action-midpoint error.
```

But the absolute action-valid rate is still very low because the candidate set is weak:

```text
oracle any action-valid fraction = 0.0547
```

That means the current cached candidate-action set rarely contains a truly valid action-subgoal pair. Policy smoke would be premature.

## What this result means

### Positive signal

At 512 queries, own-state mean scoring shows:

```text
A Q/V action-valid: 0.0039
B Q/V action-valid: 0.0098
F Q/V action-valid: 0.0039

A Q/V state-valid: 0.0586
B Q/V state-valid: 0.0801
F Q/V state-valid: 0.0586
```

B also has better source/next path stretch and lower midpoint/action-midpoint error than A/F.

This supports the pivot hypothesis:

```text
BMM is more promising as a high-level action-subgoal selector than as a flat Q scorer.
```

### Negative / limiting signal

The candidate set is too weak for policy conclusions:

```text
oracle any action-valid = 0.0547
```

Even an oracle selector can only find an action-valid candidate in about 5.5% of queries. Learned policies cannot look good if useful candidate actions are rarely present.

Also, the V/V teacher is still much stronger than learned Q/V on state-valid and midpoint metrics, which means the value/subgoal side is better developed than the action-conditioned first branch.

## Decision rule now

Continue for one more focused phase only if the next diagnostic can improve candidate coverage.

```text
Go:    candidate generator has oracle action-valid ceiling >= 0.20 and B still beats A/F.
Pivot: B improves subgoal quality but action-valid remains low; use BMM for high-level subgoal planning with a separate low-level controller.
Stop/pause: oracle/V-teacher baselines are strong, candidate coverage is good, but B does not beat A/F.
```

## Main problem to solve next: candidate-action coverage

The current candidate source is inherited from the action-ranking cache. It was useful for debugging, but it is not enough for a policy-facing joint diagnostic.

The next tests should be **evaluation-only** and should focus on candidate generation.

## Milestone 1: add candidate-generation modes to joint diagnostic

Extend `scripts/eval_bmm_joint_action_subgoal.py` with:

```text
--candidate_action_mode=same_cell_cached|neighbor_cell|directional|oracle_diverse|dataset_global_oracle
```

### same_cell_cached

Current behavior. Keep as baseline.

### neighbor_cell

For each source cell, include logged transitions from:

```text
source cell
adjacent free cells
possibly 2-hop neighboring cells
```

This increases candidate action diversity while still staying local.

### directional

Pick candidate transitions that cover distinct next-cell directions:

```text
north / south / east / west / stay / diagonal-ish / farthest-progress / closest-progress
```

This should increase unique next-cell count and next-distance spread.

### oracle_diverse

Diagnostic-only mode. For each query, choose candidate transitions whose next-state distances to the goal span a wide range, including good and bad actions when available.

This is not a deployable policy sampler. It answers:

```text
If useful candidate actions are present, can BMM select them better than A/F?
```

### dataset_global_oracle

Upper-bound diagnostic. Select candidate transitions from the dataset that maximize action-valid coverage for the query.

This should not be used for policy claims, only to determine whether the learned Q/V score contains useful signal.

## Milestone 2: candidate coverage thresholds

For every candidate mode, report:

```text
oracle any action-valid fraction
oracle any state-valid fraction
unique next-cell count
next-distance spread
source-position spread
oracle best action-midpoint error
oracle best selected distance
```

Use these thresholds:

```text
oracle action-valid < 0.10: diagnostic too weak; do not judge learned critics.
oracle action-valid 0.10-0.20: useful for debugging only.
oracle action-valid >= 0.20: meaningful enough for A/B/F comparison.
```

Run 128 queries first. Only use 512 if coverage is good and B looks better than A/F.

## Milestone 3: rerun A/B/F only on better candidate sets

Use existing checkpoints only:

```text
A: no parent labels, no transitive
B: Q/V transitive
F: V-next distill
V/V teacher
oracle
random
```

No new training.

Compare:

```text
selected action-valid fraction
selected state-valid fraction
source/next path stretch
midpoint/action-midpoint error
selected unique subgoals
selected action diversity
```

Decision:

```text
If B > A/F and approaches V/V teacher under a candidate mode with oracle action-valid >= 0.20:
    run tiny high-level policy smoke.
else:
    do not run policy; pivot to value-only subgoal planning or pause.
```

## Milestone 4: separate high-level subgoal from low-level action

If candidate action coverage remains poor, stop trying to solve action selection through BMM Q directly.

Use a two-level design:

```text
High level: choose subgoal w using BMM/V/V score.
Low level: use BC / existing actor / local controller to reach w.
```

High-level score:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

or, if action candidates are available:

```text
score(a,w) = min(Q_h(s,a,w), V_{H-h}(w,g))
```

But do not require Q to solve all low-level control. This is closer to HIGL-style usage and aligns with the original BMM extension plan.

## Milestone 5: tiny policy smoke only after coverage improves

Policy smoke is allowed only if:

```text
candidate mode has oracle action-valid >= 0.20
B beats A/F on joint diagnostic
```

First policy smoke should be tiny:

```text
one env
one seed
few eval episodes
no baseline comparison yet
```

Policy sketch:

```text
1. Choose H_hat(s,g) by budget scan over V.
2. Choose subgoal w by max_w min(V_h(s,w), V_{H-h}(w,g)).
3. Choose action either by Q_h(s,a,w) over candidates or by existing actor/BC toward w.
4. Replan every few steps.
```

Report:

```text
subgoal distance improvement
final distance to goal
success rate
failure examples
```

Do not claim policy improvement until this smoke is sensible.

## Runtime plan

Use this strict schedule:

```text
<5 min: 128-query cached candidate coverage check
<10 min: A/B/F eval on candidate mode with good coverage
<20 min: 512-query confirmation if 128-query result is positive
>30 min: only for one tiny policy smoke after positive diagnostics
```

Do not run any new training in the next phase.

## What this means for the research goal

The research goal was O(log T)-style error compounding via budgeted max-min reachability. The critic-level budget-holdout results support the compositional bootstrapping story. The current joint result suggests the control interface should likely be:

```text
BMM for reachability / subgoal planning
not flat Q_H(s,a,g) action ranking
```

This is not a failure of the research goal, but it is a narrowing of the viable contribution.

A realistic paper story would now be:

```text
1. Logged-offset labels are behavior-time labels and fail.
2. Support/geodesic reachability gives clean budgeted targets.
3. Max-min BMM improves heldout long-budget reachability bootstrapping.
4. BMM is best used for high-level subgoal/action-subgoal selection, not raw flat action ranking.
```

## Immediate task list

1. Add candidate-action modes to `eval_bmm_joint_action_subgoal.py`.
2. Add coverage summary by candidate mode.
3. Run 128-query coverage-only diagnostics for `same_cell_cached`, `neighbor_cell`, `directional`, and `oracle_diverse`.
4. Pick the first mode with oracle action-valid >= 0.20.
5. Run A/B/F/V-teacher/oracle on that mode with 128 queries.
6. If B > A/F, confirm with 512 queries.
7. If still positive, run one tiny high-level policy smoke.
8. If not, pivot to value-only subgoal planning or pause the neural Q/V policy path.

## Bottom line

You are making progress, but the decision is now narrow:

```text
The BMM critic helps reachability bootstrapping.
The flat Q path is not working.
The joint action-subgoal path has a weak positive signal but is candidate-limited.
```

The next fast test is not more training. It is better candidate generation for the joint diagnostic. If BMM still wins when useful candidates are actually present, continue. If not, pivot away from policy extraction and keep BMM as a high-level planning/reachability method.
