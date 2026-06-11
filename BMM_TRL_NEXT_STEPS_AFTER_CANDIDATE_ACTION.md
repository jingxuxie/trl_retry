# BMM-TRL next steps after candidate-action diagnostics

Date: 2026-06-11

This plan follows `BMM_TRL_CANDIDATE_ACTION_RESULTS_20260611_135918.md`.

## Executive decision

The latest candidate-action diagnostics give a clear decision:

```text
Continue flat Q extraction: no.
Run policy smoke now: no.
Stop the whole project: no.
Pivot: yes.
```

Use BMM primarily for:

```text
high-level reachability / subgoal planning
```

and pair it with:

```text
a separate low-level controller, BC policy, or action proposal mechanism.
```

The current neural Q/V first-branch action selector should remain diagnostic-only until it shows a much larger and more consistent advantage.

## What the latest results mean

### Candidate coverage is no longer the blocker

The improved candidate-action modes show that useful candidates can be made available:

```text
same_cell_cached oracle action-valid: 0.0547
neighbor_cell oracle action-valid:   1.0000
directional oracle action-valid:     0.9922
oracle_diverse oracle action-valid:  1.0000
```

So the earlier weak action-ranking result was partly caused by the same-cell candidate set, but not entirely.

### BMM Q/V does not robustly win the full joint objective

With useful candidates present, BMM Q/V improves some action-facing metrics:

```text
higher action-valid rate
lower action-midpoint error
```

but it does not consistently improve:

```text
state-valid rate
path stretch
midpoint error
subgoal quality
```

In `oracle_diverse`, all methods can choose action-valid candidates, but BMM does not improve subgoal/midpoint quality over A/F. This weakens the case for continuing neural Q/V policy extraction.

## Main conclusion

The BMM critic still has a coherent role as a reachability/subgoal-planning module. The max-min structure is naturally about choosing a witness/subgoal:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

Trying to make the same module solve flat one-step action selection is not paying off in the current prototype.

## Recommended pivot

Move to a two-level design:

```text
High level: BMM/V selects a subgoal w.
Low level: a separate controller reaches w.
```

The low-level controller can be:

```text
BC / goal-conditioned BC
existing actor from the repo
nearest-neighbor local action policy
a short-horizon Q/critic trained only for local control
```

This is closer to the original longer-term BMM plan, which included budget scan and HIGL-style high-level subgoal selection.

## Next diagnostic: value-only high-level subgoal smoke

Do not train anything new yet. Use existing value critics/checkpoints and cached query sets.

### Goal

Test whether BMM/V subgoal selection produces useful intermediate goals.

### Score

For each `(s,g,H)` and candidate subgoal `w`:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

Use `H=160`, `h=80`, or the graph/budget setting already used in diagnostics.

### Compare

```text
random subgoal
oracle midpoint
V/V teacher
A-derived V if available
BMM/V score
```

If only the V teacher is currently available, start with:

```text
V/V teacher vs random/oracle
```

This answers whether the high-level planning interface is viable independent of Q.

### Metrics

Report:

```text
state-valid fraction
path stretch
d(s,w)
d(w,g)
midpoint error
selected subgoal cell diversity
progress-to-goal if a local controller moves toward w
```

## Low-level controller diagnostic

Before running environment policy, evaluate whether the proposed subgoals are reachable by a simple local controller.

### Option A: dataset nearest-neighbor controller

For a selected subgoal `w`, choose a logged action from nearby states whose next state reduces distance to `w`.

Metrics:

```text
first-step distance improvement to w
fraction reducing distance to w
next-state distance to w
```

### Option B: existing goal-conditioned actor / BC

Use the repo actor or a simple BC policy conditioned on `w`.

Metrics:

```text
short rollout distance-to-subgoal reduction
subgoal reach rate within K steps
```

This separates:

```text
subgoal quality
```

from:

```text
low-level control quality
```

## Tiny policy smoke only after two checks

Run a tiny policy smoke only if both are true:

```text
1. BMM/V subgoals are better than random and competitive with V/V teacher.
2. A simple low-level controller makes progress toward selected subgoals.
```

Policy sketch:

```text
1. Estimate H_hat(s,g) by budget scan over V.
2. Choose subgoal w = argmax_w min(V_h(s,w), V_{H-h}(w,g)).
3. Use low-level controller toward w for K steps.
4. Replan.
```

Report:

```text
subgoal progress
final distance to goal
success rate as secondary
failure examples
```

Use one seed and a small number of episodes only.

## Runtime rule

From now on:

```text
No new training sweep.
No broad A-G matrix.
No policy benchmark until high-level diagnostics pass.
```

Use:

```text
cached queries
existing checkpoints
one seed
128 queries first
512 only if positive
```

## Go / pause criteria

### Continue as high-level BMM project if

```text
BMM/V subgoal selection beats random and simple baselines,
and a low-level controller can make progress toward selected subgoals.
```

### Pause neural Q/V policy extraction if

```text
Q/V action-subgoal selection remains mixed even with strong candidates.
```

This is already the current state.

### Pause the whole project if

```text
BMM/V subgoal selection is not useful,
low-level controller cannot exploit selected subgoals,
and only critic-level budget-holdout metrics remain positive.
```

## Immediate task list

1. Implement or clean up a value-only high-level subgoal diagnostic.
2. Reuse existing V checkpoints and cached query sets.
3. Report random/oracle/V/V teacher subgoal quality.
4. Add a nearest-neighbor or BC low-level controller diagnostic toward selected subgoals.
5. Only if both are positive, run one tiny high-level policy smoke.
6. Do not run more Q/V policy-facing sweeps unless a new action-selection idea appears.

## Bottom line

The evidence is clear enough to pivot:

```text
BMM helps reachability bootstrapping.
BMM does not yet give robust flat or joint neural action selection.
BMM is most plausible as a high-level subgoal/reachability planner.
```

The next useful test is not more Q/V action training. It is whether BMM-selected subgoals can be paired with a separate low-level controller to make progress.
