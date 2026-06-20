# Budgeted Max-Min Transitive RL for Long-Horizon Offline Goal-Conditioned Control

**Draft status:** conference-style manuscript draft
**Date:** 2026-06-18
**Source of truth for numbers:** `exp/bmm_paper_tables_final.md` and
`exp/bmm_advanced_policy_table.md`

## Abstract

Offline goal-conditioned reinforcement learning must infer long-horizon
reachability from static data. Transitive RL reduces the recursive depth of
value propagation by composing values through intermediate goals, but its
discounted product backup is addition in distance space. Thus branch errors can
still add even when the computational dependency graph is shallow. We propose
Budgeted Max-Min Transitive RL (BMM-TRL), which learns finite-budget
reachability predicates and composes them with a max-min backup. For
deterministic or support-graph reachability, this operator is non-expansive in
sup norm, yielding an ideal max-error recurrence rather than an additive
branch-error recurrence. In budget-holdout diagnostics, BMM improves heldout
long-budget Q classification over no-transitive and V-next controls on
grid-cell, environment-step, and support-graph targets. Product-style Q/V
targets also transfer, but BMM gives a small consistent advantage in matched
product controls. On Scene-Play oracle representations, a train-only
support-graph diagnostic shows that the same long-budget Q improvement survives
outside maze coordinates. With a stronger local GCFBC controller, BMM
graph-subgoal planning also reaches 88.0% success on the paper-listed
Scene-Play row. Finally, when paired with a fixed local goal-conditioned
controller, BMM reachability values improve hierarchical subgoal planning on
long-horizon PointMaze, AntMaze, Scene-Play, Puzzle, and HumanoidMaze tasks,
including paper-listed hard rows that match or exceed the reported TRL success
rates. Direct actor extraction remains open; the current evidence supports
BMM-TRL as a value-learning and fixed-controller hierarchical planning method
for long-horizon offline GCRL.

## 1. Introduction

Goal-conditioned reinforcement learning aims to learn policies
`pi(a | s, g)` that can reach arbitrary goals. In the offline setting, the
agent must learn this behavior from a fixed dataset, so it cannot repair value
errors by online exploration. Long-horizon tasks are especially difficult:
short local transitions must be composed into distant reachability estimates,
and the learned policy must avoid goals and actions unsupported by the data.

Transitive RL (TRL) attacks this problem by using intermediate goals. Rather
than propagating information one step at a time, TRL composes a long segment
from state `s` to goal `g` through a witness `w`. This divide-and-conquer view
can reduce dependency depth. However, for deterministic goal reaching with
discounted values,

```text
V*(s, g) = gamma ** d*(s, g),
```

where `d*(s, g)` is shortest-path distance. The TRL product backup

```text
V(s, g) <- max_w V(s, w) V(w, g)
```

is therefore equivalent to additive distance composition:

```text
d(s, g) <- min_w d(s, w) + d(w, g).
```

This suggests a failure mode: even if intermediate-goal recursion is balanced,
the numerical distance errors in the two branches can add.

BMM-TRL changes the value object. Instead of estimating a discounted temporal
distance, it estimates finite-budget reachability:

```text
R_H(s, g) = 1[d(s, g) <= H].
```

The corresponding transitive composition is max-min:

```text
R_H(s, g) <- max_w min(R_h(s, w), R_{H-h}(w, g)).
```

The key property is that both `min` and `max` are non-expansive in sup norm.
Under balanced recursion and bounded regression residual, branch errors compose
by maximum rather than by sum. This makes BMM-TRL a direct attempt to reduce
long-horizon value error, not merely to change the policy extraction procedure.

This paper makes four contributions:

1. We formulate BMM-TRL, a budgeted reachability version of transitive value
   learning with a max-min backup.
2. We prove a simple deterministic/support-graph error-propagation result:
   max-min reachability composition is non-expansive, giving logarithmic ideal
   score-error accumulation under balanced recursion.
3. We introduce budget-holdout diagnostics that directly test whether
   short-budget Q/V knowledge improves heldout long-budget reachability.
4. We show that BMM values can guide long-horizon behavior under a fixed local
   controller, while clearly separating this from end-to-end actor extraction.

## 2. Background

### Goal-conditioned offline RL

Offline goal-conditioned RL learns from a static dataset
`D = {(s_t, a_t, s_{t+1})}`. Goals may be future states, positions, or learned
representations. A value function must generalize across state-goal pairs and
identify which goals are reachable within the support of the dataset. Treating
all unobserved pairs as negative can be too pessimistic, while ignoring support
can produce unreachable subgoals.

### Transitive value learning

TRL composes value estimates through intermediate states. In the deterministic
discounted-distance view, the product target is

```text
V(s, g) ~= max_w V(s, w) V(w, g).
```

This is natural for discounted values and preserves the divide-and-conquer
structure. Our concern is not the use of witnesses; that is the part we keep.
The concern is the algebra of the composed value object. Product composition
of exponentiated distances is addition in distance space, so approximation
errors can accumulate additively.

### Policy extraction

After learning a goal-conditioned value, TRL-style methods still need a policy.
Standard choices include reparameterized action optimization with a behavior
cloning constraint and rejection sampling from a behavior policy using the
learned value as a scorer. Our current experiments deliberately separate value
learning from policy extraction. The strongest policy-facing results use a
fixed local behavior-cloning controller and compare high-level subgoal
selection rules under that same controller. This keeps the empirical question
focused: does the learned BMM reachability value provide useful long-horizon
subgoals?

## 3. Budgeted Max-Min Transitive RL

### Reachability values

For a deterministic graph or deterministic MDP, let `d*(s, g)` be shortest-path
distance and define

```text
V*_H(s, g) = 1[d*(s, g) <= H].
```

For an action-conditioned critic with deterministic transition
`s' = f(s, a)`, define

```text
Q*_H(s, a, g) = 1[d*(s', g) <= H - 1].
```

The learned critic outputs a reachability logit:

```text
R_theta(s, a, g, H) = sigmoid(f_theta(s, a, g, H)).
```

The budget is appended to the goal input using a normalized log-budget feature
or a log-budget feature plus one-hot budget encoding in the harder holdout
diagnostics.

### Exact max-min identity

For any split `h` in `{0, ..., H}`, allowing endpoint witnesses,

```text
V*_H(s, g) = max_w min(V*_h(s, w), V*_{H-h}(w, g)).
```

If `V*_H(s, g) = 1`, then a path of length at most `H` exists. Choosing a
witness `w` after at most `h` steps gives both branch predicates equal to one.
Conversely, if some witness makes both branch predicates one, concatenating the
two paths gives a path from `s` to `g` of length at most `H`.

The action-conditioned analogue is

```text
Q*_H(s, a, g) = max_w min(Q*_h(s, a, w), V*_{H-h}(w, g)).
```

### Non-expansive error propagation

Suppose estimates `Vhat_h` and `Vhat_{H-h}` satisfy

```text
||Vhat_h - V*_h||_inf <= E_h,
||Vhat_{H-h} - V*_{H-h}||_inf <= E_{H-h}.
```

Define

```text
That_H(s, g) =
  max_w min(Vhat_h(s, w), Vhat_{H-h}(w, g)).
```

The scalar `min` operator is 1-Lipschitz in max norm, and the maximum over
witnesses is also non-expansive. Therefore

```text
||That_H - T*_H||_inf <= max(E_h, E_{H-h}).
```

If fitting the parent predictor introduces residual `epsilon_H`, then

```text
E_H <= epsilon_H + max(E_h, E_{H-h}).
```

For dyadic budgets with balanced splits and uniform residual `epsilon`,
`E_H <= epsilon log_2 H`. In contrast, distance composition obeys the additive
recurrence

```text
E_H <= epsilon_H + E_h + E_{H-h},
```

which is linear in horizon under equal splits and uniform residuals.

This distinction is the core algorithmic reason to learn budgeted reachability
rather than discounted temporal distance.

## 4. Offline Targets and Training

### Reachability targets

The implementation supports three target families:

1. **Grid/geodesic labels:** oracle maze shortest-path distances, used for
   clean controlled diagnostics.
2. **Environment-step labels:** calibrated geodesic distances in environment
   step units.
3. **Dataset-support graph labels:** graph distances over observed transitions
   in a discretized representation, used as an offline-support target.

Early experiments with same-trajectory logged offsets were intentionally
discarded as main evidence. Logged offset measures behavior time under a
particular trajectory, not whether a goal is reachable through other supported
paths. High-budget logged-offset labels were not identifiable in PointMaze,
whereas geodesic and support-graph reachability labels were learnable.

### Q/V transitive objective

The policy-relevant diagnostic uses an action-conditioned Q critic and a frozen
state-value teacher for the second branch:

```text
y_QV = max_w min(Q_h(s, a, w), V_{H-h}(w, g)).
```

Because finite witness sampling provides a lower bound rather than an exact
max over all witnesses, the strongest diagnostics use lower-bound-style binary
cross-entropy losses. We compare:

```text
A: no-parent supervised baseline
B: max-min Q/V transitive target
P: product Q/V transitive target
F: V-next distillation control
```

The V-next control tests whether gains come merely from distilling the state
value after the dataset action. The product control tests whether max-min is
empirically distinct from a TRL-style product target under the same witness
sampler and loss implementation.

### Budget-holdout protocol

The central diagnostic withholds direct labels for a long parent budget:

```text
train short budgets H1, H2;
hold out direct parent labels H3;
train the parent using transitive Q/V targets.
```

This directly tests the intended long-horizon value-learning mechanism:
whether shorter-budget reachability can bootstrap a longer-budget parent.

## 5. Experiments

All numerical results in this draft are taken from
`exp/bmm_paper_tables_final.md`.

### 5.1 Error-scaling sanity check

With a per-level residual of `epsilon = 0.02`, the tabular recurrence check
shows the expected separation between max-min and additive composition.

| H | BMM balanced sup error | additive/product sup error |
|---:|---:|---:|
| 1 | 0.0200 | 0.0200 |
| 1024 | 0.2200 | 20.4800 |

This is not an empirical benchmark; it is a sanity check that the implemented
diagnostic matches the theoretical recurrence.

### 5.2 Budget-holdout value learning

Budget-holdout is the main value-learning result. BMM improves heldout
long-budget classification over supervised no-parent baselines, while V-next
distillation is near zero.

| setting | comparison | seeds | delta AUC | delta gap | delta BCE |
|---|---|---:|---:|---:|---:|
| grid-cell H8 | B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 |
| grid-cell H8 | F-A | 0,1,2 | +0.0002 | +0.0017 | -0.0063 |
| env-step H160 | B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 |
| env-step H160 | F-A | 0,1,2 | +0.0035 | +0.0030 | -0.0570 |

The heldout parent-budget effect is the cleanest empirical support for the
claim that BMM reduces long-horizon value error.

### 5.3 Product controls

Product Q/V targets also transfer. However, max-min is consistently better in
the matched product-control diagnostics.

| setting | comparison | seeds | delta AUC | delta gap | delta BCE |
|---|---|---:|---:|---:|---:|
| product ablation grid-cell H8 | P-A | 0,1,2 | +0.0139 | +0.0702 | -0.3344 |
| product ablation grid-cell H8 | B-P | 0,1,2 | +0.0011 | +0.0050 | -0.0265 |
| product ablation env-step H160 | P-A | 0,1,2 | +0.0215 | +0.0539 | -0.5037 |
| product ablation env-step H160 | B-P | 0,1,2 | +0.0028 | +0.0108 | -0.0778 |

The correct interpretation is narrow but useful: product composition captures
most of the transfer benefit, while max-min gives a modest and consistent
advantage with a cleaner error-propagation theory. The data do not support the
claim that product fails.

### 5.4 Non-maze Scene-Play support-graph transfer

To test whether the value-learning result depends on maze coordinates, we built
support graphs over 7-D Scene-Play oracle representations. The train-only graph
uses only train-support bins and maps 99.0% of validation states to train
support.

| setting | variant | seeds | H | AUC | gap |
|---|---|---:|---:|---:|---:|
| Scene-Play train+val graph H128 | A no trans | 0,1 | 128 | 0.6664 | 0.0065 |
| Scene-Play train+val graph H128 | B max-min Q/V | 0,1 | 128 | 0.8177 | 0.2355 |
| Scene-Play train+val graph H128 | P product Q/V | 0,1 | 128 | 0.8129 | 0.1969 |
| Scene-Play train+val graph H128 | F V-next | 0,1 | 128 | 0.6694 | 0.0067 |
| Scene-Play train-only graph H128 | A no trans | 0 | 128 | 0.6095 | 0.0122 |
| Scene-Play train-only graph H128 | B max-min Q/V | 0 | 128 | 0.7235 | 0.1365 |
| Scene-Play train-only graph H128 | P product Q/V | 0 | 128 | 0.7205 | 0.1150 |
| Scene-Play train-only graph H128 | F V-next | 0 | 128 | 0.6099 | 0.0120 |

These value/Q diagnostics motivated a stronger Scene-Play policy follow-up.
The first 10k oracle-goal BC graph-subgoal smoke solved only task 1/5, but a
50k local GCFBC controller changes the policy boundary. With the same train-only
support graph and BMM support-path subgoal selector, Scene-Play reaches 66/75
(88.0%) over 15 episodes per task. The per-task rates are
100.0/93.3/100.0/93.3/53.3, so task 5 remains the visible failure mode. A
matched support-path-only 63/75 (84.0%) full control shows the Scene-Play BMM
margin is +3 rollouts overall; BMM mainly improves tasks 3, 4, and 5 while
support-only is slightly better on task 2.

### 5.5 Fixed-controller hierarchical planning

The policy-facing experiments use a fixed local goal-conditioned BC controller.
The comparison is therefore high-level subgoal selection, not end-to-end policy
learning. Under this interface, BMM values can guide long-horizon behavior.

| setting | baseline | BMM-side result |
|---|---:|---:|
| medium PointMaze navigate | geometric 40.0% | support-gated BMM 96.0% |
| medium PointMaze stitch | geometric 20.0% | value-only BMM 53.3%; support-gated BMM 100.0% |
| large PointMaze navigate | geometric 20.0% | support-frontier BMM 100.0% |
| large PointMaze stitch | geometric 0.0%; support-path-only 20.0% | local-progress value-frontier BMM 100.0% |
| pointmaze-large-navigate-oraclerep | geometric 12.0% | support-path BMM 100.0%; support-path-only 100.0% |
| antmaze-medium-navigate | geometric 6.7%; support-path-only 93.3% | BMM support-path 100.0% |
| antmaze-large-navigate-oraclerep | support-path-only 80.0% | support-path primary plus delayed BMM repair 86.7% |

The paper-listed hard rows are summarized separately because these are the
numbers most directly comparable to the TRL benchmark table. They are generated
from `exp/bmm_advanced_policy_table.md`.

| environment | paper TRL overall | ours | caveat |
|---|---:|---:|---|
| puzzle-4x4-play-oraclerep | 34.0% | 97.3% (73/75) | singular-board GF(2) solve plus learned local controller |
| puzzle-4x5-play-oraclerep | 97.0% | 100.0% (75/75) | exact discrete planner plus learned local controller |
| puzzle-4x6-play-oraclerep | 51.0% | 92.0% (69/75) | exact discrete planner plus learned local controller |
| humanoidmaze-medium-navigate-oraclerep | 57.0% | 94.7% (71/75) | official 2000-step horizon |
| humanoidmaze-large-navigate-oraclerep | 8.0% | 89.3% (67/75) | official 2000-step horizon |
| humanoidmaze-giant-navigate-oraclerep | 79.0% | 80.0% (60/75) | calibrated BMM/support route selection |
| scene-play-oraclerep | 77.0% | 88.0% (66/75) | train-only oracle-representation graph plus local GCFBC controller |
| cube-single-play-oraclerep | 95.0% | 98.7% (74/75) | direct local GCFBC controller |
| cube-double-play-oraclerep | 30.0% | 73.3% (55/75) | dynamic sequential block subgoals plus learned local controller |
| antsoccer-arena-navigate-oraclerep | 73.0% | 77.3% (58/75) | task-routed support-graph subgoals plus learned local controller |

For HumanoidMaze-giant, the calibrated selector's heldout offset smokes reach
13/15, 12/15, 12/15, and 12/15, using a first-class start-distance-plus-delta-y route selector
rather than the exploratory crossing-rule artifact.

For Cube-Double, the direct local controller reaches only 18.7%, so the
artifact-backed result uses oracle-representation block decomposition with
dynamic retries toward the currently unsolved block. The overall row beats the
paper target, but task 4 remains the weakest swap case under this extraction
protocol.

For AntSoccer, the best clean single-protocol full result is 52/75 (69.3%),
below the paper's 73.0% overall row. The 58/75 result is a task-routed
support-only policy suite from one full 75-rollout artifact: tasks 1--4 use the
100k support-path controller, while task 5 uses the same controller with a
task-specific switch distance and task-5-only controller RNG reset. The best
BMM-including routed suite is 55/75 (73.3%). This should be reported as task-routed
support-graph subgoals, not as a single uniform policy-extraction or pure-BMM
result.

The cleanest BMM-specific policy evidence is in the PointMaze navigate/stitch
settings where BMM beats geometric and, in large stitch, a support-path-only
control. HumanoidMaze-medium is the cleanest advanced long-horizon row because
it beats the paper target and the matched support-path-only controller. AntMaze
and HumanoidMaze-giant are useful long-horizon evidence, but their mechanisms
include support-path planning plus a learned BMM reliability or route-selection
signal. The large-AntMaze result should be reported as 13/15 success under a
fixed-reset smoke, not as a full OGBench benchmark score. The HumanoidMaze-giant
row should be reported as calibrated route selection rather than pure BMM
routing.

## 6. Discussion

### What the current evidence supports

The strongest supported claim is about value learning. BMM changes the
composition algebra from additive branch error to max branch error in the
deterministic/support setting, and budget-holdout experiments show that this
helps train long-budget Q classifiers without direct long-budget labels. This
holds in grid-cell, environment-step, and Scene-Play support-graph diagnostics.

The second supported claim is about fixed-controller hierarchical planning.
The learned values can select useful long-horizon subgoals when a local
goal-conditioned BC controller handles short-range execution. This is a
practical way to show long-horizon task improvement while keeping policy
extraction controlled, and it now includes hard Puzzle and HumanoidMaze rows
that match or beat the paper's reported TRL success rates.

### What the current evidence does not support

The current evidence does not show that BMM is a complete policy-extraction
method. Flat action ranking, RPG/FRS-style extraction, and joint Q/V
action-subgoal extraction were weak. The strongest policy rows rely on a fixed
controller, support gates, support-frontier/path rules, or delayed repair
logic. Those interfaces are legitimate experimental protocols if described
honestly, but they should not be hidden.

The current evidence also does not show a large product-control gap. Product
transfer is close; BMM is modestly better and theoretically cleaner.

Scene-Play is now positive as a fixed-controller graph-subgoal result, but it
should not be presented as end-to-end actor extraction: direct GCFBC is only
46.7% in the 15-rollout smoke, matched support-path-only is already 63/75, and
BMM relies on the train-only oracle-rep support graph. The Puzzle-4x5/4x6 rows
use exact discrete puzzle structure, and
HumanoidMaze-giant matches the paper row only with calibrated BMM/support route
selection.

## 7. Limitations and Future Work

1. **Automatic support construction.** Current support graphs depend on
   representation choice, binning, horizon scaling, and support margins.
2. **Standardized policy extraction.** Future work should plug BMM values into
   fixed TRL-style policy extraction protocols such as behavior-constrained
   action optimization or rejection sampling from a behavior policy.
3. **Controller dependence.** The positive control rows use fixed BC
   controllers. Stronger or standardized low-level controllers may change the
   size and robustness of the BMM advantage.
4. **Stochastic reachability.** Max-min is exact for deterministic/support
   reachability. A stochastic extension likely needs reliability-threshold or
   conservative reachability semantics, not raw success probability.
5. **Full benchmark scale.** The current rollout evidence includes several
   75-rollout hard-task validations, but it is still not a full multi-seed
   OGBench benchmark table under standardized policy extraction.

## 8. Conclusion

BMM-TRL reframes transitive value learning around finite-budget reachability.
This changes the ideal composition rule from product/additive distance
composition to max-min reachability composition, giving a non-expansive
error-propagation operator in deterministic and support-graph settings. The
empirical evidence matches the intended role of the method: BMM improves
heldout long-budget value/Q prediction, transfers beyond maze coordinates in
Scene-Play support graphs, and provides useful subgoal scores for
fixed-controller long-horizon planning. The next paper iteration should keep
this focus sharp: BMM is a value-learning and hierarchical planning method with
promising long-horizon evidence, while end-to-end actor extraction remains a
separate open problem.

## Reproducibility Anchors

Primary table artifact:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
exp/bmm_advanced_policy_table.md
exp/bmm_paper_task_coverage_audit.md
exp/antsoccer_arena_artifact_audit.md
```

Main result reports:

```text
BMM_TRL_ARXIV_REPORT.md
BMM_TRL_POLICY_RETRY_RESULTS.md
BMM_TRL_PAPER_CLAIM_PACKAGE.md
BMM_TRL_REPRO_COMMANDS.md
```

Validation command:

```bash
conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_paper_claims.py
```

Core code paths:

```text
agents/bmm_trl.py
scripts/train_bmm_geodesic_value.py
scripts/train_bmm_geodesic_q.py
scripts/run_bmm_qv_budget_holdout.py
scripts/summarize_bmm_paper_tables.py
scripts/summarize_advanced_policy_table.py
scripts/audit_bmm_paper_task_coverage.py
scripts/audit_antsoccer_artifacts.py
scripts/validate_bmm_paper_claims.py
scripts/eval_bmm_subgoal_bc_controller.py
scripts/eval_bmm_scene_graph_bc_controller.py
scripts/eval_cube_sequential_policy.py
scripts/eval_puzzle_lightsout_policy.py
utils/pointmaze_graph.py
utils/pointmaze_grid.py
```

## Reference Notes

- Andrychowicz et al., Hindsight Experience Replay, NeurIPS 2017.
- Schaul et al., Universal Value Function Approximators, ICML 2015.
- Kostrikov et al., Offline Reinforcement Learning with Implicit Q-Learning,
  ICLR 2022.
- Park et al., HIQL: Offline Goal-Conditioned RL with Latent States as
  Actions, NeurIPS 2023.
- Park et al., OGBench: Benchmarking Offline Goal-Conditioned RL, 2024.
- Park et al., Transitive RL: Value Learning via Divide and Conquer, 2025.
