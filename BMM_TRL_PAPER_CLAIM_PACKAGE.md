# BMM-TRL paper claim package

Date: 2026-06-18

This file is the paper-facing contract for the current BMM-TRL evidence. It is
intended to keep the manuscript focused on claims that are actually supported by
the local artifacts.

## One-sentence thesis

Budgeted Max-Min Transitive RL replaces product composition of discounted
goal-reaching values with max-min composition of finite-budget reachability,
changing the ideal error recurrence from additive branch error to max branch
error; in offline diagnostics this improves heldout long-budget value/Q
classification, and the learned reachability values can guide long-horizon
behavior when paired with a fixed local controller.

## Recommended title

Budgeted Max-Min Transitive RL for Long-Horizon Offline Goal-Conditioned Control

## Paper scope

The paper should be framed as:

1. A new value-learning target and backup for offline goal-conditioned RL.
2. A theory/diagnostic paper about long-horizon value error propagation.
3. A fixed-controller hierarchical planning demonstration showing that the
   learned values can be useful for long-horizon control, including
   paper-listed Puzzle-3x3, Scene-Play, and HumanoidMaze success-rate targets.
4. A separate policy-interface result showing that structured decomposition
   plus a learned local controller can match or beat the hard Puzzle-4x5 and
   Puzzle-4x6 rows from the TRL paper.
5. A transparent hard-task follow-up showing that HumanoidMaze-giant can match
   the paper row with calibrated BMM/support route selection, while pure BMM
   routing remains below the paper target.

The paper should not be framed as:

1. A complete replacement for TRL policy extraction.
2. A full OGBench benchmark sweep.
3. An end-to-end actor-extraction method.
4. A claim that Scene-Play is solved by direct actor extraction.
5. A claim that the Puzzle-4x5/4x6 wins are pure BMM value-ranking wins; those
   rows currently use an exact Lights Out planner.

## Core claims and evidence

### Claim 1: BMM changes the error algebra

Claim:
For deterministic or support-graph reachability, the max-min backup is
non-expansive in sup norm. Under balanced recursion and bounded per-level
regression residual, the ideal accumulated score error is logarithmic in the
horizon.

Evidence:

- Mathematical identity and non-expansive proof in `BMM_TRL_ARXIV_REPORT.md`.
- Tabular diagnostic in `exp/bmm_tabular_error_scaling.json`.
- Consolidated table in `exp/bmm_paper_tables_final.md`:
  - H=1: BMM balanced error 0.0200, additive/product error 0.0200.
  - H=1024: BMM balanced error 0.2200, additive/product error 20.4800.

Use in paper:
This is the first main contribution. It justifies learning budgeted
reachability instead of discounted temporal distance.

### Claim 2: BMM improves heldout long-budget value/Q prediction

Claim:
When direct long-budget labels are held out, Q/V transitive BMM improves
long-budget reachability classification over no-transitive and V-next-only
controls.

Evidence from `exp/bmm_paper_tables_final.md`:

| setting | main comparison | result |
|---|---|---|
| grid-cell H8 | B-A over seeds 0,1,2 | delta AUC +0.0150, delta gap +0.0751, delta BCE -0.3609 |
| env-step H160 | B-A over seeds 0,1,2 | delta AUC +0.0243, delta gap +0.0647, delta BCE -0.5815 |
| V-next control | F-A | near-zero gap gains: +0.0017 at H8 and +0.0030 at H160 |

Use in paper:
This is the strongest empirical value-learning result. It directly supports
the "reduces long-horizon value error" story.

### Claim 3: Product controls improve too, but max-min is consistently better

Claim:
Product Q/V targets also transfer, but BMM has a small consistent advantage in
the current matched product-control diagnostics.

Evidence from `exp/bmm_paper_tables_final.md`:

| setting | B-P over seeds 0,1,2 |
|---|---|
| product ablation grid-cell H8 | delta AUC +0.0011, delta gap +0.0050, delta BCE -0.0265 |
| product ablation env-step H160 | delta AUC +0.0028, delta gap +0.0108, delta BCE -0.0778 |

Use in paper:
Phrase as "consistent but modest advantage over product under the matched
lower-bound BCE implementation." Do not overstate it as a large product-control
win.

### Claim 4: The value/Q result extends beyond maze coordinates

Claim:
The support-graph construction and Q/V transfer work on non-maze Scene-Play
oracle representations.

Evidence from `exp/bmm_paper_tables_final.md`:

| setting | no trans gap | BMM gap | product gap | V-next gap |
|---|---:|---:|---:|---:|
| Scene-Play train+val graph H128, seeds 0,1 | 0.0065 | 0.2355 | 0.1969 | 0.0067 |
| Scene-Play train-only graph H128, seed 0 | 0.0122 | 0.1365 | 0.1150 | 0.0120 |

Additional support:

- Train-only graph maps 99.0% of validation states to train-support bins.
- Train-only H128 value teacher reaches AUC/gap 0.9734/0.7569.

Use in paper:
This is non-maze value/Q evidence. It should be in the main paper if space
allows, but clearly labeled as a diagnostic rather than policy control.

### Claim 5: BMM values can guide long-horizon behavior under a fixed controller

Claim:
When paired with the same fixed local controller, BMM-based subgoal planning
substantially improves long-horizon behavior over geometric midpoint,
support-path-only, or flat-controller baselines on several long-horizon tasks.

Evidence from `exp/bmm_paper_tables_final.md` and
`exp/bmm_advanced_policy_table.md`:

| setting | control | BMM-side result |
|---|---:|---:|
| medium PointMaze navigate | geometric 40.0% | support-gated BMM 96.0% |
| medium PointMaze stitch | geometric 20.0% | support-gated BMM 100.0%; value-only BMM 53.3% |
| large PointMaze navigate | geometric 20.0% | support-frontier BMM 100.0% |
| large PointMaze stitch | geometric 0.0%, support-path-only 20.0% | local-progress value-frontier BMM 100.0% |
| pointmaze-large-navigate-oraclerep | geometric 12.0% | BMM support-path 100.0%; support-path-only 100.0% |
| antmaze-medium-navigate | geometric 6.7%, support-path-only 93.3% | BMM support-path 100.0% |
| antmaze-large-navigate-oraclerep | support-path-only 80.0% | support-path primary plus delayed BMM repair 86.7% |
| puzzle-3x3-play-oraclerep | paper target 99.0%, flat TRL/RPG 88.0% | BMM support-path 100.0% over 75 rollouts |
| scene-play-oraclerep | paper target 77.0%, direct local GCFBC 46.7% in the 15-rollout smoke, matched support-path-only 63/75 (84.0%) | BMM support-path 88.0% over 75 rollouts with a 50k local GCFBC controller |
| humanoidmaze-medium-navigate-oraclerep | paper target 57.0%, earlier 1000-step eval understated the row | BMM support-path 94.7% over 75 official-horizon rollouts |
| humanoidmaze-large-navigate-oraclerep | paper target 8.0%, direct transfer 8.0% with a local-finetuned giant controller | BMM support-path 89.3% over 75 official-horizon rollouts |
| humanoidmaze-giant-navigate-oraclerep | paper target 79.0%, corrected deterministic direct-goal TRL/RPG 0.0% over 15-rollout smoke, support-path-only 76.0% over corrected deterministic 75 rollouts | fixed calibrated start-distance-plus-crossing BMM/support route gate reaches 80.0% over corrected deterministic 75 rollouts; the fitted distance-plus-delta-y selector reaches 86.7%, 80.0%, 80.0%, and 80.0% on four heldout offset smokes; BMM/support oracle union is 90.67% |

Hard paper-listed rows from `exp/bmm_advanced_policy_table.md`:

| environment | paper TRL | ours | artifact-backed caveat |
|---|---:|---:|---|
| puzzle-4x4-play-oraclerep | 34.0% | 97.3% (73/75) | singular-board GF(2) solve + learned local controller |
| puzzle-4x5-play-oraclerep | 97.0% | 100.0% (75/75) | exact discrete planner + learned local controller |
| puzzle-4x6-play-oraclerep | 51.0% | 92.0% (69/75) | exact discrete planner + learned local controller |
| scene-play-oraclerep | 77.0% | 88.0% (66/75) | train-only oracle-representation graph plus local GCFBC controller; matched support-path-only 63/75 |
| humanoidmaze-medium-navigate-oraclerep | 57.0% | 94.7% (71/75) | official 2000-step OGBench horizon |
| humanoidmaze-large-navigate-oraclerep | 8.0% | 89.3% (67/75) | official 2000-step horizon |
| humanoidmaze-giant-navigate-oraclerep | 79.0% | 80.0% (60/75) | calibrated BMM/support route selection; heldout offset smokes 13/15, 12/15, 12/15, and 12/15 |
| cube-single-play-oraclerep | 95.0% | 98.7% (74/75) | direct local GCFBC controller |
| cube-double-play-oraclerep | 30.0% | 73.3% (55/75) | dynamic sequential block subgoals + learned local controller |
| antsoccer-arena-navigate-oraclerep | 73.0% | 77.3% (58/75) | task-routed support-graph subgoals + learned local controller |

The current task-coverage audit is `exp/bmm_paper_task_coverage_audit.md`.
It is the canonical answer to whether the present artifacts beat the paper on
all tasks: all 10 promoted rows beat or match the paper on overall success, but
only 6 of 9 rows with paper per-task references beat or match every individual
task entry. The remaining promoted per-task gaps are HumanoidMaze-giant task 1,
task 2, and task 4, Scene-Play task 2, and AntSoccer tasks 2 and 3. AntSoccer
beats the paper overall only as a task-routed support-only policy suite; the
best clean single-protocol AntSoccer result is 69.3% versus the paper's 73.0%
overall row.

Use in paper:
Present as hierarchical fixed-controller evidence, not end-to-end policy
extraction. The cleanest BMM-specific policy rows are medium navigate, medium
stitch, large navigate, large stitch, Puzzle-3x3, and HumanoidMaze-medium. The
AntMaze and large-navigate-oraclerep rows are useful but have support-only
caveats.

Cube-Double should be presented as a policy-interface/decomposition result, not
a pure value-ranking result. The direct local GCFBC controller is 18.7%, while
one-pass sequential block subgoals reach 29/75 (38.7%) overall. Dynamic
sequential block subgoals with the same learned controller reach 55/75 (73.3%),
although task 4 remains the weakest case.

HumanoidMaze-medium uses `final_goal_switch_distance=128`; this setting was
chosen after short switch-distance smokes. A more aggressive switch at 192 fell
back to 50.67%, so the switch rule should be reported as part of the controller
protocol rather than hidden.

HumanoidMaze-large should be reported with the official OGBench 2000-step
horizon. The direct-transfer shortcut only matched the paper's weak 8% overall
row, and a 1000-step graph smoke was artificially short. With the official
horizon, BMM graph subgoals reach 67/75 (89.3%) and solve tasks 3 and 5
perfectly over 15 episodes/task.

AntSoccer should be reported only with the task-routing caveat. The best clean
single-protocol full result is 52/75 (69.3%) from the 100k local GCFBC
support-path controller, below the paper's 73.0% row. A task-routed
support-only policy suite reaches 58/75 (77.3%) in a single full artifact:
tasks 1--4 use the 100k support-path controller, while task 5 uses the same
controller with a task-specific switch distance and task-5-only controller RNG
reset. The AntSoccer artifact audit is `exp/antsoccer_arena_artifact_audit.md`;
it also confirms that no saved AntSoccer artifact contains a 66/75 result. The
best BMM-including routed suite remains 55/75 (73.3%). This is evidence that the
overall row can be beaten with controller/protocol selection, but it is not a
single uniform policy-extraction or pure-BMM result.

HumanoidMaze-giant should be reported as a calibrated hard-row success, not as a
fully robust solved row. The paper target is 79.0%. With deterministic-reset
correction, BMM switch128 reaches 72.0% over 75 rollouts and support-path-only
switch128 reaches 76.0%. Their paired failures are complementary: an oracle
per-episode choice between BMM and support-only reaches 68/75 (90.67%). A
start-distance route gate, using BMM when the initial graph distance is at least
1480 and support-only otherwise, reaches 60/75 (80.0%) on the corrected
deterministic 75-rollout protocol but only 73.3% on offset episodes 15--17. The
fixed crossing-aware route gate also uses BMM for right-to-left crossings with
start x >= 50 and goal_y - start_y >= 35; it reaches 60/75 (80.0%) on the same
75-rollout protocol and 13/15 (86.7%) plus 12/15 (80.0%) on the first two
offset smokes.
A constrained calibration-split fitter over route-choice diagnostics selects
the closely related `source_to_goal >= 1480 OR delta_y >= 35.3224` rule; this
first-class selector reaches 13/15 on the offset-15 rollout and 12/15 on each
of the offset-18, offset-21, and offset-24 rollouts. Continuing to 800k/1M, a local-goal fine-tune, and small
stochastic-controller temperatures did not improve the quick check. We also
found that OGBench locomaze reset noise uses global `np.random`, so seed-labeled
switch comparisons are not paired unless the evaluator seeds global NumPy
before reset.

### Claim 6: The hard Puzzle table rows can be solved by hierarchical policy interfaces

Claim:
The difficult TRL-paper Puzzle-4x5 and Puzzle-4x6 rows are not blocked by the
offline dataset or local policy class. A structured puzzle planner plus a
learned local goal-conditioned controller can match or beat the reported TRL
success rates.

Evidence:

| setting | paper TRL | our structured-controller result |
|---|---:|---:|
| puzzle-4x4-play-oraclerep | overall 34.0%; per-task 47/17/38/34/32 | 97.3% overall over 75 rollouts; 86.7/100/100/100/100 |
| cube-single-play-oraclerep | overall 95.0%; per-task 98/97/99/93/87 | 98.7% overall over 75 rollouts; 100/100/100/100/93.3 |
| cube-double-play-oraclerep | overall 30.0%; per-task 73/23/30/3/18 | 73.3% overall over 75 rollouts; 100/80/80/6.7/100 |
| puzzle-4x5-play-oraclerep | overall 97.0%; per-task 100/99/100/99/88 | 100.0% overall over 75 rollouts; 100/100/100/100/100 |
| puzzle-4x6-play-oraclerep | overall 51.0%; per-task 100/66/67/23/0 | 92.0% overall over 75 rollouts; 100/100/100/100/60 |

Artifacts:

```text
exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.json
exp/puzzle4x6_lightsout_gcfbc_local100k_nearest_ep15.json
scripts/eval_puzzle_lightsout_policy.py
```

Use in paper:
Use this as evidence for the importance of the policy-extraction/control
interface on very long-horizon discrete tasks. Do not merge it into the
BMM-specific value-improvement claim unless a follow-up replaces the exact
Lights Out planner with learned BMM value ranking or reports it explicitly as
an oracle decomposition baseline.

## Claims not supported

Do not claim:

1. "BMM solves Scene-Play through direct actor extraction." It does not. The
   50k local GCFBC controller is only 46.7% in the direct 15-rollout smoke;
   the 88.0% result uses a train-only oracle-representation support graph and
   BMM graph subgoals, with task 5 still weak.
2. "BMM is a stronger low-level controller." The low-level controller is fixed
   and external to the value algorithm.
3. "Direct actor extraction works." Flat Q/RPG/FRS and joint Q/V action-subgoal
   extraction remain weak.
4. "BMM uniquely explains all support-path policy success." Some support-path-only
   controls are very strong, including pointmaze-large-navigate-oraclerep.
5. "The support graph is fully automatic." Current support construction uses
   representation choice, bin size, horizon, and extraction rules.
6. "The product baseline fails." Product transfer is often close; BMM is
   better but the margin is modest in matched product controls.
7. "HumanoidMaze is uniformly solved." The medium official-horizon 94.7% BMM result
   beats the paper target, but task 4 remains weak and must be reported.
   HumanoidMaze-large is strong at 89.3% under the official 2000-step horizon.
   HumanoidMaze-giant only matches the paper target with a calibrated
   BMM/support route selector; pure BMM remains below target over 75 rollouts.
8. "The hard Puzzle-4x5/4x6 results are pure BMM." They use an exact Lights Out
   planner plus a learned local GCFBC controller.

## Main paper outline

### Abstract

Use a short abstract centered on value error:

Offline goal-conditioned RL must compose short-horizon experience into
long-horizon value estimates. Product-style transitive backups reduce dependency
depth, but for discounted goal-reaching values they are additive in distance
space, so branch errors can add. We propose Budgeted Max-Min Transitive RL
(BMM-TRL), which learns finite-budget reachability predicates and composes them
with a max-min backup. For deterministic/support reachability, the max-min
operator is non-expansive, yielding an ideal max-error recurrence under balanced
decompositions. In budget-holdout diagnostics, BMM improves heldout long-budget
Q classification over no-transitive and V-next controls on grid, env-step, and
support-graph targets, with a small but consistent advantage over product
controls. The learned values also improve fixed-controller hierarchical
planning on long-horizon PointMaze and AntMaze smokes, and reach or exceed the
reported TRL success rates on Puzzle-3x3, Scene-Play, and HumanoidMaze under
fixed low-level controllers. On the harder Puzzle-4x5 and Puzzle-4x6 rows, a
structured Lights Out planner with a learned local controller matches or beats
the reported TRL numbers, underscoring the importance of policy extraction. On
HumanoidMaze-giant, pure BMM improves short smokes over flat and support-only
controls but remains below the paper row over 75 rollouts; the paper-row match
comes from the calibrated BMM/support route selector.
Direct actor extraction remains open, but the results support BMM-TRL as a
value-learning and hierarchical subgoal-planning method for long-horizon
offline GCRL.

### Introduction

1. Problem: long-horizon offline GCRL needs compositional values.
2. TRL insight: intermediate goals reduce recursive dependency depth.
3. Gap: product of discounted values is addition in distance space; numeric
   distance/log-value error can add.
4. BMM idea: classify finite-budget reachability and compose with max-min.
5. Summary of theory and empirical findings.

### Method

1. Define \(R_H(s,g)\) and \(Q_H(s,a,g)\).
2. Exact max-min identity for deterministic/support reachability.
3. Non-expansive error bound.
4. Offline target construction:
   - geodesic/grid labels for maze diagnostics;
   - support-graph labels for offline data;
   - train-only support graph for offline-clean Scene-Play.
5. Q/V transitive objective and controls:
   - BMM max-min Q/V;
   - product Q/V;
   - V-next distillation.

### Experiments

Recommended table order:

1. Tabular error scaling.
2. Budget-holdout value/Q results.
3. Product-control ablation.
4. Scene-Play non-maze Q/V transfer.
5. Fixed-controller policy results, including Puzzle-3x3 and HumanoidMaze.
6. Limitations/boundaries table.

Use `exp/bmm_paper_tables_final.md` plus
`exp/bmm_advanced_policy_table.md` as the source for the table values.

### Discussion

Emphasize:

1. BMM's strongest result is value/Q error reduction.
2. Hierarchical policy use is promising but currently controller-dependent.
3. Support-only controls are necessary and should remain in the paper.
4. Scene-Play is positive only as fixed-controller graph-subgoal extraction;
   direct local GCFBC is much weaker and task 5 remains the main failure.
5. Future work: automatic support graph construction, learned representations,
   standardized policy extraction, stochastic reliability budgets.

## Recommended figures and tables

| item | source | purpose |
|---|---|---|
| Fig. 1: backup algebra diagram | manuscript schematic | product distance addition vs max-min reachability |
| Fig. 2: tabular error scaling | `exp/bmm_tabular_error_scaling.json` | visual theory sanity check |
| Table 1: budget-holdout values | `exp/bmm_paper_tables_final.md` | main value/Q evidence |
| Table 2: product controls | `exp/bmm_paper_tables_final.md` | BMM vs product specificity |
| Table 3: fixed-controller policy results | `exp/bmm_paper_tables_final.md`, `BMM_TRL_ADVANCED_TASK_RESULTS.md` | long-horizon control evidence |
| Table 4: limitation/boundary table | `exp/bmm_paper_tables_final.md` | keep claims defensible |

## Minimal remaining work before writing

No further fast experiment is required before drafting. The next useful work is:

1. Convert `BMM_TRL_ARXIV_REPORT.md` from a running report into a concise
   conference-style paper.
2. Move detailed policy caveats to an appendix or limitations table.
3. Keep `BMM_TRL_REPRO_COMMANDS.md` and `scripts/validate_bmm_paper_claims.py`
   synchronized when new headline artifacts are promoted.

## Reproducibility anchors

Primary summary:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
exp/bmm_advanced_policy_table.md
exp/bmm_advanced_policy_table.json
exp/bmm_paper_task_coverage_audit.md
exp/bmm_paper_task_coverage_audit.json
exp/antsoccer_arena_artifact_audit.md
exp/antsoccer_arena_artifact_audit.json
```

Validation command:

```bash
conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_paper_claims.py
```

Reproduction command sheet:

```text
BMM_TRL_REPRO_COMMANDS.md
```

Main result report:

```text
BMM_TRL_ARXIV_REPORT.md
BMM_TRL_POLICY_RETRY_RESULTS.md
```

Key code paths:

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
