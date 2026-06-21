# BMM-TRL policy retry results

## Decision

The flat BMM policy path is still weak, but the hierarchical retry is promising:

```text
High level: BMM value reachability selects a subgoal.
Low level: a small goal-conditioned BC controller moves toward the selected subgoal.
Near goal: switch the BC target from sampled subgoal to the actual task goal.
```

This is the first retry result where BMM produces actual PointMaze task success
rather than only reachability diagnostics or distance progress.

## Code changes

- Added `actor_budget_mode=scan` for FRS action reranking in `agents/bmm_trl.py`.
  - Restored-checkpoint smokes showed this was not enough by itself.
- Added `scripts/eval_policy_checkpoint.py` for fast checkpoint policy evals.
  - It now supports `bmm_trl`, `trl`, `gcfbc`, and `gciql` checkpoints restored
    from their saved `flags.json`.
- Added `--final_goal_switch_distance` to `scripts/eval_bmm_subgoal_bc_controller.py`.
  - When the high-level grid/geodesic distance to the task goal is below this
    threshold, the low-level BC controller targets the real goal coordinate.
- Added `--controller_type=agent` to `scripts/eval_bmm_subgoal_bc_controller.py`
  so a restored repo agent can replace the custom BC low-level controller.
  This lets us test TRL-style policy extraction as the shared controller.
- Added `--subgoal_commit_steps` and `--subgoal_replan_distance` so restored
  actor controllers can hold a selected subgoal for several environment steps
  instead of being retargeted every step.
- Added streaming per-episode progress lines to the value-subgoal policy smoke
  path. Use `conda run --no-capture-output` for longer checks, otherwise
  `conda run` buffers those lines until process exit.
- Added `scripts/inspect_bmm_eval_task_subgoals.py` to inspect top-ranked
  subgoals at OGBench eval-task reset states.
- Added `scripts/eval_bmm_scene_graph_bc_controller.py` for Scene-Play
  graph-subgoal policy smokes using online oracle representations, a train-only
  support graph, and a fixed oracle-goal BC controller.
- Added feasibility-gated BMM selectors, including
  `BMM_V_min_budget_scan_left_gate`, which scans right budgets while keeping
  the selected subgoal inside the low-level controller's left-budget reach.
- Added `BMM_V_min_budget_scan_right_progress`, which scans learned right-branch
  reachability over budgets and prefers locally reachable candidates with a
  smaller learned remaining horizon.
- Added diagnostic early stopping to the rollout evaluator. It is disabled by
  default; use it only for fast screens because it can stop late recoveries.
- Extended `scripts/test_bmm_agent_shapes.py` to cover scan-mode action sampling.

## Fast checks

Passed:

```bash
JAX_PLATFORMS=cpu conda run -n bmm-trl python -m py_compile \
  agents/bmm_trl.py \
  scripts/eval_policy_checkpoint.py \
  scripts/eval_bmm_subgoal_bc_controller.py \
  scripts/test_bmm_agent_shapes.py

JAX_PLATFORMS=cpu conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
```

The CPU test still prints the known sandbox CUDA discovery warning, then exits 0.

## Negative fast result: scan-only reranking

Restored existing BMM checkpoints and compared `actor_budget_mode=max` versus
`actor_budget_mode=scan` on `pointmaze-medium-navigate-v0`, tasks 1-5, two
episodes per task.

```text
exp/mrl/Debug/sd000_20260610_155926:10000
max:  0.2000 success
scan: 0.2000 success

exp/mrl/Debug/sd000_20260610_130441:20000
max:  0.1000 success
scan: 0.0000 success
```

Interpretation: execution-time budget scan alone does not solve the policy
bottleneck. The low-level action generator/controller is the limiting piece.

## Positive result: hierarchical BMM + BC controller

Setup:

```bash
conda run -n bmm-trl python scripts/eval_bmm_subgoal_bc_controller.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,160 \
  --left_budget=80 \
  --right_budget=80 \
  --controller_hops=0 \
  --num_subgoal_candidates=64 \
  --selectors=random,geometric_midpoint,BMM_V \
  --task_ids=1,2,3,4,5 \
  --episodes_per_task=3 \
  --max_steps=200 \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000 \
  --bc_offsets=1,2,4,8,16,32,64,80 \
  --bc_steps=2000 \
  --bc_batch_size=512 \
  --bc_hidden_dims="(256, 256)" \
  --final_goal_switch_distance=20
```

Result over 15 episodes:

| selector | success | final_d | improve | mean_step_goal | subgoal_valid |
|---|---:|---:|---:|---:|---:|
| random | 0.0000 | 121.4142 | 52.7888 | 0.2639 | 0.1103 |
| geometric_midpoint | 0.1333 | 101.6184 | 72.5846 | 0.3653 | 0.2981 |
| BMM_V | 0.8000 | 39.5916 | 134.6114 | 0.7520 | 0.6962 |

Artifact:

```text
exp/policy_retry_subgoal_bc_offsets80_goal_switch20_tasks1_5_ep3.json
exp/policy_retry_subgoal_bc_offsets80_goal_switch20_tasks1_5_ep3.md
```

Per-task breakdown for BMM_V:

| task | success | final_d |
|---:|---:|---:|
| 1 | 1.000 | 0.000 |
| 2 | 1.000 | 0.000 |
| 3 | 1.000 | 0.000 |
| 4 | 0.000 | 197.958 |
| 5 | 1.000 | 0.000 |

Task 4 is the main failure. In that task, mean BMM selected false-positive
subgoals with `source_to_subgoal` around 148 despite `left_budget=80`.

## Conservative ensemble selector

Added `BMM_V_min`, which scores a candidate by the lower ensemble branch
probability instead of the mean:

```text
score(w) = min(
    min_e V_h^e(s,w),
    min_e V_{H-h}^e(w,g)
)
```

Focused comparison over 15 episodes:

| selector | success | final_d | improve | mean_step_goal | subgoal_valid |
|---|---:|---:|---:|---:|---:|
| BMM_V | 0.7333 | 39.5916 | 134.6114 | 0.7552 | 0.6988 |
| BMM_V_min | 0.8000 | 39.5916 | 134.6114 | 0.7700 | 0.6958 |

Artifact:

```text
exp/policy_retry_subgoal_bc_bmm_min_tasks1_5_ep3.json
exp/policy_retry_subgoal_bc_bmm_min_tasks1_5_ep3.md
```

Conservative scoring is at least not worse and is slightly better in this
fast confirmation, so use `BMM_V_min` as the next default selector.

One-episode full-selector smoke:

| selector | success | final_d | improve |
|---|---:|---:|---:|
| random | 0.0000 | 126.6930 | 47.5099 |
| geometric_midpoint | 0.2000 | 98.9789 | 75.2240 |
| BMM_V | 0.8000 | 39.5916 | 134.6113 |
| oracle_midpoint | 0.0000 | 102.9381 | 71.2648 |

Artifact:

```text
exp/policy_retry_subgoal_bc_offsets80_goal_switch20_tasks1_5_steps200.json
exp/policy_retry_subgoal_bc_offsets80_goal_switch20_tasks1_5_steps200.md
```

## Fixed-controller confirmation

Saved the BC controller once and reused it for selector comparison:

```text
exp/policy_retry_bc_offsets80_seed0.pkl
```

Fixed-controller result over 15 episodes:

| selector | success | final_d | final_xy |
|---|---:|---:|---:|
| random | 0.0000 | 122.7339 | 14.2898 |
| geometric_midpoint | 0.1333 | 100.2987 | 9.7998 |
| BMM_V_min | 0.7333 | 39.5916 | 3.8637 |

Artifact:

```text
exp/policy_retry_fixed_bc_bmm_min_tasks1_5_ep3.json
exp/policy_retry_fixed_bc_bmm_min_tasks1_5_ep3.md
```

Longer fixed-controller comparison over 25 episodes:

| selector | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|
| geometric_midpoint | 0.1200 | 100.5626 | 9.4999 | 73.6403 | 0.2829 |
| BMM_V_min | 0.7200 | 39.5916 | 3.8179 | 134.6113 | 0.6933 |

Artifact:

```text
exp/policy_retry_fixed_bc_geom_vs_bmm_min_tasks1_5_ep5.json
exp/policy_retry_fixed_bc_geom_vs_bmm_min_tasks1_5_ep5.md
```

Per-task success:

| selector | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---:|---:|---:|---:|---:|
| geometric_midpoint | 0.0000 | 0.6000 | 0.0000 | 0.0000 | 0.0000 |
| BMM_V_min | 0.8000 | 0.8000 | 1.0000 | 0.0000 | 1.0000 |

This fixed-split result identified task 4 as the systematic failure.

Task-4 inspection showed the reason: with `left_budget=80` and `right_budget=80`,
there is no feasible one-hop subgoal for a start-goal distance of about 198
environment steps. Switching task 4 to an `80/160` split gives feasible frontier
subgoals, but a fixed `80/160` split hurts some other tasks. The next selector
therefore scans available right budgets and gates candidates by a left-budget
grid-geodesic feasibility check. This gate is an oracle diagnostic device, not
yet a deployable gate.

I then added `BMM_V_min_budget_scan_support_gate`, which replaces the
grid-geodesic gate with an offline dataset-support gate:

```text
left branch: candidate cell must be observed within support_gate_left_frac * left_budget
right branch: goal cell must be observed within the scanned right budget
```

This uses the offline trajectory support rather than the maze geodesic oracle.
On navigate, `support_gate_left_frac=0.75` avoids rare optimistic support
shortcuts near the BC controller's 80-step limit.

Validated adaptive budget-scan selectors over 25 episodes, holding each selected
subgoal for up to 10 environment steps and replanning early when the controller
gets within geodesic distance 20 of the subgoal:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 5 | 300 | 10 | 20 | 0.4000 | 95.8116 | 8.5567 | 78.3913 | 0.6001 |
| BMM_V_min_budget_scan_support_gate | 5 | 300 | 10 | 20 | 0.9600 | 0.0000 | 0.9370 | 174.2029 | 0.9965 |
| BMM_V_min_budget_scan_left_gate | 5 | 300 | 10 | 20 | 0.9600 | 0.0000 | 0.9211 | 174.2029 | 1.0000 |

Artifact:

```text
exp/policy_retry_fixed_bc_geom_bmm_support075_vs_oracle_gate_commit10_replan20_step300_ep5.json
exp/policy_retry_fixed_bc_geom_bmm_support075_vs_oracle_gate_commit10_replan20_step300_ep5.md
```

Per-task success:

| selector | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---:|---:|---:|---:|---:|
| geometric_midpoint | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| BMM_V_min_budget_scan_support_gate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8000 |
| BMM_V_min_budget_scan_left_gate | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The support-gated miss reached `final_d=0` but did not trigger the environment
success flag before the cap, so success and final geodesic distance should both
be reported. A previous oracle-gated run reached 100% success, but the
support-gated result is now the cleaner main policy evidence.

The earlier conservative `commit=1` scan over 15 rollouts showed the same
direction but one task-5 rollout reached `final_d=0` without triggering the
environment success flag before the 300-step cap:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 300 | 1 | None | 0.4000 | 95.0198 | 9.0560 | 79.1831 | 0.7115 |
| BMM_V_min_budget_scan_left_gate | 3 | 300 | 1 | None | 0.9333 | 0.0000 | 0.9304 | 174.2029 | 1.0000 |

Artifact:

```text
exp/policy_retry_fixed_bc_geom_vs_bmm_budget_scan_80_160_step300_ep3.json
exp/policy_retry_fixed_bc_geom_vs_bmm_budget_scan_80_160_step300_ep3.md
```

Terminology: one "episode" here means one complete environment rollout for one
task, not one action. With `max_steps=300`, a 5-episode, 5-task evaluation can
execute up to 7500 low-level actions per selector.

The `commit=10, replan_d=20` result is the current preferred policy smoke: it
is faster than rescoring every environment step and no longer collapses on task
5. The BMM-only one-rollout fast check is retained as a quick development
artifact:

```text
exp/policy_retry_bmm_budget_scan_commit10_replan20_step300_ep1.json
exp/policy_retry_bmm_budget_scan_commit10_replan20_step300_ep1.md
```

## OGBench PointMaze stitch expansion

To test a paper-listed OGBench variant beyond the original navigate dataset, I
downloaded `pointmaze-medium-stitch-v0` and trained a matching state-value
teacher instead of reusing the navigate checkpoint:

```text
exp/bmm_ogbench_pointmaze_medium_stitch_value_teacher_40_80_160/params_1000.pkl
exp/bmm_ogbench_pointmaze_medium_stitch_value_teacher_40_80_160.json
```

Heldout value metrics after 1000 updates:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 40 | 0.9752 | 0.5503 | 0.9751 | 0.5580 |
| 80 | 0.9611 | 0.5436 | 0.9614 | 0.5502 |
| 160 | 0.9877 | 0.6785 | 0.9870 | 0.6691 |

Using a freshly trained stitch BC controller
`exp/policy_retry_stitch_bc_offsets80_seed0.pkl`, the 15-rollout policy smoke
with full left support budget (`support_gate_left_frac=1.0`) was:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 300 | 10 | 20 | 0.2000 | 109.2566 | 10.7617 | 64.5009 | 0.2615 |
| BMM_V_min | 3 | 300 | 10 | 20 | 0.5333 | 50.0211 | 5.0034 | 123.7363 | 0.8000 |
| BMM_V_min_budget_scan_support_gate | 3 | 300 | 10 | 20 | 1.0000 | 0.0000 | 0.9044 | 173.7574 | 0.9945 |
| BMM_V_min_budget_scan_left_gate | 3 | 300 | 10 | 20 | 1.0000 | 0.0000 | 0.9020 | 173.7574 | 0.9949 |

Artifacts:

```text
exp/policy_retry_stitch_fixed_bc_geom_bmm_support_vs_oracle_gate_step300_ep3.json
exp/policy_retry_stitch_fixed_bc_geom_bmm_support_vs_oracle_gate_step300_ep3.md
```

Interpretation: the value-only BMM selector transfers enough to improve over
geometric midpoint on stitch, but it still fails the hardest split-feasibility
cases. The dataset-support gate makes the planner complete on this smoke and
matches the grid-geodesic oracle gate. A conservative
`support_gate_left_frac=0.75` was too strict on stitch and fell to 60% success,
so the support margin is an algorithm hyperparameter rather than a universal
constant.

## OGBench PointMaze large navigate probe

I also moved to the next PointMaze task in the handoff progression,
`pointmaze-large-navigate-v0`. The large maze has 46 free cells and a maximum
grid-geodesic diameter of about 376.6 environment steps.

The first large teacher with budgets `(80, 160, 320)` learned large-maze
reachability well, but the corresponding policy smoke exposed a planning
interface problem: `left_budget=80` waypoints were too coarse for the offset-BC
controller on the hard large tasks. Even a geodesic-oracle progress selector
with 80-step or 160-step waypoints stayed at 20% success. This was a useful
negative diagnostic, but not the final large-maze result.

I then switched to short high-level waypoints. A geodesic-oracle progress
selector with `left_budget=20`, `commit=5`, and `replan_d=5` solved all five
large tasks with the same offsets-80 BC controller. This revealed the real
failure mode: for the hardest task, the remaining distance after a 20-step
waypoint is still about 356.8 steps, larger than the trained H320 value budget.
The previous support-gated selector therefore had no feasible complete
left/right decomposition and fell back to ungated scores.

To handle this, I added `BMM_V_min_budget_scan_support_frontier`. It uses the
BMM value only for the short left branch, gates candidate waypoints by offline
dataset support, and then chooses the supported waypoint with the best remaining
support-distance frontier progress. This avoids requiring an ill-posed H400
classifier; H400 has too few negative pairs because the large maze diameter is
only about 376.6.

I trained a short-budget large value teacher with budgets
`(20, 40, 80, 160, 320)`:

```text
exp/bmm_ogbench_pointmaze_large_navigate_value_teacher_20_40_80_160_320/params_1000.pkl
exp/bmm_ogbench_pointmaze_large_navigate_value_teacher_20_40_80_160_320.json
```

Heldout value metrics after 1000 updates:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 20 | 0.9232 | 0.4412 | 0.9238 | 0.5491 |
| 40 | 0.9585 | 0.4865 | 0.9585 | 0.5787 |
| 80 | 0.9258 | 0.4908 | 0.9250 | 0.5499 |
| 160 | 0.9450 | 0.4950 | 0.9333 | 0.5176 |
| 320 | 0.9992 | 0.8052 | 0.9989 | 0.7790 |

With the same fixed offsets-80 BC controller, the 15-rollout large policy smoke
was:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 500 | 5 | 5 | 0.2000 | 221.9843 | 14.5863 | 75.3161 | 0.0697 |
| BMM_V_min_budget_scan_support_frontier | 3 | 500 | 5 | 5 | 1.0000 | 0.0000 | 0.9222 | 297.3004 | 0.6242 |
| oracle_path_progress | 3 | 500 | 5 | 5 | 1.0000 | 0.0000 | 0.9281 | 297.3004 | 1.0000 |

Artifacts:

```text
exp/policy_retry_large_nav_bc_offsets80_seed0.pkl
exp/policy_retry_large_nav_support_frontier_step500_ep3.json
exp/policy_retry_large_nav_support_frontier_step500_ep3.md
exp/policy_retry_large_nav_support_frontier_start_subgoal_inspection.json
exp/policy_retry_large_nav_support_frontier_start_subgoal_inspection.md
```

Interpretation: large PointMaze is now a positive fixed-controller result. The
algorithmic change is not simply "use a shorter budget"; it is to separate local
reachability from global frontier progress when the remaining distance exceeds
the largest reliable value budget. This is a stronger paper story than the
earlier medium-only result because it shows the BMM value can be used in a
multi-step frontier planner on a longer OGBench PointMaze task.

## OGBench paper-listed large navigate oraclerep smoke

The TRL paper's standard OGBench table uses oraclerep task names. I therefore
repeated the large-navigate value diagnostic on
`pointmaze-large-navigate-oraclerep-v0`, which is the first environment listed
in the paper's full OGBench table. This uses the same short budgets and 1k
training updates as the non-oraclerep large navigate run:

```text
exp/bmm_ogbench_pointmaze_large_navigate_oraclerep_value_teacher_20_40_80_160_320/params_1000.pkl
exp/bmm_ogbench_pointmaze_large_navigate_oraclerep_value_teacher_20_40_80_160_320.json
```

Heldout value metrics after 1000 updates:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 20 | 0.9290 | 0.5814 | 0.9301 | 0.5859 |
| 40 | 0.9625 | 0.6444 | 0.9608 | 0.6443 |
| 80 | 0.9554 | 0.6756 | 0.9548 | 0.6818 |
| 160 | 0.9813 | 0.7301 | 0.9795 | 0.7393 |
| 320 | 0.9969 | 0.9126 | 0.9974 | 0.9263 |

A 500-step policy smoke was too short for the oraclerep controller: the learned
support-frontier selector improved mean final geodesic distance from 237.8
to 142.7, but did not trigger success. With a 1000-step cap and the same fixed
offsets-80 BC controller, the first 15-rollout support-frontier validation
improved over geometric midpoint but still failed some task-2/task-3 rollouts.
The failure came from the frontier score preferring high left-branch BMM scores
over path-consistent progress.

I therefore added `BMM_V_min_budget_scan_support_path`, which uses dataset
support distances as the primary path-progress objective and the learned BMM
left score only as a small tie-break. With a local grid feasibility gate for the
short first hop (`support_frontier_left_gate=grid`), the matched validation was:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 1000 | 5 | 5 | 0.1333 | 244.4470 | 17.4104 | 52.8534 | 0.0535 |
| BMM_V_min_budget_scan_support_frontier | 3 | 1000 | 5 | 5 | 0.8000 | 31.7120 | 5.3726 | 265.5883 | 1.0000 |
| BMM_V_min_budget_scan_support_path | 3 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9649 | 297.3004 | 1.0000 |
| oracle_path_progress | 3 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9662 | 297.3004 | 1.0000 |

A support-path-only control, which removes the BMM left-score tie-break, also
solved the same 15-rollout smoke. I then reran the three matched comparators at
five episodes per task. The 25-rollout validation confirmed the large
geometric-vs-support-path gap:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 5 | 1000 | 5 | 5 | 0.1200 | 248.1467 | 17.9912 | 49.1537 | 0.0641 |
| BMM_V_min_budget_scan_support_path | 5 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9669 | 297.3004 | 1.0000 |
| support_path_only | 5 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9640 | 297.3004 | 1.0000 |

Artifacts:

```text
exp/policy_retry_large_nav_oraclerep_bc_offsets80_seed0.pkl
exp/policy_retry_large_nav_oraclerep_support_path_grid_step1000_ep3.json
exp/policy_retry_large_nav_oraclerep_support_path_grid_step1000_ep3.md
exp/policy_retry_large_nav_oraclerep_support_only_vs_bmm_support_path_grid_step1000_ep3.json
exp/policy_retry_large_nav_oraclerep_support_only_vs_bmm_support_path_grid_step1000_ep3.md
exp/policy_retry_large_nav_oraclerep_geom_bmm_support_only_grid_step1000_ep5.json
exp/policy_retry_large_nav_oraclerep_geom_bmm_support_only_grid_step1000_ep5.md
```

Interpretation: this is the closest result so far to the TRL paper's standard
OGBench table. It remains a smoke, not the full 15-episode-per-goal protocol,
but the 25-rollout validation preserves the main story that conservative
support-path subgoal planning solves all five long-horizon eval tasks under the
same local BC controller, while a geometric midpoint baseline remains mostly
stuck. This row should not be over-read as a BMM-only policy win: on large
PointMaze oraclerep, the support-path-only control matches BMM support-path.

Follow-up task-3 diagnostic: adding a minimum progress filter
(`support_frontier_min_progress_frac=0.5`) made the start-state BMM ranking
choose the same first progress cell as the oracle, but the closed-loop task-3
smoke still failed 0/3 while oracle solved 3/3. This suggests the remaining
failure is not just the initial no-op candidate; it likely needs a better
closed-loop path-progress/support target or subgoal-state selection within the
chosen cell.

The support-path selector fixed that issue in the 15-rollout validation by
following support-distance path progress while enforcing local first-hop
feasibility.

```text
exp/policy_retry_large_nav_oraclerep_task3_progress05_start_subgoal_inspection.md
exp/policy_retry_large_nav_oraclerep_task3_progress05_step1000_ep3.md
exp/policy_retry_large_nav_oraclerep_task3_support_path_grid_step1000_ep3.md
```

## AntMaze medium OGBench probe

I then moved to the next non-PointMaze environment in the handoff progression,
`antmaze-medium-navigate-v0`. The dataset download succeeded, and the env
exposes the same ingredients needed by the current geodesic scripts: XY dims
`(0, 1)`, an 8x8 `maze_map`, 29-D observations, and 8-D actions. The calibrated
grid has 26 free cells, about 29.01 environment steps per cell, and diameter
about 319.11 steps. This means H320 is saturated on medium AntMaze and is not a
useful balanced binary label; the first smoke failed to sample enough H320
negatives for exactly this reason.

I trained a 1k-update BMM state-value teacher with cell-aligned budgets
`(40, 80, 160, 240)` and a smaller 256x256 value network:

```text
exp/bmm_ogbench_antmaze_medium_navigate_value_teacher_40_80_160_240_1000/params_1000.pkl
exp/bmm_ogbench_antmaze_medium_navigate_value_teacher_40_80_160_240_1000.json
```

Heldout value metrics after 1000 updates:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 40 | 0.8286 | 0.2691 | 0.8238 | 0.2674 |
| 80 | 0.9220 | 0.4411 | 0.9177 | 0.4506 |
| 160 | 0.9169 | 0.4577 | 0.9114 | 0.4588 |
| 240 | 0.9355 | 0.4646 | 0.9376 | 0.4388 |

The long-horizon value result is positive, though H40 remains below the old
strict pass threshold. This is not surprising: H40 is only about 1.4 grid cells,
and the observation/action dimensionality is much higher than PointMaze.

The first policy smoke used a lightweight AntMaze goal-conditioned BC controller
(`bc_offsets=1,2,4,8,16,32`, 2k updates). That controller exposed a real
low-level bottleneck: even the grid oracle comparator did not trigger
environment success. Increasing the controller to a 512x512x512 layer-norm BC
model trained for 10k updates on offsets `4,8,16,32,64`, and switching to the
final goal once the remaining calibrated grid distance is at most 80, made the
oracle comparator reliable. Under that fixed stronger BC controller, the
15-rollout AntMaze validation smoke over all five eval tasks was:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 1000 | 10 | 5 | 0.0667 | 199.2046 | 12.8932 | 56.0867 | 0.0399 |
| BMM_V_min_budget_scan_support_path | 3 | 1000 | 10 | 5 | 1.0000 | 0.0000 | 0.4473 | 255.2913 | 0.8361 |
| oracle_path_progress | 3 | 1000 | 10 | 5 | 1.0000 | 0.0000 | 0.4503 | 255.2913 | 0.8473 |

I then added a `support_path_only` control that removes the learned BMM
left-branch tie-break and uses only conservative dataset-support path progress.
Under the same controller and rollout settings, support-only was also strong
but slightly less reliable:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| support_path_only | 3 | 1000 | 10 | 5 | 0.9333 | 7.7361 | 1.0403 | 247.5552 | 0.8587 |
| BMM_V_min_budget_scan_support_path | 3 | 1000 | 10 | 5 | 1.0000 | 0.0000 | 0.4308 | 255.2913 | 0.8312 |

Artifacts:

```text
exp/antmaze_medium_nav_bc_offsets_4_8_16_32_64_h512_ln_steps10000.pkl
exp/policy_retry_antmaze_medium_nav_support_path_bc_h512_ln_switch80_step1000_tasks1_5_ep3.json
exp/policy_retry_antmaze_medium_nav_support_path_bc_h512_ln_switch80_step1000_tasks1_5_ep3.md
exp/policy_retry_antmaze_medium_nav_support_only_vs_bmm_bc_h512_ln_switch80_step1000_tasks1_5_ep3.json
exp/policy_retry_antmaze_medium_nav_support_only_vs_bmm_bc_h512_ln_switch80_step1000_tasks1_5_ep3.md
```

Interpretation: AntMaze confirms the value/subgoal story beyond PointMaze once
the fixed local controller is strong enough. BMM support-path matches the
grid-geodesic oracle comparator on success and final distance, while the same
controller with a geometric midpoint planner solves only 1/15 rollouts. The
support-only control clarifies the mechanism: conservative support-path planning
is doing most of the work in AntMaze, and the learned BMM left-score tie-break
improves reliability from 14/15 to 15/15 in this smoke. Therefore AntMaze should
be framed as a strong fixed-controller hierarchical planning result, not as
standalone evidence that BMM values alone solve AntMaze.

## Paper-listed AntMaze large oraclerep probe

I then tried the paper-listed `antmaze-large-navigate-oraclerep-v0`. The large
AntMaze grid has 46 free cells, about 27.78 environment steps per cell, and
diameter about 527.91 environment steps. H640 is saturated, and H480 has only
22 negative free-cell pairs, so the first balanced value diagnostic used
budgets `(40, 80, 160, 240, 320)`.

This uncovered an oraclerep-specific shape issue in the diagnostic scripts:
the critic should receive 2-D `oracle_reps` as goals, while grid labels and any
full-observation controller still need 29-D observations. I patched the value
diagnostic and policy scorer to keep those representations separate. I also
disabled JAX GPU preallocation for these runs:
`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.35`. That
reduced observed GPU memory from about 20 GB to about 2.8 GB for this critic.

Artifacts:

```text
exp/bmm_ogbench_antmaze_large_navigate_oraclerep_value_teacher_40_80_160_240_320_1000/params_1000.pkl
exp/bmm_ogbench_antmaze_large_navigate_oraclerep_value_teacher_40_80_160_240_320_1000.json
```

Heldout value metrics after 1000 updates:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 40 | 0.8987 | 0.4495 | 0.9018 | 0.4640 |
| 80 | 0.9387 | 0.4721 | 0.9406 | 0.4870 |
| 160 | 0.9308 | 0.4585 | 0.9270 | 0.4713 |
| 240 | 0.9668 | 0.5802 | 0.9669 | 0.6031 |
| 320 | 0.9810 | 0.7027 | 0.9816 | 0.7019 |

The value result is positive: H40 narrowly misses the old 0.90 AUC threshold,
but every longer-horizon budget is strong. Policy control is harder. Reusing
the medium AntMaze BC controller moved task 1 substantially but did not solve
it. Training a matching large-AntMaze 512x512x512 layer-norm BC controller for
5k updates on full 29-D future observations produced only a mixed policy smoke:
support-only outperformed BMM in the all-task ep1 aggregate. This pointed to a
controller-interface mismatch rather than a value-learning failure: the
paper-listed environment is an `oraclerep` task, but the local BC controller was
being conditioned on arbitrary full future states inside each selected cell.

I therefore added `--bc_goal_rep=oracle`, which trains and applies the local BC
controller on compact 2-D oracle/XY goals when `dataset['oracle_reps']` is
available. With the same 512x512x512 layer-norm architecture, offsets
`4,8,16,32,64`, and 5k updates, the first matched 15-rollout smoke over tasks
1-5 used `left_budget=80/right_budget=320`:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 1500 | 10 | 5 | 0.0000 | 366.7609 | 20.2000 | 50.0129 | 0.0613 |
| BMM_V_min_budget_scan_support_path | 3 | 1500 | 10 | 5 | 0.6000 | 148.1862 | 5.2295 | 268.5875 | 0.7460 |
| support_path_only | 3 | 1500 | 10 | 5 | 0.4667 | 148.1862 | 7.2996 | 268.5875 | 0.7449 |

Per-task BMM success was 2/3 on task 1, 0/3 on task 2, 3/3 on task 3, 2/3 on
task 4, and 2/3 on task 5. Task 2 is the remaining hard failure: both BMM and
support-only made essentially no progress there under the current interface.

Inspecting the task-2 reset state showed that this was an interface bug in the
support-path horizon. Task 2 has source-to-goal distance 527.91. With
`right_budget=320`, no path-consistent first waypoint could certify the
remaining support distance. Raising the support right horizon exposed valid
local candidates without retraining the value function. The tighter corrected
setting `left_budget=80/right_budget=480` was enough to make the task-2 first
waypoint valid while avoiding the looser global choices seen with 560.

With `right_budget=480`, the matched 15-rollout smoke over tasks 1-5 becomes:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 1500 | 10 | 5 | 0.0000 | 366.7609 | 20.2000 | 50.0129 | 0.0613 |
| BMM_V_min_budget_scan_support_path | 3 | 1500 | 10 | 5 | 0.8000 | 64.8315 | 4.6980 | 351.9423 | 0.9927 |
| support_path_only | 3 | 1500 | 10 | 5 | 0.5333 | 103.7304 | 6.8577 | 313.0435 | 0.9877 |

Per-task BMM success with `right_budget=480` was 3/3 on task 1, 2/3 on task 2,
2/3 on task 3, 3/3 on task 4, and 2/3 on task 5. The support-only control was
2/3, 2/3, 1/3, 2/3, and 1/3 on the same tasks. A looser `right_budget=560`
also fixed task 2 but destabilized tasks 1 and 4, leaving BMM at 60.0%;
therefore 480 is the current best fixed corrected horizon.

I then repeated the corrected-horizon comparison with 5 episodes per task
instead of 3. This larger 25-rollout validation kept the main geometric
baseline separation but overturned the BMM-vs-support-only margin:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 5 | 1500 | 10 | 5 | 0.0000 | 374.5406 | 20.4864 | 42.2331 | 0.0456 |
| BMM_V_min_budget_scan_support_path | 5 | 1500 | 10 | 5 | 0.4800 | 144.4816 | 11.0565 | 272.2922 | 0.9894 |
| support_path_only | 5 | 1500 | 10 | 5 | 0.5200 | 102.2485 | 6.9151 | 314.5253 | 0.9860 |

The ep5 per-task pattern is mixed. BMM is better on tasks 4 and 5, support-only
is better on tasks 1, 2, and 3, and support-only is slightly better overall.
Therefore the defensible large-AntMaze claim is not that BMM's tie-break beats
support-only. It is that the learned value transfers to the paper-listed
oraclerep setup and that corrected support-path hierarchical planning, with or
without the BMM tie-break, strongly outperforms geometric midpoint. The BMM
specific margin remains clear on medium AntMaze and PointMaze smokes, but not
on this larger AntMaze-large validation.

I also tried two automatic support-horizon rules with nominal `right_budget=320`.
`source_goal_grid` uses the current grid source-goal distance as the support
horizon; it made task-2 candidates valid but over-admitted alternatives and
reduced BMM to 40.0% success versus 53.3% for support-only. A tighter
`local_grid_min_right` rule chooses the smallest right horizon that admits at
least one local grid-progress candidate; it improved BMM to 60.0%, but
support-only reached 73.3%. These runs are useful negative diagnostics:
automatic support-horizon selection is still open, and the corrected-horizon
large-AntMaze result should be reported as a fixed-horizon smoke rather than as
an automatically tuned planner.

I also ran an oracle path-progress upper-bound check with the same oracle-goal
BC controller and rollout settings, one episode per task. Even this high-level
oracle solved only task 4:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_path_progress | 1 | 1500 | 10 | 5 | 0.2000 | 200.0514 | 14.6562 | 216.7224 | 0.0873 |

This is an important controller diagnostic. It means the 25-rollout
AntMaze-large instability should not be interpreted purely as a BMM selector
failure: the current oracle-goal BC controller is not a reliable executor even
when the high-level path target is oracle-derived.

I then trained the same oracle-goal BC architecture for 20k updates, keeping
offsets `4,8,16,32,64` and the 512x512x512 layer-norm architecture. The final
supervised BC loss dropped from about 0.0966 in the 5k checkpoint to about
0.0664. This improved the same oracle-path upper-bound check from 1/5 to 3/5
tasks:

| selector | controller steps | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_path_progress | 5000 | 1 | 1500 | 10 | 5 | 0.2000 | 200.0514 | 14.6562 | 216.7224 | 0.0873 |
| oracle_path_progress | 20000 | 1 | 1500 | 10 | 5 | 0.6000 | 161.1525 | 9.4619 | 255.6213 | 0.1744 |

Under this stronger 20k controller, the matched 15-rollout support-path smoke
was:

| selector | controller steps | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 20000 | 3 | 1500 | 10 | 5 | 0.2000 | 288.9631 | 10.7373 | 127.8106 | 0.0815 |
| BMM_V_min_budget_scan_support_path | 20000 | 3 | 1500 | 10 | 5 | 0.6667 | 103.7304 | 6.5073 | 313.0434 | 0.9962 |
| support_path_only | 20000 | 3 | 1500 | 10 | 5 | 0.8000 | 57.4222 | 3.3755 | 359.3516 | 0.9955 |

This is better support-path planning evidence than the 5k-controller run, but
it still is not a BMM-specific win. The stronger controller helps support-only
as much as, or more than, it helps the BMM tie-break. Task 5 remains the main
hard case: BMM solved 0/3, support-only solved 1/3, and geometric solved 0/3.

Artifacts:

```text
exp/antmaze_large_oraclerep_bc_oraclegoal_offsets_4_8_16_32_64_h512_ln_steps5000.pkl
exp/antmaze_large_oraclerep_bc_oraclegoal_offsets_4_8_16_32_64_h512_ln_steps20000.pkl
exp/policy_retry_antmaze_large_oraclerep_geometric_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/inspect_antmaze_large_oraclerep_task2_bmm_support_oracle_start_right480.json
exp/inspect_antmaze_large_oraclerep_task2_bmm_support_oracle_start_right560.json
exp/policy_retry_antmaze_large_oraclerep_task2_right560_oraclegoal_bc_h512_ln_steps5000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right560_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.json
exp/inspect_antmaze_large_oraclerep_task2_bmm_support_oracle_start_auto_sourcegoal.json
exp/inspect_antmaze_large_oraclerep_task2_bmm_support_oracle_start_auto_local_grid_min_right.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_auto_sourcegoal_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_auto_local_grid_min_right_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep1.json
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep1.json
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.md
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.json
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.md
exp/antmaze_large_oraclerep_bc_offsets_4_8_16_32_64_h512_ln_steps5000.pkl
exp/policy_retry_antmaze_large_oraclerep_oracle_path_large_bc_h512_ln_steps5000_switch80_step1500_task1_ep1.json
exp/policy_retry_antmaze_large_oraclerep_selectors_large_bc_h512_ln_steps5000_switch80_step1500_task1_ep1.json
exp/policy_retry_antmaze_large_oraclerep_selectors_large_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep1.json
```

Interpretation: large AntMaze oraclerep is now useful evidence in both parts of
the story. The value critic learns long-horizon labels, and the oracle-goal
fixed-controller smoke shows corrected support-path planning strongly beating
geometric midpoint on a paper-listed long-horizon OGBench task. The 5-episode
validation weakens the BMM-specific claim: BMM is 48.0% versus 52.0% for
support-only. This should still be presented as a fast smoke rather than a
completed OGBench benchmark, and large AntMaze should be used as support-path
planning evidence plus positive value-learning evidence, not as a robust
BMM-tie-break win. The oracle-path upper-bound check further suggests that the
next AntMaze-large method step should improve or standardize the low-level
controller before spending more compute on high-level selector variants. The
20k-controller follow-up confirms this: stronger BC improves both oracle-path
and support-path execution, but support-only still beats BMM in the 15-rollout
follow-up.

I then isolated the remaining task-5 failure under the 20k controller. The
15-rollout support-path run showed that both BMM and support-only usually chose
local, support-valid waypoints whose remaining distance stayed around
370-380 environment steps; the closed loop then made little progress on task 5.
Start-state inspection showed that BMM support-path and support-only rank the
same first cells, while the oracle path-progress ranking points to a different
progress pattern. A small left-budget slack test (`left_budget=84`) did not fix
the issue:

| selector | task | left_budget | episodes | success | final_d | improve |
|---|---:|---:|---:|---:|---:|---:|
| BMM_V_min_budget_scan_support_path | 5 | 84 | 3 | 0.0000 | 407.5121 | 9.2616 |
| support_path_only | 5 | 84 | 3 | 0.0000 | 398.2505 | 18.5233 |
| oracle_path_progress | 5 | 84 | 3 | 0.6667 | 83.3548 | 333.4190 |

Existing value-driven selectors made more geodesic progress than support-path
BMM, but still did not solve task 5:

| selector | task | episodes | success | final_d | improve |
|---|---:|---:|---:|---:|---:|
| BMM_V_min | 5 | 3 | 0.0000 | 240.8026 | 175.9711 |
| BMM_V_min_left_gate | 5 | 3 | 0.0000 | 240.8026 | 175.9711 |
| BMM_V_min_budget_scan_left_gate | 5 | 3 | 0.0000 | 185.2328 | 231.5410 |
| oracle_path_progress | 5 | 3 | 1.0000 | 0.0000 | 416.7738 |

I then tested a one-grid-step value-frontier selector,
`BMM_V_min_budget_scan_value_frontier`. It gates candidates by learned left
reachability but allows one graph-step slack around the left horizon, which
causes the task-5 start ranking to select the same 83-step frontier cells that
oracle path-progress ranks first. This did not solve the closed-loop problem:

| selector | task | episodes | success | final_d | improve |
|---|---:|---:|---:|---:|---:|
| BMM_V_min_budget_scan_value_frontier | 5 | 3 | 0.0000 | 305.6341 | 111.1397 |
| BMM_V_min_budget_scan_left_gate | 5 | 3 | 0.3333 | 120.4013 | 296.3725 |
| oracle_path_progress | 5 | 3 | 1.0000 | 0.0000 | 416.7738 |

The next targeted variant, `BMM_V_min_budget_scan_right_progress`, uses the
learned value differently: after a local left-reachability gate, it scans the
right branch over budgets and prefers candidates whose learned remaining
horizon is smallest. At the task-5 start state, this selector ranks the same
83-step frontier cell as oracle path-progress. In a fast diagnostic run with
`--early_stop_patience=250 --early_stop_min_steps=350`, its own episodes did
not early-stop, while weak baseline failures were cut to 350-395 steps:

| selector | task | episodes | early stop | success | final_d | improve | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| BMM_V_min_budget_scan_right_progress | 5 | 3 | 250 | 0.6667 | 9.2616 | 407.5121 | 765.7 |
| BMM_V_min_budget_scan_left_gate | 5 | 3 | 250 | 0.0000 | 351.9423 | 64.8315 | 365.0 |
| oracle_path_progress | 5 | 3 | 250 | 0.3333 | 185.2328 | 231.5410 | 414.0 |

This is the first task-5 selector diagnostic where a learned non-oracle BMM
variant nearly solves the remaining large-AntMaze task. Because early stopping
can stop late recoveries, this artifact should be treated as a fast screen, not
a final paper metric; the useful signal is that right-progress BMM made two
successful full episodes and one near miss, while the left-gated value selector
was cut off early after little progress.

Full-horizon follow-up was more mixed but still positive for task 5. With no
early stopping, right-progress BMM reached 2/5 success and final distance
116.6967 on task 5. In contrast, using right-progress as the global selector
over tasks 1-5 failed badly in a one-episode all-task smoke: it solved only task
3 and got 20.0% overall success. So right-progress is not a replacement for the
support-path selector; it is a fallback candidate for the specific stall mode.

I therefore added a generic runtime fallback interface to the evaluator:
`--fallback_selector`, `--fallback_patience`, `--fallback_min_steps`,
`--fallback_min_delta`, and `--fallback_max_action_frac`. The primary selector
remains support-path, but if geodesic goal distance does not improve enough for
the configured patience, the policy replans with the fallback selector. With
`fallback_selector=BMM_V_min_budget_scan_right_progress`,
`fallback_patience=250`, `fallback_min_steps=300`, and
`fallback_min_delta=30.0`, the focused task-5 full-horizon smoke solved 3/3.
Fallback was actually used: task-5 episodes spent about 8.5%-14.0% of actions
under right-progress fallback.

The same fallback setting also solved all five tasks in a one-episode all-task
smoke. Fallback did not fire on tasks 1-4 and fired only on task 5, where it
used right-progress for 12.9% of actions. A later 15-rollout confirmation was
interrupted early because task 2 already failed twice, so the current status is:
promising fallback mechanism, but not yet robust enough for a paper table.

The task-2 failure was caused by fallback overuse. In a focused task-2 rerun,
the failed episode used fallback for 50.8% of actions and subgoal validity
dropped to 0.527. Adding `fallback_max_action_frac=0.2` fixed task 2 at 3/3,
but reduced task 5 to 1/3 because two task-5 episodes hit the cap exactly.
Relaxing the cap to 0.35 kept task 2 at 3/3 and improved task 5 to 2/3:

| setting | task 2 success | task 5 success | task 5 final_d | note |
|---|---:|---:|---:|---|
| fallback delta30, no cap | 0.6667 | 1.0000 | 0.0000 | task-2 failed episode used 50.8% fallback |
| fallback delta30, cap 0.20 | 1.0000 | 0.3333 | 268.5876 | cap too strict for task 5 |
| fallback delta30, cap 0.35 | 1.0000 | 0.6667 | 129.6629 | best current tradeoff |

After speeding up the evaluator, I reran the focused task-2/task-5 setting and
found that the exact extraction flags matter. The earlier best AntMaze-large
fallback checks used `support_frontier_left_gate=grid`,
`support_path_horizon_mode=fixed`, `final_goal_switch_distance=80`,
`subgoal_commit_steps=10`, and the environment's 1000-step truncation. Under
those corrected flags, the faster evaluator reproduced the cap-0.35 tradeoff:
task 2 stayed at 3/3 and task 5 stayed at 2/3.

I then added two opt-in controls for adaptive fallback:
`fallback_max_goal_distance`, which only allows fallback when the current
geodesic remaining distance is inside the intended right-branch horizon, and
`fallback_burst_steps`, which bounds a fallback pulse and then resets the
patience timer. The horizon gate is principled for the right-progress fallback:
with `fallback_max_goal_distance=480`, task-2 starts at distance 527.9, so
support-path primary must first bring the state into the learned right horizon
before right-progress can take over. In focused current-code reruns:

| setting | task 2 success | task 5 success | task 5 final_d | note |
|---|---:|---:|---:|---|
| cap 0.35, corrected flags | 1.0000 | 0.6667 | 138.9246 | reproduces the useful fixed-cap tradeoff; failing task-5 episode hit cap/thrashing |
| goal-distance gate 480, no cap | 1.0000 | 0.6667 | 129.6629 | suppresses task-2 fallback, but one task-5 episode overuses fallback at 70.0% actions |
| goal-distance gate 480 + burst 120 | 1.0000 | 0.6667 | 37.0466 | same success, much closer failed task-5 episode (`final_d=111.1397`) and high task-5 validity |
| goal-distance gate 480 + burst 200 | n/a | 0.3333 | 277.8492 | task-5-only screen; too long and worse |

I then added two evaluator controls to reduce policy-extraction noise. First,
`fallback_burst_min_delta` extends a fallback burst only if the previous burst
made real geodesic progress. Second, `subgoal_sample_mode=center` replaces
random per-cell waypoint samples with a deterministic in-dataset representative
closest to the cell's mean XY position. With `fallback_max_goal_distance=480`,
`fallback_burst_steps=120`, `fallback_burst_min_delta=30`, and the corrected
grid/fixed-horizon flags, the focused task-2/task-5 diagnostic solved 6/6 with
zero final distance. In that particular run, the burst extension counter stayed
at zero, so the clean result should not be over-attributed to burst extension;
it is best treated as evidence that the current extraction interface can solve
the hard pair when waypoint sampling and fallback overuse are controlled.

The deterministic waypoint diagnostic was informative but not yet a robust
all-task result:

| setting | scope | success | final_d | note |
|---|---|---:|---:|---|
| random waypoints, goal480 + burst120 + extend30 | tasks 2,5 ep3 | 1.0000 | 0.0000 | focused hard-pair success; no burst extension actually triggered |
| random waypoints, same setting | tasks 1-5 ep1 | 0.8000 | 77.7978 | tasks 1-4 solved; task 5 failed |
| center waypoints, same setting | task 5 ep3 | 1.0000 | 0.0000 | deterministic in-cell waypoints remove one source of task-5 variance |
| center waypoints, same setting | tasks 1-5 ep1 | 0.8000 | 83.3548 | tasks 1-4 solved; task 5 failed |
| center waypoints + `reset_seed_base=0` | tasks 1-5 ep1 | 0.6000 | 177.8235 | fixed hard reset exposed task-2 and task-5 failures |

This suggests that task-5 instability is not only random waypoint sampling:
the reset distribution and controller/path interaction still matter. The next
high-leverage step is to log compact trajectory traces for the fixed hard
task-2/task-5 resets and learn a fallback-continuation/confidence rule from
closed-loop progress, instead of treating the 6/6 focused smoke as final.

This makes the current status sharper: the right-progress fallback is not ready
as a final benchmark policy, but the horizon-gated burst version is a better
failure mode than a fixed cap. It preserves task 2, keeps task-5 subgoals mostly
valid, and turns one failure from no progress or cap-thrashing into a near miss.
The next cheap method step should add a learned confidence/progress criterion
for continuing a fallback burst, then confirm on all tasks 1-5.

This narrows the remaining large-AntMaze method issue: the controller can
execute an oracle-derived path on task 5, and the learned value can even choose
oracle-like first frontier cells. A learned right-branch progress scan can
recover much more of the closed-loop path when used as a fallback, but the
fallback trigger still needs a better adaptive cap or confidence signal. The
next algorithmic step should therefore refine this learned/conservative
right-progress fallback, not more seed repeats or more BC updates.

Additional artifacts:

```text
exp/inspect_antmaze_large_oraclerep_task5_start_selectors_right480.json
exp/inspect_antmaze_large_oraclerep_task5_start_selectors_right480.md
exp/policy_retry_antmaze_large_oraclerep_task5_left84_right480_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_left84_right480_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_task5_value_selectors_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_value_selectors_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/inspect_antmaze_large_oraclerep_task5_value_frontier_right480.json
exp/inspect_antmaze_large_oraclerep_task5_value_frontier_right480.md
exp/policy_retry_antmaze_large_oraclerep_task5_value_frontier_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_value_frontier_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/inspect_antmaze_large_oraclerep_task5_right_progress_right480.json
exp/inspect_antmaze_large_oraclerep_task5_right_progress_right480.md
exp/policy_retry_antmaze_large_oraclerep_task5_right_progress_faststop_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_right_progress_faststop_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_task5_right_progress_full_oraclegoal_bc_h512_ln_steps20000_ep5.json
exp/policy_retry_antmaze_large_oraclerep_task5_right_progress_full_oraclegoal_bc_h512_ln_steps20000_ep5.md
exp/policy_retry_antmaze_large_oraclerep_right_progress_full_oraclegoal_bc_h512_ln_steps20000_tasks1_5_ep1.json
exp/policy_retry_antmaze_large_oraclerep_right_progress_full_oraclegoal_bc_h512_ln_steps20000_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_tasks1_5_ep1.json
exp/policy_retry_antmaze_large_oraclerep_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_task2_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task2_support_path_right_progress_fallback_delta30_full_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap02_full_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap02_full_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap035_full_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap035_full_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap035_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_cap035_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_burst120_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_burst120_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_goal480_burst200_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_goal480_burst200_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks2_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.md
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_task5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.md
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_reset0_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_delta30_goal480_burst120_extend30_center_reset0_currentfast_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep1.md
```

## Large stitch local-progress extraction milestone

I also started the next handoff task, `pointmaze-large-stitch-v0`. The matching
short-budget value teacher learned the heldout labels well:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 20 | 0.9872 | 0.6762 | 0.9869 | 0.6846 |
| 40 | 0.9705 | 0.5676 | 0.9662 | 0.5454 |
| 80 | 0.9379 | 0.4942 | 0.9283 | 0.4667 |
| 160 | 0.9551 | 0.5088 | 0.9473 | 0.4894 |
| 320 | 0.9999 | 0.7702 | 0.9990 | 0.6728 |

The first fixed-controller smoke was not yet positive:

| selector | episodes/task | max_steps | commit | replan_d | success | final_d | final_xy | improve | subgoal_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 1 | 500 | 5 | 5 | 0.0000 | 276.7055 | 20.7765 | 19.7647 | 0.0200 |
| BMM_V_min_budget_scan_support_frontier | 1 | 500 | 5 | 5 | 0.2000 | 237.1761 | 18.8437 | 59.2940 | 0.7103 |
| oracle_path_progress | 1 | 500 | 5 | 5 | 1.0000 | 0.0000 | 0.9013 | 296.4702 | 1.0000 |

The next value-frontier smoke, still without a no-backtracking extraction rule,
was also negative despite using center subgoal representatives and stratified
candidate ordering:

| selector | episodes/task | max_steps | success | final_d | improve | subgoal_valid | select_s/ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 1 | 500 | 0.0000 | 276.7055 | 19.7647 | 0.0200 | 0.0076 |
| BMM_V_min_budget_scan_right_progress | 1 | 500 | 0.0000 | 280.6584 | 15.8117 | 1.0000 | 66.8671 |
| BMM_V_min_budget_scan_value_frontier | 1 | 500 | 0.0000 | 256.9408 | 39.5294 | 1.0000 | 66.1325 |
| oracle_path_progress | 1 | 500 | 1.0000 | 0.0000 | 296.4702 | 1.0000 | 0.0035 |

The failure was closed-loop policy extraction, not value learning. A task-1
trace showed that the learned value-frontier selector initially followed the
right path cells `0 -> 9 -> 14 -> 15 -> 16`, then oscillated between cells 15
and 16. The oracle path comparator kept moving monotonically
`0 -> 1 -> 2 -> 3 -> 10 -> 17 -> 18 -> 19 -> ...`.

I added an opt-in `--require_goal_progress` extraction gate. The first version
still allowed backtracking when the learned selector masked all forward
candidates; the corrected version uses a diagnostic local-progress fallback:
when no learned-scored candidate both survives and reduces geodesic distance to
the goal, choose a forward-progress candidate that is locally reachable within
the left budget. With `choice_cache_mode=cell`, this also makes evaluation fast
enough for iteration.

Under the same fixed offset-BC controller and the same 5-task set, the matched
large-stitch result is now:

| selector | episodes/task | progress gate | cache | success | final_d | improve | subgoal_valid | select_s/ep |
|---|---:|---|---|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | yes | cell | 0.0000 | 245.0820 | 51.3882 | 0.0181 | 0.0020 |
| support_path_only | 3 | yes | cell | 0.2000 | 241.1291 | 55.3411 | 0.0922 | 0.0538 |
| BMM_V_min_budget_scan_value_frontier | 1 | yes | cell | 1.0000 | 0.0000 | 296.4702 | 1.0000 | 2.7169 |
| BMM_V_min_budget_scan_value_frontier | 3 | yes | cell | 1.0000 | 0.0000 | 296.4702 | 1.0000 | 0.8814 |

This is a useful paper-facing milestone for the PointMaze large-stitch task:
the learned BMM value-frontier score is needed under the matched gate, because
geometric midpoint and support-path-only controls remain weak. The caveat is
important: this is not pure ungated actor extraction. It is hierarchical
policy extraction with a local no-backtracking/geodesic-progress rule, fixed
low-level BC, and cached deterministic subgoal choices.

```text
exp/policy_retry_large_stitch_right_progress_center_stratified_step500_ep1.json
exp/inspect_large_stitch_right_progress_center_stratified.md
exp/trace_large_stitch_task1_value_frontier_step150.md
exp/trace_large_stitch_task1_oracle_path_step150.md
exp/trace_large_stitch_task1_value_frontier_localprogress_step150.md
exp/policy_retry_large_stitch_value_frontier_localprogress_step500_ep1.json
exp/policy_retry_large_stitch_value_frontier_localprogress_step500_ep3.json
exp/policy_retry_large_stitch_controls_localprogress_step500_ep1.json
exp/policy_retry_large_stitch_controls_localprogress_step500_ep3.json
exp/policy_retry_large_stitch_support_path_grid_start_subgoal_inspection.md
```

## Policy extraction clarification

The BC controller is not the same object as policy extraction in the TRL code.

- In `agents/trl.py`, policy extraction is the actor side of the algorithm:
  `pe_type=rpg`, `pe_type=frs`, or `pe_type=discrete`. PointMaze in
  `hyperparameters.sh` uses RPG with `--agent.pe_type=rpg` and
  `--agent.rpg.alpha=10`.
- The scripted BC controller here is a separate goal-conditioned local
  controller trained on dataset action targets for fixed future offsets.
- For an apples-to-apples value-function paper, it makes sense to fix the
  policy-extraction/controller layer. But if the method is hierarchical subgoal
  planning, the fixed low-level controller must actually follow intermediate
  subgoals; the paper-style final-goal actor is not automatically suitable.

Fast controls:

| setup | controller / extraction | success | note |
|---|---|---:|---|
| BMM direct | BMM-RPG, 5k updates | 0.2000 | `exp/policy_retry_bmm_rpg_fast_eval_tasks1_5_ep1.*` |
| TRL direct | TRL-RPG final-goal actor, 5k updates | 0.4000 | `exp/policy_retry_trl_rpg_fast_eval_tasks1_5_ep1.*` |
| TRL direct | TRL-RPG geometric future-goal actor, 5k updates | 0.6000 | `exp/policy_retry_trl_rpg_geom_eval_tasks1_5_ep1.*` |
| TRL direct | TRL-FRS geometric future-goal actor, 5k updates | 0.2000 | `exp/policy_retry_trl_frs_geom_eval_tasks1_5_ep1.*` |
| BMM subgoals | GCFBC controller, 5k/50k updates | 0.0000 | low-level controller failed |
| BMM subgoals | TRL-RPG final-goal actor | 0.0000 | not trained for subgoals |
| BMM subgoals | TRL-RPG geometric-goal actor | 0.0000 | BMM improved distance but not exact success |
| BMM subgoals | TRL-FRS geometric-goal actor | 0.0000 | held subgoals, BMM improved distance but not exact success |
| BMM subgoals | custom offset-BC controller | 0.7333 to 0.8000 | current successful path |

Interpretation: in these fast controls, simply swapping in TRL's direct policy
extraction does not recover the hierarchical BC result. RPG is the stronger
direct policy among these smokes, but neither RPG nor FRS is currently a good
drop-in low-level controller for BMM subgoals. The current positive result
depends on a local controller trained to follow intermediate goals. That
controller can be framed as a fixed low-level policy layer for the hierarchical
method, but it should not be described as the same extraction used by the TRL
paper.

## No-controller ablation

With the nearest-neighbor/base policy path and only the near-goal switch, BMM did
not solve the tasks:

| selector | success | final_d | final_xy | improve |
|---|---:|---:|---:|---:|
| BMM_V_min | 0.0000 | 89.7409 | 11.5062 | 84.4620 |

Artifact:

```text
exp/policy_retry_nn_goal_switch20_bmm_min_tasks1_5_ep3.json
exp/policy_retry_nn_goal_switch20_bmm_min_tasks1_5_ep3.md
```

This answers the near-goal-switch concern: the switch alone is not enough. The
low-level controller is doing real work.

## Interpretation

This changes the paper direction. The new viable algorithm is not a flat
action-conditioned BMM actor. It is:

```text
Budgeted Max-Min Subgoal Planning:
learn budgeted reachability with the BMM objective,
select subgoals with min(V_h(s,w), V_{H-h}(w,g)),
execute them with a local goal-conditioned controller,
and switch to the final goal when near enough.
```

The important point for a paper is that BMM is now improving a long-horizon
control outcome, not only a heldout classifier:

```text
navigate, support-gated BMM: 96.0% success, final_d 0.0
navigate, oracle-gated BMM: 96.0-100.0% success, final_d 0.0
navigate, geometric midpoint: 40.0% success, final_d 95.8
stitch, value-only BMM: 53.3% success, final_d 50.0
stitch, support-gated BMM: 100.0% success, final_d 0.0
stitch, geometric midpoint: 20.0-40.0% success, final_d about 100
large navigate value critic: H320 AUC 0.9996, gap 0.8325
large navigate short-budget value critic: H320 AUC 0.9992, gap 0.8052
large navigate support-frontier BMM: 100.0% success, final_d 0.0
large navigate geometric midpoint: 20.0% success, final_d 222.0
large navigate oraclerep value critic: H320 AUC 0.9969, gap 0.9126
large navigate oraclerep ep5 support-path BMM: 100.0% success, final_d 0.0
large navigate oraclerep ep5 support-path-only control: 100.0% success, final_d 0.0
large navigate oraclerep support-frontier BMM: 80.0% success, final_d 31.7
large navigate oraclerep ep5 geometric midpoint: 12.0% success, final_d 248.1
antmaze medium value critic: H80/H160/H240 AUC 0.9220/0.9169/0.9355
antmaze medium support-path BMM: 100.0% success, final_d 0.0
antmaze medium support-path-only control: 93.3% success, final_d 7.7
antmaze medium geometric midpoint: 6.7% success, final_d 199.2
antmaze large oraclerep value critic: H80/H160/H240/H320 AUC 0.9387/0.9308/0.9668/0.9810
antmaze large oraclerep oracle-goal BC ep5 policy: geometric 0.0%, BMM 48.0%, support-only 52.0%
large stitch ungated support-frontier/value-frontier BMM: 20.0%/0.0% success; oracle path: 100.0%
large stitch local-progress value-frontier BMM: 100.0% success, final_d 0.0 over 15 rollouts
large stitch local-progress controls: geometric 0.0%, support-path-only 20.0%
```

The evidence is still a fast smoke, not a full benchmark. But it is now a
fixed-controller comparison on navigate plus a matching-value OGBench stitch
expansion, a 15-rollout large-navigate support-frontier result, and a
paper-listed large-navigate oraclerep 25-rollout validation, with a preliminary but positive
large-AntMaze oraclerep value-and-policy diagnostic and a positive large-stitch
local-progress extraction result. The key
improvement over the previous state is that the strongest policy rows no longer
require a grid-geodesic oracle gate; they can use offline dataset-support gates
and frontier distances, with geodesic selectors retained as oracle comparators.

## Connection to value-error diagnostics

The revised paper story should keep two claims separate:

1. Value-function target claim: BMM's max-min target reduces heldout
   reachability error for long budgets in the controlled diagnostics.
2. Control claim: when a fixed local BC controller is available, the BMM value
   function can select better long-horizon subgoals than a geometric midpoint
   baseline.

Relevant existing value-error artifacts:

```text
BMM_TRL_PAPER_EXPERIMENT_RESULTS_20260612_013139.md
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
```

Key value-error rows from the current paper result note:

| setting | comparison | seeds | delta AUC | delta gap | delta BCE | delta ECE |
|---|---|---:|---:|---:|---:|---:|
| grid-cell H8 | B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 |
| env-step H160 | B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 | -0.0380 |
| product ablation H8 | B-P | 0,1,2 | +0.0011 | +0.0050 | -0.0265 | -0.0057 |

These rows support "lower heldout value/reachability error." The adaptive
budget-scan fixed-controller results support "the lower-error value function can
improve long-horizon subgoal control" on PointMaze medium, with an explicit
caveat that the current controller is fixed and the support-gate margin needs
task-level validation.

## Next fast iterations

1. Use `BMM_V_min_budget_scan_support_gate` as the default paper selector for
   medium PointMaze policy tables, with the grid-geodesic gate reported as an
   oracle comparator.
2. Use `subgoal_commit_steps=10` and `subgoal_replan_distance=20` for further
   adaptive-selector smokes; this setting now has a 25-rollout comparison
   against geometric midpoint.
3. Tune or learn the support-gate margin instead of hand-setting it per task.
   Navigate used `support_gate_left_frac=0.75`; stitch used `1.0`.
4. Treat `BMM_V_min_budget_scan_support_path` with local grid feasibility as the
   default large-oraclerep PointMaze selector. The current oraclerep result is
   15/15, but it is still one value seed and one controller checkpoint.
5. Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` or a lower
   `XLA_PYTHON_CLIENT_MEM_FRACTION` for future JAX runs. The current small
   experiments can otherwise reserve about 20GB on a 24GB GPU due to JAX
   preallocation.
6. Keep BC as the fixed low-level controller for now:
   - conservative path: fixed local goal-conditioned controller, same across all
     subgoal selectors;
   - TRL RPG/FRS extraction remains a secondary control to revisit later.
7. For large PointMaze, keep the short left budget and use support-frontier or
   support-path progress rather than the old complete left/right gate unless
   the right budget covers the full remaining task distance.
8. For AntMaze, keep the stronger fixed-BC result as a positive smoke, but
   standardize the controller protocol next. The current large-oraclerep result
   uses a 5k-step 512x512x512 layer-norm oracle-goal BC controller, a final-goal
   switch at distance 80, and a corrected support right horizon of 480.
9. If medium remains positive after task-4 debugging, train the value critic and
   BC controller from scratch in a single scripted pipeline and compare against
   geometric midpoint.
10. Keep runs short and use `conda run --no-capture-output` for live progress on
   any check that may exceed a minute.

## Fast-evaluation notes for AntMaze-large

The slow AntMaze-large evaluations are not slow because `env.step` is expensive.
In the 15-rollout fresh-env commit-10 run, the aggregate timing was:

```text
wall_s/ep 7.6962
select_s/ep 4.9381
action_s/ep 2.4164
env_s/ep 0.2595
```

So the bottleneck is policy extraction during rollout: repeated value-network
subgoal scoring plus one-at-a-time BC action inference. The environment itself
is only a small fraction of the wall time.

I added two disabled-by-default diagnostic speed knobs:

```text
--choice_cache_mode cell
--stop_on_grid_goal_distance 0
```

The cell cache is useful for triage but should not be treated as an official
policy score: on the fixed 5-task ep1 AntMaze-large smoke it reduced wall time
from `7.5451` to `4.9393` seconds/episode and selection time from `5.0638` to
`2.4743`, but changed task 5 from success to failure. It is therefore a
debugging accelerator, not a reporting setting.

A cleaner speed/behavior knob is a longer subgoal commitment. With
`subgoal_commit_steps=20`, no cache, and grid stopping disabled, the same fixed
5-task ep1 fresh-env smoke solved all tasks and reduced average wall time to
`6.0279` seconds/episode:

```text
commit10 no cache: success 1.0, final_d 0.0, wall_s/ep 7.5451, select_s/ep 5.0638
commit20 no cache: success 1.0, final_d 0.0, wall_s/ep 6.0279, select_s/ep 3.4471
cell cache: success 0.8, final_d 50.0129, wall_s/ep 4.9393, select_s/ep 2.4743
```

For fast debugging, commit-20 is acceptable for quick mechanism checks, but
rerun any promising setting with commit-10 before interpreting success rates.
Use the cell cache only for mechanism inspection or quick failure localization,
and rerun promising settings with `--choice_cache_mode none` before reporting
them.

I also added `--bc_inference {numpy,jax}` to
`scripts/eval_bmm_subgoal_bc_controller.py`, defaulting to `numpy`. The custom
BC controller is a deterministic MLP, so NumPy inference avoids one JAX dispatch
per environment step while matching JAX actions. On the restored AntMaze-large
512x512x512 layer-norm oracle-goal BC checkpoint, the NumPy path matched JAX
within `7.2e-7` max absolute error and reduced CPU single-action latency from
about `0.322 ms` to `0.137 ms`. This reduces the action-inference overhead, but
the main bottleneck for BMM-primary runs remains value-based subgoal selection.

## AntMaze-large switch/candidate/fallback follow-up

The commit-20 speed result did not hold up as a robustness improvement. In a
15-rollout fresh-env validation with the same no-cache settings, commit-20
dropped to 66.7% success:

```text
commit10, switch80, cand64, fallback: env 80.0%, grid 86.7%, final_d 33.3
commit20, switch80, cand64, fallback: env 66.7%, grid 66.7%, final_d 74.1
```

So commit-20 is only a diagnostic speed knob, not the paper setting.

I then tested fixes targeted at the known task-1/task-2 failures:

```text
switch40 on tasks 1,2: task 1 fixed at 3/3, task 2 stayed 2/3
switch40 + fallback cap 0.2 on tasks 1,2: task 1 3/3, task 2 stayed 2/3
switch40 + no fallback on tasks 1,2: task 1 3/3, task 2 stayed 2/3
switch40 + no fallback + cand128 on task 2: task 2 fixed at 3/3
switch40 + no fallback + cand128 on task 5: task 5 fell to 1/3
```

The interpretation is that task 1 benefits from a later final-goal switch
(`40` instead of `80`), task 2 needs wider deterministic candidate coverage,
and task 5 still needs learned right-progress fallback or another learned
closed-loop repair.

Full 15-rollout fresh-env results:

```text
BMM support-path + right-progress fallback, cand64, switch40:
  env 80.0%, grid 86.7%, final_d 29.6
  per task env: task1 2/3, task2 3/3, task3 3/3, task4 3/3, task5 1/3

BMM support-path + right-progress fallback, cand128, switch40:
  env 80.0%, grid 80.0%, final_d 75.9
  per task env: task1 3/3, task2 2/3, task3 2/3, task4 3/3, task5 2/3

support-path-only control, cand64, switch40:
  env 80.0%, grid 80.0%, final_d 55.6
  per task env: task1 2/3, task2 3/3, task3 3/3, task4 3/3, task5 1/3
```

This is a modest learned-value advantage in grid success/final distance, not an
AntMaze-large success-rate win: BMM+fallback improves final distance from
`55.6` to `29.6` and grid success from `80.0%` to `86.7%` over support-only, but
env success remains tied at `80.0%`. The safe paper claim is still that
AntMaze-large is a partially positive stress test, not a solved benchmark.

New artifacts:

```text
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_center_stratified64_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_center_stratified128_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_only_center_stratified64_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/diagnose_antmaze_large_oraclerep_task2_commit10_switch40_nofallback_cand128_freshenv_center_stratified_reset0.json
exp/diagnose_antmaze_large_oraclerep_task5_commit10_switch40_nofallback_cand128_freshenv_center_stratified_reset0.json
```

## AntMaze-large conditional fallback dead end

I tried simple conditional fallback variants to repair task 5 without hurting
the support-path successes. None improved the task-5 bottleneck:

```text
BMM support-path primary, right-progress fallback, min active right distance 320:
  task5 env 1/3, grid 1/3, final_d 176.0

BMM support-path primary, right-progress fallback, fallback action cap 0.2:
  task5 env 1/3, grid 1/3, final_d 268.6

BMM support-path primary, support-path-only fallback:
  task5 env 0/3, grid 0/3, final_d 296.4

support-path-only primary, right-progress fallback:
  task5 env 1/3, grid 1/3, final_d 268.6
```

These are worse than the best cand64/switch40 BMM+right-progress result for
task 5 (`env 1/3`, `grid 2/3`, `final_d 138.9`) and worse than the broad
cand64/switch40 comparison (`env 80.0%`, `grid 86.7%`, `final_d 29.6`).

The immediate conclusion is that the remaining AntMaze-large bottleneck is not
solved by simple fallback gating. The next promising direction is a better
closed-loop repair policy for the task-5 bottleneck, or an explicitly learned
fallback trigger trained/evaluated on heldout starts, rather than thresholding
the current heuristic features.

Negative artifacts:

```text
exp/diagnose_antmaze_large_oraclerep_task5_condfallback_right320_switch40_cand64_freshenv_center_stratified_reset0.json
exp/diagnose_antmaze_large_oraclerep_task5_fallback_cap02_switch40_cand64_freshenv_center_stratified_reset0.json
exp/diagnose_antmaze_large_oraclerep_task5_supportonly_fallback_switch40_cand64_freshenv_center_stratified_reset0.json
exp/diagnose_antmaze_large_oraclerep_task5_supportonly_primary_rightprogress_fallback_switch40_cand64_freshenv_center_stratified_reset0.json
```

## AntMaze-large delayed learned repair milestone

The latest fixed-reset full-horizon AntMaze-large result is good enough to stop
tuning this task for now. The best current setting uses a conservative
`support_path_only` primary selector, then switches to the learned
`BMM_V_min_budget_scan_right_progress` repair only after a long stall
(`fallback_patience=500`, `fallback_min_steps=500`). Under the same fresh-env,
fixed-reset, no-cache, `commit=10`, `switch40`, `cand64` protocol:

```text
support-path primary + delayed learned right-progress repair:
  env/grid success 86.7% (13/15), final_d 35.2
  per task env: task1 3/3, task2 3/3, task3 3/3, task4 2/3, task5 2/3

support-path-only, no learned repair:
  env/grid success 80.0% (12/15), final_d 55.6
  per task env: task1 2/3, task2 3/3, task3 3/3, task4 3/3, task5 1/3

BMM support-path primary + immediate right-progress fallback:
  env success 80.0% (12/15), grid success 86.7%, final_d 29.6
  per task env: task1 2/3, task2 3/3, task3 3/3, task4 3/3, task5 1/3
```

This is the right paper-facing status: learned BMM reachability contributes as
a delayed closed-loop repair on top of the conservative support path. It is not
yet evidence that BMM-primary extraction alone is robust; a focused BMM-primary
delayed-fallback task-5 rerun still reached only 1/3 success. For the next
iteration, freeze AntMaze-large at 13/15 and broaden the benchmark/control set
instead of chasing 15/15 on this single task.

Artifacts:

```text
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_supportonly_primary_rightprogress_fallback_pat500_switch40_cand64_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_only_center_stratified64_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/policy_retry_antmaze_large_oraclerep_tasks1_5_support_path_right_progress_fallback_center_stratified64_switch40_reset0_freshenv_commit10_nocache_gridfixed_oraclegoal_bc_h512_ln_steps20000_ep3.json
exp/diagnose_antmaze_large_oraclerep_task5_bmm_primary_rightprogress_fallback_pat500_switch40_cand64_freshenv_center_stratified_reset0.json
```

## Scene-play oraclerep graph-value first pass

After freezing AntMaze-large and validating large-stitch, I started the next
handoff environment, `scene-play-oraclerep-v0`. The existing maze grid path does
not apply here, so I added a generic OGBench dataset inspector and generalized
the dataset-transition graph builder to bin arbitrary vector representations.

Dataset inspection:

```text
train observations: (1001000, 40), actions: (1001000, 5)
train oracle_reps: (1001000, 7)
train valid transitions: 1000000 / 1001000, terminals: 2000
validation observations: (100100, 40), actions: (100100, 5)
validation oracle_reps: (100100, 7)
validation valid transitions: 100000 / 100100, terminals: 200
oracle_rep step norm median: train 0.0972, validation 0.0942
```

Using `oracle_reps`, all seven dimensions, and graph bin size factor 8, the
observed-transition support graph is compact enough for all-pairs graph
distances:

```text
nodes: 6070
edges: 14662
env_steps_per_graph_edge: 8.0
max graph distance: 248.0
mean graph distance: 83.9
```

A 500-update state-value BMM smoke on graph-reachability labels is positive:

| H | heldout AUC | gap | ensemble-min AUC | ensemble-min gap | Euclidean AUC |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8781 | 0.3714 | 0.8771 | 0.3850 | 0.4493 |
| 32 | 0.9264 | 0.4975 | 0.9276 | 0.5130 | 0.5723 |
| 64 | 0.9382 | 0.5404 | 0.9411 | 0.5441 | 0.4847 |

This is not yet a scene-play policy result. It is a useful non-maze value
diagnostic: the same budgeted reachability classifier can learn support-graph
labels in a 7-D oracle representation, and Euclidean distance in that
representation is not a strong baseline. The next method step is to add either
budget-holdout comparisons or a fixed low-level controller/extraction interface
for scene-play before claiming long-horizon control beyond maze domains.

Artifacts:

```text
scripts/inspect_ogbench_dataset.py
exp/bmm_scene_play_oraclerep_graph_factor8.npz
exp/bmm_scene_play_oraclerep_graph_factor8_step0.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_16_32_64_200.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_16_32_64_500.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_16_32_64_500/params_500.pkl
```

## Scene-play Q/V budget-holdout first pass

I extended the action-value Q/V budget-holdout trainer to the same generic
support-graph interface used by the scene-play value diagnostic. Two fixes were
needed before the run was meaningful:

- cache graph source/goal lookup tables in the Q trainer, otherwise sampling
  repeatedly rebuilt million-row bin maps;
- use the same goal representation as `GCDataset` for Q labels and Q/V
  witnesses, i.e. `oracle_reps` when the dataset provides them.

The first scalar-budget smoke is a useful implementation check but not a
discriminative algorithmic result. With direct Q labels only at H=16 and H=32
and H=64 held out, all variants reached similar H64 AUC:

| Budget feature | Variant | H64 AUC | H64 gap | Note |
|---|---|---:|---:|---|
| `log_scalar` | A no parent/no trans | 0.8759 | 0.2048 | smooth budget extrapolation already strong |
| `log_scalar` | B max-min Q/V | 0.8699 | 0.2151 | tied with product |
| `log_scalar` | P product Q/V | 0.8696 | 0.2133 | tied with max-min |
| `log_scalar` | F V-next distill | 0.8773 | 0.2077 | tied with A |

I then made the holdout harder by using `budget_feature=log_scalar_onehot`.
This removes the strongest smooth interpolation path for the heldout H64 output.
A onehot state-value teacher trained cleanly with direct graph labels:

| H | onehot value AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 16 | 0.8767 | 0.4522 | 0.8785 | 0.4586 |
| 32 | 0.9490 | 0.6194 | 0.9495 | 0.6182 |
| 64 | 0.9471 | 0.6103 | 0.9441 | 0.6042 |

The onehot Q/V holdout is more informative. H64 AUC improves only modestly, but
the confidence gap recovers strongly when Q/V transitive labels are used:

| Variant | H64 AUC | H64 gap | ensemble-min AUC | ensemble-min gap |
|---|---:|---:|---:|---:|
| A no parent/no trans | 0.8781 | 0.1331 | 0.8745 | 0.1094 |
| B max-min Q/V | 0.8898 | 0.4807 | 0.8931 | 0.4829 |
| P product Q/V | 0.8900 | 0.4678 | 0.8958 | 0.4642 |
| F V-next distill | 0.8793 | 0.1347 | 0.8753 | 0.1101 |

Interpretation: scene-play now gives a non-maze support-graph budget-holdout
sanity check where transitive Q/V supervision transfers shorter-budget
knowledge to a heldout longer budget. It is not a max-min-specific win, because
the product target is tied with BMM on this smoke. The result is still useful
for the paper as broader evidence that the support-graph reachability setup and
Q/V holdout interface work beyond maze coordinates.

Artifacts:

```text
exp/bmm_scene_play_graph_qv_holdout_h64_seed0_smoke/summary.csv
exp/bmm_scene_play_graph_qv_holdout_h64_seed0_smoke/summary.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_500.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_500/params_500.pkl
exp/bmm_scene_play_graph_qv_holdout_h64_onehot_seed0_smoke/summary.csv
exp/bmm_scene_play_graph_qv_holdout_h64_onehot_seed0_smoke/summary.json
```

## Scene-play H128 onehot Q/V follow-up

To iterate faster, I added an all-pairs graph distance-matrix cache next to the
Scene-Play support graph:

```text
exp/bmm_scene_play_oraclerep_graph_factor8_distance_matrix.npz
```

This avoids recomputing 6070-source BFS and full graph statistics for every
variant. A zero-update smoke verified the cached matrix path:

```text
exp/bmm_scene_play_oraclerep_graph_factor8_cache_smoke_step0.json
```

I then trained a matching onehot value teacher for budgets H=16,32,64,128. The
teacher is strong at H128, so failures in the Q holdout are not caused by an
invalid frozen V branch:

| H | onehot value AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 16 | 0.8802 | 0.4619 | 0.8783 | 0.4555 |
| 32 | 0.9503 | 0.6094 | 0.9510 | 0.6085 |
| 64 | 0.9396 | 0.5927 | 0.9391 | 0.5895 |
| 128 | 0.9552 | 0.7068 | 0.9545 | 0.7120 |

The harder Q holdout supervises only H=16,32 directly and uses transitive
parents at H=64,128. This is a better long-horizon diagnostic than the H64-only
smoke: the no-transitive baseline almost collapses at H128 confidence.

| Variant | H64 AUC | H64 gap | H128 AUC | H128 gap | H128 ensemble-min AUC | H128 ensemble-min gap |
|---|---:|---:|---:|---:|---:|---:|
| A no parent/no trans | 0.8823 | 0.1378 | 0.7149 | 0.0075 | 0.7076 | 0.0055 |
| B max-min Q/V | 0.8768 | 0.4501 | 0.8254 | 0.2752 | 0.8306 | 0.2393 |
| P product Q/V | 0.8790 | 0.4370 | 0.8220 | 0.2363 | 0.8261 | 0.1998 |
| F V-next distill | 0.8824 | 0.1390 | 0.7189 | 0.0077 | 0.7087 | 0.0056 |

I reran the same H64/H128 onehot holdout with seed 1. It preserves the same
ordering:

| Variant | H64 AUC | H64 gap | H128 AUC | H128 gap | H128 ensemble-min AUC | H128 ensemble-min gap |
|---|---:|---:|---:|---:|---:|---:|
| A no parent/no trans | 0.8785 | 0.1274 | 0.6179 | 0.0054 | 0.6003 | 0.0031 |
| B max-min Q/V | 0.8916 | 0.4757 | 0.8100 | 0.1959 | 0.8171 | 0.1803 |
| P product Q/V | 0.8943 | 0.4702 | 0.8038 | 0.1576 | 0.8132 | 0.1422 |
| F V-next distill | 0.8804 | 0.1297 | 0.6199 | 0.0056 | 0.6090 | 0.0032 |

Two-seed H128 means:

| Variant | H128 AUC mean | H128 gap mean | H128 ensemble-min AUC mean | H128 ensemble-min gap mean |
|---|---:|---:|---:|---:|
| A no parent/no trans | 0.6664 | 0.0065 | 0.6540 | 0.0043 |
| B max-min Q/V | 0.8177 | 0.2355 | 0.8239 | 0.2098 |
| P product Q/V | 0.8129 | 0.1969 | 0.8197 | 0.1710 |
| F V-next distill | 0.6694 | 0.0067 | 0.6589 | 0.0044 |

Interpretation: this is the strongest Scene-Play diagnostic so far. Transitive
Q/V supervision reliably recovers heldout H128 confidence where both the
no-transitive baseline and V-next-only distillation fail. Max-min BMM has a
consistent but modest edge over product on H128 gap in the two-seed smoke. The
paper-facing claim should remain cautious: Scene-Play supports non-maze
support-graph value learning and long-budget Q/V transfer, with suggestive
evidence that max-min is better calibrated than product at the longest heldout
budget.

## Scene-play train-only support-graph check

The previous Scene-Play graph was built from train plus validation
representations. That is fine as an engineering diagnostic, but it weakens the
offline-clean interpretation. I therefore added `--graph_build_mode=train_only`
for both the state-value and Q trainers. Validation representations are mapped
to existing train-support bins; unmapped validation states are excluded from
graph-distance labels rather than adding validation nodes or edges.

With the same `oracle_reps`, all seven dimensions, and graph bin size factor 8,
the train-only graph is still well covered:

```text
nodes: 5864
edges: 13836
validation mapped states: 99090 / 100100 = 0.9899
env_steps_per_graph_edge: 8.0
max graph distance: 240.0
mean graph distance: 84.2
```

A 500-update onehot state-value teacher trained cleanly on the train-only graph:

| H | onehot value AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 16 | 0.9103 | 0.4815 | 0.9097 | 0.4695 |
| 32 | 0.9352 | 0.6086 | 0.9370 | 0.6023 |
| 64 | 0.9493 | 0.6384 | 0.9485 | 0.6303 |
| 128 | 0.9734 | 0.7569 | 0.9722 | 0.7520 |

The train-only Q/V holdout uses direct Q labels only at H=16 and H=32, with
H=64 and H=128 held out. Seeds 0, 1, and 2 now give:

| Variant | H64 AUC | H64 gap | H128 AUC | H128 gap | H128 ensemble-min AUC | H128 ensemble-min gap |
|---|---:|---:|---:|---:|---:|---:|
| A no parent/no trans | 0.8687 | 0.1253 | 0.6171 | 0.0128 | 0.6357 | 0.0093 |
| B max-min Q/V | 0.8781 | 0.3959 | 0.7618 | 0.1896 | 0.7594 | 0.1650 |
| P product Q/V | 0.8793 | 0.3794 | 0.7568 | 0.1563 | 0.7530 | 0.1317 |
| F V-next distill | 0.8688 | 0.1265 | 0.6181 | 0.0130 | 0.6360 | 0.0096 |

Interpretation: this addresses the main offline-clean concern for the
Scene-Play diagnostic. Even when the support graph is built from train only,
transitive Q/V supervision transfers to the heldout H128 budget, while
V-next-only distillation does not. Max-min BMM remains directionally better than
product on H128 confidence, but the gap is modest. The stronger claim is
transfer through the budgeted reachability/Q-V interface; the max-min-over-product
claim should rely on the broader aggregate as suggestive evidence, not a
standalone conclusion.

Artifacts:

```text
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_128_500.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_128_500/params_500.pkl
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.csv
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.json
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed1_smoke/summary.csv
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed1_smoke/summary.json
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8.npz
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_distance_matrix.npz
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_step0.json
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500.json
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500/params_500.pkl
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.csv
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.json
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed1_smoke/summary.csv
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed1_smoke/summary.json
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed2_smoke/summary.csv
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed2_smoke/summary.json
```

## Scene-play graph-subgoal BC policy smoke

I added a narrow Scene-Play policy evaluator rather than reusing the maze
evaluator. The new script reads the online 7-D oracle representation from
`env.unwrapped.compute_oracle_observation()`, maps it to the train-only
support-graph bins, trains or restores an oracle-goal BC controller, and
compares:

```text
direct_goal:        BC targets the final task goal directly.
support_path_only:  graph path-progress subgoals, no learned value tie-break.
BMM_support_path:   graph path-progress subgoals with BMM min-score tie-break.
```

A two-step interface smoke passed first and verified reset, oracle-rep mapping,
BC action inference, graph-distance logging, and JSON/markdown writing.

The first real policy smoke used a 512x512x512 layer-norm oracle-goal BC
controller trained for 2k updates and a 300-step cap. It did not solve any
Scene-Play task:

| selector | success | final graph d | graph improve | final rep d | rep improve |
|---|---:|---:|---:|---:|---:|
| direct_goal | 0.0000 | 86.4 | 4.8 | 4.2830 | 0.3665 |
| support_path_only | 0.0000 | 92.8 | 0.0 | 4.5477 | -0.0019 |
| BMM_support_path | 0.0000 | 92.8 | 3.2 | 4.4490 | 0.1865 |

I then trained the same controller for 10k updates and evaluated the full
750-step Scene-Play task horizon, one episode per task. This solves only the
easiest task:

| selector | success | final graph d | graph improve | final rep d | rep improve |
|---|---:|---:|---:|---:|---:|
| direct_goal | 0.2000 | 59.2 | 33.6 | 3.1293 | 1.4337 |
| support_path_only | 0.2000 | 76.8 | 16.0 | 3.4005 | 1.1625 |
| BMM_support_path | 0.2000 | 65.6 | 27.2 | 3.6101 | 0.9529 |

Per-task success was 1/1 only on task 1 for all three selectors. BMM reached
task 1 faster than the controls and improved task 3 and task 5 graph distance
relative to support-only, but it regressed task 4 in this seed.

A selector-only follow-up with the saved 10k BC controller and longer
`left_budget=64` did not recover policy success:

| selector | success | final graph d | graph improve | final rep d | rep improve |
|---|---:|---:|---:|---:|---:|
| support_path_only | 0.0000 | 81.6 | 11.2 | 3.6028 | 0.9603 |
| BMM_support_path | 0.0000 | 68.8 | 24.0 | 3.0636 | 1.4995 |

Interpretation: Scene-Play is still a value/Q result, not a policy result. The
10k oracle-goal BC controller can solve the easiest task and make progress, so
the evaluator is meaningful, but graph subgoals do not yet solve long-horizon
manipulation. BMM is sometimes better than support-only on final graph distance
and progress, but the effect is not reliable enough to claim Scene-Play control.
The next policy step for Scene-Play should be a stronger low-level extraction
method or a controller pretraining protocol, not more BMM selector tuning.

Artifacts:

```text
scripts/eval_bmm_scene_graph_bc_controller.py
exp/scene_graph_bc_interface_smoke.json
exp/scene_graph_bc_interface_smoke.md
exp/scene_play_oraclerep_bc_offsets128_h512_ln_steps2000.pkl
exp/scene_play_graph_bc_h512_ln_steps2000_ep1_smoke.json
exp/scene_play_graph_bc_h512_ln_steps2000_ep1_smoke.md
exp/scene_play_oraclerep_bc_offsets128_h512_ln_steps10000.pkl
exp/scene_play_graph_bc_h512_ln_steps10000_ep1_step750_smoke.json
exp/scene_play_graph_bc_h512_ln_steps10000_ep1_step750_smoke.md
exp/scene_play_graph_bc_h512_ln_steps10000_left64_ep1_step750_smoke.json
exp/scene_play_graph_bc_h512_ln_steps10000_left64_ep1_step750_smoke.md
```
