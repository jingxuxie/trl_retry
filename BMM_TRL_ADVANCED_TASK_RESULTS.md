# BMM-TRL advanced-task readiness and first smokes

Date: 2026-06-19

This note tracks the move from PointMaze/AntMaze/Scene-Play diagnostics to
harder TRL-paper tasks such as HumanoidMaze and Puzzle. The primary metric is
now rollout success rate, not only value/Q diagnostics.

Paper-ready hard-task table:

```text
exp/bmm_advanced_policy_table.md
exp/bmm_advanced_policy_table.json
```

Coverage audit for the paper-table question:

```text
exp/bmm_paper_task_coverage_audit.md
exp/bmm_paper_task_coverage_audit.json
```

Current status: the promoted hard-task table beats or matches the paper overall
on all 10 promoted rows. It does not beat every individual paper task entry:
6 of the 9 promoted rows with per-task paper references clear every task, while
HumanoidMaze-giant, Scene-Play, and AntSoccer have remaining per-task gaps.
AntSoccer beats the paper overall only as a task-routed support-only policy
suite; the best clean single-protocol full result is 52/75 (69.3%) versus the
paper's 73.0% overall row, and the best BMM-including routed suite is 55/75
(73.3%).

## Short answer

Puzzle now has strong paper-table success-rate results beyond 3x3. On the
standard Table-2 `puzzle-4x4-play-oraclerep-v0` row, a 50k-update learned local
GCFBC controller plus a structured one-press Lights Out planner reaches 97.3%
success over 75 rollouts, beating the paper TRL row's 34% overall target. On
the harder Table-1 rows, a learned local GCFBC controller plus the same
structured one-press planner reaches 100.0% success on
`puzzle-4x5-play-oraclerep-v0` over 75 rollouts, beating the paper TRL row's
97% overall target. The same protocol reaches 92.0% success on
`puzzle-4x6-play-oraclerep-v0`, beating the paper TRL row's 51% overall target
and improving every task row, including task 5 from 0% in the paper to 60%.
On the standard manipulation rows, direct local GCFBC reaches 98.7% success on
`cube-single-play-oraclerep-v0`, beating the paper TRL row's 95% overall
target. For `cube-double-play-oraclerep-v0`, direct local GCFBC is weak at
18.7%, and a one-pass sequential wrapper reaches only 38.7%. A dynamic
block-subgoal retry wrapper reaches 73.3% over 75 rollouts, beating the paper
row's 30% overall target and clearing the 70% internal target.

Puzzle-3x3 also has a strong policy-control result: BMM support-path planning
with a paper-style TRL/RPG controller reaches 100.0% success over 75 rollouts,
matching/beating the paper's 99% Puzzle-3x3 target. HumanoidMaze-medium has a
target-beating result: with the 1M-total TRL/RPG controller and an earlier
final-goal switch at graph distance 128, BMM support-path planning reaches
94.7% success over 75 rollouts at the official 2000-step OGBench horizon. This
is above the paper's 57% target. The earlier 1000-step evaluation understated
this row and made medium look anomalously weaker than large/giant.
HumanoidMaze-large now has a stronger standard Table-2 result under the same
graph-subgoal interface: BMM support-path planning with the fixed 600k
HumanoidMaze-giant TRL/RPG controller reaches 67/75 (89.3%) over the official
2000-step OGBench horizon, compared with the paper TRL row's 8% overall target.
The earlier weak large result came from a direct-transfer shortcut, and the
0/15-to-2/15 graph smoke used an artificially short 1000-step cap.
Scene-Play is now also positive as a fixed-controller graph-subgoal result:
direct local GCFBC reaches 7/15 (46.7%) in a 3-episode/task smoke, while BMM
support-path graph subgoals with the same 50k local GCFBC controller reach
66/75 (88.0%) over 15 episodes/task. This beats the paper TRL row's 77% overall
target. A matched support-path-only full control reaches 63/75 (84.0%), so the
BMM margin is modest but positive; task 5 remains weak at 53.3%.

Recommended progression:

1. Keep Puzzle-3x3 as the first positive advanced success-rate result.
2. Add Puzzle-4x4, Puzzle-4x5, and Puzzle-4x6 as puzzle success-rate wins, but
   label them as structured puzzle decomposition plus learned local control,
   not pure BMM value evidence.
3. Treat HumanoidMaze-medium as a positive advanced success-rate result, while
   still reporting its per-task failure mode.
4. Add HumanoidMaze-large as the clean standard locomaze row: 89.3% over 75
   official-horizon rollouts with BMM graph subgoals and a fixed controller.
5. Add Scene-Play as a positive fixed-controller graph-subgoal row: 88.0% over
   75 rollouts with a train-only oracle-representation graph and 50k local
   GCFBC controller, but report the matched support-only, direct-GCFBC, and
   task-5 caveats.
6. Use the 1M controller as the current HumanoidMaze-medium baseline; tune task 4 only
   if we need margin beyond the paper target.
7. Treat `humanoidmaze-giant` as a calibrated hard-row match, not a fully
   solved row. After fixing OGBench locomaze reset determinism, pure BMM
   switch128 reaches 72.00% and support-path-only reaches 76.00% over 75
   deterministic rollouts. A fixed start-distance-plus-crossing route selector
   reaches 60/75 (80.00%), just above the paper TRL row's 79% target. The
   fitted distance-plus-delta-y route selector now has four heldout offset
   smokes: 13/15 (86.67%) at offset 15 and 12/15 (80.00%) at offsets 18, 21,
   and 24.

## Data and infrastructure status

Downloaded and inspected:

```text
~/.ogbench/data/humanoidmaze-medium-navigate-v0.npz
~/.ogbench/data/humanoidmaze-medium-navigate-v0-val.npz
~/.ogbench/data/puzzle-3x3-play-v0.npz
~/.ogbench/data/puzzle-3x3-play-v0-val.npz
~/.ogbench/data/puzzle-4x4-play-v0.npz
~/.ogbench/data/puzzle-4x4-play-v0-val.npz
~/.ogbench/data/puzzle-4x5-play-v0.npz
~/.ogbench/data/puzzle-4x5-play-v0-val.npz
~/.ogbench/data/puzzle-4x6-play-v0.npz
~/.ogbench/data/puzzle-4x6-play-v0-val.npz
~/.ogbench/data/cube-single-play-v0.npz
~/.ogbench/data/cube-single-play-v0-val.npz
~/.ogbench/data/cube-double-play-v0.npz
~/.ogbench/data/cube-double-play-v0-val.npz
~/.ogbench/data/humanoidmaze-large-navigate-v0.npz
~/.ogbench/data/humanoidmaze-large-navigate-v0-val.npz
~/.ogbench/data/humanoidmaze-giant-navigate-v0.npz
~/.ogbench/data/humanoidmaze-giant-navigate-v0-val.npz
```

There is also an interrupted stale file:

```text
~/.ogbench/data/humanoidmaze-medium-navigate-v0.npz.tmp
```

It is ignored by OGBench, but it consumes about 472 MB.

GPU check:

```text
RTX 4090, 24564 MiB total, about 1264 MiB used before advanced runs.
```

During the HumanoidMaze Q/V smoke, GPU memory rose to about 20.1 GiB. This is
large but expected for the current JAX/XLA process shape plus loaded data and
compiled networks; it is not a policy-memory leak by itself.

Later short evaluations used `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which
kept observed GPU memory much lower during Puzzle diagnostics. Use this flag
for fast iteration unless a run becomes unstable.

After the 75-rollout HumanoidMaze support-path-only control finished, an
escalated `nvidia-smi` check reported about 2.25 GiB used on the RTX 4090. This
supports the interpretation that the earlier ~20 GiB reading was mostly JAX/XLA
preallocation/reservation rather than the true footprint of these small
evaluations.

During the full 75-rollout HumanoidMaze-large BMM graph evaluation, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, observed GPU memory was about 2.2 GiB.
This again supports treating the earlier 20 GiB reading as a process
reservation/preallocation artifact rather than the model's actual footprint.

During the new Puzzle-4x5/4x6 local-controller training and evaluation runs,
GPU memory was about 2.2-3.1 GiB with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
The hard-puzzle result therefore does not require the earlier 20 GiB footprint.

## Table-2 Puzzle-4x4 follow-up

`puzzle-4x4-play-oraclerep-v0` is the standard OGBench puzzle row in Table 2.
The board transition system is singular, so this row uses a direct GF(2)
linear solve rather than a matrix inverse inside
`scripts/eval_puzzle_lightsout_policy.py`. The low-level controller is a
learned local GCFBC policy trained for 50k updates with `discount=0.95`.

Training/evaluation artifacts:

```text
exp/mrl/GCFBC_puzzle4x4_local_d095_50k/sd000_20260619_131236/params_50000.pkl
exp/puzzle4x4_lightsout_gcfbc_local50k_nearest_ep15.json
exp/puzzle4x4_lightsout_gcfbc_local50k_nearest_ep15.md
```

Paper Table-2 comparison:

| environment | metric | paper TRL | ours |
|---|---:|---:|---:|
| `puzzle-4x4` | task 1 | 47% | 86.7% |
| `puzzle-4x4` | task 2 | 17% | 100% |
| `puzzle-4x4` | task 3 | 38% | 100% |
| `puzzle-4x4` | task 4 | 34% | 100% |
| `puzzle-4x4` | task 5 | 32% | 100% |
| `puzzle-4x4` | overall | 34% | 97.3% |

Rollout details:

| environment | episodes/task | max steps | press order | commit steps | success |
|---|---:|---:|---|---:|---:|
| `puzzle-4x4-play-oraclerep-v0` | 15 | 1000 | nearest | 50 | 73/75 |

## Table-2 Cube follow-up

The standard Table-2 manipulation rows are now covered by learned local GCFBC
controllers. `cube-single` works directly, while `cube-double` needs a
sequential oracle-representation block decomposition: target one cube at a
time, then refine on the final two-block goal. This should be reported as a
policy-interface/decomposition result rather than pure BMM value evidence.

Training/evaluation artifacts:

```text
exp/mrl/GCFBC_cube_single_local_d095_50k/sd000_20260619_133236/params_50000.pkl
exp/cube_single_gcfbc_local50k_eval_ep15.json
exp/cube_single_gcfbc_local50k_eval_ep15.md

exp/mrl/GCFBC_cube_double_local_d095_50k/sd000_20260619_133834/params_50000.pkl
exp/cube_double_gcfbc_local50k_eval_ep15.json
exp/cube_double_seq_gcfbc_local50k_finalzfarthest_b160_f180_ep15.json
exp/cube_double_seq_gcfbc_local50k_finalzfarthest_b160_f180_ep15.md

exp/mrl/GCFBC_cube_double_local_d095_continue200k/sd000_20260619_164614/params_100000.pkl
exp/cube_double_seq_gcfbc_local200k_dynamic_finalzfarthest_r80_p5_f100_ep15.json
exp/cube_double_seq_gcfbc_local200k_dynamic_finalzfarthest_r80_p5_f100_ep15.md

exp/mrl/GCFBC_cube_double_local_d095_continue250k/sd000_20260619_175101/params_50000.pkl
exp/cube_double_seq_gcfbc_local250k_dynamic_finalzfarthest_r80_p5_f100_ep15.json
exp/cube_double_seq_gcfbc_local250k_dynamic_finalzfarthest_r80_p5_f100_ep15.md
exp/cube_double_seq_gcfbc_local250k_dynamic_finalzfarthest_r80_p5_f100_ep3.json
exp/cube_double_seq_gcfbc_local250k_dynamic_finalzfarthest_r80_p5_f100_ep3.md

exp/mrl/GCFBC_cube_double_local_d095_continue300k/sd000_20260619_174253/params_100000.pkl
exp/cube_double_seq_gcfbc_local300k_dynamic_finalzfarthest_r80_p5_f100_ep3.json
exp/cube_double_seq_gcfbc_local300k_dynamic_finalzfarthest_r80_p5_f100_ep3.md
```

Paper Table-2 comparison:

| environment | metric | paper TRL | ours |
|---|---:|---:|---:|
| `cube-single` | task 1 | 98% | 100% |
| `cube-single` | task 2 | 97% | 100% |
| `cube-single` | task 3 | 99% | 100% |
| `cube-single` | task 4 | 93% | 100% |
| `cube-single` | task 5 | 87% | 93.3% |
| `cube-single` | overall | 95% | 98.7% |
| `cube-double` | task 1 | 73% | 100.0% |
| `cube-double` | task 2 | 23% | 80.0% |
| `cube-double` | task 3 | 30% | 80.0% |
| `cube-double` | task 4 | 3% | 6.7% |
| `cube-double` | task 5 | 18% | 100.0% |
| `cube-double` | overall | 30% | 73.3% |

Useful negative/control result: direct local GCFBC on `cube-double` reaches
only 14/75 (18.7%), and the old one-pass sequential wrapper reaches 29/75
(38.7%). The dynamic retry wrapper is therefore the component that makes this
row substantially paper-comparable, although task 4 remains the weakest case.

Longer-controller follow-up: continuing the local GCFBC controller beyond the
promoted 200k total update checkpoint does not improve the matched
dynamic-retry smoke. On the same 3-episode/task protocol, the 200k checkpoint
gets 12/15, the 250k checkpoint gets 11/15, and the 300k checkpoint gets 10/15.
Both longer checkpoints improve task 4 from 0/3 to 1/3, but they trade away
success on easier tasks. I fully evaluated the stronger 250k candidate anyway:
it ties the promoted 200k row at 55/75 (73.3%), with task 4 improving from
1/15 to 2/15 but task 1 and task 5 each losing one success. The 200k checkpoint
therefore remains the promoted cube-double row.

Flow-extraction follow-up: the stock GCFBC `sample_actions` path ignores the
`temperature` argument, so I added evaluator-only zero-noise and
temperature-scaled flow-sampling modes in `scripts/eval_cube_sequential_policy.py`.
These are useful diagnostics but do not improve the claim. Zero-noise dynamic
retry gets 7/15, zero-noise park-then-dynamic gets 8/15, temperature-scaled
dynamic retry at 0.25 gets 7/15, and temperature-scaled dynamic retry at 0.5
gets 9/15. A full task-4-only 0.5 run gets 1/15, matching the promoted task-4
count rather than improving it.

## Table-1 hard Puzzle follow-up

The paper highlights `puzzle-4x5` and `puzzle-4x6` as difficult long-horizon
rows. These are discrete Lights Out tasks, and the oracle representation is the
binary board state. I added `scripts/eval_puzzle_lightsout_policy.py`, which
solves the current binary board exactly over GF(2), then gives the learned
policy a sequence of one-press oracle-representation subgoals. The low-level
controller is a learned local GCFBC policy trained for 100k updates with
`discount=0.95` to bias actor goals toward short local moves.

This is a structured policy-extraction/decomposition result: the planner uses
known puzzle transition structure, while the learned controller executes each
local button-press subgoal. It should not be presented as pure BMM value
evidence. It is nevertheless strong evidence that the long-horizon puzzle rows
can be solved by the right hierarchical policy interface.

Training/evaluation artifacts:

```text
exp/mrl/GCFBC_puzzle4x5_local_d095_100k/sd000_20260618_221300/params_100000.pkl
exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.json
exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.md

exp/mrl/GCFBC_puzzle4x6_local_d095_100k/sd000_20260618_222520/params_100000.pkl
exp/puzzle4x6_lightsout_gcfbc_local100k_nearest_ep15.json
exp/puzzle4x6_lightsout_gcfbc_local100k_nearest_ep15.md
```

Paper Table-1 comparison:

| environment | metric | paper TRL | ours |
|---|---:|---:|---:|
| `puzzle-4x5` | task 1 | 100% | 100% |
| `puzzle-4x5` | task 2 | 99% | 100% |
| `puzzle-4x5` | task 3 | 100% | 100% |
| `puzzle-4x5` | task 4 | 99% | 100% |
| `puzzle-4x5` | task 5 | 88% | 100% |
| `puzzle-4x5` | overall | 97% | 100% |
| `puzzle-4x6` | task 1 | 100% | 100% |
| `puzzle-4x6` | task 2 | 66% | 100% |
| `puzzle-4x6` | task 3 | 67% | 100% |
| `puzzle-4x6` | task 4 | 23% | 100% |
| `puzzle-4x6` | task 5 | 0% | 60% |
| `puzzle-4x6` | overall | 51% | 92% |

Rollout details:

| environment | episodes/task | max steps | press order | commit steps | success |
|---|---:|---:|---|---:|---:|
| `puzzle-4x5-play-oraclerep-v0` | 15 | 1000 | nearest | 50 | 75/75 |
| `puzzle-4x6-play-oraclerep-v0` | 15 | 1000 | nearest | 50 | 69/75 |

The main remaining Puzzle-4x6 failure is task 5: the policy usually reduces
the board from 24 required presses to about 1.27 remaining presses, but 6 of 15
episodes hit the 1000-step cap before clearing the final buttons. This suggests
more reliable local execution, a longer cap, or a reattempt rule could close the
remaining gap.

Useful negative/control result:

```text
exp/mrl/TRL_puzzle4x5_frs_smoke100k/sd000_20260618_214939/params_100000.pkl
exp/trl_puzzle4x5_frs_smoke100k_eval_ep1.json
exp/puzzle4x5_lightsout_trl100k_nearest_ep1.json
```

A flat 100k TRL/FRS Puzzle-4x5 policy was 0/5 under direct evaluation, and only
1/5 when wrapped by the same Lights Out planner. The local GCFBC controller is
therefore the key low-level interface improvement for these hard puzzle rows.

## HumanoidMaze-giant dataset probe

Environment:

```text
humanoidmaze-giant-navigate-oraclerep-v0
```

Dataset shapes:

```text
train observations: (4001000, 69)
train actions:      (4001000, 21)
train oracle_reps:  (4001000, 2)
valid transitions:  4000000 / 4001000

validation observations: (400100, 69)
validation actions:      (400100, 21)
validation oracle_reps:  (400100, 2)
valid transitions:       400000 / 400100
```

Oracle representation:

```text
2-D XY-like representation
median one-step oracle_rep norm: about 0.050 train and validation
```

Interpretation:
HumanoidMaze-giant is the hardest continuous long-horizon target in this batch.
Unlike Puzzle-4x5/4x6, it does not have a known discrete transition solver; it
needs a support-graph/BMM planner plus a robust humanoid low-level controller.
We now have graph/value/controller results, but not a stable paper-target match.

## HumanoidMaze-giant graph/value/controller follow-up

Graph artifacts:

```text
exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_smoke.npz
exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_smoke_distance_matrix.npz
```

Graph statistics:

| metric | value |
|---|---:|
| nodes | 8,861 |
| edges | 26,919 |
| validation mapped | 400,047 / 400,100 |
| validation mapped fraction | 0.999868 |
| max hops | 224 |
| max env-step units | 1,792 |
| mean env-step units | 726.7 |

Best value teacher so far:

```text
exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_value_256_512_1024_1536_500/params_500.pkl
```

Validation metrics at 500 updates:

| H | AUC | gap |
|---:|---:|---:|
| 256 | 0.8797 | 0.1158 |
| 512 | 0.7794 | 0.1117 |
| 1024 | 0.8864 | 0.2137 |
| 1536 | 0.9610 | 0.3457 |

Controller/evaluation artifacts:

```text
exp/mrl/TRL_humanoidmaze_giant_rpg_actor_continue600k/sd000_20260619_012559/params_300000.pkl
exp/trl_humanoidmaze_giant_rpg_actor_total600k_eval_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_support_switch256_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_ep15_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch128_ep15_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_temp01_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_temp005_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch64_task1_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch64_ep3_seed10_interrupted.md
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_recover128_p200_task1_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_recover128_p200_ep3_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_recover128_p200_start1550_ep3_seed10_interrupted.md
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_ep3_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch128_ep3_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch128_ep15_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_direct_goal_ep3_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_support_switch128_ep3_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_support_switch128_ep15_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_then_support_nodirect800_ep3_seed10_detreset.json
```

Key success rates:

| protocol | episodes | success | final graph d |
|---|---:|---:|---:|
| flat TRL/RPG, 600k total | 15 | 6.67% | n/a |
| support_path_only, 600k controller, switch256 | 15 | 53.33% | 215.47 |
| BMM_support_path, 600k controller, switch256 | 15 | 86.67% | 56.00 |
| BMM_support_path, 600k controller, switch256, unpaired reset noise | 75 | 74.67% | 87.04 |
| BMM_support_path, 600k controller, switch128, unpaired reset noise | 75 | 72.00% | 73.07 |
| BMM_support_path, 800k total, switch256 | 15 | 80.00% | 27.73 |
| BMM_support_path, 1M total, switch256 | 15 | 66.67% | 204.80 |
| BMM_support_path, 600k plus local-goal fine-tune, switch256 | 15 | 53.33% | 94.40 |
| BMM_support_path, 600k controller, switch256, controller temp 0.10 | 15 | 80.00% | 36.27 |
| BMM_support_path, 600k controller, switch256, controller temp 0.05 | 15 | 73.33% | 62.93 |
| BMM_support_path, 600k controller, switch64, task 1 only | 3 | 66.67% | 80.00 |
| BMM_support_path, switch256 plus conservative recovery-to-128, task 1 only | 3 | 100.00% | 8.00 |
| BMM_support_path, switch256 plus conservative recovery-to-128 | 15 | 73.33% | 42.13 |
| BMM_support_path, switch256, deterministic reset noise | 15 | 60.00% | 120.53 |
| BMM_support_path, switch128, deterministic reset noise | 15 | 86.67% | 77.33 |
| BMM_support_path, switch128, deterministic reset noise | 75 | 72.00% | 136.96 |
| direct_goal, deterministic reset noise | 15 | 0.00% | 1181.87 |
| support_path_only, switch128, deterministic reset noise | 15 | 80.00% | 101.87 |
| support_path_only, switch128, deterministic reset noise | 75 | 76.00% | 85.55 |
| BMM then support if no direct-goal choice by 800 steps | 15 | 73.33% | 61.33 |
| start-distance gate, BMM if start graph d >= 1480 else support | 15 | 93.33% | 60.27 |
| start-distance gate, BMM if start graph d >= 1480 else support | 75 | 80.00% | 85.33 |
| start-distance gate, offset episodes 15-17 | 15 | 73.33% | 92.27 |
| start-distance gate plus force BMM on task 2, offset episodes 15-17 | 15 | 80.00% | 48.53 |
| start-distance plus crossing gate, offset episodes 15-17 | 15 | 86.67% | 48.53 |
| start-distance plus crossing gate, offset episodes 18-20 | 15 | 80.00% | 43.20 |
| fitted distance-plus-delta-y gate, offset episodes 15-17 | 15 | 86.67% | 48.53 |
| fitted distance-plus-delta-y gate, offset episodes 18-20 | 15 | 80.00% | 43.20 |
| start-distance plus crossing gate | 75 | 80.00% | 78.93 |

Additional negative controller check:

```text
exp/humanoidmaze_giant_oraclerep_bc_offsets64_h1536_ln_steps10000.pkl
```

A plain supervised oracle-goal BC controller trained for 10k updates with
offsets 1/2/4/8/16/32/64 reached BC MSE about 0.0867 but failed closed-loop
Giant control. In the interrupted smoke, `support_path_only` completed at 0/15
with mean final graph distance about 893; BMM was then stopped after starting
0/2 with large remaining distances. This suggests the current missing piece is
not a simple deterministic BC head trained on short offsets.

Evaluator/protocol note:

```text
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_nopadscore64_smoke_task1_ep1_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_padscore64_smoke_task1_ep1_seed10.json
```

OGBench locomaze reset determinism: `MazeEnv.add_noise()` uses global
`np.random.uniform`, not Gym's per-environment RNG. Therefore
`env.reset(seed=s, options={"task_id": t})` alone does not reproduce the same
start/goal noise across repeated resets. The evaluator now has
`--seed_global_reset_noise`, which calls `np.random.seed(reset_seed)` immediately
before each reset. A direct reset check confirmed this makes the same
task/episode reset graph distance, start XY, and goal XY identical across
repeated resets. Earlier Giant switch128/switch256 comparisons without this flag
should be treated as stochastic reset-noise estimates, not paired per-episode
comparisons.

`scripts/eval_bmm_scene_graph_bc_controller.py` now has an explicit
`--pad_score_batches` option. This pads BMM candidate scoring calls to a fixed
JAX batch shape and slices away padded scores before ranking. It materially
reduces per-episode evaluation overhead in the one-episode Giant task-1 smoke:
the no-padding run took 69.37s episode time and failed with final graph distance
480, while fixed-shape scoring took 31.54s episode time and succeeded with
final graph distance 8. However, this is not a transparent speed-only change:
near-tie BMM candidate scores can shift with batch shape, so padded and
unpadded BMM results must be reported as separate extraction protocols. A
follow-up fixed-shape 15-rollout screen was interrupted after task 1 reached
only 1/3 success with two far failures, so it did not justify a 75-rollout
claim run.

Switch-distance follow-up:

Lowering the Giant final-goal switch to graph distance 64 improved the isolated
task-1 screen to 2/3 success with mean final graph distance 80.0, but the
broader 15-rollout screen was stopped after reaching only 3/6 through tasks 1
and 2. This makes switch64 a negative screen rather than a new Giant candidate.

Switch-complementarity and direct-recovery follow-up:

The two earlier 75-rollout switch runs appeared to have substantial
complementary failures: switch256 succeeded on 56/75 and switch128 on 54/75,
with an apparent oracle per-episode choice of 69/75. The reset-determinism check
showed that this was not a clean paired oracle bound, because repeated
`reset(seed=...)` calls sampled different start/goal noise. Under the corrected
paired deterministic 15-rollout smoke, switch128 dominated switch256: 13/15
versus 9/15 on identical starts, with no switch256-only successes. However, the
75-rollout deterministic switch128 verification reached only 54/75 (72.00%),
so this still does not match the paper target.

Corrected deterministic controls on the same 15 starts are now available:
direct-goal TRL/RPG reaches 0/15 with mean final graph distance 1181.87,
support-path-only switch128 reaches 12/15 with mean final graph distance 101.87,
and BMM support-path switch128 reaches 13/15 with mean final graph distance
77.33. Thus BMM still has a matched planning margin over support-only in the
corrected smoke, but it is a small 1-episode margin rather than the much larger
unpaired-reset contrast.

The full corrected deterministic support-path-only control is now available:
57/75 (76.00%) with mean final graph distance 85.55. Per-task success is
40.00%, 80.00%, 66.67%, 93.33%, and 100.00% for tasks 1--5. This beats the
75-rollout BMM switch128 aggregate (54/75, 72.00%) but does not beat the paper
TRL row's 79.00%. The paired overlap with BMM is strongly complementary:
BMM-only successes account for 11 episodes, support-only successes account for
14 episodes, both solve 43, and neither solves 7. The oracle per-episode union
is therefore 68/75 (90.67%), which is a useful upper bound showing that the
current graph/controller stack can solve Giant if we can learn or diagnose the
route-family choice.

Two disabled-by-default hybrid route selectors were screened. A
`support_then_bmm` stall trigger tied support-only on a hard 6-episode window
(3/6) but did not recover the support-only failure where pure BMM succeeds. A
`bmm_then_support` no-direct timeout at 800 environment steps also tied that
hard window (3/6) and solved one episode that neither pure selector solved, but
it regressed the standard 15-rollout smoke to 11/15 (73.33%). A more aggressive
400-step timeout solved some task-1 failures but regressed other BMM-only
successes. These are negative screens for the current simple timeout rules, but
they identify the right next problem: learn or derive a reliable route selector
between conservative support-path routing and BMM reranking.

A cheaper initial route-choice diagnostic is now available:

```text
scripts/analyze_scene_graph_route_choice.py
exp/humanoidmaze_giant_route_choice_initial_features_seed10_detreset.json
```

It computes reset-time BMM and support choices without rolling out the
environment, then joins those features with completed BMM/support success
labels. The diagnostic showed that reset-time BMM and support choices are often
identical (57/75 same first bin), so reset-only route features do not explain
the full 90.67% oracle union. However, a simple calibrated rule, "use BMM when
the initial graph distance is at least 1480, otherwise use support", predicts
60/75 and was verified by an actual 75-rollout run:

```text
exp/humanoidmaze_giant_graph_trl_total600k_startdist_gate1480_ep15_seed10_detreset.json
```

This start-distance gate reaches 60/75 (80.00%) with mean final graph distance
85.33, just above the paper TRL row's 79.00% overall. Per-task success is
53.33%, 80.00%, 73.33%, 93.33%, and 100.00%. It uses BMM on about 30.67% of
route choices. A heldout offset smoke on episodes 15--17 reached only 11/15
(73.33%), showing that the threshold alone is calibration-sensitive. A
crossing-aware variant, `start_distance_cross_gate_bmm_support`, uses BMM when
the initial graph distance is at least 1480, or when the initial geometry is a
right-to-left crossing with start x >= 50 and goal_y - start_y >= 35. The fixed
implementation reaches 13/15 (86.67%) on offset episodes 15--17, 12/15
(80.00%) on offset episodes 18--20, and 60/75 (80.00%) on the original
corrected deterministic 75-rollout set:

```text
exp/humanoidmaze_giant_graph_trl_total600k_startdist_cross_gate_fixed_ep3_offset15_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_startdist_cross_gate_fixed_ep3_offset18_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_startdist_cross_gate_fixed_ep15_seed10_detreset.json
```

The earlier non-`fixed` crossing-gate offset artifact should be ignored for
claims: a wiring bug computed the route decision but did not pass it into
subgoal replanning, so that run was effectively pure BMM (`route_bmm_frac=1`).

We added an offline route-selector fitter:

```text
scripts/fit_scene_graph_route_selector.py
exp/humanoidmaze_giant_route_selector_fit_deltay_offset15_seed10.json
```

It splits the original 75 route-choice diagnostic into 60 calibration rows and
15 tuning rows, then evaluates on the offset-15 BMM/support labels. An
unrestricted three-threshold tree overfits the tuning rows and transfers poorly
to offset 15 (10/15). Restricting the candidate family to a simple
distance-plus-vertical-displacement rule selects
`source_to_goal >= 1480 OR delta_y >= 35.3224`, with label scores 50/60 on
calibration, 11/15 on tuning, and 13/15 on offset 15. Executing that fitted rule
with the first-class evaluator selector `start_distance_deltay_gate_bmm_support`
reaches 13/15 (86.67%) on offset episodes 15--17 and 12/15 (80.00%) on offset
episodes 18--20, matching the fixed crossing gate on both heldout smokes.
Two additional heldout windows reached 12/15 (80.00%) on offset episodes
21--23 and 12/15 (80.00%) on offset episodes 24--26:

```text
exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_clean_ep3_offset15_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_clean_ep3_offset18_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_clean_ep3_offset21_seed10_detreset.json
exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_clean_ep3_offset24_seed10_detreset.json
```

The fixed route selector is promising, and the constrained fitter makes the
selection procedure more defensible, but it should still be framed as calibrated
policy-extraction/route-selection until validated with a separate calibration
split or a larger heldout set.

We implemented a disabled-by-default direct-goal recovery rule in
`scripts/eval_bmm_scene_graph_bc_controller.py`: after direct-goal control
stalls, it can lower the active final-goal switch to 128 and replan through
graph subgoals. The conservative stall detector did not fire in the completed
15-rollout smoke (`direct_recovery_triggers=0`), and that run reached only
11/15 despite a 3/3 task-1 mini-screen. A forced start-gated recovery screen
for very long starts was worse, reaching only 2/6 before stopping. The lesson
is that Giant likely needs an online predictor for switch choice or a better
near-goal recovery signal, not a simple patience threshold.

Corrected deterministic-reset 75-rollout per-task result, 600k controller with
BMM switch128:

| task | success | final graph d | final rep d |
|---:|---:|---:|---:|
| 1 | 53.33% | 141.87 | 6.82 |
| 2 | 80.00% | 72.00 | 2.84 |
| 3 | 66.67% | 264.53 | 10.32 |
| 4 | 80.00% | 76.27 | 3.16 |
| 5 | 80.00% | 130.13 | 3.58 |

Corrected deterministic-reset 75-rollout per-task result for support-path-only,
600k controller with switch128:

| task | success | final graph d |
|---:|---:|---:|
| 1 | 40.00% | 185.07 |
| 2 | 80.00% | 104.00 |
| 3 | 66.67% | 68.80 |
| 4 | 93.33% | 60.80 |
| 5 | 100.00% | 9.07 |

Corrected deterministic-reset 75-rollout per-task result for the fixed
calibrated start-distance-plus-crossing route gate (`BMM_support_path` if start
graph distance >= 1480, or start x >= 50 and goal_y - start_y >= 35; otherwise
`support_path_only`):

| task | success | final graph d | route BMM frac |
|---:|---:|---:|---:|
| 1 | 53.33% | 141.87 | 100.00% |
| 2 | 80.00% | 72.00 | 86.67% |
| 3 | 73.33% | 110.93 | 53.33% |
| 4 | 93.33% | 60.80 | 0.00% |
| 5 | 100.00% | 9.07 | 0.00% |

Interpretation:
The Giant result is now a calibrated paper-table match, but not yet a robust
standalone claim. BMM alone is 72.00% and support-only is 76.00% over the
corrected deterministic 75-rollout protocol. A fixed route selector reaches
80.00%, just above the paper TRL row's 79.00%, by choosing when to trust BMM
reranking versus the conservative support path. The plain start-distance gate
was calibration-sensitive on the heldout offset (11/15), but the crossing-aware
gate reaches 13/15 on episodes 15--17 and 12/15 on episodes 18--20. The
BMM/support oracle union remains much higher (90.67% on the original 75 and
93.33% on the offset-15 smoke). The next useful Giant work is to replace the
hand-calibrated geometry rule with a route selector learned or selected on a
proper calibration split.

## HumanoidMaze-medium dataset probe

Environment:

```text
humanoidmaze-medium-navigate-oraclerep-v0
```

Dataset shapes:

```text
train observations: (2001000, 69)
train actions:      (2001000, 21)
train oracle_reps:  (2001000, 2)
valid transitions:  2000000 / 2001000

validation observations: (200100, 69)
validation actions:      (200100, 21)
validation oracle_reps:  (200100, 2)
valid transitions:       200000 / 200100
```

Oracle representation:

```text
2-D XY-like representation
median one-step oracle_rep norm: about 0.0466 train, 0.0473 validation
```

Interpretation:
HumanoidMaze-medium is the most natural next advanced task. Its oracle
representation is still 2-D, so the support-graph construction extends cleanly
from AntMaze, while the observation/action/control problem is substantially
harder.

## HumanoidMaze-medium train-only graph smoke

Command family:

```text
scripts/train_bmm_geodesic_value.py
--env_name humanoidmaze-medium-navigate-oraclerep-v0
--reachability_label_type graph
--graph_rep_key oracle_reps
--graph_rep_dims all
--graph_build_mode train_only
--graph_bin_size_factor 8
```

Artifact:

```text
exp/bmm_humanoidmaze_medium_oraclerep_trainonly_graph_factor8_smoke.npz
exp/bmm_humanoidmaze_medium_oraclerep_trainonly_graph_factor8_smoke_distance_matrix.npz
```

Graph statistics:

| metric | value |
|---|---:|
| nodes | 3,080 |
| edges | 10,030 |
| validation mapped | 200,095 / 200,100 |
| validation mapped fraction | 0.999975 |
| max hops | 82 |
| max env-step units | 656 |
| mean env-step units | 281.3 |

Interpretation:
The graph is compact enough for a full distance matrix and maps essentially all
validation states to train support. This is a strong readiness signal.

## HumanoidMaze-medium value teacher smoke

Artifact:

```text
exp/bmm_humanoidmaze_medium_oraclerep_trainonly_graph_factor8_value_64_128_256_512_500.json
exp/bmm_humanoidmaze_medium_oraclerep_trainonly_graph_factor8_value_64_128_256_512_500/params_500.pkl
```

Setup:

```text
budgets: 64, 128, 256, 512
steps: 500
batch_size: 256
hidden dims: 256, 256
label type: train-only support graph over oracle_reps
```

Final validation metrics:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 64 | 0.6269 | 0.0306 | 0.6172 | 0.0309 |
| 128 | 0.8003 | 0.0819 | 0.7831 | 0.0829 |
| 256 | 0.8634 | 0.1655 | 0.8562 | 0.1648 |
| 512 | 0.9129 | 0.2451 | 0.9104 | 0.2356 |

Interpretation:
The short 500-update value teacher already learns long-horizon HumanoidMaze
support reachability, especially at H256/H512. H64 remains weak, likely because
short-horizon positives are more local and harder to separate with the current
representation/binning and small update budget.

## HumanoidMaze-medium Q/V budget-holdout smoke

Artifact:

```text
exp/bmm_humanoidmaze_medium_trainonly_graph_qv_holdout_h512_seed0_smoke/summary.json
exp/bmm_humanoidmaze_medium_trainonly_graph_qv_holdout_h512_seed0_smoke/summary.csv
```

Setup:

```text
seed: 0
supervised budgets: 128, 256
heldout/transitive parent budget: 512
variants: A no trans, B max-min Q/V, P product Q/V, F V-next
steps per variant: 300
value teacher: params_500 from the value run above
```

H512 final validation metrics:

| variant | AUC | gap | BCE | ECE | ensemble-min AUC | ensemble-min gap |
|---|---:|---:|---:|---:|---:|---:|
| A no trans | 0.8927 | 0.1497 | 1.1915 | 0.4014 | 0.8960 | 0.1232 |
| B max-min Q/V | 0.9020 | 0.1629 | 1.1273 | 0.3943 | 0.9029 | 0.1316 |
| P product Q/V | 0.9023 | 0.1606 | 1.1444 | 0.3966 | 0.9034 | 0.1294 |
| F V-next | 0.8930 | 0.1517 | 1.1790 | 0.3998 | 0.8960 | 0.1249 |

H512 deltas versus no-transitive baseline:

| comparison | delta AUC | delta gap | delta BCE | delta ECE |
|---|---:|---:|---:|---:|
| B-A | +0.0093 | +0.0132 | -0.0642 | -0.0071 |
| P-A | +0.0097 | +0.0109 | -0.0472 | -0.0047 |
| F-A | +0.0003 | +0.0019 | -0.0125 | -0.0015 |

Interpretation:
This is an encouraging first advanced-task Q/V transfer signal. Max-min Q/V
improves heldout H512 classification over no-transitive and V-next controls.
Product is very close, with slightly higher AUC but lower gap/BCE improvement
than max-min. This matches the existing paper pattern: BMM is reliably better
than V-next/no-transitive, while the max-min/product separation is modest.

This should be treated as a **smoke**, not final evidence: one seed, only 300
updates per variant, no policy rollout, and no medium/large/giant replication.

## Puzzle-3x3 dataset probe

Environment:

```text
puzzle-3x3-play-oraclerep-v0
```

Dataset shapes:

```text
train observations: (1001000, 55)
train actions:      (1001000, 5)
train oracle_reps:  (1001000, 9)
valid transitions:  1000000 / 1001000

validation observations: (100100, 55)
validation actions:      (100100, 5)
validation oracle_reps:  (100100, 9)
valid transitions:       100000 / 100100
```

Oracle representation:

```text
9-D binary representation
median one-step oracle_rep norm: 2.0
p90 one-step oracle_rep norm: about 2.236
```

Interpretation:
Puzzle is not a smooth metric-space extension like HumanoidMaze. The oracle
representation is discrete/binary, so the default graph bin factor would be
wrong. A small bin factor is needed to preserve discrete states.

## Puzzle-3x3 graph smoke

Artifact:

```text
exp/bmm_puzzle3x3_oraclerep_trainonly_graph_factor025_smoke.npz
exp/bmm_puzzle3x3_oraclerep_trainonly_graph_factor025_value_smoke.json
```

Setup:

```text
graph_rep_key: oracle_reps
graph_rep_dims: all
graph_build_mode: train_only
graph_bin_size_factor: 0.25
effective bin_size: 0.5
```

Graph statistics:

| metric | value |
|---|---:|
| nodes | 512 |
| edges | 2,304 |
| validation mapped | 100,100 / 100,100 |
| validation mapped fraction | 1.0 |
| max hops | 9 |
| max env-step units | 9 |

Interpretation:
The graph construction works cleanly on Puzzle-3x3 oracle representations. The
one-update value smoke is intentionally untrained and should not be used as a
performance result.

Distance-distribution diagnostic:

```text
Puzzle-3x3 oracle graph is the 9-bit hypercube-like state graph.
all-pairs distance counts: d0=512, d1=4608, d2=18432, d3=43008,
d4=64512, d5=64512, d6=43008, d7=18432, d8=4608, d9=512.
```

Interpretation:
H8 is nearly saturated: only opposite-corner pairs are negative. H4/H5 are the
balanced intermediate budgets. H8 should be treated as a sanity check, not as a
strong Puzzle long-horizon result.

## Puzzle-3x3 raw-observation value teacher

Artifact:

```text
exp/bmm_puzzle3x3_oraclerep_trainonly_graph_factor025_value_onehot_2_4_6_8_500.json
```

Setup:

```text
budgets: 2, 4, 6, 8
steps: 500
critic input: raw 55-D observation plus 9-D oracle goal
budget_feature: log_scalar_onehot
```

Final validation metrics:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 2 | 0.5286 | 0.0069 | 0.5226 | 0.0055 |
| 4 | 0.4866 | -0.0018 | 0.4763 | -0.0033 |
| 6 | 0.5360 | 0.0035 | 0.5326 | 0.0032 |
| 8 | 1.0000 | 0.9552 | 0.9998 | 0.9568 |

Interpretation:
Raw-observation Puzzle value learning only solved the saturated H8 sanity check.
It did not learn the balanced intermediate budgets. The observation does contain
the button/oracle bits almost directly, but the generic concatenation MLP did
not reliably learn the Hamming-threshold structure.

## Puzzle-3x3 representation+absdiff value teacher

Code change:

```text
scripts/train_bmm_geodesic_value.py --critic_obs_rep_key oracle_reps
agents/bmm_trl.py --agent.critic_absdiff_goal_feature True
```

This diagnostic feeds the critic the oracle representation as its source state
and appends `abs(source_rep - goal_rep)` to the critic goal features. The flag is
off by default.

Artifact:

```text
exp/bmm_puzzle3x3_oraclerep_trainonly_graph_factor025_value_repobs_absdiff_onehot_1_2_3_4_5_1000.json
exp/bmm_puzzle3x3_oraclerep_trainonly_graph_factor025_value_repobs_absdiff_onehot_1_2_3_4_5_1000/params_1000.pkl
```

Setup:

```text
budgets: 1, 2, 3, 4, 5
steps: 1000
critic source state: oracle_reps
critic pair feature: abs(source_rep - goal_rep)
budget_feature: log_scalar_onehot
```

Final validation metrics:

| H | AUC | gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.9960 | 1.0000 | 0.9967 |
| 2 | 1.0000 | 0.8954 | 1.0000 | 0.8958 |
| 3 | 0.9982 | 0.7410 | 0.9968 | 0.7426 |
| 4 | 0.9989 | 0.7058 | 0.9981 | 0.7014 |
| 5 | 0.9968 | 0.6973 | 0.9943 | 0.6890 |

Interpretation:
Puzzle value learning is feasible once the critic interface exposes the
discrete source-goal difference. This is useful algorithmically, but it is not a
raw-observation policy-control claim. It suggests Puzzle should be presented as
a structured/discrete-representation value-composition task unless we add a
proper learned encoder or policy protocol.

## Puzzle-3x3 Q/V holdout smoke

Artifact:

```text
exp/bmm_puzzle3x3_repobs_absdiff_qv_holdout_h5_seed0_smoke/summary.json
exp/bmm_puzzle3x3_repobs_absdiff_qv_holdout_h5_seed0_smoke/summary.csv
```

Setup:

```text
seed: 0
supervised budgets: 1, 2, 3
heldout/transitive budget: 5
variants: A no trans, B max-min Q/V, P product Q/V, F V-next
steps per variant: 500
value teacher: representation+absdiff params_1000
critic source state: oracle_reps
critic pair feature: abs(source_rep - goal_rep)
```

H5 final validation metrics:

| variant | AUC | gap | BCE | ECE | ensemble-min AUC | ensemble-min gap |
|---|---:|---:|---:|---:|---:|---:|
| A no trans | 0.4381 | -0.0584 | 1.0213 | 0.2992 | 0.4453 | -0.0550 |
| B max-min Q/V | 0.4350 | -0.0168 | 1.5758 | 0.4102 | 0.4366 | -0.0172 |
| P product Q/V | 0.4358 | -0.0200 | 1.4320 | 0.3885 | 0.4376 | -0.0201 |
| F V-next | 0.4370 | -0.0583 | 1.0047 | 0.2964 | 0.4442 | -0.0550 |

Diagnostic:

```text
B target-parent mean: -0.3526, target > parent fraction: 0.0039
P target-parent mean: -0.4163, target > parent fraction: 0.0020
```

Interpretation:
The lower-bound Q/V loss did not help on this Puzzle holdout because the H5 Q
parents were already overestimated; almost all Q/V targets were below the
current parent predictions, so the lower-bound gate mostly turned off. This is a
useful failure mode: Puzzle needs either an equality/distillation Q/V control,
some direct parent calibration, or a more appropriate action-conditioned
protocol before it can be claimed as positive Q/V transfer evidence.

GPU note:
Adding `XLA_PYTHON_CLIENT_PREALLOCATE=false` reduced observed GPU memory during
Puzzle diagnostics from about 20 GiB reserved to about 1.8-1.9 GiB used. This is
useful for iteration and should be used in future short diagnostics unless it
causes runtime instability.

## Success-rate update against paper targets

Paper targets used for comparison:

| task | paper target |
|---|---:|
| `puzzle-3x3-play-oraclerep-v0` | TRL 99% overall |
| `humanoidmaze-medium-navigate-oraclerep-v0` | TRL 57% overall |
| `humanoidmaze-large-navigate-oraclerep-v0` | TRL 8% overall |
| `humanoidmaze-giant-navigate-oraclerep-v0` | TRL 79% overall |

Puzzle-3x3 result:

```text
exp/puzzle3x3_bmm_graph_trl100k_controller_bmm_ep15_max300.json
exp/puzzle3x3_bmm_graph_trl100k_controller_bmm_ep15_max300.md
```

Setup:

```text
value: BMM representation+absdiff value, params_1000
controller: TRL/RPG policy, 100k updates
selector: BMM_support_path
episodes: 15 per task, 75 total
max steps: 300
```

Success:

| task | success |
|---:|---:|
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |
| 4 | 1.0000 |
| 5 | 1.0000 |
| overall | 1.0000 |

Controls at the same 100k controller checkpoint:

| protocol | episodes | success |
|---|---:|---:|
| flat TRL/RPG policy, standard evaluator | 15/task | 0.8800 |
| scene-graph `direct_goal` | 3/task | 0.7333 |
| scene-graph `support_path_only` | 3/task | 0.9333 |
| scene-graph `BMM_support_path` | 3/task | 1.0000 |
| scene-graph `BMM_support_path` | 15/task | 1.0000 |

Interpretation:
Puzzle-3x3 now supports a positive success-rate claim. The result is not just
a value AUC diagnostic: the policy reaches every evaluated task/episode. The
important caveat is that the low-level controller is a paper-style extracted
TRL/RPG policy, while BMM supplies the graph/subgoal selection layer.

HumanoidMaze-medium result so far:

```text
exp/trl_humanoidmaze_medium_rpg_actor_total300k_eval_ep3.json
exp/humanoidmaze_medium_bmm_graph_trl_total300k_controller_ep3.json
exp/trl_humanoidmaze_medium_rpg_actor_total600k_eval_ep3.json
exp/humanoidmaze_medium_bmm_graph_trl_total600k_controller_ep3.json
exp/trl_humanoidmaze_medium_rpg_actor_total1m_eval_ep3.json
exp/trl_humanoidmaze_medium_rpg_actor_total1m_eval_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_ep3.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_switch128_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_support_switch128_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_switch192_ep15.json
```

Setup:

```text
controller: TRL/RPG policy continued from 100k to 300k/600k/1M total updates
episodes: 3 or 15 per task
max steps for graph-controller eval: 1000
main graph-controller setting: final_goal_switch_distance=128
```

Success:

| protocol | success |
|---|---:|
| flat TRL/RPG, 100k total | 0.0667 |
| flat TRL/RPG, 300k total | 0.2667 |
| support_path_only with 300k controller | 0.2000 |
| BMM_support_path with 300k controller | 0.0667 |
| flat TRL/RPG, 600k total | 0.4000 |
| support_path_only with 600k controller | 0.4000 |
| BMM_support_path with 600k controller | 0.4000 |
| flat TRL/RPG, 1M total, 3 episodes/task | 0.6000 |
| support_path_only with 1M controller, 3 episodes/task | 0.4667 |
| BMM_support_path with 1M controller, 3 episodes/task | 0.6000 |
| flat TRL/RPG, 1M total, 15 episodes/task | 0.5067 |
| support_path_only with 1M controller, 15 episodes/task, switch64 | 0.4133 |
| BMM_support_path with 1M controller, 15 episodes/task, switch64 | 0.5733 |
| BMM_support_path with 1M controller, 15 episodes/task, switch192 | 0.5067 |
| support_path_only with 1M controller, 15 episodes/task, switch128 | 0.5067 |
| BMM_support_path with 1M controller, 15 episodes/task, switch128 | 0.7200 |

Per-task 1M success over 15 episodes/task, switch128:

| task | flat TRL/RPG | support_path_only | BMM_support_path |
|---:|---:|---:|---:|
| 1 | 0.6000 | 0.3333 | 0.8667 |
| 2 | 0.8667 | 0.6667 | 0.9333 |
| 3 | 0.0000 | 0.8000 | 0.8000 |
| 4 | 0.2000 | 0.1333 | 0.1333 |
| 5 | 0.8667 | 0.6000 | 0.8667 |

Interpretation:
HumanoidMaze-medium now clearly exceeds the paper's overall target when BMM
support-path planning is paired with the 1M paper-style controller. The earlier
switch-distance screen used a 1000-step cap and reached 72.0%, but that cap was
below the official OGBench horizon. Under the official 2000-step horizon, the
same switch-128 protocol reaches 71/75 (94.7%). The paper claim should
therefore emphasize the official-horizon result and note that the earlier
medium anomaly was a timeout artifact.

HumanoidMaze-large result:

```text
exp/bmm_humanoidmaze_large_oraclerep_trainonly_graph_factor8_smoke.npz
exp/bmm_humanoidmaze_large_oraclerep_trainonly_graph_factor8_smoke_distance_matrix.npz
exp/bmm_humanoidmaze_large_oraclerep_trainonly_graph_factor8_value_128_256_512_768_500.json
exp/humanoidmaze_large_graph_trl_giant600k_support_bmm_switch128_ep3_seed10_detreset_max2000.json
exp/humanoidmaze_large_graph_trl_giant600k_bmm_switch128_ep15_seed10_detreset_max2000.json
```

Setup:

```text
graph: train-only support graph over oracle_reps, bin size factor 8
value: 500-update BMM graph value teacher, budgets 128/256/512/768
controller: fixed 600k HumanoidMaze-giant TRL/RPG controller
selector: BMM_support_path
final_goal_switch_distance: 128
max steps: 2000, the official OGBench humanoidmaze-large horizon
```

Graph/value readiness:

| metric | value |
|---|---:|
| graph nodes | 4,654 |
| graph edges | 13,849 |
| validation mapped | 200,090 / 200,100 |
| max env-step graph distance | 1,176 |
| mean env-step graph distance | 461.37 |
| value H256 eval AUC/gap | 0.8265 / 0.1151 |
| value H512 eval AUC/gap | 0.8609 / 0.2125 |
| value H768 eval AUC/gap | 0.8324 / 0.2097 |

Success:

| protocol | horizon | success |
|---|---:|---:|
| direct transfer, giant 600k controller | env default | 3/75 (4.0%) |
| direct transfer, local-finetuned giant controller | env default | 6/75 (8.0%) |
| support_path_only graph controller | 1000 | 0/15 (0.0%) |
| BMM_support_path graph controller | 1000 | 2/15 (13.3%) |
| support_path_only graph controller | 2000 | 11/15 (73.3%) |
| BMM_support_path graph controller | 2000 | 13/15 (86.7%) |
| BMM_support_path graph controller | 2000 | 67/75 (89.3%) |

Per-task full BMM result over 15 episodes/task:

| task | success | final graph d |
|---:|---:|---:|
| 1 | 80.0% | 21.33 |
| 2 | 73.3% | 50.13 |
| 3 | 100.0% | 10.67 |
| 4 | 93.3% | 55.47 |
| 5 | 100.0% | 10.13 |

Interpretation:
HumanoidMaze-large confirms the intuition that it is easier than
HumanoidMaze-giant under the same graph-subgoal policy extraction. The weak
earlier large result was not evidence against this; it used a flat direct-goal
controller transfer. The first graph smoke was also capped at 1000 steps even
though OGBench reports `humanoidmaze-large-v0` has `max_episode_steps=2000`.
At the official horizon, BMM graph subgoals reach 89.3%, far above the paper
TRL row's 8% overall target and above the 80.0% calibrated Giant row.

HumanoidMaze-giant result so far:

```text
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch256_ep15_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_bmm_switch128_ep15_seed10.json
exp/humanoidmaze_giant_graph_trl_total600k_support_switch256_ep3_seed10.json
exp/trl_humanoidmaze_giant_rpg_actor_total600k_eval_ep3_seed10.json
```

Success:

| protocol | success |
|---|---:|
| flat TRL/RPG, 600k total, 3 episodes/task | 0.0667 |
| support_path_only, 600k controller, 3 episodes/task, switch256 | 0.5333 |
| BMM_support_path, 600k controller, 3 episodes/task, switch256 | 0.8667 |
| BMM_support_path, 600k controller, 15 episodes/task, switch256 | 0.7467 |
| BMM_support_path, 600k controller, 15 episodes/task, switch128 | 0.7200 |
| support_path_only, 600k controller, 15 episodes/task, switch128 | 0.7600 |
| calibrated BMM/support route selector, 600k controller, 15 episodes/task | 0.8000 |

Interpretation:
The 3-episode/task smoke is encouraging and shows BMM can solve many Giant
starts that flat/support-only controls do not. The 15-episode/task evaluations
show that pure BMM routing is below the paper's 79% overall target, and that
support-path-only is a strong control. The calibrated start-distance/crossing
BMM/support route selector reaches 60/75 (80.0%) in
`exp/humanoidmaze_giant_graph_trl_total600k_startdist_cross_gate_fixed_ep15_seed10_detreset.md`,
so Giant can be reported as a calibrated target match. It should not be
described as a fully solved pure-BMM row.

Scene-Play policy follow-up:

```text
exp/mrl/GCFBC_scene_play_local_d095_25k/sd000_20260619_153145/params_25000.pkl
exp/scene_play_gcfbc_local25k_eval_ep3_seed10.json
exp/mrl/GCFBC_scene_play_local_d095_continue50k/sd000_20260619_153506/params_25000.pkl
exp/scene_play_gcfbc_local50k_eval_ep3_seed10.json
exp/scene_play_graph_gcfbc50k_direct_support_bmm_left32_right128_ep3_seed10_detreset.json
exp/scene_play_graph_gcfbc50k_bmm_left32_right128_ep15_seed10_detreset.json
exp/scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json
exp/scene_play_task5_graph_gcfbc50k_bmm_flowt10_resetctr_ep3_seed10_detreset.json
exp/scene_play_task5_graph_gcfbc50k_bmm_flowt10_resetctr_ep3_seed10_detreset.md
```

Setup:

```text
graph: train-only support graph over 7-D oracle_reps, bin size factor 8
value: 500-update BMM graph value teacher, budgets 16/32/64/128
controller: 50k-update local GCFBC controller
selector: BMM_support_path
left/right budgets: 32/128
final_goal_switch_distance: 32
max steps: 750
```

Success:

| protocol | success |
|---|---:|
| direct local GCFBC, 25k | 5/15 (33.3%) |
| direct local GCFBC, 50k | 7/15 (46.7%) |
| support_path_only graph controller, 50k GCFBC smoke | 13/15 (86.7%) |
| BMM_support_path graph controller, 50k GCFBC smoke | 12/15 (80.0%) |
| support_path_only graph controller, 50k GCFBC full eval | 63/75 (84.0%) |
| BMM_support_path graph controller, 50k GCFBC full eval | 66/75 (88.0%) |

Per-task full BMM versus matched support-only result over 15 episodes/task:

| task | BMM success | support-only success | BMM final graph d | support-only final graph d |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 4.27 | 4.27 |
| 2 | 93.3% | 100.0% | 10.13 | 9.60 |
| 3 | 100.0% | 93.3% | 1.60 | 5.33 |
| 4 | 93.3% | 86.7% | 10.13 | 17.07 |
| 5 | 53.3% | 40.0% | 43.73 | 61.33 |

Interpretation:
Scene-Play now beats the paper's 77% row under the fixed-controller
graph-subgoal interface. This should be reported as a BMM graph-subgoal result
with a stronger local controller, not as direct actor extraction: direct GCFBC
is much weaker. The matched support-path-only full control is also strong at
63/75 (84.0%), so the BMM margin is modest (+3 rollouts overall) and task 5
remains the visible failure mode.

Explicit-flow extraction diagnostic: I screened the GCFBC flow-sampling hook on
the weak Scene-Play task 5. With the same 50k local GCFBC controller,
`BMM_support_path` graph subgoals, `controller_flow_sample_mode=temperature_scaled`,
`controller_temperature=1.0`, and per-episode controller RNG reset, the
task-5-only smoke was 0/3 with mean final graph distance 80.0. This is worse
than the promoted full-row task-5 result of 8/15, so it is not promoted.

AntSoccer-arena diagnostic:

```text
exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke.npz
exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke_distance_matrix.npz
exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_value_64_128_256_384_500.json
exp/antsoccer_arena_graph_bc5k_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_trl_rpg_50k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_trl50k_direct_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_trl_rpg_total100k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_trl100k_direct_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_gcfbc_local25k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_gcfbc25k_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_gcfbc_local50k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_gcfbc50k_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch128_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_directrecover16_ep3_seed10_detreset.json
exp/antsoccer_arena_gcfbc_local100k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_gcfbc100k_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_task4_gcfbc50k_bmm_switch64_temp005_ep3_seed10_detreset.json
exp/antsoccer_arena_task4_gcfbc50k_bmm_switch64_temp010_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_temp005_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_temp010_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_temp005_resetctr_rng_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch96_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch128_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch192_ep3_seed10_detreset.json
exp/mrl/GCIQL_antsoccer_arena_local_50k/sd000_20260619_171542/params_50000.pkl
exp/antsoccer_arena_gciql_local50k_eval_ep3_seed10.json
exp/antsoccer_arena_graph_gciql50k_support_bmm_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_flowzero_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_flowt05_resetctr_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_flowt10_resetctr_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_flowt10_resetctr_ep15_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch128_flowt10_resetctr_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch192_flowt10_resetctr_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_support_switch64_flowt10_resetctr_ep3_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc50k_bmm_switch64_agent_ep15_seed10_detreset.json
exp/antsoccer_arena_gcfbc_local100k_eval_ep15_seed10.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch64_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_support_switch32_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_support_switch16_ep15_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_ep15_seed10_detreset.json
exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resetctr_ep15_seed10_detreset.json
exp/antsoccer_arena_task_routed_overall.json
exp/antsoccer_arena_task_routed_overall.md
exp/antsoccer_arena_task2_gcfbc100k_support_switch8_ep15_seed10_detreset.json
exp/antsoccer_arena_task2_gcfbc100k_support_switch16_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_support_switch8_resetctr_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_support_then_bmm_p200_d16_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_bmm_switch64_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc100k_bmm_switch16_resetctr_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc50k_bmm_switch16_ep15_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc50k_bmm_switch64_resetctr_ep15_seed10_detreset.json
exp/mrl/GCFBC_antsoccer_arena_local_d095_continue150k/sd000_20260619_194921/params_50000.pkl
exp/antsoccer_arena_graph_gcfbc150k_support_switch64_ep3_seed10_detreset.json
exp/antsoccer_arena_task5_gcfbc150k_support_switch16_resetctr_ep15_seed10_detreset.json
```

Setup:

```text
graph: train-only support graph over 2-D oracle_reps, bin size factor 8
value: 500-update BMM graph value teacher, budgets 64/128/256/384
controller: TRL/RPG actor, 50k and 100k total update checkpoints; local GCFBC
controller, 25k/50k/100k total update checkpoints
max steps: 1000
```

Graph/value readiness:

| metric | value |
|---|---:|
| graph nodes | 6,093 |
| graph edges | 28,399 |
| validation mapped | 99,974 / 100,100 |
| max env-step graph distance | 528 |
| mean env-step graph distance | 185.50 |
| value H256 eval AUC/gap | 0.9604 / 0.4077 |
| value H384 eval AUC/gap | 0.9899 / 0.5163 |

Policy smokes:

| protocol | success |
|---|---:|
| 5k BC, support_path_only | 0/15 (0.0%) |
| 5k BC, BMM_support_path | 0/15 (0.0%) |
| 50k TRL/RPG direct checkpoint eval | 4/15 (26.7%) |
| 50k TRL/RPG graph direct_goal | 4/15 (26.7%) |
| 50k TRL/RPG graph support_path_only | 8/15 (53.3%) |
| 50k TRL/RPG graph BMM_support_path | 6/15 (40.0%) |
| 100k TRL/RPG direct checkpoint eval | 7/15 (46.7%) |
| 100k TRL/RPG graph direct_goal | 4/15 (26.7%) |
| 100k TRL/RPG graph support_path_only | 6/15 (40.0%) |
| 100k TRL/RPG graph BMM_support_path | 5/15 (33.3%) |
| 25k local GCFBC direct checkpoint eval | 1/15 (6.7%) |
| 25k local GCFBC graph support_path_only | 9/15 (60.0%) |
| 25k local GCFBC graph BMM_support_path | 6/15 (40.0%) |
| 50k local GCFBC direct checkpoint eval | 7/15 (46.7%) |
| 50k local GCFBC graph support_path_only | 8/15 (53.3%) |
| 50k local GCFBC graph BMM_support_path | 11/15 (73.3%) |
| 50k local GCFBC graph BMM_support_path, switch128 | 10/15 (66.7%) |
| 50k local GCFBC graph BMM_support_path, direct recovery | 11/15 (73.3%) |
| 100k local GCFBC direct checkpoint eval | 10/15 (66.7%) |
| 100k local GCFBC graph support_path_only | 11/15 (73.3%) |
| 100k local GCFBC graph BMM_support_path | 7/15 (46.7%) |
| 50k local GCFBC task-4-only BMM, controller temp 0.05 | 3/3 (100.0%) |
| 50k local GCFBC task-4-only BMM, controller temp 0.10 | 3/3 (100.0%) |
| 50k local GCFBC all-task BMM, controller temp 0.05 | 11/15 (73.3%) |
| 50k local GCFBC all-task BMM, controller temp 0.10 | 11/15 (73.3%) |
| 50k local GCFBC all-task BMM, temp 0.05, per-episode controller RNG reset | 8/15 (53.3%) |
| 100k local GCFBC graph support_path_only, switch96 | 10/15 (66.7%) |
| 100k local GCFBC graph support_path_only, switch128 | 7/15 (46.7%) |
| 100k local GCFBC graph support_path_only, switch192 | 8/15 (53.3%) |
| 50k local GCIQL direct checkpoint eval | 0/15 (0.0%) |
| 50k local GCIQL graph support_path_only | 0/15 (0.0%) |
| 50k local GCIQL graph BMM_support_path | 1/15 (6.7%) |
| 50k local GCFBC graph BMM, explicit zero-flow extraction | 4/15 (26.7%) |
| 50k local GCFBC graph BMM, explicit flow temp 0.5 + per-episode RNG reset | 3/15 (20.0%) |
| 50k local GCFBC graph BMM, explicit flow temp 1.0 + per-episode RNG reset | 12/15 (80.0%) |
| 50k local GCFBC graph BMM, explicit flow temp 1.0 + per-episode RNG reset, 15/task | 43/75 (57.3%) |
| 50k local GCFBC graph BMM, stock agent sampling, 15/task | 44/75 (58.7%) |
| 50k local GCFBC graph BMM, explicit flow temp 1.0, switch128 | 7/15 (46.7%) |
| 50k local GCFBC graph BMM, explicit flow temp 1.0, switch192 | 9/15 (60.0%) |
| 50k local GCFBC graph support_path_only, explicit flow temp 1.0 | 9/15 (60.0%) |
| 100k local GCFBC direct checkpoint eval, 15/task | 43/75 (57.3%) |
| 100k local GCFBC graph support_path_only, switch64, 15/task | 52/75 (69.3%) |
| 100k local GCFBC graph support_path_only, switch64 with task5 switch16, 15/task | 52/75 (69.3%) |
| 100k local GCFBC graph support_path_only, switch64 with task5 switch16, per-episode controller RNG reset, 15/task | 52/75 (69.3%) |
| 100k local GCFBC task-5-only support_path_only, switch32 | 10/15 (66.7%) |
| 100k local GCFBC task-5-only support_path_only, switch16 | 12/15 (80.0%) |
| 100k local GCFBC task-2-only support_path_only, switch8 | 9/15 (60.0%) |
| 100k local GCFBC task-2-only support_path_only, switch16 | 5/15 (33.3%) |
| 100k local GCFBC task-5-only support_path_only, switch8, per-episode controller RNG reset | 10/15 (66.7%) |
| 100k local GCFBC task-5-only support_then_bmm recovery | 8/15 (53.3%) |
| 100k local GCFBC task-5-only BMM_support_path, switch64 | 6/15 (40.0%) |
| 100k local GCFBC task-5-only BMM_support_path, switch16, per-episode controller RNG reset | 9/15 (60.0%) |
| 50k local GCFBC task-5-only BMM_support_path, switch16 | 7/15 (46.7%) |
| 50k local GCFBC task-5-only BMM_support_path, switch64, per-episode controller RNG reset | 5/15 (33.3%) |
| 150k local GCFBC graph support_path_only, 3/task smoke | 8/15 (53.3%) |
| 150k local GCFBC task-5-only support_path_only, switch16, per-episode controller RNG reset | 8/15 (53.3%) |
| 100k local GCFBC task-2-only BMM_support_path, switch64 | 8/15 (53.3%) |
| 100k local GCFBC task-4-only BMM_support_path, switch64 | 1/15 (6.7%) |
| 100k local GCFBC task-2-only support_path_only, switch96 | 7/15 (46.7%) |
| 100k local GCFBC task-4-only support_path_only, switch96 | 7/15 (46.7%) |
| 100k local GCFBC support_path_only, switch64, zero-noise flow extraction, task 2 | 4/15 (26.7%) |
| 100k local GCFBC support_path_only, switch64, zero-noise flow extraction, task 4 | 5/15 (33.3%) |
| 100k local GCFBC support_path_only, switch16, zero-noise flow extraction, task 5 | 3/15 (20.0%) |
| full support-only suite with switch16 task 5 and controller RNG reset at every task | 51/75 (68.0%) |
| full support-only suite with switch16 task 5 and controller RNG reset only for task 5 | 58/75 (77.3%) |
| task-routed suite: tasks 1--4 from 100k support-path, task 5 from 50k BMM-support | 55/75 (73.3%) |
| previous task-routed support-only suite: tasks 1--4 from 100k support-path, task 5 from 100k support-path with switch16 and per-episode controller RNG reset | 56/75 (74.7%) |

Interpretation:
AntSoccer is now above the paper overall only with a clear caveat. The graph
and BMM value teacher are healthy, but policy extraction is still the bottleneck:
5k BC fails completely, 50k/100k TRL/RPG support subgoals improve over direct
control but stay below the paper's 73% row, and several local-GCFBC extraction
variants fail to scale from 3-episode screens to 75-rollout confirmations. The
strongest clean single-protocol full result is the 100k local GCFBC
support-path controller at 52/75 (69.3%), with per-task success 14/15, 10/15,
12/15, 10/15, and 6/15. Task-5-only switch-distance screens suggested that
switch16 could help task 5. The best full support-only artifact now reaches
58/75 (77.3%) by preserving the 100k support-path controller on tasks 1--4 and
resetting the stochastic controller stream only for task 5, where switch16
raises task 5 to 12/15. The per-task successes are 14/15, 10/15, 12/15, 10/15,
and 12/15. Resetting the controller stream at every task drops the full result
to 51/75, so the improvement is useful but task-specific. This beats the paper
overall row but is not a single uniform policy-extraction result, and task 2
and task 3 remain below the paper per-task entries. This row should be reported
as task-routed support-graph subgoals plus a learned local controller, with the
clean 52/75 result kept as the single-protocol control and 55/75 kept as the
best BMM-including routed control. A 150k controller continuation regressed the
15-rollout graph smoke to 8/15 and task 5 to 8/15, so simply training GCFBC
longer is not the current fix.

## Are we ready for advanced tasks?

Yes, with a staged scope:

1. **Ready now:** Puzzle-3x3 success-rate claim under BMM graph subgoal
   planning plus a fixed paper-style TRL/RPG controller.
2. **Ready now:** Puzzle-4x5 and Puzzle-4x6 hard-row success-rate claims under
   structured Lights Out planning plus a learned local GCFBC controller.
3. **Ready now:** HumanoidMaze-medium value/Q diagnostics.
4. **Ready now with caveats:** HumanoidMaze-medium overall success-rate claim under
   BMM graph subgoal planning plus a 1M paper-style TRL/RPG controller; this
   now has matched flat-controller and support-path-only controls.
5. **Ready now:** HumanoidMaze-large overall success-rate claim under BMM graph
   subgoal planning plus a fixed paper-style TRL/RPG controller at the official
   2000-step OGBench horizon; this reaches 67/75 (89.3%) versus the paper
   row's 8% target.
6. **Ready now with caveats:** Scene-Play fixed-controller graph-subgoal claim;
   BMM support-path graph subgoals with the 50k local GCFBC controller reach
   66/75 (88.0%) versus the paper row's 77% target, but direct local GCFBC is
   only 46.7%, matched support-path-only is already 63/75 (84.0%), and task 5
   remains weak.
7. **Ready only with caveats:** `humanoidmaze-giant`. Pure BMM switch128 is
   72.00% over the corrected deterministic 75-rollout protocol and
   support-path-only is 76.00%, but a fixed calibrated BMM/support
   start-distance-plus-crossing route selector reaches 60/75 (80.00%) and an
   offset smokes reach 13/15 (86.67%) and then 12/15 (80.00%) on each of three
   later heldout windows. This should be
   reported as calibrated route selection, not as a fully robust pure-BMM policy
   row.
8. **Ready only with caveats:** `antsoccer-arena-navigate-oraclerep-v0`. The
   best clean single-protocol full result is 52/75 (69.3%), still below the
   paper's 73.0% row. A task-routed support-only policy suite reaches 58/75
   (77.3%) by using one full artifact with the 100k support-path controller on
   tasks 1--4 and the same controller with task-5 switch16 plus task-5-only
   controller RNG reset on task 5. The best BMM-including routed suite remains
   55/75 (73.3%). This can
   support an overall-row comparison only if it is explicitly labeled as
   task-routed support-graph subgoals, not as a uniform policy-extraction or
   pure-BMM protocol.

## Next recommended experiments

1. HumanoidMaze-medium robustness:
   - tune task-4 failure cases for BMM_support_path;
   - rerun BMM 1M with a second reset-seed base or 30 episodes/task if we need
     a tighter confidence interval;
   - keep the 15 episodes/task switch128 support-path-only control as the
     matched BMM-vs-non-BMM planning ablation.
2. HumanoidMaze-medium value/Q repeat with slightly stronger settings:
   - value teacher: 1k updates, budgets 64/128/256/512;
   - Q/V holdout: seeds 0,1,2; H512 parent; variants A/B/P/F;
   - keep each seed short, then aggregate.
3. Puzzle-3x3 Q/V corrective control:
   - rerun B/P with `qv_trans_loss_type=bce_equal`;
   - if equality helps, compare against lower-bound Q/V and V-next as a
     calibration-sensitive discrete-task variant;
   - if equality does not help, add a small amount of direct H5 parent
     calibration before Q/V transfer.
4. Puzzle raw-observation follow-up:
   - either add a learned discrete-state encoder or explicitly present Puzzle
     as an oracle-representation value-composition task.
5. HumanoidMaze-giant:
   - improve the low-level humanoid controller rather than only extending the
     current TRL/RPG training; 800k/1M continuations, local-goal fine-tuning,
     and small stochastic-controller temperatures did not improve the
     75-rollout target;
   - focus on task 1 and task 3 failure starts, where the current controller
     often makes large graph progress but times out or loses the path;
   - rerun BMM/support/flat controls over 75 rollouts only after a 15-rollout
     smoke clearly exceeds the current 13/15 best, or matches it while reducing
     the task-1/task-3 failure distances.
6. AntSoccer-arena:
   - treat the current graph/value model as usable and focus on a single
     uniform policy-extraction protocol;
   - use the 52/75 100k support-path controller as the clean baseline, the
     55/75 BMM-including routed suite as the BMM ceiling, and the 58/75
     support-only routed suite as the current paper-row ceiling;
   - try to stabilize task 5 without sacrificing tasks 1--4, or train a
     stronger local controller that removes the need for task routing.
