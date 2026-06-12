# BMM-TRL value-subgoal controller decision - 20260611_170712

## Summary

Following `BMM_TRL_NEXT_STEPS_AFTER_VALUE_SUBGOAL_NEXT_STEPS.md`, I ran the
next focused controller milestones:

```text
1. one-step low-level controller comparison;
2. tiny hierarchical smoke with selector comparison;
3. tiny goal-conditioned BC low-level controller smoke.
```

The result is mixed:

```text
High-level BMM/V subgoals remain useful.
The low-level controller remains the bottleneck.
The tiny BC controller did not make BMM/V clearly better than geometric.
```

Decision:

```text
Do not continue toward broad OGBench policy benchmarks.
Do not return to flat Q/QV action extraction.
Pause broad project expansion.
Continue only if the next work is explicitly low-level-controller research.
```

## Milestone 1: one-step controller diagnostic

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_value_subgoal_controller.py \
    --geodesic_budget_unit=env_steps \
    --budgets=40,80,160 \
    --budget=160 \
    --left_budget=80 \
    --right_budget=80 \
    --num_queries=128 \
    --num_candidates=64 \
    --controller_hops=0 \
    --query_cache_path=exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz \
    --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
    --value_restore_epoch=1000
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_low_level_h160_128.md
```

Key rows:

| scorer | state_valid | source_stretch | local_subgoal_improve | local_goal_improve | local_goal_reduce | random_subgoal_improve |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0156 | 48.2522 | 19.4865 | 14.8468 | 0.8672 | 0.3043 |
| geometric_midpoint | 0.1172 | 8.9700 | 19.7958 | 17.6306 | 0.9453 | 0.3897 |
| BMM_V_value | 0.1875 | 1.5465 | 19.7958 | 19.4865 | 0.9922 | 0.3455 |

Interpretation:

```text
BMM/V is best among non-oracle selectors for one-step final-goal improvement.
BMM/V also has much better path stretch than random/geometric.
This passes the one-step continue gate.
```

## Milestone 3: tiny hierarchical NN-controller smoke

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_value_subgoal_policy_smoke.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --geodesic_budget_unit=env_steps \
    --budgets=40,80,160 \
    --left_budget=80 \
    --right_budget=80 \
    --controller_hops=0 \
    --num_subgoal_candidates=64 \
    --selectors=random,geometric_midpoint,BMM_V,oracle_midpoint \
    --task_ids=1,2,3,4,5 \
    --episodes_per_task=1 \
    --max_steps=100 \
    --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
    --value_restore_epoch=1000
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/value_subgoal_policy_decision_tasks12345_hops0.md
```

Result:

| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 122.7339 | 51.4690 | 0.5147 | 0.1160 | 0.1480 | 0.0960 |
| geometric_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.5147 | 0.3740 | 0.1520 | 0.1480 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6731 | 0.6820 | 0.1680 | 0.1680 |
| oracle_midpoint | 0.0000 | 110.8564 | 63.3465 | 0.6335 | 0.4220 | 0.2840 | 0.2040 |

Interpretation:

```text
BMM/V beats random and geometric under the same strict same-cell NN controller.
No selector solves the tasks.
This supports the hierarchical BMM subgoal-planning direction, but only weakly.
```

## Milestone 2: tiny goal-conditioned BC controller

Added:

```text
scripts/eval_bmm_subgoal_bc_controller.py
```

The BC controller trains:

```text
input:  (s_t, w)
target: a_t
w = s_{t+k}, k in {1,2,4,8}
```

Run:

```bash
conda run -n bmm-trl python scripts/eval_bmm_subgoal_bc_controller.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --geodesic_budget_unit=env_steps \
    --budgets=40,80,160 \
    --left_budget=80 \
    --right_budget=80 \
    --controller_hops=0 \
    --num_subgoal_candidates=64 \
    --selectors=random,geometric_midpoint,BMM_V,oracle_midpoint \
    --task_ids=1,2,3,4,5 \
    --episodes_per_task=1 \
    --max_steps=100 \
    --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
    --value_restore_epoch=1000 \
    --bc_offsets=1,2,4,8 \
    --bc_steps=1000 \
    --bc_batch_size=256 \
    --bc_hidden_dims="(256, 256)"
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/subgoal_bc_controller_1k_tasks12345.md
```

BC training loss:

| step | loss |
|---:|---:|
| 1 | 0.829524 |
| 500 | 0.236592 |
| 1000 | 0.222748 |

BC-controller smoke result:

| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 134.6113 | 39.5916 | 0.3959 | 0.1040 | 0.0440 | 0.0340 |
| geometric_midpoint | 0.0000 | 102.9381 | 71.2648 | 0.7126 | 0.2780 | 0.0380 | 0.0380 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6731 | 0.6480 | 0.0420 | 0.0420 |
| oracle_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.5147 | 0.1820 | 0.2960 | 0.1800 |

Interpretation:

```text
The tiny BC controller learned a nontrivial imitation loss.
However, with this controller, geometric midpoint beats BMM/V on final distance
and improvement.
BMM/V still beats random but no longer beats the simple geometric baseline.
```

This fails the stronger "BMM/V + better controller clearly beats simple
baselines" criterion.

## Decision

Based on these results:

```text
Continue flat Q extraction: no.
Continue Q/V action extraction: no.
Continue broad OGBench policy benchmarks: no.
Continue BMM as high-level subgoal planning: only narrowly.
Continue immediately with more sweeps: no.
Pause or ask for advice before spending more compute: yes.
```

The most accurate status is:

```text
BMM/V has a real high-level subgoal signal, including graph-support evidence.
But policy-facing gains are not robust to the low-level controller choice.
The project is not ready for larger policy experiments.
```

## Recommended next question for advisors

Ask whether this should be reframed as:

```text
1. a critic/reachability/subgoal-selection diagnostic project;
2. a hierarchical RL project requiring a separately trained low-level controller;
3. or a project to pause because the controller bottleneck makes the main policy claim too weak.
```

I would not spend more time on BMM loss tuning or action-value extraction until
that decision is made.
