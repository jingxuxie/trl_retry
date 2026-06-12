# BMM-TRL hierarchical pivot quick try - 20260611_173408

## Summary

Following `BMM_TRL_NEXT_STEPS_AFTER_CONTROLLER_DECISION.md`, I ran the one
remaining cheap go/no-go test for the hierarchical RL pivot:

```text
selectors = random, geometric_midpoint, BMM_V, oracle_midpoint
controller = stronger local goal-conditioned BC controller
env = pointmaze-medium-navigate-v0
tasks = 1,2,3,4,5
episodes/task = 1
max_steps = 100
one seed
```

There were no saved strong goal-conditioned actor checkpoints in `exp/`, so I
used the existing lightweight subgoal BC controller diagnostic and made it
stronger than the previous 1k run:

```text
bc_steps = 5000
hidden_dims = (512, 512)
bc_offsets = 1,2,4,8,16
```

## Command

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
    --bc_offsets=1,2,4,8,16 \
    --bc_steps=5000 \
    --bc_batch_size=256 \
    --bc_hidden_dims="(512, 512)"
```

Output artifact:

```text
exp/bmm_qv_budget_holdout_20260611_021352/subgoal_bc_controller_5k_h512_tasks12345.md
```

## BC training

The controller trained without numerical issues:

| step | loss | action_norm |
|---:|---:|---:|
| 1 | 0.949735 | 0.8352 |
| 500 | 0.302062 | 0.7876 |
| 1000 | 0.260519 | 0.7925 |
| 2000 | 0.250770 | 0.7858 |
| 3000 | 0.260773 | 0.8666 |
| 4000 | 0.274990 | 0.8438 |
| 5000 | 0.248530 | 0.8367 |

## Selector results

| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 134.6113 | 39.5916 | 0.3959 | 0.1020 | 0.0400 | 0.0320 |
| geometric_midpoint | 0.0000 | 106.8972 | 67.3057 | 0.6731 | 0.2820 | 0.0360 | 0.0360 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6731 | 0.6100 | 0.0340 | 0.0340 |
| oracle_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.5147 | 0.1960 | 0.2800 | 0.1720 |

## Decision

The plan's go/no-go criterion was:

```text
If BMM_V > geometric_midpoint: continue hierarchical pivot.
If BMM_V <= geometric_midpoint: pause the project.
```

This run gives:

```text
BMM_V final_d == geometric_midpoint final_d
BMM_V improve == geometric_midpoint improve
BMM_V does not beat geometric_midpoint
```

So the quick hierarchical pivot does **not** clear the continue gate.

## Recommendation

Pause active policy-facing experimentation.

The evidence still supports a narrower writeup around:

```text
BMM as budgeted reachability / subgoal-selection diagnostics.
```

It does not currently support spending more compute on:

```text
flat Q extraction;
Q/V action extraction;
broad OGBench policy benchmarks;
more controller sweeps;
larger hierarchical policy runs.
```

If the project continues, it should be explicitly reframed as a hierarchical RL
project with a serious low-level controller implementation, not as more BMM
critic tuning.
