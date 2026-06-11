# BMM-TRL joint action-subgoal results

Date: 2026-06-11

This follows `BMM_TRL_NEXT_STEPS_AFTER_FAST_DECISION_RESULTS.md`.

## Decision

Recommendation: continue the project only as a high-level reachability / subgoal-planning direction for now.

Do not return to flat one-step Q policy extraction. Do not run policy benchmarks yet. The joint diagnostic gives a real but still weak positive signal for BMM Q/V over A/F, while also showing that the current cached candidate-action set has low oracle action-valid coverage.

Short version:

- B Q/V beats A/F on joint `(a,w)` own-state selection metrics.
- The advantage is consistent at 128 and 512 cached queries.
- Absolute action-valid rates are very low because the oracle action-valid ceiling is only `0.0547`.
- This is enough to continue with high-level BMM diagnostics, but not enough to start policy evaluation.

## Code changes

- Added `scripts/eval_bmm_joint_action_subgoal.py`.
- Added `scripts/test_bmm_joint_action_subgoal.py`.
- The diagnostic reuses cached action-ranking queries and existing A/B/F checkpoints.
- For every query, it samples candidate subgoals and scores candidate pairs with:

```text
min(Q_h(s_or_candidate_state, a, w), V_{H-h}(w, g))
```

- It reports both source-state and own-state Q modes, plus mean and ensemble-min scores.
- It reports oracle, random, and V/V teacher baselines.

## Verification

Commands run:

```bash
conda run -n bmm-trl python scripts/test_bmm_joint_action_subgoal.py
conda run -n bmm-trl python scripts/test_bmm_action_ranking.py
conda run -n bmm-trl python scripts/test_bmm_subgoal_selection.py
```

All passed. The non-escalated JAX imports printed the known sandbox CUDA warning but exited successfully.

## 128-query joint diagnostic

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_joint_action_subgoal.py \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,160 \
  --budget=160 \
  --left_budget=80 \
  --right_budget=80 \
  --num_queries=128 \
  --num_subgoals=64 \
  --query_cache_path=exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000 \
  --critics \
    A=exp/bmm_qv_budget_holdout_20260611_021352/seed0_A_no_parent_no_trans_qv0_vnext0_parent0:1000 \
    B=exp/bmm_qv_budget_holdout_20260611_021352/seed0_B_no_parent_qv_trans_qv0p01_vnext0_parent0:1000 \
    F=exp/bmm_qv_budget_holdout_20260611_021352/seed0_F_no_parent_vnext_distill_qv0_vnext0p01_parent0:1000
```

Artifacts:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_h160_128.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_h160_128.md`

Candidate set:

| metric | value |
|---|---:|
| oracle any action-valid fraction | 0.0547 |
| oracle any state-valid fraction | 0.3594 |

Own-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| A Q/V | 0.0781 | 0.0156 | 0.9279 | 2.7838 | 36.2913 | 38.6366 |
| B Q/V | 0.1016 | 0.0156 | 0.3093 | 1.8559 | 32.3341 | 36.8289 |
| F Q/V | 0.0781 | 0.0156 | 0.6186 | 2.1652 | 35.6854 | 38.1698 |

Interpretation:

- B is better than A/F on state-valid fraction and path stretch.
- B ties A/F on action-valid fraction.
- The action-valid ceiling is low, so this result was only a weak go signal for the larger cached diagnostic.

## 512-query joint diagnostic

Command:

```bash
conda run -n bmm-trl python scripts/eval_bmm_joint_action_subgoal.py \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,160 \
  --budget=160 \
  --left_budget=80 \
  --right_budget=80 \
  --num_queries=512 \
  --num_subgoals=64 \
  --query_cache_path=exp/bmm_qv_budget_holdout_20260611_021352/action_ranking_h160_queries.npz \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000 \
  --critics \
    A=exp/bmm_qv_budget_holdout_20260611_021352/seed0_A_no_parent_no_trans_qv0_vnext0_parent0:1000 \
    B=exp/bmm_qv_budget_holdout_20260611_021352/seed0_B_no_parent_qv_trans_qv0p01_vnext0_parent0:1000 \
    F=exp/bmm_qv_budget_holdout_20260611_021352/seed0_F_no_parent_vnext_distill_qv0_vnext0p01_parent0:1000
```

Artifacts:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_h160_512.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_h160_512.md`

Candidate set:

| metric | value |
|---|---:|
| oracle any action-valid fraction | 0.0547 |
| oracle any state-valid fraction | 0.3672 |

Main own-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| V/V teacher | 0.1680 | 0.0059 | 1.0826 | 1.0826 | 24.6779 | 24.2006 |
| A Q/V | 0.0586 | 0.0039 | 2.0878 | 3.0158 | 37.1586 | 40.4633 |
| B Q/V | 0.0801 | 0.0098 | 1.0053 | 1.5465 | 33.0659 | 36.5462 |
| F Q/V | 0.0586 | 0.0039 | 2.0105 | 2.8611 | 36.6978 | 40.0767 |

Main own-state ensemble-min rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| V/V teacher | 0.1738 | 0.0059 | 1.0053 | 1.0826 | 24.2235 | 23.6660 |
| A Q/V | 0.0566 | 0.0059 | 1.9332 | 2.7065 | 37.3906 | 40.7367 |
| B Q/V | 0.0762 | 0.0098 | 1.0053 | 1.5465 | 33.2883 | 36.8027 |
| F Q/V | 0.0586 | 0.0059 | 1.9332 | 2.7065 | 37.0071 | 40.4739 |

Interpretation:

- B beats A/F on own-state state-valid fraction: `0.0801` vs `0.0586`.
- B beats A/F on own-state action-valid fraction: `0.0098` vs `0.0039`.
- B has the best own-state source and next path stretch among A/B/F.
- B has the lowest own-state midpoint and action-midpoint error among A/B/F.
- V/V teacher remains substantially better on state-valid fraction and midpoint error, which means the value-side subgoal signal is stronger than the learned Q/V joint signal.
- The oracle action-valid ceiling is only `0.0547`, so the current action candidates are too sparse for a meaningful policy smoke.

## Final signal

Continue flat Q extraction: no.

Stop the project: no.

Continue high-level BMM planning diagnostics: yes, but fix candidate-action coverage before policy.

The project still has a coherent path as BMM-assisted subgoal/action selection. The next milestone should not be a policy benchmark. It should be a stronger candidate-generation diagnostic where oracle action-valid coverage is high enough that learned differences can matter.

Recommended next steps:

1. Improve or replace candidate action generation for the joint diagnostic.
2. Target an oracle action-valid ceiling well above `0.0547` before running policy.
3. Re-run the 512-query joint diagnostic after candidate coverage improves.
4. Only run a tiny high-level policy smoke if B still beats A/F and the oracle/V-teacher baselines are strong.
