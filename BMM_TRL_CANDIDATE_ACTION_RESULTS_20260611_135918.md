# BMM-TRL candidate-action diagnostic results

Date: 2026-06-11

This follows `BMM_TRL_NEXT_STEPS_AFTER_JOINT_ACTION_SUBGOAL.md`.

## Decision

Recommendation: pivot, do not stop.

The improved candidate-action diagnostics answer the immediate question:

```text
Can better candidate actions rescue the current BMM Q/V policy-facing path?
```

Current answer:

```text
Not enough for policy smoke.
```

Candidate coverage is no longer the blocker under `neighbor_cell`, `directional`, or `oracle_diverse`. Those modes raise oracle action-valid coverage from `0.0547` to about `0.99-1.00`. However, BMM Q/V does not robustly beat A/F on the full joint action-subgoal metrics once useful candidates are present.

The next direction should be:

```text
BMM for high-level reachability / subgoal planning.
Use a separate low-level controller or BC/action proposal mechanism.
Pause flat neural Q/V policy extraction.
```

## Code changes

- Extended `scripts/eval_bmm_joint_action_subgoal.py` with candidate-action modes:
  - `same_cell_cached`
  - `neighbor_cell`
  - `directional`
  - `oracle_diverse`
  - `dataset_global_oracle`
- Added `--coverage_only`.
- Added candidate coverage metrics:
  - oracle any action-valid fraction
  - oracle any state-valid fraction
  - unique next-cell count
  - next-distance spread
  - source-position spread
  - oracle best selected distance
  - oracle best action-midpoint error
- Extended `scripts/test_bmm_joint_action_subgoal.py` for coverage metrics.

## Verification

Commands run:

```bash
conda run -n bmm-trl python -m py_compile \
  scripts/eval_bmm_joint_action_subgoal.py \
  scripts/test_bmm_joint_action_subgoal.py

conda run -n bmm-trl python scripts/test_bmm_joint_action_subgoal.py
conda run -n bmm-trl python scripts/test_bmm_action_ranking.py
```

All passed. Non-escalated JAX imports still print the known sandbox CUDA warning, but the commands exited successfully.

## Candidate coverage sweep

All coverage runs used:

```text
H=160, split 80/80, 128 queries, 8 candidate actions, 64 subgoals
```

Artifacts:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_candidate_coverage_same_cell_cached_128.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/joint_candidate_coverage_neighbor_cell_128.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/joint_candidate_coverage_directional_128.json`
- `exp/bmm_qv_budget_holdout_20260611_021352/joint_candidate_coverage_oracle_diverse_128.json`

| mode | oracle any action-valid | oracle any state-valid | unique next cells | next-distance spread | source-position spread | oracle best selected distance |
|---|---:|---:|---:|---:|---:|---:|
| same_cell_cached | 0.0547 | 0.3594 | 2.0781 | 21.0330 | 2.6618 | 157.2837 |
| neighbor_cell | 1.0000 | 0.3594 | 5.1641 | 75.0075 | 9.4583 | 115.5270 |
| directional | 0.9922 | 0.3594 | 5.3984 | 73.3063 | 9.8015 | 114.7537 |
| oracle_diverse | 1.0000 | 0.3594 | 7.5938 | 202.5975 | 25.1146 | 0.0000 |

Interpretation:

- The original cached same-cell candidate set was too weak for policy-facing conclusions.
- `neighbor_cell` is the first usable candidate mode by the plan threshold: oracle action-valid `>= 0.20`.
- `directional` also clears the threshold.
- `oracle_diverse` is only an upper-bound diagnostic, not a deployable policy sampler.

## Neighbor-cell A/B/F result

512-query artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_neighbor_cell_h160_512.json`

Candidate coverage:

```text
oracle any action-valid = 1.0000
oracle any state-valid  = 0.3672
unique next cells       = 5.2832
next-distance spread    = 76.5540
```

Own-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| A Q/V | 0.2012 | 0.7188 | 1.7012 | 2.0878 | 22.3031 | 46.8694 |
| B Q/V | 0.1777 | 0.7402 | 1.9332 | 2.1652 | 24.1143 | 46.0571 |
| F Q/V | 0.1992 | 0.7207 | 1.7785 | 2.1652 | 22.2999 | 46.4512 |

Source-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| A Q/V | 0.0430 | 0.0410 | 1.8559 | 4.0210 | 41.7791 | 56.3149 |
| B Q/V | 0.0625 | 0.0469 | 1.1599 | 2.5518 | 37.5835 | 51.7672 |
| F Q/V | 0.0449 | 0.0410 | 1.9332 | 3.8664 | 41.4698 | 56.2007 |

Interpretation:

- In source-state mode, B beats A/F on all listed metrics, but the absolute action-valid rate remains low.
- In own-state mode, B has the highest action-valid rate and lowest action-midpoint error, but A/F have better state-valid, path-stretch, and midpoint metrics.
- This is not a robust enough win to justify a policy smoke.

## Directional A/B/F result

128-query artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_directional_h160_128.json`

Candidate coverage:

```text
oracle any action-valid = 0.9922
oracle any state-valid  = 0.3594
unique next cells       = 5.3984
next-distance spread    = 73.3063
```

Own-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| A Q/V | 0.1797 | 0.7109 | 2.4745 | 2.1652 | 23.4024 | 50.6598 |
| B Q/V | 0.1641 | 0.7500 | 2.4745 | 2.1652 | 27.0886 | 48.2776 |
| F Q/V | 0.1797 | 0.7031 | 2.4745 | 2.1652 | 23.7117 | 50.5081 |

Interpretation:

- This repeats the neighbor-cell pattern.
- B improves action-valid and action-midpoint error.
- B does not improve state-valid, path stretch, or midpoint error.

## Oracle-diverse A/B/F result

128-query artifact:

- `exp/bmm_qv_budget_holdout_20260611_021352/joint_action_subgoal_oracle_diverse_h160_128.json`

Candidate coverage:

```text
oracle any action-valid = 1.0000
oracle any state-valid  = 0.3594
unique next cells       = 7.5938
next-distance spread    = 202.5975
```

Own-state mean rows:

| scorer | state valid | action valid | source stretch | next stretch | midpoint err | action midpoint err |
|---|---:|---:|---:|---:|---:|---:|
| A Q/V | 0.0234 | 1.0000 | 0.3093 | 5.5676 | 108.6058 | 118.3259 |
| B Q/V | 0.0234 | 1.0000 | -0.0000 | 6.4955 | 114.7920 | 120.9550 |
| F Q/V | 0.0234 | 1.0000 | 0.3093 | 5.2583 | 108.9151 | 118.6352 |

Interpretation:

- If globally useful actions are inserted, all learned own-state Q/V methods can pick action-valid pairs.
- B does not improve subgoal quality or midpoint metrics over A/F in this upper-bound mode.
- This weakens the case for continuing neural Q/V policy extraction.

## Final signal

Continue flat Q extraction: no.

Run policy smoke now: no.

Stop the whole project: no.

Pivot: yes.

The BMM critic still has a coherent role in reachability and high-level planning, but the current neural Q/V action-subgoal extraction is not strong enough. Candidate coverage can be fixed, yet B does not robustly beat A/F on the full selection objective.

Recommended next steps:

1. Stop running Q/V policy-facing sweeps for now.
2. Use BMM/V for high-level subgoal selection:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g))
```

3. Pair high-level subgoals with a separate low-level controller or BC policy.
4. Treat Q/V first-branch action selection as diagnostic only until it shows a larger and more consistent advantage.
5. If the project needs a final go/no-go quickly, build one value-only high-level subgoal smoke rather than a Q/V policy smoke.
