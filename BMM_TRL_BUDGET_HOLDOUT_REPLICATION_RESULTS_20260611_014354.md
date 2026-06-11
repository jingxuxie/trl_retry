# BMM-TRL budget-holdout replication results

Date: 2026-06-11

This follows `BMM_TRL_NEXT_STEPS_AFTER_BUDGET_HOLDOUT.md`.

## What was run

I finished the focused budget-holdout replication rather than running another broad sparse-label sweep.

Grid-cell H8 holdout:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --seeds=2 \
  --variants=A,B,C,D,F
```

This completed the existing seed 0 and seed 1 evidence with seed 2 for:

- A: no H8 parent labels, no transitive loss
- B: no H8 parent labels + Q/V transitive
- C: 16 H8 parent labels, no transitive loss
- D: 16 H8 parent labels + Q/V transitive
- F: no H8 parent labels + V-next distillation

Env-step H160 holdout:

```bash
conda run -n bmm-trl python scripts/run_bmm_qv_budget_holdout.py \
  --geodesic_budget_unit=env_steps \
  --budgets=40,80,160 \
  --eval_budgets=40,80,160 \
  --supervised_budgets=40,80 \
  --trans_budgets=160 \
  --seeds=0 \
  --variants=A,B,C,D,F \
  --value_restore_path=exp/bmm_grid_value_qv_teacher_40_80_160 \
  --value_restore_epoch=1000
```

Because env-step seed 0 was positive, I then ran seeds 1 and 2 with the same env-step A/B/C/D/F setup.

## Code added

- Added `--variants` to `scripts/run_bmm_qv_budget_holdout.py`, so the focused A/B/C/D/F matrix can be run without spending time on E/G rows.
- Added `scripts/summarize_bmm_budget_holdout.py`.
  - Reads finished `summary.json` files or interrupted per-run JSON directories.
  - Aggregates B-A, D-C, F-A, and optional G-B deltas.
  - Reports mean delta AUC, gap, BCE, ECE, Q-V abs diff, and per-seed rows.
- Added `scripts/test_bmm_budget_holdout_summary.py`.
- Extended `scripts/test_bmm_qv_budget_holdout.py` to check variant filtering.

## Grid-cell H8 aggregate

Source run directories:

- `exp/bmm_qv_budget_holdout_20260611_000227`
- `exp/bmm_qv_budget_holdout_20260611_001949`
- `exp/bmm_qv_budget_holdout_20260611_005700`

Generated aggregate:

- `exp/bmm_qv_budget_holdout_20260611_005700/aggregate_h8_all_seeds.md`
- `exp/bmm_qv_budget_holdout_20260611_005700/aggregate_h8_all_seeds.json`

| comparison | seeds | delta H8 AUC | delta H8 gap | delta H8 BCE | delta H8 ECE | delta Q-V abs | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 | -0.0208 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0116 | +0.0507 | -0.1503 | -0.0758 | +0.0011 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0002 | +0.0017 | -0.0063 | -0.0010 | -0.0005 | V-next distill control |
| G-B | 0 | +0.0014 | +0.0139 | -0.0560 | -0.0165 | -0.0036 | oracle branch control |

Per-seed H8 deltas:

| comparison | seed | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs |
|---|---:|---:|---:|---:|---:|---:|
| B-A | 0 | +0.0175 | +0.0703 | -0.3365 | -0.0527 | -0.0277 |
| B-A | 1 | +0.0172 | +0.0571 | -0.2986 | -0.0400 | -0.0176 |
| B-A | 2 | +0.0102 | +0.0980 | -0.4478 | -0.0798 | -0.0169 |
| D-C | 0 | +0.0082 | +0.0533 | -0.1446 | -0.0795 | -0.0078 |
| D-C | 1 | +0.0089 | +0.0217 | -0.0946 | -0.0555 | +0.0012 |
| D-C | 2 | +0.0177 | +0.0771 | -0.2118 | -0.0923 | +0.0099 |
| F-A | 0 | +0.0002 | +0.0014 | -0.0054 | -0.0008 | -0.0006 |
| F-A | 1 | +0.0004 | +0.0012 | -0.0053 | -0.0007 | -0.0002 |
| F-A | 2 | +0.0001 | +0.0025 | -0.0082 | -0.0015 | -0.0006 |

Interpretation:

- The grid-cell H8 holdout signal replicated over three seeds.
- B-A is positive for AUC, gap, BCE, ECE, and Q-V abs diff on every seed.
- D-C is positive for AUC and gap on every seed.
- F-A is essentially zero, so the effect is not explained by generic V-next distillation.

## Env-step H160 aggregate

Source run directories:

- `exp/bmm_qv_budget_holdout_20260611_010904`
- `exp/bmm_qv_budget_holdout_20260611_012102`

Generated aggregate:

- `exp/bmm_qv_budget_holdout_20260611_012102/aggregate_h160_all_seeds.md`
- `exp/bmm_qv_budget_holdout_20260611_012102/aggregate_h160_all_seeds.json`

| comparison | seeds | delta H160 AUC | delta H160 gap | delta H160 BCE | delta H160 ECE | delta Q-V abs | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 | -0.0380 | -0.0102 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0041 | +0.0498 | -0.1815 | -0.0547 | -0.0293 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0035 | +0.0030 | -0.0570 | -0.0016 | +0.0026 | V-next distill control |

Per-seed H160 deltas:

| comparison | seed | delta AUC | delta gap | delta BCE | delta ECE | delta Q-V abs |
|---|---:|---:|---:|---:|---:|---:|
| B-A | 0 | +0.0400 | +0.0640 | -0.5976 | -0.0398 | -0.0245 |
| B-A | 1 | +0.0029 | +0.0812 | -0.6050 | -0.0470 | -0.0175 |
| B-A | 2 | +0.0300 | +0.0489 | -0.5418 | -0.0272 | +0.0115 |
| D-C | 0 | +0.0043 | +0.0354 | -0.1159 | -0.0412 | -0.0132 |
| D-C | 1 | -0.0087 | +0.0610 | -0.2310 | -0.0661 | -0.0315 |
| D-C | 2 | +0.0169 | +0.0531 | -0.1977 | -0.0566 | -0.0431 |
| F-A | 0 | +0.0035 | +0.0067 | -0.0774 | -0.0038 | -0.0027 |
| F-A | 1 | +0.0013 | +0.0043 | -0.0671 | -0.0024 | -0.0006 |
| F-A | 2 | +0.0058 | -0.0019 | -0.0264 | +0.0013 | +0.0110 |

Interpretation:

- The env-step H160 no-parent comparison B-A is positive over all three seeds for AUC and gap.
- The env-step H160 few-parent comparison D-C is positive on average, but seed 1 has a small negative AUC delta. Gap, BCE, ECE, and Q-V abs diff still improve for all D-C seeds.
- F-A remains much smaller than B-A, so generic V-next distillation still does not explain the BMM transitive improvement.
- This is the first evidence that the budget-holdout effect survives the policy-facing env-step scale.

## Current conclusion

The budget-holdout hypothesis now has a clean 3-seed result at both grid-cell H8 and env-step H160:

```text
Shorter-budget Q/V knowledge helps heldout longer-budget parent classification.
The effect is strongest in the no-parent-label setting.
The V-next distillation control is much smaller, so the result looks BMM-specific rather than just teacher smoothing.
```

This is stronger evidence than the earlier uniform sparse-Q sweep.

## Caveats

- These are critic/reachability diagnostics, not policy performance.
- Env-step D-C AUC is mixed across seeds, even though gap/BCE/ECE improve.
- Runs are still 1000-step diagnostics, not tuned final training.
- The full all-budget gate is less important here than heldout-parent metrics; H160 is the primary target for env-step holdout.

## Recommended next step

Do not start a large benchmark yet.

The next useful step is a small policy-facing smoke that uses budget-conditioned action scoring instead of max-budget-only scoring:

```text
H_hat(s,g) = smallest H where V_H(s,g) >= tau
H_action = clamp(H_hat, min_budget, max_budget)
a = argmax_a Q_H_action(s,a,g)
```

Before that, it may be worth adding one cheap non-BMM baseline:

```text
score_H160_baseline = max(score_H40, score_H80)
```

If the simple monotone extrapolation baseline is below B, the BMM-specific story is cleaner.

## Verification

Passed:

```bash
conda run -n bmm-trl python scripts/test_bmm_qv_budget_holdout.py
conda run -n bmm-trl python scripts/test_bmm_budget_holdout_summary.py
```
