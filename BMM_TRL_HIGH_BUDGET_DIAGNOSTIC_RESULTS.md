# BMM-TRL High-Budget Diagnostic Results

Date: 2026-06-10

This note records the follow-up diagnostics from
`BMM_TRL_HIGH_BUDGET_FAILURE_REVIEW.md`.

## Implemented

- Added `diagnostic_critic_mode={action,state}` to `agents/bmm_trl.py`.
- Added `value_only=True` to train the critic without actor loss.
- Added optional `oracle_offset_feature` plumbing for debug-only checks.
- Added `scripts/debug_bmm_fixed_batch_overfit.py`.
- Added `scripts/test_bmm_neural_chain.py`.
- Added simple diagnostic baselines to `scripts/bmm_reachability_utils.py`:
  - `offset_oracle`
  - `euclidean`
  - `action_goal`

## Checks Run

```bash
conda run -n bmm-trl python scripts/test_bmm_reachability_eval.py
conda run -n bmm-trl python scripts/test_bmm_agent_shapes.py
conda run -n bmm-trl python scripts/test_bmm_neural_chain.py
```

All three passed. The CPU/sandbox checks still print the known JAX CUDA plugin
warning when GPU access is unavailable.

## Neural Chain Result

`scripts/test_bmm_neural_chain.py` trains the real BMM JAX critic on a
deterministic chain where same-trajectory offset equals graph distance.

Fixed-batch state-only results:

| step | H | AUC | gap |
|---:|---:|---:|---:|
| 300 | 128 | 0.9930 | 0.5354 |
| 300 | 256 | 0.9993 | 0.7505 |
| 300 | 512 | 1.0000 | 0.8339 |

Interpretation: the JAX critic, budget augmentation, supervised BCE path, and
high-budget labels can work when the label is a clean function of `(s, g, H)`.

## PointMaze Fixed-Batch Overfit

These runs froze one balanced supervised PointMaze batch with budgets 256 and
512, disabled actor loss, disabled monotonicity, disabled rank/transitive/old
negative losses, and trained on the same batch repeatedly.

### State-Only Critic

Command used `diagnostic_critic_mode=state`, hidden dims `(128, 128)`, batch
size 128.

| step | H | mean AUC | mean gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|
| 1000 | 256 | 0.9179 | 0.4097 | 0.9162 | 0.4069 |
| 1000 | 512 | 0.8740 | 0.3489 | 0.8745 | 0.3525 |
| 2000 | 256 | 0.9761 | 0.6138 | 0.9743 | 0.6152 |
| 2000 | 512 | 0.9683 | 0.5752 | 0.9649 | 0.5748 |

The fixed-batch target was reached at 2000 steps.

### Action-Conditioned Critic

Same setup, but `diagnostic_critic_mode=action`.

| step | H | mean AUC | mean gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|---:|
| 500 | 256 | 0.8704 | 0.2991 | 0.8672 | 0.3007 |
| 500 | 512 | 0.8362 | 0.2643 | 0.8348 | 0.2644 |
| 1000 | 256 | 0.9739 | 0.5710 | 0.9726 | 0.5700 |
| 1000 | 512 | 0.9591 | 0.5255 | 0.9590 | 0.5331 |

The fixed-batch target was reached at 1000 steps.

Interpretation: the critic can memorize high-budget PointMaze labels in both
state-only and action-conditioned modes. This argues against a basic loss,
network-shape, or budget-feature plumbing bug.

## PointMaze State-Only Heldout

A short heldout run was trained with:

```text
offline_steps=5000
diagnostic_critic_mode=state
value_only=True
lambda_sup=1.0
lambda_mono=0.0
lambda_rank=0.0
num_rank_pairs=0
lambda_trans=lambda_pos=lambda_budget_neg=lambda_hard_neg=lambda_rand_hinge=0.0
max_budget=512
budgets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
```

Checkpoint:

```text
exp/mrl/Debug/sd000_20260610_170743/params_5000.pkl
```

Diagnostic report:

```text
exp/mrl/Debug/sd000_20260610_170743/bmm_reachability_state_5000.json
```

Balanced heldout results:

| H | mean AUC | mean gap | ensemble-min AUC | ensemble-min gap |
|---:|---:|---:|---:|---:|
| 64 | 0.7356 | 0.1920 | 0.7337 | 0.1919 |
| 128 | 0.5974 | 0.0600 | 0.5955 | 0.0595 |
| 256 | 0.4684 | -0.0089 | 0.4720 | -0.0081 |
| 512 | 0.4898 | -0.0024 | 0.4928 | -0.0018 |

Gate result:

```text
FAIL mean_monotonicity_violation=0.2518 exceeds 0.0500
FAIL min_monotonicity_violation=0.2523 exceeds 0.0500
FAIL H=256 mean: auc=0.4684 below 0.8500
FAIL H=256 mean: pos-neg gap=-0.0089 below 0.1500
FAIL H=512 mean: auc=0.4898 below 0.8000
FAIL H=512 mean: pos-neg gap=-0.0024 below 0.1000
```

The monotonicity value here is a diagnostic statistic, not a training penalty;
`lambda_mono` was set to zero.

## Baseline Evidence

The same heldout report includes non-neural baseline AUCs on balanced pairs.

| H | offset oracle AUC | Euclidean AUC | action-goal AUC |
|---:|---:|---:|---:|
| 64 | 1.0000 | 0.6977 | 0.5195 |
| 128 | 1.0000 | 0.5382 | 0.5767 |
| 256 | 1.0000 | 0.4845 | 0.4998 |
| 512 | 1.0000 | 0.5050 | 0.4986 |

Interpretation: the offset label is perfectly separable if the true offset is
leaked, but simple geometry is near chance at high budgets. Combined with the
fixed-batch overfit success, this points toward heldout generalization,
trajectory-offset label semantics, or maze topology representation as the main
failure, not a basic BMM implementation bug.

## Current Diagnosis

The strongest current evidence is:

1. Clean deterministic chain succeeds through `H=512`.
2. PointMaze fixed-batch overfit succeeds through `H=512`.
3. PointMaze heldout state-only fails at `H=256/512`.
4. Euclidean and action-goal baselines are near chance at `H=256/512`.

The likely issue is that high-budget same-trajectory offset in PointMaze is not
a clean heldout function of the vector inputs. It may encode behavior-policy
detours, maze topology, phase/history, or alternate shorter paths that are not
captured by `(s, g, H)` or `(s, a, g, H)`.

## Recommended Next Diagnostics

- Add a position-only diagnostic if the observation layout is reliable.
- Add a geodesic or graph-distance proxy diagnostic for PointMaze.
- Measure label ambiguity directly by finding nearby `(s, g)` pairs with
  conflicting high-budget labels.
- Try a small train/eval split from the same frozen balanced pair pool to
  separate memorization from generalization.
- Keep ranking disabled until the sampler is vectorized; it is not needed to
  explain the current high-budget heldout failure.
