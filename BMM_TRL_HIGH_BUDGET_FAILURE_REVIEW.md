# BMM-TRL high-budget failure review and next diagnostic plan

Date: 2026-06-10
Repo reviewed: `jingxuxie/trl`, branch `main`

Relevant current summary: `BMM_TRL_DIAGNOSTIC_SUMMARY.md`

## Executive diagnosis

The prototype is probably failing the current high-budget PointMaze gate because the current diagnostic target is harder and noisier than it looks. Balanced high-budget labels rule out the earlier simple explanation of "not enough H=256/512 negatives," but they do **not** rule out conditional label noise or representation mismatch.

The current target is:

```text
label = 1[offset <= H]
```

where `offset` is a same-trajectory future index difference. The model input is approximately:

```text
(observation_i, action_i, goal_j, budget_H)
```

For low budgets, same-trajectory offset is strongly correlated with local motion and geometry. For high budgets, especially `H=256/512`, the label can depend on behavior-policy wandering, trajectory phase, detours, and maze topology. Those variables are not fully encoded in `(s_i, a_i, g_j, H)`. If two visually/geometrically similar pairs can have different labels because one trajectory reached the goal state after 480 logged steps and another after 700 logged steps, the Bayes-optimal classifier may genuinely be close to 0.5.

So the current result does **not** disprove BMM. It mostly says the current PointMaze same-trajectory-offset diagnostic is not yet a clean first verification target.

The next step should be to temporarily simplify the problem until the desired classifier is definitely learnable:

1. pure supervised BCE only, no monotonicity, no actor, no transitive;
2. state-only `R(s,g,H)` diagnostic critic;
3. fixed-batch overfit and oracle-feature sanity checks;
4. synthetic neural chain/grid using the same JAX network path;
5. only then return to PointMaze high-budget heldout diagnostics.

## Direct answers to the five questions

### 1. Is the action-conditioned critic label well-posed?

For the current diagnostic, only weakly.

The code labels a pair by `offset <= H`, while the diagnostic batch uses the logged action at the source state. In `scripts/bmm_reachability_utils.py`, labels are built only from `offsets <= budget`, while actions are simply `dataset["actions"][idxs]`.

That means the action is an input feature but not part of the label definition. This is fine if the intended semantics are:

```text
starting from this logged transition, will this exact future state appear within H logged steps along the same logged trajectory?
```

It is not a clean label for:

```text
is g reachable within H after taking action a under an optimal/learned policy?
```

For small budgets, the logged action probably matters enough to help. For high budgets, the first action is often a weak predictor of whether a future state appears within 256/512 logged steps. The label is closer to a behavior-trajectory time-to-hit label than an action-conditioned reachability label.

### 2. Should we first test a state-only diagnostic critic `R(s,g,H)`?

Yes. This is now the highest-priority diagnostic.

If state-only works at `H=256/512` and action-conditioned fails, then the action-conditioned label/objective is the mismatch. If state-only also fails, the issue is not the action input; it is the offset label, representation, or the PointMaze task geometry.

State-only should be implemented as a diagnostic-only mode, not a replacement for the eventual policy critic.

### 3. Is balanced BCE enough, or is ranking essential?

Balanced BCE should be enough on a clean deterministic label. Pairwise budget ranking is useful, but it should not be required to pass a simple supervised classifier test.

The fact that balanced BCE gets nearly random AUC at `H=256/512` suggests one of these:

- the label is noisy or not a function of the inputs;
- the representation lacks the information needed to infer long-horizon maze distance;
- the model/training path has a hidden bug;
- the auxiliary losses still make the run not actually pure supervised BCE.

Ranking should be added after the fixed-batch and state-only checks. It is not the first tool to debug this failure.

### 4. Could same-trajectory offset be a bad high-budget label in PointMaze?

Yes. This is currently the leading hypothesis.

Same-trajectory offset is a clean label on a deterministic chain or on a dataset graph where the logged trajectory is the object of interest. It is not necessarily a clean proxy for true shortest-path reachability in a maze. Long logged offsets can reflect detours or behavior-policy inefficiency rather than long geodesic distance. Conversely, a pair labeled negative because it occurs after more than `H` logged steps might still be reachable in fewer than `H` steps through an alternate path.

This issue gets worse at large `H`: the label becomes increasingly about trajectory history and behavior-policy time, not just local geometry.

### 5. What minimal diagnostic should run next?

Run this ladder before any policy evaluation:

```text
A. Fixed-batch overfit, action-conditioned.
B. Fixed-batch overfit, state-only.
C. Synthetic neural chain/grid with same network and sampler API.
D. PointMaze state-only, same train examples used for eval.
E. PointMaze state-only, heldout val examples.
F. PointMaze action-conditioned only if B-E pass.
```

The fastest decisive test is **fixed-batch overfit**. Freeze a balanced supervised dataset for `H=256/512`, train on exactly those examples, and evaluate on the same examples. If the network cannot drive training AUC/gap near 1.0 on a fixed finite dataset, there is a code/model optimization bug. If it can overfit but cannot generalize, the issue is label/representation/generalization rather than plumbing.

## Code review observations

### Observation 1: the current diagnostic action input is not reflected in the label

`make_pair_batch` creates labels from offset and budget:

```python
labels = (offsets <= budget).astype(np.float32)
```

but uses the logged source action:

```python
actions = gc_dataset.dataset["actions"][idxs]
```

This is not necessarily a bug, but it means the current `R(s,a,g,H)` diagnostic is not a clean action-conditional reachability target. It is a trajectory-index target with the action appended.

### Observation 2: default config still has monotonicity on

`get_config()` currently has:

```python
lambda_trans=0.0
lambda_pos=0.0
lambda_budget_neg=0.0
lambda_hard_neg=0.0
lambda_rand_hinge=0.0
lambda_mono=0.05
lambda_sup=1.0
```

So a run described as "supervised BCE only" is not pure BCE unless `--agent.lambda_mono=0.0` is explicitly passed. Monotonicity probably does not fully explain the random high-budget AUC, but it can absolutely obscure the diagnosis, especially because the one-hot-budget run showed worse monotonicity.

For the next debugging run, set:

```text
--agent.lambda_mono=0.0
--agent.lambda_rank=0.0
--agent.lambda_sup=1.0
--agent.lambda_trans=0.0
--agent.lambda_pos=0.0
--agent.lambda_budget_neg=0.0
--agent.lambda_hard_neg=0.0
--agent.lambda_rand_hinge=0.0
```

### Observation 3: ranking sampler is predictably slow

`add_bmm_rank_fields` loops over `num_pairs * batch_size` and recomputes an eligible set depending on each sampled offset. With `num_rank_pairs=8` and `batch_size=512`, this is thousands of Python eligibility scans per training step. That explains why ranking probes stall before the first train log.

Do not use ranking until this is vectorized or precomputed.

### Observation 4: the current critic is a plain concatenation MLP

`GCValue` concatenates observations, goals, and actions, then feeds them through an MLP. That is a good minimal baseline, but it has no built-in maze topology, graph distance, or path-planning bias. A high-budget PointMaze boundary may require learning a nonlocal wall-aware distance function from raw vector states.

This is another reason to test synthetic chain/grid and state-only PointMaze first.

### Observation 5: the tabular test is algebraic, not a neural supervised test

`scripts/test_bmm_tabular.py` verifies the exact max-min backup on chain/grid and prints an error-scaling table. That is useful, but it does not test whether the JAX critic, sampler, budget features, and BCE loss can learn high-budget thresholds. Add a neural chain/grid test next.

## Recommended code changes

### 1. Add diagnostic state-only mode

Add a config option:

```python
diagnostic_critic_mode = "action"  # "action" or "state"
```

Then route all BMM critic calls through:

```python
def maybe_actions(self, actions):
    if self.config.get("diagnostic_critic_mode", "action") == "state":
        return None
    return actions
```

In `critic_logits_for`:

```python
return self.network.select(module)(
    observations,
    goals=aug_goals,
    actions=self.maybe_actions(actions),
    ...
)
```

In `create`, initialize the critic with no action argument in state-only mode:

```python
if config.get("diagnostic_critic_mode", "action") == "state":
    critic_args = (ex_observations, ex_critic_goals)
else:
    critic_args = (ex_observations, ex_critic_goals, ex_actions)
```

Keep the actor untouched or skip actor loss entirely in this mode.

### 2. Add value-only training mode for diagnostics

Add:

```python
value_only = False
```

In `total_loss`:

```python
critic_loss, critic_info = self.critic_loss(...)
if self.config.get("value_only", False):
    return critic_loss, info
```

Use this for all classifier diagnostics. The actor is not the target right now and can slow/confuse experiments.

### 3. Add fixed-batch overfit script

Create:

```text
scripts/debug_bmm_fixed_batch_overfit.py
```

Behavior:

1. Build one balanced supervised batch only for selected budgets, e.g. `H=(256,512)`.
2. Reuse the exact same batch for every update.
3. Evaluate on the same batch every 100 steps.
4. Stop when train AUC > 0.98 and gap > 0.5, or after a small max step count.

Run both:

```text
state-only, value-only, lambda_mono=0
state+action, value-only, lambda_mono=0
```

Expected interpretation:

- Cannot overfit fixed batch: implementation/model/optimization bug.
- Can overfit fixed batch but cannot generalize: label or representation issue.

### 4. Add oracle-feature sanity mode

For one run only, append the true normalized offset to the goal/budget feature:

```text
oracle_offset_feature = offset / H or log2(offset) / log2(max_budget)
```

Do not use this for policy or final experiments. It is only a plumbing test. With offset leaked, balanced BCE should trivially pass all budgets. If it does not, the loss/eval path is wrong.

### 5. Add neural chain/grid test using the same JAX path

The existing tabular script proves the algebra. Add a neural supervised test:

```text
scripts/test_bmm_neural_chain.py
```

Construct synthetic transitions:

```text
state i: [i / N]
action: +1 or zero
trajectory: 0 -> 1 -> ... -> N-1
label: 1[j - i <= H]
```

Train the same BMM critic on balanced pairs for budgets up to 512. This should pass. Then add a 2D grid with BFS distance labels. If chain passes but grid fails, the problem is topology/geodesic representation. If chain fails, the implementation is still wrong.

### 6. Add simple baselines to the report

For each balanced diagnostic budget, compute AUC/gap for non-neural scores:

```text
score_offset_oracle = -offset
score_euclidean = -||obs_xy - goal_xy||
score_action_goal = dot(action, goal_xy - obs_xy)
```

`score_offset_oracle` should be perfect by construction. If Euclidean AUC is near 0.5 at `H=256/512`, the learned MLP has little chance without a topology-aware signal. If Euclidean AUC is high but the MLP is random, look for optimization/model bugs.

### 7. Vectorize or precompute rank fields before using ranking

Do not sample ranking pairs with nested Python loops inside every training step. Instead:

- precompute eligible source indices per budget interval once in `GCDataset.__post_init__`; or
- sample `pos_budget_idx` in vectorized chunks, group by interval, and fill arrays group-wise; or
- generate a fixed rank replay buffer for diagnostics.

Only then test:

```text
lambda_rank=0.25 or 0.5
rank_margin=1.0
```

## Minimal next experiment matrix

### Experiment 0: truly pure BCE on current action-conditioned critic

Purpose: remove monotonicity as a confound.

```bash
--agent.value_only=True \
--agent.diagnostic_critic_mode=action \
--agent.lambda_sup=1.0 \
--agent.lambda_mono=0.0 \
--agent.lambda_rank=0.0 \
--agent.lambda_trans=0.0 \
--agent.lambda_pos=0.0 \
--agent.lambda_budget_neg=0.0 \
--agent.lambda_hard_neg=0.0 \
--agent.lambda_rand_hinge=0.0
```

Expected: probably still weak at high budgets, but this makes the conclusion clean.

### Experiment 1: fixed-batch overfit, state-only

```bash
--agent.value_only=True \
--agent.diagnostic_critic_mode=state \
--agent.lambda_sup=1.0 \
--agent.lambda_mono=0.0
```

Train and evaluate on the same frozen examples. This must pass.

### Experiment 2: fixed-batch overfit, action-conditioned

Same as Experiment 1, but:

```bash
--agent.diagnostic_critic_mode=action
```

If this fails but state-only passes, action is harming optimization or the action-conditioned architecture/path has a bug.

### Experiment 3: neural chain/grid

Use synthetic data and the same BMM critic code. Budgets should pass through 512. This is the cleanest verification of the prototype idea.

### Experiment 4: PointMaze state-only heldout

If fixed-batch and synthetic pass, run PointMaze heldout state-only. If this fails at high budgets, the PointMaze same-trajectory label is the problem or the raw state representation cannot infer long-range maze distance.

### Experiment 5: PointMaze position-only state critic

Use only position-like dimensions for `s` and `g` if the observation structure supports it. Full future observations may include velocity or other components that make future-state matching noisier than goal-position reachability.

## What would count as a real bug?

Treat these as high-priority bugs if observed:

1. The critic cannot overfit a frozen balanced batch at `H=256/512`.
2. The oracle-offset-feature run does not pass.
3. The neural chain test fails.
4. Train AUC on the exact supervised batch is high, but the evaluation code reports random on that exact same batch.
5. State-only fixed-batch passes but action-conditioned fixed-batch fails with identical labels and enough capacity.

If none of those happen, the code is probably mostly correct and the current failure is caused by label semantics/generalization.

## Why this felt like it should be simple, but is not

The original BMM idea is simple in the deterministic graph setting:

```text
R_H(s,g) = 1[d(s,g) <= H]
```

and the max-min recurrence is simple when `d` is a true graph distance.

The current PointMaze diagnostic changed that into:

```text
R_H(s,a,g) = 1[g appears within H logged steps after this transition]
```

That is a different object. It is behavior-trajectory-index prediction. At high budgets, it may be weakly identifiable from state/action/goal alone. Balanced sampling fixes class imbalance, but it cannot fix non-identifiability.

So the prototype is probably too complicated at the current verification stage. The clean first verification should be:

```text
state-only + deterministic neural chain/grid + fixed-batch overfit
```

Then gradually add PointMaze, heldout generalization, action conditioning, ranking, and finally transitive max-min.

## Suggested Codex task list

1. Add `diagnostic_critic_mode={state,action}`.
2. Add `value_only=True` support to skip actor loss.
3. Add `scripts/debug_bmm_fixed_batch_overfit.py`.
4. Add optional oracle-offset feature for debugging only.
5. Add `scripts/test_bmm_neural_chain.py` using the same BMM critic path.
6. Add simple baseline AUCs to `scripts/bmm_reachability_utils.py`.
7. Rerun pure BCE with `lambda_mono=0`.
8. Rerun state-only fixed-batch and PointMaze heldout.
9. Only optimize/vectorize ranking after the above pass.

## Bottom line

Do not spend more cycles tuning high-budget action-conditioned PointMaze policy-facing losses yet. First prove the classifier can learn a clean high-budget threshold in a setting where the label is truly a function of the input. If that passes, the current `H=256/512` failure is evidence that the PointMaze same-trajectory offset label is not a clean high-budget reachability target, not that the BMM idea is broken.
