# BMM-TRL code review and next-step plan

Date: 2026-06-10
Repo reviewed: `jingxuxie/trl`, branch `main`
Primary files reviewed:

- `agents/bmm_trl.py`
- `utils/datasets.py`
- `scripts/bmm_reachability_utils.py`
- `scripts/eval_bmm_reachability.py`
- `scripts/check_bmm_reachability_report.py`
- `BMM_TRL_DIAGNOSTIC_HANDOFF.md`

## Executive diagnosis

The high-budget failure is very likely **not** a JAX shape bug. The main problem is that the current training distribution and losses do not give the critic enough high-budget boundary supervision.

The current implementation mostly trains:

1. positive parent pairs `(s_i, g_j, H)` where `H >= j - i`;
2. a positive-only one-witness transitive consistency target;
3. weak budget negatives that are usually far below the true offset;
4. one hard same-trajectory negative per sampled transition.

That is enough to avoid total collapse at small and mid budgets, but it is not enough to learn a sharp boundary at `H=256` and especially `H=512`. In fact, the current weak-negative rule makes `H=512` weak negatives essentially impossible when `max_budget=1024` and `budget_neg_frac=0.5`, because it only samples budgets satisfying

```text
H < 0.5 * offset.
```

So a weak negative at `H=512` would require `offset > 1024`, which is outside the current maximum offset. This leaves `H=512` almost entirely dependent on the single hard-negative sampler.

The next experiment should therefore **not** be another small weight tweak. It should be a diagnostic-only supervised reachability critic with balanced per-budget positives and negatives. Once that passes, reintroduce the max-min transitive target conservatively.

## Direct answers to the current questions

### 1. Is the one-witness max-min target well-posed?

The exact equation is well-posed:

```text
R_H(s, g) = max_{h,w} min(R_h(s, w), R_{H-h}(w, g)).
```

But the current implementation is not estimating that full max. It uses one sampled same-trajectory witness:

```text
y_trans = min(R_h(s, a, w), R_{H-h}(w, a_w, g)).
```

For a **positive** parent pair and a valid midpoint, this is a reasonable positive closure target. It is not enough to learn the classifier boundary, because it gives almost no negative information. For a negative parent pair, the useful condition would be that no witness should make both branches high. The current transitive loss does not train that.

Verdict: keep the one-witness target only as a later consistency regularizer. It should not be the primary objective while the diagnostic classifier is failing.

### 2. Is the hard-negative sampler strong enough?

No.

The current same-trajectory hard-negative sampler is a useful first patch, but one hard negative per sampled positive is too sparse and too indirectly distributed. The handoff already shows that weighting valid hard-negative budgets raised the mean hard-negative budget, but the `H=256/512` AUC and score gap stayed weak.

The deeper issue is that high-budget negatives require states with enough remaining trajectory length:

```text
negative for H requires offset > H.
```

For `H=512`, many sampled source states simply cannot produce such a negative. If you sample source states first and then try to attach one hard negative, `H=512` will remain under-covered.

Fix: sample supervised diagnostic pairs **by budget first**, not by source transition first. For each budget `H`, choose source indices from the subset with enough remaining horizon to produce positives and negatives.

### 3. Is BCE the right objective?

BCE is fine for direct supervised labels:

```text
label = 1[offset <= H].
```

The issue is not BCE itself. The issue is the sampling distribution and the fact that the bootstrapped transitive target is positive-only.

For the next stage, use three losses:

```text
loss_sup_bce      = balanced BCE on per-budget positive/negative pairs
loss_budget_rank  = pairwise boundary ranking across budgets for the same pair
loss_mono         = monotonicity, preferably by construction later
```

Set the transitive loss to zero until this supervised classifier passes diagnostics.

### 4. Is the budget representation too weak?

The scalar `log2(H) / log2(max_budget)` is acceptable for smoke tests, but it is probably too easy for the network to use it as a global monotone bias: larger budget means larger score. That is especially dangerous when high budgets see many easy positives and too few near-boundary negatives.

The next representation to try is not a huge embedding model. Use one of these simple upgrades:

```text
budget_feature = concat(log_budget_scalar, one_hot_budget_index)
```

or a small learned budget embedding concatenated to the goal.

Longer term, the cleanest version is a multi-head discrete-budget critic with one head per dyadic budget and monotonicity enforced across heads.

### 5. What should the next experiment be?

Do this order:

1. **Supervised-only budgeted reachability critic.** Set `lambda_trans=0`. Train balanced per-budget positives and negatives.
2. Add **pairwise budget-ranking loss** for the same `(s,g)` pair around its true offset boundary.
3. Upgrade budget input from scalar log budget to scalar-plus-one-hot or budget embedding.
4. Only after the above passes, reintroduce conservative transitive max-min with small weight.

Do not run policy comparisons yet.

## Code review: likely issues and fixes

### P0 issue: high-budget weak negatives are structurally missing

Current weak-negative sampling in `utils/datasets.py` chooses budgets below `neg_frac * offset`:

```python
def sample_bmm_negative_budgets(offsets, budgets, neg_frac):
    counts = np.searchsorted(budgets, neg_frac * offsets, side="left")
    ...
```

With `budget_neg_frac=0.5`, a weak negative at `H=512` requires `offset > 1024`. If `max_budget=1024`, this cannot happen. A weak negative at `H=256` requires `offset > 512`, which is possible but much rarer.

This explains the observed pattern: low/mid budgets learn something; `H=512` remains almost all positive.

#### Fix

Replace weak negatives as the main diagnostic signal with explicit per-budget boundary negatives:

```text
For each sampled budget H:
  positive offset: L_pos <= H
  negative offset: L_neg > H
```

Prefer near-boundary negatives first:

```text
L_neg in [H + 1, min(remaining, 2H)]
```

Then add farther negatives:

```text
L_neg in [2H, min(remaining, 4H)]
```

For `H=512`, this means explicitly sampling source states with `remaining >= 513`.

### P0 issue: high budgets see too many easy positives

Current positive budget sampling chooses any budget `H >= offset`:

```python
def sample_bmm_positive_budgets(offsets, budgets):
    min_budget_idxs = np.searchsorted(budgets, offsets, side="left")
    spans = len(budgets) - min_budget_idxs
    sampled_idxs = min_budget_idxs + random_index_in_span
    return budgets[sampled_idxs]
```

This is semantically correct but statistically dangerous. A pair with offset `1` can train `H=512` as positive. Many such easy positives allow the budget scalar to become a global high-budget bias.

#### Fix

For diagnostics, train high budgets mostly near their decision boundary:

```text
positive for H: offset in [0.5H, H] when possible
negative for H: offset in (H, 2H] when possible
```

Keep a small fraction of easy positives to teach monotonicity, but do not let them dominate.

### P0 issue: transitive loss is positive-only during classifier debugging

In `agents/bmm_trl.py`, the parent logit is trained both against the transitive target and against a direct positive target:

```python
y_trans = stop_gradient(min(first_r, second_r))
loss_trans = BCE(r_logits, y_trans)
loss_pos = BCE(r_logits, 1)
```

This is okay as a closure regularizer, but it cannot teach the high-budget negative boundary.

#### Fix

For the next diagnostic experiment:

```text
lambda_trans = 0.0
lambda_pos   = 0.0 or very small
lambda_sup   = 1.0
lambda_rank  = 0.25 to 1.0
lambda_mono  = 0.05
```

The current `loss_pos` can be replaced by the new balanced supervised BCE.

### P1 issue: one hard negative per positive is underpowered

The current sampler returns one hard negative per transition:

```text
value_hard_neg_goals
value_hard_neg_budgets
value_hard_neg_offsets
value_hard_neg_valids
```

Weighted budget sampling improved `hard_neg_budget_mean`, but the final diagnostics still show tiny gaps at `H=256/512`. That means the coverage is still insufficient.

#### Fix

Add vectorized multi-negative fields:

```text
value_sup_observations: [K, B, obs_dim]
value_sup_actions:      [K, B, action_dim]
value_sup_goals:        [K, B, goal_dim]
value_sup_budgets:      [K, B]
value_sup_labels:       [K, B]
value_sup_valids:       [K, B]
```

A single general supervised field is cleaner than adding more special-case fields for positives, weak negatives, and hard negatives.

### P1 issue: scalar budget feature can become a bias shortcut

Current critic input is:

```text
concat(goal, log2(H) / log2(max_budget))
```

This is not necessarily too weak; it may be too easy to use incorrectly. If training is imbalanced, the model can learn a monotone budget prior and ignore the pair-specific geometry.

#### Fix

Add one of these:

```text
budget_feature = "log_scalar_onehot"
budget_feature = "embedding"
budget_feature = "multi_head"
```

For the next quick experiment, use scalar plus one-hot:

```python
budget_idx = searchsorted(budgets, H)
budget_feat = concat(log_budget[:, None], one_hot(budget_idx, len(budgets)))
aug_goal = concat(goal, budget_feat)
```

This preserves the continuous ordering information while giving the network budget-specific capacity.

### P1 issue: monotonicity penalty is weak and not checked on both metrics

The training penalty is soft:

```python
loss_mono = mean(relu(R_low - R_high)^2)
```

The report gate currently checks the mean-score rows, but the useful conservative policy quantity is often the ensemble minimum.

#### Fix

For the gate, check both:

```text
row["mean"]
row["ensemble_min"]
```

For the model, later consider monotonicity by construction with discrete heads:

```text
logit_0 = raw_0
logit_k = logit_{k-1} + softplus(delta_k)
R_k = sigmoid(logit_k)
```

Do this only after the balanced supervised classifier passes; exact monotonicity can worsen high-budget collapse if negatives remain under-sampled.

### P2 cleanup: likely copy-paste bug in actor goal observations

In `utils/datasets.py`, this line appears to use `value_goal_idxs` where it likely should use `actor_goal_idxs`:

```python
batch["actor_goal_observations"] = self.get_observations(value_goal_idxs)
```

It probably does not affect the current BMM critic diagnostics, but it should be fixed:

```python
batch["actor_goal_observations"] = self.get_observations(actor_goal_idxs)
```

### P2 semantic issue: action-conditioned critic vs state-goal labels

The critic is `R(s, a, g, H)`, but the diagnostic label is based on same-trajectory offset:

```text
label = 1[offset <= H]
```

This is acceptable if the intended diagnostic is "reachable along the logged trajectory after taking the logged action." It is not a perfect label for true shortest-path reachability in PointMaze, because a far future state along a wandering trajectory may be reachable more quickly via another route.

For debugging architecture, add one of these checks:

1. a deterministic chain or grid diagnostic where offset equals graph distance;
2. a state-only reachability critic `R(s,g,H)` trained only for diagnostics;
3. PointMaze diagnostics stratified by Euclidean/geodesic proxy distance to detect shortcut false negatives.

## Proposed implementation patch

### 1. Add new config fields

In `agents/bmm_trl.py:get_config()`:

```python
# Diagnostic supervised classifier.
lambda_sup = 1.0
lambda_rank = 0.5
num_sup_pairs = 8
sup_budget_sample = "uniform"      # uniform over dyadic budgets
sup_pos_boundary_frac = 0.5         # prefer L in [0.5H, H]
sup_neg_min_factor = 1.0            # prefer L > H
sup_neg_max_factor = 2.0            # first hard band: (H, 2H]
sup_include_far_negs = True

# During classifier debugging.
lambda_trans = 0.0
lambda_pos = 0.0
lambda_budget_neg = 0.0
lambda_hard_neg = 0.0
lambda_rand_hinge = 0.0
lambda_mono = 0.05

# Budget representation.
budget_feature = "log_scalar_onehot"
```

Keep the old fields for ablations, but do not rely on them for the next diagnostic gate.

### 2. Add a general budgeted supervised sampler

In `utils/datasets.py`, add a sampler that directly returns labeled pairs.

Pseudo-code:

```python
def sample_bmm_supervised_pairs(self, batch_size, num_pairs):
    budgets = get_bmm_budgets(self.config)
    valid_idxs = self.dataset.valid_idxs
    remaining_all = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)] - valid_idxs

    sup_obs = []
    sup_actions = []
    sup_goals = []
    sup_budgets = []
    sup_labels = []
    sup_valids = []

    for _ in range(num_pairs):
        H = np.random.choice(budgets, size=batch_size)
        label = np.random.randint(0, 2, size=batch_size)  # balanced pos/neg

        src = np.empty(batch_size, dtype=np.int32)
        goal = np.empty(batch_size, dtype=np.int32)
        valid = np.zeros(batch_size, dtype=np.float32)

        for b in range(batch_size):
            if label[b] == 1:
                # Positive: choose source with at least one future state.
                eligible = valid_idxs[remaining_all >= 1]
                i = np.random.choice(eligible)
                rem = self.terminal_locs[np.searchsorted(self.terminal_locs, i)] - i
                hi = min(int(H[b]), int(rem))
                if hi < 1:
                    continue
                lo = max(1, int(np.ceil(self.config.sup_pos_boundary_frac * int(H[b]))))
                lo = min(lo, hi)  # fallback for short remaining horizon
                L = np.random.randint(lo, hi + 1)
            else:
                # Negative: choose source that can realize offset > H.
                eligible = valid_idxs[remaining_all >= int(H[b]) + 1]
                if len(eligible) == 0:
                    continue
                i = np.random.choice(eligible)
                rem = self.terminal_locs[np.searchsorted(self.terminal_locs, i)] - i
                lo = int(H[b]) + 1
                hi = min(int(rem), int(np.floor(self.config.sup_neg_max_factor * int(H[b]))))
                if lo > hi:
                    continue
                L = np.random.randint(lo, hi + 1)

            src[b] = i
            goal[b] = i + L
            valid[b] = 1.0

        sup_obs.append(self.get_observations(src))
        sup_actions.append(self.dataset["actions"][src])
        sup_goals.append(self.goal_vectors(goal))
        sup_budgets.append(H)
        sup_labels.append(label.astype(np.float32))
        sup_valids.append(valid)

    batch["value_sup_observations"] = stack_tree(sup_obs)
    batch["value_sup_actions"] = np.stack(sup_actions)
    batch["value_sup_goals"] = stack_tree(sup_goals)
    batch["value_sup_budgets"] = np.stack(sup_budgets).astype(np.int32)
    batch["value_sup_labels"] = np.stack(sup_labels).astype(np.float32)
    batch["value_sup_valids"] = np.stack(sup_valids).astype(np.float32)
```

Notes for Codex:

- Use `jax.tree_util.tree_map` for stacking if observations/goals can be pytrees, although the current BMM path only supports vector goals.
- For speed, this can be vectorized later. Correctness matters more now.
- Log per-budget valid fractions to ensure `H=256/512` negatives are actually present.

### 3. Add supervised BCE in the critic

In `agents/bmm_trl.py`, add a helper to score arbitrary budgeted pairs:

```python
def critic_logits_for(self, obs, actions, goals, budgets, grad_params=None, target=False):
    aug_goals = augment_goal_with_budget(goals, budgets, self.config["max_budget"])
    module = "target_critic" if target else "critic"
    kwargs = {} if grad_params is None or target else {"params": grad_params}
    return self.network.select(module)(obs, goals=aug_goals, actions=actions, **kwargs)
```

Then add:

```python
sup_logits = critic_logits_for(
    batch["value_sup_observations"],
    batch["value_sup_actions"],
    batch["value_sup_goals"],
    batch["value_sup_budgets"],
    grad_params=grad_params,
)

# sup_logits shape should be [E, K, B] or [K, B] depending network behavior.
# Make mask broadcasting explicit.
loss_sup = masked_mean(
    self.bce_loss(sup_logits, batch["value_sup_labels"]),
    batch["value_sup_valids"],
)
```

Be explicit about broadcasting. The existing `masked_mean` handles `[E,B]`, but for `[E,K,B]` it should be generalized:

```python
def masked_mean(x, mask, eps=1e-8):
    mask = jnp.asarray(mask, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask[None, ...]
    return (x * mask).sum() / (jnp.broadcast_to(mask, x.shape).sum() + eps)
```

### 4. Add pairwise budget-ranking loss

For a same-trajectory pair with offset `L`, define:

```text
H_plus  = smallest budget >= L
H_minus = largest budget < L, if it exists
```

Train:

```python
logit_plus  = R_logit(s, a, g, H_plus)
logit_minus = R_logit(s, a, g, H_minus)
loss_rank = softplus(margin - (logit_plus - logit_minus))
```

Use logits, not probabilities. Suggested margin: `1.0`.

This directly teaches the budget threshold. It is especially useful for high budgets:

```text
L in (256, 512]   => positive at H=512, negative at H=256
L in (512, 1024]  => positive at H=1024, negative at H=512
```

### 5. Make the transitive target conservative when re-enabled

Once the supervised classifier passes, change the transitive target to use a lower-confidence ensemble target:

```python
first_lcb = first_r.min(axis=0)      # [B]
second_lcb = second_r.min(axis=0)    # [B]
y_trans = jnp.minimum(first_lcb, second_lcb)
y_trans = jax.lax.stop_gradient(y_trans)[None, :]  # broadcast to ensembles
```

Then re-enable cautiously:

```text
lambda_trans = 0.025 or 0.05
```

Do not make the transitive loss dominant until the classifier boundary is already good.

## Diagnostics to add before the next run

### Training metrics

Log these every training step or every eval interval:

```text
critic/sup_loss
critic/sup_pos_r_mean_by_budget/H=...
critic/sup_neg_r_mean_by_budget/H=...
critic/sup_pos_count_by_budget/H=...
critic/sup_neg_count_by_budget/H=...
critic/sup_valid_frac_by_budget/H=...
critic/rank_loss
critic/rank_gap_mean
critic/rank_valid_frac
```

The most important metric is coverage. For each budget, confirm that the training batch actually contains both positives and negatives.

### Diagnostic report changes

Keep the current random-pair diagnostic, but add a balanced diagnostic:

```text
For each H:
  sample N positives with offset in [0.5H, H]
  sample N negatives with offset in (H, 2H]
```

Report AUC/gap on this balanced set. This is a cleaner classifier gate than a random offset distribution, which can be heavily one-class at high budgets.

Also update `check_bmm_reachability_report.py` to check both `mean` and `ensemble_min` metric dictionaries.

### Budget-ignored diagnostic

For each fixed pair `(s_i, s_j)` with offset `L`, scan all budgets:

```text
R(s_i, a_i, s_j, H), H in [1,2,4,...,1024]
```

Expected behavior:

```text
low before H ~= L, high after H >= L, monotone in H.
```

This directly shows whether the network learned a threshold or just a high-budget prior.

## Recommended next experiment matrix

Run these on PointMaze after adding the balanced supervised fields. Do not evaluate policy yet.

### Experiment A: supervised BCE only

```text
lambda_sup=1.0
lambda_rank=0.0
lambda_trans=0.0
lambda_pos=0.0
lambda_budget_neg=0.0
lambda_hard_neg=0.0
lambda_rand_hinge=0.0
lambda_mono=0.05
budget_feature=log_scalar
```

Goal: prove the architecture and balanced sampler can learn the label.

### Experiment B: supervised BCE + one-hot budget

```text
same as A
budget_feature=log_scalar_onehot
```

Goal: test whether scalar budget conditioning is the bottleneck.

### Experiment C: BCE + pairwise budget ranking

```text
lambda_sup=1.0
lambda_rank=0.5
lambda_trans=0.0
lambda_mono=0.05
budget_feature=log_scalar_onehot
```

Goal: sharpen the threshold at `H=256/512`.

### Experiment D: conservative transitive consistency

```text
lambda_sup=1.0
lambda_rank=0.5
lambda_trans=0.05
trans_target=ensemble_lcb
budget_feature=log_scalar_onehot
```

Goal: test whether max-min consistency helps once direct classification works.

### Experiment E: current sampler ablation

```text
Use current sampler, but lambda_trans=0.
```

Goal: separate sampler failure from transitive bootstrapping failure. If this still fails high budgets, the sampler is confirmed as the main problem.

## Suggested gate thresholds for the supervised stage

For the balanced diagnostic, use stronger thresholds than the current random-pair gate:

```text
H <= 128:  AUC >= 0.90, gap >= 0.20
H = 256:  AUC >= 0.85, gap >= 0.15
H = 512:  AUC >= 0.80, gap >= 0.10
monotonicity violation <= 0.05 initially, <= 0.01 later
```

For the current random-pair diagnostic, keep the existing gate as a secondary smoke test, not the main acceptance test.

## Minimal Codex task list

1. Fix `actor_goal_observations` to use `actor_goal_idxs`.
2. Add `budget_feature="log_scalar_onehot"` support.
3. Generalize `masked_mean` to support `[ensemble, K, batch]` tensors.
4. Add `value_sup_*` supervised budgeted pair fields to `GCDataset`.
5. Add `loss_sup` in `BMMTRLAgent.critic_loss`.
6. Add pairwise budget-ranking fields and `loss_rank`.
7. Add per-budget training coverage metrics.
8. Add balanced per-budget diagnostic rows to `scripts/bmm_reachability_utils.py`.
9. Update `check_bmm_reachability_report.py` to check both `mean` and `ensemble_min` metrics.
10. Run Experiments A-D above before any policy comparison.

## Bottom line

The high-budget failure is expected from the current objective. The implementation tests pass because the tensor plumbing is mostly correct, but the current data/loss setup does not force the classifier to learn the boundary near large budgets.

The highest-value next step is:

```text
balanced supervised budgeted reachability first,
then pairwise budget ranking,
then stronger budget representation,
then conservative transitive max-min.
```

Only after the budgeted classifier passes per-budget diagnostics should policy tuning resume.
