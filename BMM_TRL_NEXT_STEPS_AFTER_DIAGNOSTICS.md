# BMM-TRL next steps after high-budget diagnostics

Date: 2026-06-10

This plan follows `BMM_TRL_HIGH_BUDGET_DIAGNOSTIC_RESULTS.md`.

## TL;DR

The current evidence is good news for the implementation and bad news for the original PointMaze logged-offset diagnostic.

You have now shown:

1. The same JAX BMM critic path succeeds on a deterministic neural chain through `H=512`.
2. The PointMaze critic can overfit a frozen high-budget batch through `H=512` in both state-only and action-conditioned modes.
3. PointMaze heldout state-only pure BCE fails at `H=256/512`.
4. Offset oracle is perfect, but Euclidean and action-goal baselines are near chance at `H=256/512`.

That combination strongly suggests this is **not** a basic BMM plumbing bug. The likely issue is that high-budget same-trajectory offset is not a clean heldout function of the available vector inputs.

The project is not dead. The first prototype target was too convenient mathematically but too noisy statistically. The next step is to replace the diagnostic label or quantify its ambiguity before returning to policy-facing BMM losses.

## Why this is happening

The original prototype used:

```text
label = 1[offset <= H]
```

where `offset` is the number of logged trajectory steps between `s_i` and `s_j`.

This is clean on a deterministic chain because:

```text
offset == graph distance == shortest controllable distance.
```

It is not clean in PointMaze. In PointMaze, high-budget offset can encode:

- behavior-policy detours;
- trajectory phase/history;
- which path the behavior happened to take;
- whether the episode wandered before reaching the same physical location;
- alternate shorter paths that make a logged long offset a false negative for true reachability;
- wall/topology effects that are not captured by Euclidean distance.

So the model is asked to predict a partly trajectory-index-specific relation from only `(s, g, H)` or `(s, a, g, H)`. At high budgets, the Bayes-optimal classifier may genuinely be close to 0.5 for many heldout pairs.

The fixed-batch overfit result is important: it says the network can memorize these labels. The heldout failure says the labels do not generalize cleanly.

## Answers to the current questions

### How to test whether high-budget same-trajectory offset is label-noisy or non-identifiable?

Run a **local label ambiguity analysis**. For each heldout pair `(s, g, H)`, find nearest training pairs in feature space and measure whether nearby examples agree with the heldout label.

Use several feature spaces:

```text
xy_pair        = [x_s, y_s, x_g, y_g]
xy_delta       = [x_s, y_s, x_g, y_g, x_g - x_s, y_g - y_s]
full_pair      = [obs_s, obs_g]
full_pair_act  = [obs_s, action_s, obs_g]
```

For each `H`, report:

```text
kNN majority AUC
kNN probability AUC
mean local label entropy
contradiction rate: fraction with neighbor positive rate in [0.25, 0.75]
near-duplicate contradiction rate under small xy thresholds
```

Interpretation:

- If kNN is also near chance and local entropy is high at `H=256/512`, the label is non-identifiable from the chosen inputs.
- If kNN is strong but the neural critic is weak, then the issue is architecture/training.
- If xy-pair kNN is weak but full-pair kNN is strong, observation components beyond position matter.
- If full-pair kNN is also weak, logged offset is probably not a useful heldout target.

### Should the next diagnostic use geodesic/grid distance instead of logged offset?

Yes, for verification.

The clean diagnostic target should be:

```text
label_geo = 1[d_geo(s, g) <= H]
```

or a calibrated version:

```text
label_geo = 1[d_geo(s, g) <= c * H]
```

where `d_geo` is a maze-aware shortest-path distance. This aligns with the BMM theory better than logged behavior offset.

Two practical options:

1. **Maze grid BFS** if the map/layout is accessible.
   - Extract or hardcode free cells for PointMaze medium.
   - Convert `(x, y)` to grid cells.
   - Run BFS/Dijkstra between cells.
   - Calibrate steps by median one-step displacement in the dataset.

2. **Dataset position graph** if the map is inconvenient.
   - Nodes are dataset states or position bins.
   - Add edges between consecutive observations.
   - Add local kNN edges only if they are short and likely unobstructed.
   - Use graph shortest path as `d_graph`.

For the next diagnostic, graph/geodesic labels are better than logged offset labels. Logged offset can remain as a weak positive signal, not the main high-budget verification label.

### Would I try position-only, graph-distance labels, or a local train/test split?

Run all three, but in this order:

1. **Local train/test split from the same frozen pair pool.** This separates label memorization from pair-distribution generalization.
2. **Position-only critic.** Full observations may include velocity or other components that make future-state labels noisier than goal-position reachability.
3. **Graph/geodesic labels.** This is the likely replacement target if the ambiguity analysis confirms logged-offset noise.

### Is there a better minimal diagnostic before returning to policy-facing BMM losses?

Yes: build an empirical Bayes-ceiling diagnostic for the label.

Before training another neural critic, answer:

```text
Given the features we allow the critic to see, how predictable is label=1[offset<=H] at all?
```

The simplest proxy is kNN. If a kNN classifier cannot predict the label at high budgets, a neural policy critic should not be expected to do it either.

## Immediate implementation tasks for Codex

### Task 1: add `scripts/analyze_bmm_label_ambiguity.py`

Purpose: quantify whether logged-offset labels are identifiable from the inputs.

Inputs:

```bash
python scripts/analyze_bmm_label_ambiguity.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --budgets=64,128,256,512 \
  --num_train_pairs=100000 \
  --num_eval_pairs=20000 \
  --balanced_pairs=True \
  --k=32 \
  --features=xy_pair,xy_delta,full_pair,full_pair_action \
  --output_json=exp/bmm_label_ambiguity.json
```

Outputs per budget and feature mode:

```text
knn_auc
knn_gap
mean_neighbor_entropy
median_neighbor_entropy
contradiction_rate_25_75
contradiction_rate_10_90
near_duplicate_count
ear_duplicate_mixed_frac
```

Implementation notes:

- Reuse the existing balanced pair sampler from `scripts/bmm_reachability_utils.py`.
- Standardize features using train-pair mean/std.
- Use chunked NumPy distance computation first; no need for FAISS initially.
- For each eval pair, compute labels of `k` nearest train pairs.
- `p_hat = mean(neighbor_labels)`.
- AUC is computed from `p_hat` against eval labels.
- Entropy is `-p log p - (1-p) log(1-p)`.

Acceptance criteria:

```text
If kNN AUC <= 0.60 and contradiction_rate_25_75 is high at H=256/512,
stop treating logged offset as a clean high-budget heldout label.
```

### Task 2: add a same-pool train/test diagnostic

Purpose: separate generalization failure from label non-identifiability.

Script:

```text
scripts/debug_bmm_pair_pool_split.py
```

Procedure:

1. Sample one large balanced pair pool for budgets `256` and `512`.
2. Randomly split pairs into train/test rows.
3. Train state-only pure BCE on train rows.
4. Evaluate on heldout rows from the same pool.
5. Repeat with harder splits:
   - split by source trajectory;
   - split by source position bins;
   - split by goal position bins.

Interpretation:

- Random row split succeeds, trajectory/bin split fails: label has local memorization structure but poor spatial/trajectory generalization.
- All splits fail: model/training issue or severe label ambiguity.
- All splits pass but val-dataset split fails: train/val distribution shift.

### Task 3: add position-only diagnostic mode

Add config:

```python
diagnostic_obs_mode = "full"  # "full", "xy", "xy_delta"
```

For PointMaze diagnostics only:

```text
xy mode critic input:        [x_s, y_s], [x_g, y_g]
xy_delta mode critic input:  [x_s, y_s, x_g - x_s, y_g - y_s], or append delta to goal
```

Start with state-only. Do not include the actor.

Run:

```bash
python main.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --agent=agents/bmm_trl.py \
  --agent.value_only=True \
  --agent.diagnostic_critic_mode=state \
  --agent.diagnostic_obs_mode=xy \
  --agent.lambda_sup=1.0 \
  --agent.lambda_mono=0.0 \
  --agent.lambda_rank=0.0 \
  --agent.num_rank_pairs=0 \
  --agent.max_budget=512 \
  --agent.budgets="(1,2,4,8,16,32,64,128,256,512)"
```

Important: if the observation layout is not guaranteed, add a small utility that prints example observation dimensions and manually verifies which indices are position.

### Task 4: add graph/geodesic label diagnostic

Create:

```text
scripts/debug_bmm_geodesic_labels.py
```

Start with a dataset-position graph because it does not require knowing the maze map file.

Procedure:

1. Extract position features, likely first two dims after verification.
2. Bin positions or subsample nodes.
3. Add edges:
   - consecutive dataset transitions, weight 1;
   - optional short local edges, weight proportional to distance, only below a conservative radius.
4. Run shortest paths from sampled source nodes to sampled goal nodes.
5. Produce labels:

```text
label_graph = 1[d_graph <= H]
```

6. Evaluate simple baselines and train the state-only critic.

Acceptance:

```text
If graph/geodesic labels are learnable at H=256/512, use them as the next BMM verification target.
```

### Task 5: stop using logged-offset high-budget gate as the main success criterion

Keep logged-offset diagnostics, but relabel them:

```text
behavior-time diagnostic
```

Add a separate gate:

```text
geodesic reachability diagnostic
```

Do not require high-budget logged-offset AUC to pass before trying policy-facing BMM again. It is probably measuring behavior trajectory idiosyncrasy rather than reachability.

## Decision tree

### Branch A: kNN ambiguity is high at H=256/512

Conclusion: logged offset is not a clean high-budget label.

Next:

- switch supervised anchors to graph/geodesic labels;
- treat same-trajectory future goals only as positives, not as exact horizon labels;
- use random/budget negatives as weak or PU-style constraints;
- resume BMM transitive experiments on the geodesic diagnostic.

### Branch B: kNN predicts logged-offset labels well, but neural critic does not

Conclusion: model/training issue.

Next:

- add pairwise input features explicitly: `g - s`, `||g - s||`, maybe Fourier features;
- try bilinear/two-tower architecture or residual MLP;
- add pairwise ranking after vectorizing sampler;
- inspect normalization and goal/state feature scaling.

### Branch C: same-pool random split works, trajectory/bin split fails

Conclusion: logged-offset labels are locally learnable but distribution-specific.

Next:

- do not use trajectory split as a hard verifier;
- use geodesic labels for heldout verification;
- if behavior-time prediction is still interesting, include trajectory context/history, but this is not the right object for control.

### Branch D: geodesic labels also fail

Conclusion: raw MLP is not learning maze topology.

Next:

- add topology-aware features or a graph neural/nearest-neighbor value module;
- use position-grid encoding;
- learn on easier PointMaze or smaller budgets first;
- consider HIGL-like graph planning where BMM predicts local reachability and graph search handles long horizon.

## How to return to BMM losses after these diagnostics

Only after one clean high-budget diagnostic passes:

1. state-only supervised graph/geodesic labels;
2. action-conditioned one-step variant;
3. transitive max-min consistency;
4. policy-facing actor extraction.

For the action-conditioned version, consider a more Bellman-consistent finite-horizon label:

```text
R_Q(s, a, g, H) = R_V(s_next, g, H-1)
```

rather than asking `R(s,a,g,H)` to directly predict a logged offset label that does not depend cleanly on `a`.

## Why this struggle is normal

You are not stuck because the idea is bad. You are seeing a classic offline RL trap: the easy supervised proxy is not the actual object you want.

The BMM theory wants a reachability relation that behaves like a graph metric or finite-horizon controllability predicate. The first PointMaze proxy was logged same-trajectory time. That proxy is simple to sample, but at high horizons it may stop being a stable function of state and goal.

In hindsight, the prototype mixed two simplifications:

- **oversimplification:** treating logged offset as true reachability distance;
- **overcomplication:** adding transitive/ranking/monotonicity losses before proving the label was identifiable on heldout PointMaze.

The latest diagnostics are progress. They have narrowed the failure from "maybe BMM is broken" to a much sharper claim:

```text
BMM learns clean high-budget reachability labels, but PointMaze high-budget logged-offset labels do not generalize from vector inputs.
```

That is a useful research result and a clear next direction.

## Recommended immediate next run order

1. `analyze_bmm_label_ambiguity.py` on logged-offset labels.
2. `debug_bmm_pair_pool_split.py` with random row split and trajectory/bin splits.
3. Position-only state critic on logged-offset labels.
4. Graph/geodesic label construction and simple baselines.
5. State-only critic on graph/geodesic labels.
6. Action-conditioned Bellman-style critic after state-only geodesic works.
7. Re-enable max-min BMM consistency.

Do not tune policy performance until step 5 passes.
