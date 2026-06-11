# BMM-TRL PointMaze reachability target plan

Date: 2026-06-11

This plan follows the latest high-budget diagnostics and the new kNN result:

```text
Logged-offset label on balanced PointMaze pairs:
H=256: best kNN AUC ~= 0.54
H=512: best kNN AUC ~= 0.52
```

Together with the previous results, this is enough to stop treating high-budget same-trajectory offset as the main PointMaze verification label.

## Current conclusion

The current BMM implementation is probably not the blocker.

Evidence so far:

1. The same JAX BMM critic path learns a deterministic neural chain through `H=512`.
2. PointMaze fixed-batch overfit succeeds through `H=512` in both state-only and action-conditioned modes.
3. PointMaze heldout logged-offset labels fail at `H=256/512`.
4. Euclidean and action-goal baselines are near chance at `H=256/512`.
5. kNN on balanced pairs is also near chance at `H=256/512`.

The best interpretation is:

```text
same-trajectory offset in PointMaze is a behavior-time label, not a clean high-budget reachability label.
```

It is useful as a source of positive examples and short-horizon supervision, but it is not a reliable high-budget negative/threshold label.

## Best next PointMaze target

Use **maze-aware geodesic reachability** as the next clean diagnostic target.

The preferred target is:

```text
R_geo(s, g, H) = 1[d_geo(xy_s, xy_g) <= H]
```

where `d_geo` is a shortest-path distance through the maze, converted to environment-step units.

If the environment layout is accessible, use **grid/BFS geodesic distance from the maze layout** as the primary target. If layout access is annoying or brittle, start with a **dataset-position graph shortest path** as a fast proxy, but treat it as an offline-support reachability target rather than true environment reachability.

Recommended order:

1. **Grid/BFS geodesic from maze layout** if available.
2. **Conservative dataset-position graph** if layout extraction is not available yet.
3. **Action-conditioned finite-horizon controllability proxy** only after state-only geodesic works.

Do not use logged offset as a high-budget hard negative anymore.

## Why grid/BFS geodesic is the best clean target

A good diagnostic target should be:

- deterministic from `(s, g, H)`;
- topology-aware;
- independent of behavior-policy detours;
- monotone in `H`;
- compatible with the BMM max-min recurrence;
- easy to evaluate with a non-neural oracle.

Grid/BFS geodesic satisfies these. Logged offset does not.

The BMM theory wants a relation like:

```text
1[shortest controllable distance <= H]
```

not:

```text
1[the behavior trajectory happened to hit this state within H logged steps]
```

## Target definitions

### Target A: layout/grid geodesic

Use this if the PointMaze map can be extracted from the env or hardcoded safely.

```text
cell_s = xy_to_cell(xy_s)
cell_g = xy_to_cell(xy_g)
d_grid = BFS_distance(cell_s, cell_g)
d_step = d_grid * step_scale
label = 1[d_step <= H]
```

`step_scale` converts grid-cell distance to environment-step distance. Estimate it from the dataset:

```text
median_step_xy = median(||xy_{t+1} - xy_t|| over non-terminal transitions)
cell_size      = inferred cell width in xy units
env_steps_per_cell = cell_size / median_step_xy
```

Then:

```text
d_step = d_grid * env_steps_per_cell
```

Alternative calibration:

Fit a single scale factor using only short horizons, e.g. `H <= 64`, where logged offset is still relatively identifiable. Do not fit the scale on high-budget logged-offset labels.

### Target B: dataset-position graph distance

Use this as the fastest implementation path if map extraction is not ready.

Build graph nodes from position bins or representative dataset states.

Edges:

1. Add consecutive transition edges:

```text
bin(x_t) <-> bin(x_{t+1}), weight = 1
```

2. Optionally add local adjacency edges only if conservative:

```text
bin_i <-> bin_j if distance(bin_i, bin_j) <= radius
```

but avoid wall shortcuts. If the map is unavailable, prefer only observed transition edges plus very small local edges that are supported by many observed transitions.

Label:

```text
label_graph = 1[d_graph(bin_s, bin_g) <= H]
```

Unreachable or disconnected pairs should be treated as `unknown` for supervised diagnostics unless the diagnostic explicitly wants hard offline-support negatives.

This graph target answers:

```text
reachable within H steps under dataset-support topology
```

not necessarily true environment reachability.

### Target C: action-conditioned Bellman proxy

After a state-only target works, define action-conditioned labels through next state:

```text
R_Q(s_t, a_t, g, H) = R_V(s_{t+1}, g, H - 1)
```

Use the logged `s_{t+1}` for supervised action-conditioned diagnostics. This is more Bellman-consistent than assigning `R(s,a,g,H)` a label based directly on future logged offset.

For policy-facing BMM, this is the right direction:

```text
Q_H(s,a,g) should mean: after taking a, is g reachable within H-1 more steps?
```

## Code review notes

### 1. Current code already supports the right debugging switches

`BMMTRLAgent` now has:

```python
diagnostic_critic_mode = "action" or "state"
value_only = True or False
oracle_offset_feature = True or False
```

`maybe_actions()` correctly drops actions in state-only mode, and `create()` initializes the critic with or without actions depending on `diagnostic_critic_mode`.

Keep using these switches.

### 2. The current evaluator is still logged-offset first

`make_pair_batch()` still defines:

```python
labels = (offsets <= budget).astype(np.float32)
```

That is now a behavior-time diagnostic. Keep it, but rename reports mentally and in code:

```text
logged_offset / behavior_time
```

Add separate geodesic label paths. Do not overload the current logged-offset evaluator.

### 3. Defaults can still accidentally re-enable expensive/confounding losses

`get_config()` currently defaults to:

```python
lambda_mono = 0.05
lambda_rank = 0.5
num_rank_pairs = 8
```

For diagnostics, override these every time:

```bash
--agent.value_only=True \
--agent.lambda_mono=0.0 \
--agent.lambda_rank=0.0 \
--agent.num_rank_pairs=0 \
--agent.lambda_trans=0.0 \
--agent.lambda_pos=0.0 \
--agent.lambda_budget_neg=0.0 \
--agent.lambda_hard_neg=0.0 \
--agent.lambda_rand_hinge=0.0
```

Suggested code cleanup: change diagnostic defaults to `lambda_rank=0.0` and `num_rank_pairs=0` until ranking is vectorized.

### 4. Same-trajectory positives are still useful

Do not throw away logged trajectories. Use them as:

- positive anchors: if `offset <= H`, then likely reachable within `H` under the behavior path;
- local edge construction for graph distance;
- transition edges for action-conditioned Bellman labels;
- witness candidates for transitive max-min.

But stop using `offset > H` as a hard negative at high budgets.

## Minimal implementation plan

### Step 1: add a label abstraction

Add a label type parameter to diagnostics and supervised sampling:

```python
reachability_label_type = "logged_offset"  # "logged_offset", "graph", "grid_geodesic"
```

Do not modify the logged-offset path. Add new functions:

```python
make_logged_offset_pair_batch(...)
make_graph_pair_batch(...)
make_grid_geodesic_pair_batch(...)
```

Each should return:

```python
observations
actions
goals
budgets
labels
valids
aux_distances
```

### Step 2: verify PointMaze position extraction

Add:

```text
scripts/inspect_pointmaze_observation_layout.py
```

It should print:

```text
obs_dim
action_dim
first 10 observations
first 10 next-observation deltas
candidate xy dims and ranges
median one-step xy displacement
terminal counts
```

Acceptance:

```text
We know which observation dimensions are x,y.
```

Do not build geodesic labels until this is verified.

### Step 3: implement dataset-position graph first

Add:

```text
scripts/build_bmm_pointmaze_graph.py
utils/pointmaze_graph.py
```

Recommended graph construction:

```text
1. Extract xy from all train+val states.
2. Bin xy into a grid with bin size around median_step_xy or 2 * median_step_xy.
3. Add undirected edges for consecutive non-terminal transitions between bins.
4. Collapse duplicate edges and store minimum/median weight.
5. Optionally add local bin-neighbor edges only when supported by observed transitions.
6. Compute sparse shortest paths from sampled source bins on demand.
```

Do not add broad kNN geometric edges initially; they can jump through walls.

Outputs:

```text
exp/bmm_pointmaze_graph.npz
  bin_centers
  state_to_bin
  adjacency
  median_step_xy
  metadata
```

### Step 4: graph-label diagnostic script

Add:

```text
scripts/eval_bmm_graph_reachability.py
```

Sampling:

For each budget `H`:

```text
positives: d_graph in [0.5H, H]
negatives: d_graph in (H, 2H]
unknown/disconnected: skipped initially
```

Report:

```text
H
pos_count / neg_count
-distance oracle AUC
Euclidean AUC
kNN AUC on graph labels
state-only critic AUC/gap
monotonicity by budget
```

Expected result:

```text
-distance oracle AUC = 1.0
Euclidean may be imperfect
kNN should be meaningfully above chance
state-only critic should pass H=256/512 if graph labels are a clean function of xy
```

### Step 5: train state-only BMM critic on graph labels

Use pure BCE first:

```bash
python main.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --agent=agents/bmm_trl.py \
  --agent.value_only=True \
  --agent.diagnostic_critic_mode=state \
  --agent.lambda_sup=1.0 \
  --agent.lambda_mono=0.0 \
  --agent.lambda_rank=0.0 \
  --agent.num_rank_pairs=0 \
  --agent.lambda_trans=0.0 \
  --agent.lambda_pos=0.0 \
  --agent.lambda_budget_neg=0.0 \
  --agent.lambda_hard_neg=0.0 \
  --agent.lambda_rand_hinge=0.0 \
  --agent.max_budget=512 \
  --agent.budgets="(1,2,4,8,16,32,64,128,256,512)" \
  --agent.dataset.reachability_label_type=graph
```

This requires adding the dataset config field and graph-label sampler first.

Passing threshold for the next milestone:

```text
H=64:  AUC >= 0.90, gap >= 0.20
H=128: AUC >= 0.88, gap >= 0.18
H=256: AUC >= 0.85, gap >= 0.15
H=512: AUC >= 0.80, gap >= 0.10
```

These are diagnostic thresholds, not final paper claims.

### Step 6: implement layout/grid BFS if graph target is ambiguous

If the dataset graph target is still not learnable or has obvious support artifacts, implement layout BFS.

Possible extraction paths:

1. inspect the OGBench environment object after `make_env_and_datasets()`;
2. search env attributes for `maze`, `maze_map`, `walls`, `map`, `layout`;
3. if not accessible, hardcode the PointMaze medium layout with a clear source note in the script.

Add a tiny unit test:

```text
scripts/test_pointmaze_grid_bfs.py
```

Test on a synthetic U-maze with a wall to ensure BFS does not use Euclidean shortcuts.

### Step 7: action-conditioned Bellman diagnostic

After state-only graph/geodesic works, train action-conditioned labels by next state:

```text
label_Q(s_t, a_t, g, H) = 1[d_graph(s_{t+1}, g) <= H - 1]
```

This turns the critic into a real finite-horizon Q-like reachability object.

Run:

```text
diagnostic_critic_mode=action
value_only=True
label_type=graph_q_next
```

Acceptance:

```text
Action-conditioned heldout AUC is close to state-only, especially for H>=128.
```

If action-conditioned fails while state-only passes, debug the Q label and action/state transition semantics.

### Step 8: return to max-min BMM

Only after state-only and action-conditioned graph/geodesic supervised labels pass:

```text
lambda_sup=1.0
lambda_trans=0.025 or 0.05
lambda_mono=0.0 initially
lambda_rank=0.0 initially
```

Use graph/geodesic midpoint witnesses where possible:

```text
choose w with d(s,w) <= h and d(w,g) <= H-h
```

For dataset trajectories, same-trajectory midpoints remain useful witness candidates, but the target should be evaluated against graph/geodesic reachability, not logged offset.

## Decision tree

### If graph/geodesic labels pass

Proceed with BMM max-min on this clean diagnostic. Logged offset becomes an auxiliary behavior-time metric only.

### If dataset graph passes but grid BFS fails

The dataset graph is easier because it follows behavior support. Use it for offline-support BMM. Be explicit that the target is dataset-support reachability, not full environment reachability.

### If grid BFS passes but dataset graph fails

The dataset graph construction is too sparse or too artifact-heavy. Use layout BFS for diagnostics and only use dataset transitions for action-conditioned `s_next` labels.

### If neither graph nor grid labels are learnable

The target may be clean, but the plain MLP lacks topology bias. Try:

- position-only inputs;
- explicit pair features: `g - s`, `||g - s||`;
- Fourier features for xy;
- a two-tower/bilinear critic;
- graph-nearest-neighbor planning instead of pure MLP classification.

### If state-only passes but action-conditioned fails

Use Bellman next-state labels:

```text
Q_H(s,a,g) = V_{H-1}(s_next,g)
```

Do not use direct logged-offset labels for `Q`.

## What to stop doing

Stop spending time on these until graph/geodesic state-only passes:

- policy evaluation;
- actor tuning;
- ranking loss;
- monotonicity penalty;
- high-budget logged-offset gate;
- hard same-trajectory negatives from `offset > H`.

## Why this is real progress

The original prototype spec intentionally started with same-trajectory labels because they were easy to sample. That was a reasonable first step. But the diagnostics have now shown the approximation breaks at high budgets in PointMaze.

This is not project failure. It is a target-definition discovery.

The revised research claim should become:

```text
BMM-TRL gives logarithmic-depth error behavior for clean budgeted reachability targets.
For offline GCRL, the hard part is choosing a reachability target that is not a noisy behavior-time artifact.
```

A clean PointMaze geodesic or dataset-support graph target is the right next test of the algorithmic idea.

## Immediate run order

1. Verify xy observation dimensions.
2. Build conservative dataset-position graph.
3. Evaluate graph distance baselines and kNN ceiling.
4. Train state-only pure BCE on graph labels.
5. If needed, implement layout/grid BFS labels.
6. Train state-only pure BCE on grid/BFS labels.
7. Train action-conditioned Bellman next-state labels.
8. Re-enable BMM max-min consistency.
9. Only then run policy-facing comparisons.
