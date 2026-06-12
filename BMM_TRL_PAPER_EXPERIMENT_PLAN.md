# BMM-TRL paper experiment plan

Date: 2026-06-11

This plan is for turning the BMM-TRL project into a complete, focused research paper. The paper should **not** be framed as an end-to-end policy-performance paper. The strongest defensible scope is:

```text
Budgeted max-min reachability is a non-expansive transitive objective for offline GCRL diagnostics.
It improves heldout long-budget reachability bootstrapping on clean geodesic and support-graph targets.
Policy extraction remains open.
```

## Proposed paper thesis

The paper's central claim should be:

> Product transitive value learning reduces recursion depth but remains additive in distance/log-value error. Budgeted max-min reachability changes the algebra: for deterministic or support-reachability targets, the max-min operator is non-expansive, yielding logarithmic-depth error accumulation under balanced decomposition. Empirically, this helps bootstrap heldout long-budget reachability labels from shorter-budget labels.

This is smaller than the original policy goal, but it is coherent and defensible.

## Claims to support

### Claim 1: BMM has the desired error algebra

Show theoretically and empirically that:

```text
BMM max-min reachability: E(H) <= epsilon + max(E(h), E(H-h))
Distance/product composition: E(H) <= epsilon + E(h) + E(H-h)
```

The theory is already in `BMM_TRL_ARXIV_REPORT.md`; the paper needs one clean empirical illustration.

### Claim 2: logged same-trajectory offset is a bad high-budget reachability target

Show that logged offset is behavior time, not true high-budget reachability:

```text
same-trajectory offset labels fail at H=256/512
kNN is near chance at high budgets
geodesic diameter says H=256/512 are one-class on PointMaze-medium
```

This justifies replacing logged offset with geodesic/support-graph reachability.

### Claim 3: BMM improves heldout parent-budget reachability

Use the budget-holdout setting as the main empirical result:

```text
train short budgets
hold out parent budget
use max-min Q/V transitive consistency
measure heldout parent classification
```

Existing grid-cell and env-step results are already strong. The paper needs a small number of additional targeted runs to complete the story.

### Claim 4: BMM is not yet a policy algorithm

Include policy extraction limitations honestly:

```text
flat Q action ranking: negative
Q/V action-subgoal extraction: mixed
hierarchical controller: not robust versus geometric baseline
```

This should be a short limitations subsection, not the main result.

---

## Experiment 1: tabular / graph error-scaling figure

### Purpose

This is the most important missing figure for the theory story. It visually demonstrates why BMM was proposed.

### Setup

Use deterministic graphs:

```text
1. chain graph, N in {32,64,128,256,512,1024}
2. 2D grid graph, side length chosen to produce similar diameters
```

Compute exact shortest-path distance by BFS.

Construct:

```text
R_H^*(s,g) = 1[d(s,g) <= H]
D_H^*(s,g) = d(s,g)
```

Inject controlled residual noise at each composition level.

Compare:

```text
A. additive distance composition / product-TRL-equivalent
B. BMM max-min reachability composition
C. one-step TD-style propagation, optional
```

### Metrics

```text
sup-norm error vs horizon
mean absolute error vs horizon
classification error for R_H
calibration error, optional
```

### Expected result

```text
BMM error grows approximately with log H.
Distance/product composition grows closer to H under worst-case residual construction.
```

### Implementation

Extend or clean up:

```text
scripts/test_bmm_tabular.py
```

Add output:

```text
exp/bmm_tabular_error_scaling.json
exp/bmm_tabular_error_scaling.md
exp/bmm_tabular_error_scaling.png
```

### Paper use

Main Figure 1 or Figure 2.

---

## Experiment 2: target-quality / identifiability table

### Purpose

Show why logged offset failed and why support/geodesic targets are the right diagnostic.

### Targets

Compare:

```text
logged same-trajectory offset
layout/grid geodesic distance
dataset-support graph distance
```

### Metrics

For each budget:

```text
class balance
oracle AUC
Euclidean baseline AUC
kNN AUC
supervised critic AUC, if available
```

Use existing results where possible.

### Table rows

Suggested table:

| target | H | pos/neg coverage | kNN AUC | Euclidean AUC | critic AUC | interpretation |
|---|---:|---:|---:|---:|---:|---|
| logged offset | 256 | balanced | ~0.54 | near chance | near chance | non-identifiable |
| logged offset | 512 | balanced | ~0.52 | near chance | near chance | non-identifiable |
| grid geodesic | 32/64/96/128 | balanced | high | high-ish | >0.93 | clean |
| support graph | 32/64/96/128 | balanced | high | high-ish | high | clean support target |

### Existing evidence

Use these existing files:

```text
BMM_TRL_STATUS_20260610_181801.md
BMM_TRL_GEODESIC_VALUE_RESULTS_20260610_184508.md
BMM_TRL_FAST_DECISION_RESULTS_20260611_124920.md
```

### Additional run needed

Probably none unless the current logged-offset table is too scattered. If needed, run one consolidated script that reproduces only the label-identifiability metrics and writes one compact summary.

### Paper use

Section: `Target design matters: logged offset is behavior time`.

---

## Experiment 3: budget-holdout main table

### Purpose

This is the main empirical result.

### Existing settings

#### Grid-cell holdout

```text
geodesic_budget_unit = grid_cells
supervised budgets = (2,4)
heldout parent = 8
```

Existing 3-seed result:

```text
B-A: +0.0150 AUC, +0.0751 gap, -0.3609 BCE, -0.0575 ECE
D-C: +0.0116 AUC, +0.0507 gap
F-A: near zero
```

#### Env-step holdout

```text
geodesic_budget_unit = env_steps
supervised budgets = (40,80)
heldout parent = 160
```

Existing 3-seed result:

```text
B-A: +0.0243 AUC, +0.0647 gap, -0.5815 BCE, -0.0380 ECE
D-C: +0.0041 AUC, +0.0498 gap
F-A: much smaller than B-A
```

### Add missing support-graph replication

The graph result is currently one-seed mild positive. For a complete paper, run graph H120 budget-holdout with seeds 1 and 2, A/B/F only.

#### Setup

```text
reachability_label_type = graph
graph_path = exp/bmm_pointmaze_graph.npz
budgets = 40,80,120
supervised_budgets = 40,80
heldout parent = 120
variants = A,B,F
steps = 500
seeds = 1,2
```

Use the existing seed 0 graph result as seed 0.

### Success criterion

The paper is strengthened if:

```text
B-A is positive on H120 AUC or BCE on average;
F-A remains near A;
witness diagnostics remain healthy.
```

If graph results are mixed, still include them honestly as a limitation:

```text
support-graph transfer is promising but smaller than oracle geodesic transfer.
```

### Paper use

Main Table 1.

---

## Experiment 4: max-min versus product transitive ablation

### Purpose

The paper is about changing the algebra from product/additive to max-min. A direct ablation will make the story much clearer.

### Setting

Use the fastest main setting:

```text
grid-cell holdout: supervised (2,4), heldout 8
or env-step holdout: supervised (40,80), heldout 160
```

Start with grid-cell H8 because it is cheap and clean.

### Variants

```text
A. supervised no parent labels, no transitive
B. BMM max-min Q/V transitive
P. product-style Q/V transitive
F. V-next distillation control
```

### Product-style target

For probabilities:

```text
y_product = Q_h(s,a,w) * V_{H-h}(w,g)
```

Train with the same lower-bound BCE or BCE-to-target protocol used by BMM, but clearly identify it as a product control.

### Metrics

```text
heldout parent AUC/gap/BCE/ECE
Q-V abs diff
calibration
stability / collapse checks
```

### Run budget

```text
seed 0 first
only add seeds 1,2 if the product comparison is informative
```

### Expected outcome

BMM should be at least as stable as product and ideally better under parent-budget holdout. If product performs similarly, the theoretical motivation remains but the empirical advantage is weaker; report honestly.

### Paper use

Ablation Table 2.

---

## Experiment 5: support-graph target with train-only graph, optional but valuable

### Purpose

Address the main generality concern: grid labels are oracle-like and train+val graph labels may leak validation support.

### Setup

Build graph from train dataset only:

```text
graph_scope = train_only
val states mapped to nearest train node/bin
report unmapped fraction and nearest-distance distribution
```

Run a small label-quality diagnostic:

```text
support graph label coverage
supervised V/Q critic AUC
budget-holdout A/B/F seed 0
```

### Success criterion

```text
train-only graph labels are learnable;
B improves heldout parent or at least does not collapse;
coverage diagnostics are acceptable.
```

### Paper use

If positive, include as a paragraph or appendix table. If not, include as future work.

### Priority

Medium. This is useful but not as important as Experiments 1, 3, and 4.

---

## Experiment 6: policy extraction limitations table

### Purpose

Make the paper honest and complete without spending more compute.

### Use existing results only

Summarize:

```text
flat action ranking: BMM does not beat A/F
joint action-subgoal: mixed even with good candidate coverage
value subgoal smoke: positive but not robust under stronger BC
hierarchical quick try: BMM ties geometric midpoint
```

### Table

| interface | result | conclusion |
|---|---|---|
| flat Q action ranking | B does not beat A/F AUC | not supported |
| Q/V joint action-subgoal | improves action-valid but not full objective | mixed |
| value-only subgoal + NN | BMM beats random/geometric in tiny smoke | weak positive |
| value-only subgoal + BC | BMM ties or loses to geometric | not robust |

### Paper use

Limitations / Discussion section. This prevents overclaiming.

---

## Minimal experiment set for a complete paper

If time is limited, run only:

```text
1. tabular error-scaling figure;
2. graph H120 seeds 1 and 2 A/B/F;
3. product-vs-max-min seed 0 on grid-cell H8;
4. consolidate target-quality table from existing logs;
5. consolidate policy-limitation table from existing logs.
```

This should be enough for a focused arXiv-style paper.

## Stretch experiments

Only run these if the minimal set is positive and time remains:

```text
1. product-vs-max-min with three seeds;
2. train-only graph support target;
3. larger maze value-only budget-holdout, no policy;
4. stochastic reliability-budget toy example.
```

Do not run large policy benchmarks for this paper.

---

## Proposed paper figures and tables

### Figure 1: error-scaling toy graph

```text
x-axis: horizon H
y-axis: sup-norm error
curves: TD/additive, product/distance, BMM max-min
```

### Figure 2: target ambiguity

```text
logged-offset kNN AUC collapses at high budgets;
geodesic/support graph labels stay identifiable below diameter.
```

### Table 1: budget-holdout main result

Rows:

```text
grid-cell H8
env-step H160
support graph H120
```

Columns:

```text
B-A delta AUC/gap/BCE/ECE
F-A control delta
seeds
```

### Table 2: ablation

```text
supervised only
BMM max-min
product target
V-next distillation
```

### Table 3: policy limitations

Compact negative-policy table.

---

## Writing plan

### Core paper message

Use this wording:

```text
BMM-TRL is not presented as a solved policy algorithm. It is a reachability objective and diagnostic showing that max-min transitive consistency can bootstrap heldout long-horizon reachability budgets with non-expansive error algebra.
```

### Do not claim

```text
SOTA offline RL
policy improvement
complete stochastic solution
true reachability from arbitrary offline data
```

### Claim

```text
non-expansive theory
clean target diagnosis
budget-holdout reachability gains
support-graph feasibility
policy extraction remains open
```

---

## Immediate task list for Codex / next coding session

1. Extend `scripts/test_bmm_tabular.py` to output error-scaling JSON/PNG/MD.
2. Add product-style Q/V transitive mode to `scripts/train_bmm_geodesic_q.py` or a small ablation wrapper.
3. Run product-vs-max-min seed 0 on grid-cell H8 holdout.
4. Run graph H120 A/B/F for seeds 1 and 2.
5. Add `scripts/summarize_bmm_paper_tables.py` to produce the final tables from JSON artifacts.
6. Update `BMM_TRL_ARXIV_REPORT.md` with the new figure/table placeholders and final numbers.

## Go / no-go for the paper

### Good enough for a paper if

```text
tabular figure clearly shows BMM log-depth error scaling;
budget-holdout table remains positive;
product ablation does not dominate max-min;
support-graph result is at least mildly positive;
limitations are clearly stated.
```

### Pause paper submission if

```text
product target dominates max-min;
graph H120 replication becomes negative;
tabular scaling does not show the expected separation.
```

In that case, keep the work as an internal research note rather than arXiv.

## Bottom line

A complete paper is still possible, but it should be scoped as:

```text
BMM as a non-expansive transitive reachability objective,
validated by target diagnosis and heldout-budget reachability bootstrapping.
```

The additional experiments should be small, targeted, and paper-focused. Do not spend more time trying to force policy improvement for this version of the work.
