# BMM-TRL next steps after paper experiment results

Date: 2026-06-12

This plan follows `BMM_TRL_PAPER_EXPERIMENT_RESULTS_20260612_013139.md`.

## Executive decision

The new paper-focused experiments make sense and are useful. They clarify the correct scope of the paper.

The project is now strong enough for a **focused arXiv-style preliminary research report** if the claims are scoped carefully:

```text
BMM-TRL is a non-expansive transitive reachability objective.
It has clean deterministic/support-reachability theory.
It improves heldout long-budget reachability in geodesic budget-holdout diagnostics.
It does not yet establish robust policy improvement or empirical dominance over product transitive targets.
```

It is **not** yet strong enough for a broad empirical claim such as:

```text
BMM-TRL outperforms TRL/product transitive learning in offline RL.
BMM-TRL solves long-horizon policy extraction.
BMM-TRL generalizes strongly to support-graph offline targets.
```

The paper can be complete, but it must be honest: the strongest story is theory + target diagnosis + budget-holdout reachability bootstrapping, with policy extraction and support-graph scaling as open problems.

## What the new experiments tell us

### 1. Tabular error scaling is strong

The tabular error-scaling result supports the core algebraic motivation:

```text
H=1024:
BMM balanced sup error:        0.2200
additive/product sup error:   20.4800
```

This is a good paper figure. It directly visualizes the difference between max-error and additive-error recurrences.

### 2. Support-graph replication is mixed

The replicated support-graph H120 result is not a strong BMM win:

```text
B-A over seeds 0,1,2:
+0.0043 AUC
-0.0194 gap
+0.0224 BCE
+0.0099 ECE
```

Interpretation:

```text
support-graph labels are feasible and learnable,
but BMM does not clearly improve all metrics on this graph H120 setup.
```

This should be reported as a mixed diagnostic, not a main positive result.

### 3. Product is a strong control

The product-vs-max-min H8 seed-0 ablation shows:

```text
B-A: +0.0175 AUC, +0.0703 gap
P-A: +0.0167 AUC, +0.0649 gap
B-P: +0.0008 AUC, +0.0054 gap
```

So product captures most of the H8 gain in this clean setting. Max-min is slightly better in seed 0, but the margin is small.

This means the paper should emphasize:

```text
BMM has a cleaner non-expansive error algebra.
BMM is competitive with product and slightly better in the tested H8 seed.
Empirical dominance over product is not yet established.
```

Do not claim:

```text
BMM empirically dominates product transitive learning.
```

## Is this enough for a complete paper?

### Enough for an arXiv preliminary report?

Yes, if framed as:

```text
A theoretical and diagnostic study of max-min budgeted reachability for offline GCRL.
```

### Enough for a strong empirical conference paper?

Probably not yet.

The missing piece is a decisive empirical advantage over product/support-graph baselines or a working policy improvement result. Current policy results are negative/mixed, and support-graph BMM gains are mixed.

## Minimal additional experiments recommended

Only run a small number of targeted experiments. Do not resume broad sweeps.

## Experiment A: replicate product-vs-max-min on H8 for seeds 1 and 2

### Why

The product control is too important to leave at one seed. Since product nearly ties BMM in seed 0, the paper needs to know whether this is consistent.

### Setup

Use the existing grid-cell H8 holdout:

```text
geodesic_budget_unit = grid_cells
supervised_budgets = (2,4)
heldout parent = 8
variants = A,B,P,F
seeds = 1,2
```

You do not need to rerun A/B/F if those artifacts already exist. Run only `P` for seeds 1 and 2 if the summarizer can combine with existing A/B/F.

### Metrics

```text
B-A
P-A
B-P
F-A
```

for:

```text
AUC
gap
BCE
ECE
Q-V abs diff
```

### Decision

If B-P is positive or near zero:

```text
Report max-min as theoretically cleaner and empirically competitive/slightly better.
```

If P beats B:

```text
Report product as a strong empirical control and avoid claiming max-min superiority.
The paper can still stand on theory/diagnostics, but the empirical method claim becomes weaker.
```

## Experiment B: optional env-step product control, seed 0 only

### Why

The main budget-holdout result includes env-step H160. A product control on H160 would check whether product also captures most of the env-step gain.

### Setup

```text
geodesic_budget_unit = env_steps
supervised_budgets = (40,80)
heldout parent = 160
variant = P
seed = 0
```

### Priority

Medium. Run only if Experiment A is quick or if reviewers/readers will ask why product was tested only on grid-cell H8.

### Decision

If product is close to BMM again, the paper's empirical claim should become:

```text
transitive reachability bootstrapping helps; max-min gives the cleanest error algebra.
```

not:

```text
max-min empirically dominates product.
```

## Experiment C: do not run more graph H120 replications

The graph H120 result is already three seeds and mixed. More seeds are unlikely to change the story enough.

Instead, present it honestly:

```text
support-graph labels are learnable;
BMM gives a small AUC gain but worsens gap/BCE/ECE;
support-graph target construction remains future work.
```

If you want a stronger support-graph result later, design a new graph holdout that is not saturated by supervised extrapolation. But this is not necessary for the current arXiv report.

## Experiment D: final table/figure generation

After Experiment A, regenerate paper tables:

```text
scripts/summarize_bmm_paper_tables.py
```

Expected final artifacts:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
exp/bmm_tabular_error_scaling.png
```

Add the final numbers to:

```text
BMM_TRL_ARXIV_REPORT.md
```

## What not to run

Do not run:

```text
more policy extraction
more controller sweeps
more large-maze policy runs
more sparse-Q tables
more witness/loss sweeps
more support-graph H120 seeds
```

These are not needed for the focused paper.

## Revised paper claims

### Safe claims

```text
1. Product TRL is additive in distance error.
2. BMM max-min reachability gives a non-expansive deterministic/support operator.
3. Logged-offset labels are poor high-budget reachability targets.
4. Clean geodesic and support-graph reachability labels are learnable.
5. Transitive budget-holdout improves long-budget reachability on geodesic settings.
6. Product is a strong control; max-min is theoretically cleaner and empirically competitive in current diagnostics.
7. Policy extraction remains open.
```

### Unsafe claims

```text
1. BMM beats product transitive learning empirically.
2. BMM solves long-horizon offline RL.
3. BMM improves policy performance.
4. Support-graph BMM is already robust.
```

## Suggested final paper structure

```text
1. Introduction
2. Related work
3. Budgeted max-min reachability
4. Deterministic/support theory
5. Stochastic interpretation and limitations
6. Experiments
   6.1 Tabular error scaling
   6.2 Target diagnosis: logged offset vs geodesic/support graph
   6.3 Supervised reachability learning
   6.4 Budget-holdout reachability bootstrapping
   6.5 Product-vs-max-min control
   6.6 Policy extraction limitations
7. Discussion and future work
8. Conclusion
```

## Final go/no-go for arXiv report

### Submit as arXiv preliminary report if

```text
Experiment A does not show product clearly dominating max-min;
tabular figure remains clean;
the report clearly states limitations.
```

### Hold back or reframe further if

```text
product dominates max-min over multiple seeds;
support-graph results are presented as central rather than mixed;
policy claims remain too strong.
```

## Immediate task list

1. Run product `P` for grid-cell H8 seeds 1 and 2.
2. Regenerate paper tables.
3. Optionally run env-step H160 product seed 0.
4. Update `BMM_TRL_ARXIV_REPORT.md` with final product-control numbers.
5. Add `exp/bmm_tabular_error_scaling.png` to the report or repository artifacts.
6. Freeze experiments and focus on writing.

## Bottom line

The additional results are enough to continue toward a focused arXiv paper, but they force a more careful claim:

```text
BMM's strongest contribution is theoretical/non-expansive reachability composition and diagnostic budget-holdout bootstrapping.
Empirically, product is a strong control and policy extraction is unresolved.
```

Run only the product-control replication, then stop experimenting and write.
