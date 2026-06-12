# Budgeted Max-Min Transitive Reinforcement Learning for Offline Goal-Conditioned Control

**Status:** arXiv-style research report / preliminary manuscript  
**Repository:** `jingxuxie/trl`  
**Date:** 2026-06-11

## Abstract

Long-horizon offline goal-conditioned reinforcement learning requires value functions that can compose short-horizon experience into long-horizon behavior. Transitive RL (TRL) addresses this problem by replacing one-step temporal-difference propagation with a divide-and-conquer product backup over intermediate goals. In deterministic goal-reaching problems, however, discounted goal-reaching values can be written as \(V^*(s,g)=\gamma^{d^*(s,g)}\), where \(d^*(s,g)\) is a shortest-path distance. TRL's product backup is therefore equivalent to addition in distance space. Consequently, although the computational dependency depth may be logarithmic under balanced decompositions, worst-case numerical distance error can still add across branches.

This report studies **Budgeted Max-Min Transitive RL (BMM-TRL)**, a reachability-based alternative designed to change the error algebra. Instead of learning a discounted temporal-distance value, BMM-TRL learns a finite-horizon reachability predicate

\[
R_H(s,g) = \mathbf{1}\{d(s,g) \le H\},
\]

and composes subproblems using a max-min backup

\[
R_H(s,g) \leftarrow \max_w \min\bigl(R_h(s,w), R_{H-h}(w,g)\bigr).
\]

The central theoretical observation is that the max-min operator is non-expansive in sup norm. Under a balanced split and a per-level regression residual \(\varepsilon\), the ideal error recurrence becomes

\[
E(H) \le \varepsilon + \max(E(h), E(H-h)),
\]

which gives \(O(\varepsilon \log H)\) error accumulation for dyadic horizons. This is in contrast to additive distance composition, where branch errors sum.

We implemented BMM-TRL in an offline goal-conditioned RL codebase and evaluated it on PointMaze diagnostics. The results support the reachability-composition side of the hypothesis: clean grid/geodesic and dataset-support graph reachability labels are learnable, and Q/V max-min transitive consistency improves heldout long-budget classification in budget-holdout experiments. However, the current prototype does not yield robust policy improvement. Flat action ranking, Q/V action-subgoal extraction, and lightweight hierarchical control remain weak. The strongest current conclusion is that BMM-TRL is a promising **budgeted reachability and subgoal-planning diagnostic**, but not yet a complete end-to-end offline RL policy algorithm.

---

## 1. Introduction

Offline goal-conditioned reinforcement learning (offline GCRL) asks an agent to learn a policy \(\pi(a\mid s,g)\) from a fixed dataset of trajectories, without online interaction during training. This setting is attractive because large unlabeled datasets can be reused for many goal-reaching tasks. It is also difficult: the agent must infer long-horizon reachability, stitch together experience across trajectories, and avoid unsupported actions.

One recurring challenge is **long-horizon error compounding**. Standard temporal-difference learning propagates information one step at a time, so a goal that is \(T\) steps away can require \(O(T)\) recursive propagation. Transitive RL (TRL) improves this by using intermediate goals and divide-and-conquer composition. A long segment from \(s\) to \(g\) can be decomposed through a witness \(w\), reducing the recursion depth when witnesses are balanced.

The starting point of this project was the observation that TRL's product backup, while shallow in recursion depth, still corresponds to **additive distance composition**. If \(V(s,g)=\gamma^{d(s,g)}\), then

\[
V(s,g) \approx V(s,w)V(w,g)
\quad\Longleftrightarrow\quad
d(s,g) \approx d(s,w)+d(w,g).
\]

Thus if the two branch distance estimates have errors \(E_1\) and \(E_2\), the parent distance error can scale like \(E_1+E_2\). For balanced splits, this gives a recurrence of the form

\[
E(T) \le 2E(T/2)+\varepsilon,
\]

which is still linear in \(T\) under uniform residuals.

BMM-TRL asks whether one can instead learn a value object whose composition has a **max-error recurrence**:

\[
E(T) \le \varepsilon + \max(E(T/2), E(T/2)).
\]

The proposed object is finite-budget reachability. Rather than estimating a precise temporal distance, BMM-TRL predicts whether a goal is reachable within a budget \(H\). The max-min composition of reachability predicates is non-expansive, leading to logarithmic-depth error accumulation under ideal assumptions.

This report presents the mathematical motivation, deterministic and stochastic interpretations, implementation, and empirical findings from a prototype. The results are mixed but informative: the reachability and bootstrapping diagnostics are positive, while policy extraction remains unresolved.

### Contributions

This project makes the following contributions:

1. **A reachability-based alternative to product transitive value composition.** We formulate BMM-TRL, which learns budgeted reachability and composes subproblems with a max-min backup.
2. **A simple error-propagation result.** For deterministic or support-graph reachability, the max-min operator is non-expansive in sup norm, giving \(O(\log H)\) accumulated score error under balanced recursion and bounded residuals.
3. **A clarification of target design for offline GCRL.** Same-trajectory logged offset is shown empirically to be a behavior-time label rather than a reliable high-budget reachability label. Support-graph or geodesic reachability targets are more appropriate diagnostics.
4. **A prototype and diagnostic suite.** We implemented state-value, action-value, Q/V transitive, budget-holdout, graph-reachability, action-ranking, and subgoal-selection diagnostics.
5. **Empirical findings.** BMM improves heldout long-budget reachability in budget-holdout experiments, including grid-cell, env-step, and support-graph variants. However, these critic-level gains do not yet translate into robust flat or hierarchical policy gains.

---

## 2. Background

### 2.1 Goal-conditioned RL

Goal-conditioned RL learns policies and value functions conditioned on a goal representation. Universal Value Function Approximators introduced the idea of approximating value functions that generalize over goals and tasks. Hindsight Experience Replay made sparse goal-reaching RL more practical by relabeling trajectories with goals achieved later in the trajectory.

Offline GCRL extends this setting to static datasets. Recent benchmarks such as OGBench emphasize stitching, long-horizon reasoning, and out-of-support generalization. Offline GCRL is particularly sensitive to target definition: not every state-goal pair absent from a trajectory should be treated as unreachable, because another supported trajectory may provide a shortcut.

### 2.2 Transitive RL

Transitive RL uses a divide-and-conquer value target. In deterministic goal-reaching, if \(d^*(s,g)\) is shortest-path distance and \(V^*(s,g)=\gamma^{d^*(s,g)}\), then the triangle inequality suggests the product composition

\[
V(s,g) \leftarrow \max_w V(s,w)V(w,g).
\]

This can reduce recursion depth, but in log-value space it becomes

\[
d(s,g) \leftarrow \min_w d(s,w)+d(w,g),
\]

so approximation errors in distance space can add across branches.

### 2.3 Hierarchical GCRL

Hierarchical methods such as HIQL use subgoals or latent states as high-level actions, with a low-level policy responsible for reaching nearby subgoals. This is closely related to BMM-TRL's final empirical direction: the max-min structure is naturally suited to choosing a witness \(w\), and the remaining challenge is to pair such high-level choices with a capable low-level controller.

### 2.4 Offline support constraints

Offline RL methods such as IQL highlight the risk of evaluating unsupported actions. For BMM-TRL, the analogous issue is evaluating unsupported reachability. From offline data alone, true MDP reachability is generally unidentifiable without assumptions. The most defensible offline target is therefore **support reachability**: reachability within the graph induced by observed transitions or conservative support-preserving stitch edges.

---

## 3. Deterministic BMM theory

Let \(\mathcal{G}\) be a deterministic graph or deterministic MDP, and let \(d^*(s,g)\) denote shortest-path distance. Define the state reachability predicate

\[
V_H^*(s,g)=\mathbf{1}\{d^*(s,g)\le H\}.
\]

For a deterministic transition \(s'=f(s,a)\), define the action-conditioned reachability predicate

\[
Q_H^*(s,a,g)=\mathbf{1}\{d^*(f(s,a),g)\le H-1\}.
\]

### Proposition 1: Exact max-min identity

For any split \(h\in\{0,\ldots,H\}\), allowing endpoint witnesses,

\[
V_H^*(s,g)=\max_w \min\left(V_h^*(s,w),V_{H-h}^*(w,g)\right).
\]

The action-conditioned analogue is

\[
Q_H^*(s,a,g)=\max_w \min\left(Q_h^*(s,a,w),V_{H-h}^*(w,g)\right).
\]

#### Proof

If \(V_H^*(s,g)=1\), then there exists a path from \(s\) to \(g\) of length \(L\le H\). Choose a witness \(w\) on this path after \(\min(h,L)\) steps. Then \(d^*(s,w)\le h\), and the remaining path length from \(w\) to \(g\) is at most \(H-h\). Hence both branch predicates equal one, so the max-min expression is one.

Conversely, if the max-min expression equals one, then there exists \(w\) such that

\[
d^*(s,w)\le h,\qquad d^*(w,g)\le H-h.
\]

Concatenating these paths gives a path from \(s\) to \(g\) of length at most \(H\), so \(V_H^*(s,g)=1\). The Q identity follows by applying the same argument after the first action transition \(s'=f(s,a)\).

### Proposition 2: Non-expansive error propagation

Suppose estimates \(\widehat V_h\) and \(\widehat V_{H-h}\) satisfy

\[
\|\widehat V_h - V_h^*\|_\infty \le E_h,
\qquad
\|\widehat V_{H-h} - V_{H-h}^*\|_\infty \le E_{H-h}.
\]

Define

\[
\widehat T_H(s,g)=\max_w \min\left(\widehat V_h(s,w),\widehat V_{H-h}(w,g)\right),
\]

and define \(T_H^*\) analogously using the true branch predicates. Then

\[
\|\widehat T_H-T_H^*\|_\infty\le \max(E_h,E_{H-h}).
\]

#### Proof

For scalars, the minimum operation is 1-Lipschitz in max norm:

\[
\left|\min(a,b)-\min(a',b')\right|\le \max(|a-a'|,|b-b'|).
\]

The maximum over witnesses is also non-expansive:

\[
\left|\max_w z_w-\max_w z'_w\right|\le \sup_w |z_w-z'_w|.
\]

Combining the two inequalities yields the result.

If fitting the parent predictor introduces a projection or regression residual \(\varepsilon_H\), then the learned error satisfies

\[
E_H\le \varepsilon_H + \max(E_h,E_{H-h}).
\]

For dyadic horizons \(H=2^k\), balanced splits, and uniform residual \(\varepsilon\), this gives

\[
E_H\le k\varepsilon = O(\varepsilon\log H).
\]

This is the central theoretical motivation for BMM-TRL.

### Contrast with distance composition

For distance composition,

\[
d_H(s,g)=\min_w d_h(s,w)+d_{H-h}(w,g),
\]

branch errors add:

\[
E_H\le \varepsilon_H+E_h+E_{H-h}.
\]

For equal splits this becomes \(E_H\le \varepsilon_H+2E_{H/2}\), which is linear in horizon under uniform residuals.

---

## 4. Stochastic environments

The deterministic result does not directly imply an exact finite-horizon success-probability Bellman equation for stochastic MDPs. Let

\[
P_H^*(s,g)=\sup_\pi \Pr_\pi(\tau_g\le H\mid s_0=s).
\]

If a policy reaches an intermediate state \(w\) within \(h\) steps with probability \(p\), and then reaches \(g\) from \(w\) within \(H-h\) steps with probability \(q\), the two-stage success probability is generally proportional to \(pq\), not \(\min(p,q)\). Thus max-min is not an exact stochastic success-probability Bellman operator.

There are two principled stochastic interpretations.

### 4.1 Support reachability

Define a support graph with an edge \(s\to s'\) if there exists a supported action \(a\) such that \(P(s'\mid s,a)>0\). Then define

\[
S_H(s,g)=\mathbf{1}\{\text{there exists a supported path from }s\text{ to }g\text{ of length}\le H\}.
\]

This is deterministic graph reachability over the support graph, so the max-min identity and non-expansive error result apply exactly. This is the most relevant interpretation for offline RL, where the dataset support is observable but full MDP reachability is not.

### 4.2 Reliability-threshold reachability

A more conservative stochastic extension uses reliability thresholds. Let \(\rho=\exp(-b)\), and define

\[
Z_{H,b}(s,g)=\mathbf{1}\{P_H^*(s,g)\ge \exp(-b)\}.
\]

If

\[
P_h(s,w)\ge \exp(-b_1),\qquad P_{H-h}(w,g)\ge \exp(-b_2),
\]

then a two-stage policy gives the sufficient condition

\[
P_H(s,g)\ge \exp(-(b_1+b_2)).
\]

Thus one can define a conservative lower-bound recurrence

\[
Z_{H,b}(s,g)\ge \max_w \min\left(Z_{h,b/2}(s,w),Z_{H-h,b/2}(w,g)\right),
\]

for balanced reliability budgets. This extension preserves the max-min structure but introduces an additional reliability-budget dimension. It was not implemented in the current prototype.

---

## 5. Algorithm

The prototype keeps a TRL-style action-conditioned critic:

\[
R_\theta(s,a,g,H)=\sigma(f_\theta(s,a,g,H)).
\]

The budget \(H\) is appended to the goal input using a normalized logarithmic feature. State-only value diagnostics use the same architecture with actions ignored.

### 5.1 Supervised labels

The final clean diagnostic labels are:

\[
V_H(s,g)=\mathbf{1}\{d(s,g)\le H\},
\]

and

\[
Q_H(s_t,a_t,g)=\mathbf{1}\{d(s_{t+1},g)\le H-1\}.
\]

The distance \(d\) was instantiated as either:

1. layout/grid geodesic distance, used as an oracle diagnostic;
2. dataset-position graph distance, used as an offline-support diagnostic.

### 5.2 Transitive Q/V target

The policy-relevant transitive target is

\[
y_{QV}=\max_w \min\left(Q_h(s,a,w),V_{H-h}(w,g)\right).
\]

A frozen state-value teacher was used for the second branch in the Q/V diagnostics. Because sampled witnesses provide a lower bound rather than an exact equality target, later experiments used lower-bound-style losses such as `bce_lower_bound`.

### 5.3 Budget-holdout protocol

The most important experimental protocol was budget holdout:

\[
\text{train short budgets }H_1,H_2;\qquad \text{hold out parent budget }H_3;\qquad \text{train BMM transitive on }H_3.
\]

This directly tests whether shorter-budget reachability knowledge can bootstrap a longer-budget parent.

---

## 6. Experiments

All experiments were performed on PointMaze-medium diagnostics unless otherwise stated. Detailed logs are available in the repository result files.

### 6.1 Tabular error scaling

The tabular sanity check now emits an error-scaling artifact for the core algebraic claim. With per-level residual \(\varepsilon=0.02\), the balanced max-min recurrence grows from 0.02 at \(H=1\) to 0.22 at \(H=1024\), while the additive/product-style recurrence grows to 20.48:

| H | BMM balanced sup error | Additive/product sup error |
|---:|---:|---:|
| 1 | 0.0200 | 0.0200 |
| 1024 | 0.2200 | 20.4800 |

The same script verifies exact max-min backups on a directed chain and undirected grid. It writes `exp/bmm_tabular_error_scaling.json`, `exp/bmm_tabular_error_scaling.md`, and `exp/bmm_tabular_error_scaling.png`; a tracked copy of the figure is available at `assets/bmm_tabular_error_scaling.png`.

### 6.2 Logged-offset labels fail

The first target used same-trajectory offset:

\[
\mathbf{1}\{\text{logged offset}\le H\}.
\]

This failed at high budgets. Diagnostics showed that the same JAX critic path could learn deterministic chains and overfit fixed PointMaze batches, but heldout PointMaze logged-offset labels were not identifiable at high budgets. kNN AUC was about 0.54 at \(H=256\) and 0.52 at \(H=512\). We concluded that logged offset is behavior time, not reliable reachability.

### 6.3 Geodesic reachability labels work

PointMaze medium has 26 free cells and calibrated grid diameter about 217.75 steps. Therefore \(H=256\) and \(H=512\) are one-class under true geodesic reachability.

The state-only geodesic value critic learned clean budgeted reachability:

| H | AUC | Gap | Ensemble-min AUC | Ensemble-min Gap |
|---:|---:|---:|---:|---:|
| 32 | 0.9344 | 0.4309 | 0.9328 | 0.4266 |
| 64 | 0.9749 | 0.6086 | 0.9743 | 0.6064 |
| 96 | 0.9588 | 0.5916 | 0.9591 | 0.5903 |
| 128 | 0.9751 | 0.5476 | 0.9755 | 0.5402 |

The action-conditioned geodesic Q critic also passed cleanly, with AUCs from 0.9458 to 0.9784 on budgets \(32,64,96,128\), and with zero monotonicity violation.

### 6.4 Q/V transitive is stable but not sufficient in abundant-label settings

Q/V transitive consistency reduced Q-V-next discrepancy but did not improve abundant-label classification overall. For budgets \((40,80,160)\):

| Mode | H40 AUC/gap | H80 AUC/gap | H160 AUC/gap | Q-V abs diff |
|---|---:|---:|---:|---:|
| supervised only | 0.9563 / 0.5359 | 0.9786 / 0.5138 | 0.9875 / 0.6421 | 0.2036 |
| Q/V transitive | 0.9564 / 0.5058 | 0.9765 / 0.4677 | 0.9857 / 0.6728 | 0.1327 |

This suggested that the right stress test was not abundant-label fitting, but parent-budget holdout.

### 6.5 Budget-holdout gives the strongest positive result

Grid-cell holdout trained budgets \((2,4)\) and held out parent \(8\). Across three seeds:

| Comparison | Delta H8 AUC | Delta H8 Gap | Delta H8 BCE | Delta H8 ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| B-A | +0.0150 | +0.0751 | -0.3609 | -0.0575 | no-parent BMM effect |
| D-C | +0.0116 | +0.0507 | -0.1503 | -0.0758 | few-parent BMM effect |
| F-A | +0.0002 | +0.0017 | -0.0063 | -0.0010 | V-next distill control |

Env-step holdout trained budgets \((40,80)\) and held out parent \(160\). Across three seeds:

| Comparison | Delta H160 AUC | Delta H160 Gap | Delta H160 BCE | Delta H160 ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| B-A | +0.0243 | +0.0647 | -0.5815 | -0.0380 | no-parent BMM effect |
| D-C | +0.0041 | +0.0498 | -0.1815 | -0.0547 | few-parent BMM effect |
| F-A | +0.0035 | +0.0030 | -0.0570 | -0.0016 | V-next distill control |

This is the strongest BMM-specific result: shorter-budget Q/V knowledge improves heldout longer-budget classification, and V-next distillation is much smaller.

### 6.6 Max-min versus product transitive ablation

A grid-cell H8 ablation compared BMM max-min Q/V transitive consistency to a product-style Q/V target using the same witness sampler and lower-bound BCE protocol. Across three seeds:

| Comparison | Delta H8 AUC | Delta H8 Gap | Delta H8 BCE | Delta H8 ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| B-A | +0.0150 | +0.0751 | -0.3609 | -0.0575 | max-min transitive over supervised only |
| P-A | +0.0139 | +0.0702 | -0.3344 | -0.0518 | product transitive over supervised only |
| B-P | +0.0011 | +0.0050 | -0.0265 | -0.0057 | max-min over product |
| F-A | +0.0002 | +0.0017 | -0.0063 | -0.0010 | V-next control |

An optional env-step H160 product-control run on seed 0 showed the same pattern:

| Comparison | Delta H160 AUC | Delta H160 Gap | Delta H160 BCE | Delta H160 ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| B-A | +0.0400 | +0.0640 | -0.5976 | -0.0398 | max-min transitive over supervised only |
| P-A | +0.0345 | +0.0517 | -0.5166 | -0.0316 | product transitive over supervised only |
| B-P | +0.0055 | +0.0122 | -0.0810 | -0.0083 | max-min over product |
| F-A | +0.0035 | +0.0067 | -0.0774 | -0.0038 | V-next control |

This does not pause the paper, because product does not dominate max-min in these diagnostics. However, it weakens the empirical distinction: product-style transitive learning captures most of the gain, while BMM is consistently but only slightly better. The safe claim is that max-min is theoretically cleaner and empirically competitive or mildly better here, not that it decisively dominates product.

### 6.7 Dataset-support graph labels are learnable, but transfer is mixed

A dataset-position graph built from observed transitions contained 1,698 nodes and 5,548 edges, with diameter 73 hops / 146 calibrated steps. Graph-label budget holdout was mildly positive:

| Variant | H120 AUC | H120 Gap | Q-V Abs Diff | Q-V Rank Corr |
|---|---:|---:|---:|---:|
| A supervised-only | 0.9783 | 0.3459 | 0.3306 | 0.7802 |
| B Q/V transitive | 0.9849 | 0.3411 | 0.3277 | 0.8084 |
| F V-next distill | 0.9780 | 0.3491 | 0.3265 | 0.7802 |

The three-seed H120 graph replication was more mixed:

| Comparison | Delta H120 AUC | Delta H120 Gap | Delta H120 BCE | Delta H120 ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| B-A | +0.0043 | -0.0194 | +0.0224 | +0.0099 | small AUC gain, worse gap/BCE/ECE |
| F-A | -0.0010 | +0.0035 | -0.0189 | -0.0056 | V-next control is small but improves BCE/ECE |

This weakens the concern that BMM only works with grid-oracle labels, but the support-graph result should be framed as diagnostic feasibility rather than a strong empirical win.

### 6.8 Policy-facing results are weak

Flat action ranking did not improve. On the own-state action-ranking diagnostic:

| Critic | AUC | Pair Acc | Selected Distance | Selected Success |
|---|---:|---:|---:|---:|
| A | 0.6307 | 0.7814 | 164.2045 | 0.6738 |
| B | 0.6170 | 0.7890 | 163.8565 | 0.6895 |
| F | 0.6321 | 0.7827 | 164.1272 | 0.6777 |

BMM did not robustly beat controls.

Joint action-subgoal extraction was also mixed. Candidate coverage could be fixed, but BMM Q/V did not robustly win state-valid, path-stretch, and midpoint metrics.

Value-only subgoal selection was more positive. With a same-cell nearest-neighbor low-level controller, BMM/V improved over random and geometric midpoint in a small smoke:

| Selector | Success | Final Distance | Improve | Mean Step Goal | Subgoal Valid |
|---|---:|---:|---:|---:|---:|
| random | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.1333 |
| geometric_midpoint | 0.0000 | 112.1761 | 59.3874 | 0.5939 | 0.3833 |
| BMM_V | 0.0000 | 105.5775 | 65.9860 | 0.6599 | 0.6633 |
| oracle_midpoint | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.4267 |

However, with a stronger 5k-step goal-conditioned BC controller, BMM/V tied the geometric midpoint baseline rather than beating it:

| Selector | Success | Final Distance | Improve | Subgoal Valid |
|---|---:|---:|---:|---:|
| random | 0.0000 | 134.6113 | 39.5916 | 0.1020 |
| geometric_midpoint | 0.0000 | 106.8972 | 67.3057 | 0.2820 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6100 |
| oracle_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.1960 |

Thus the hierarchical pivot did not clear the final policy-facing continue gate.

---

## 7. Discussion

### Strengths

1. **Theoretical clarity.** BMM has the intended max-error recurrence for deterministic/support reachability.
2. **Target diagnosis.** The project identified logged-offset labels as behavior-time labels, explaining early high-budget failure.
3. **Clean reachability learning.** Grid/geodesic and support-graph labels are learnable.
4. **BMM-specific budget-holdout gains.** Q/V transitive improves heldout long-budget classification across seeds, and V-next distillation does not explain the effect.
5. **Subgoal signal.** Value-level BMM scores can select useful subgoals in diagnostics.

### Limitations

1. **No robust policy improvement.** The prototype did not produce a reliable control method.
2. **Flat action extraction failed.** Better reachability did not imply better one-step action ranking.
3. **Joint Q/V action-subgoal extraction was not robust.** Better candidate coverage did not rescue this path.
4. **Controller bottleneck.** Hierarchical usage depends on a low-level controller; lightweight BC was insufficient.
5. **General offline target construction remains open.** Support-graph targets are promising but depend on representation and conservative graph construction.
6. **Stochastic extension is theoretical.** The current implementation handles deterministic/support reachability, not full stochastic success probability.

---

## 8. Future directions

### 8.1 Train-only support graphs

The most important offline extension is to build reachability targets from train-only support graphs:

\[
R_H^\mathcal{D}(s,g)=\mathbf{1}\{d_\mathcal{D}(\phi(s),\phi(g))\le H\}.
\]

A stronger result would show budget-holdout gains when validation states are mapped conservatively to train graph nodes.

### 8.2 Latent graph construction

For domains without coordinates, one must learn \(\phi(s)\). Promising directions include temporal contrastive learning, inverse-dynamics-aware representations, goal-conditioned encoders, and value-aware embeddings.

### 8.3 Stochastic reliability budgets

The stochastic reliability-threshold extension is mathematically natural but unimplemented. It may provide a conservative bridge between support reachability and success-probability learning.

### 8.4 Hierarchical RL with a strong controller

BMM/V may be useful as a high-level subgoal planner if paired with a strong low-level controller. This would require comparison to HIQL/HIGL-style baselines and careful subgoal proposal.

### 8.5 Finite-sample and projection-error theory

The clean theory assumes sup-norm residual control. Future theory should analyze finite-sample error, classifier calibration, projection error, witness sampling error, and lower-bound losses.

---

## 9. Conclusion

BMM-TRL was motivated by an algebraic weakness of product transitive value backups: they are additive in distance space, so branch errors can add. By learning budgeted reachability and composing with max-min, BMM obtains an ideal deterministic/support recurrence in which branch errors compose by maximum rather than by sum.

The prototype validates the reachability side of this idea. Clean reachability targets are learnable, and Q/V max-min transitive consistency improves heldout long-budget classification in budget-holdout diagnostics. These are the strongest outcomes of the project.

The prototype does not validate BMM-TRL as an end-to-end offline RL policy method. Flat action extraction, joint Q/V action-subgoal extraction, and lightweight hierarchical control did not yield robust policy gains. The current evidence supports preserving BMM-TRL as a reachability/subgoal-planning technical result, and future work should focus either on support-graph reachability or on a formal hierarchical RL system with a strong low-level controller.

---

## References

- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., and Zaremba, W. **Hindsight Experience Replay.** NeurIPS, 2017.
- Bellman, R. **Dynamic Programming.** Princeton University Press, 1957.
- Bertsekas, D. P. **Dynamic Programming and Optimal Control.** Athena Scientific.
- Kostrikov, I., Nair, A., and Levine, S. **Offline Reinforcement Learning with Implicit Q-Learning.** ICLR, 2022.
- Park, S., Ghosh, D., Eysenbach, B., and Levine, S. **HIQL: Offline Goal-Conditioned RL with Latent States as Actions.** NeurIPS, 2023.
- Park, S., Frans, K., Eysenbach, B., and Levine, S. **OGBench: Benchmarking Offline Goal-Conditioned RL.** 2024.
- Park, S., Oberai, A., Atreya, P., and Levine, S. **Transitive RL: Value Learning via Divide and Conquer.** 2025.
- Schaul, T., Horgan, D., Gregor, K., and Silver, D. **Universal Value Function Approximators.** ICML, 2015.

---

## Repository artifacts

Primary result files:

```text
BMM_TRL_STATUS_20260610_181801.md
BMM_TRL_GEODESIC_VALUE_RESULTS_20260610_184508.md
BMM_TRL_GEODESIC_Q_RESULTS_20260610_191752.md
BMM_TRL_QV_TRANSITIVE_RESULTS_20260610_213730.md
BMM_TRL_QV_LOSS_ABLATION_RESULTS_20260610_221317.md
BMM_TRL_BUDGET_HOLDOUT_REPLICATION_RESULTS_20260611_014354.md
BMM_TRL_FAST_DECISION_RESULTS_20260611_124920.md
BMM_TRL_CANDIDATE_ACTION_RESULTS_20260611_135918.md
BMM_TRL_VALUE_SUBGOAL_NEXT_STEPS_RESULTS_20260611_163357.md
BMM_TRL_VALUE_SUBGOAL_CONTROLLER_DECISION_20260611_170712.md
BMM_TRL_HIERARCHICAL_PIVOT_QUICK_TRY_20260611_173408.md
BMM_TRL_PAPER_EXPERIMENT_RESULTS_20260612_013139.md
BMM_TRL_FINAL_PAPER_CONTROL_RESULTS_20260612_020718.md
```

Implementation files:

```text
agents/bmm_trl.py
utils/pointmaze_grid.py
utils/pointmaze_graph.py
scripts/train_bmm_geodesic_value.py
scripts/train_bmm_geodesic_q.py
scripts/run_bmm_qv_budget_holdout.py
scripts/summarize_bmm_paper_tables.py
scripts/eval_bmm_action_ranking.py
scripts/eval_bmm_joint_action_subgoal.py
scripts/eval_bmm_value_subgoal_controller.py
scripts/eval_bmm_value_subgoal_policy_smoke.py
scripts/eval_bmm_graph_value_subgoal.py
scripts/eval_bmm_subgoal_bc_controller.py
assets/bmm_tabular_error_scaling.png
```
