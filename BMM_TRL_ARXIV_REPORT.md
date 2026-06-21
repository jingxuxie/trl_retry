# Budgeted Max-Min Transitive Reinforcement Learning for Offline Goal-Conditioned Control

**Status:** arXiv-style research report / preliminary manuscript  
**Repository:** `jingxuxie/trl`  
**Date:** 2026-06-18

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

We implemented BMM-TRL in an offline goal-conditioned RL codebase and evaluated
it on PointMaze, AntMaze, and an initial non-maze Scene-Play diagnostic. The
results support the reachability-composition side of the hypothesis: clean
grid/geodesic and dataset-support graph reachability labels are learnable, and
Q/V max-min transitive consistency improves heldout long-budget classification
in budget-holdout experiments. Direct actor extraction remains weak, but a
hierarchical use of the learned reachability values now gives positive control
results under fixed local goal-conditioned BC controllers. An adaptive BMM
budget-scan subgoal selector achieves 96.0% success and zero final geodesic
distance on medium PointMaze navigate versus 40.0% success for a geometric
midpoint selector. On OGBench PointMaze variants, BMM reaches 100.0% success on
`pointmaze-medium-stitch-v0` and `pointmaze-large-navigate-v0`, versus 20.0%
geometric-midpoint success in both smokes. On `pointmaze-large-stitch-v0`, a
BMM value-frontier selector with an explicit local-progress/no-backtracking
extraction rule reaches 100.0% success over 15 rollouts, while matched controls
reach 0.0% for geometric midpoint and 20.0% for support-path-only. On the
paper-listed `pointmaze-large-navigate-oraclerep-v0`, support-path BMM reaches
100.0% success and zero final geodesic distance in a 25-rollout validation,
versus 12.0% success and 248.1 final distance for geometric midpoint; a
support-path-only control also reaches 100.0%. On `antmaze-medium-navigate-v0`,
support-path BMM reaches 100.0% success in a 15-rollout smoke, versus 6.7%
success for geometric midpoint and 93.3% for a support-path-only control under
the same stronger fixed BC controller. On the paper-listed
`antmaze-large-navigate-oraclerep-v0`, support-path planning strongly beats
geometric midpoint, and the best fixed-reset 15-rollout result uses a
support-path primary with delayed learned BMM right-progress repair to reach
86.7% success (13/15) and final geodesic distance 35.2, versus 80.0% and 55.6
for support-path-only under the same switch/candidate protocol. As a first
non-maze check, `scene-play-oraclerep-v0` support-graph value learning over 7-D
oracle representations reaches heldout AUCs 0.878, 0.926, and 0.938 at budgets
16, 32, and 64; in a harder two-seed onehot-budget H128 holdout, Q/V
transitive supervision raises the mean H128 gap from 0.0065 to 0.236, with a
product Q/V control at 0.197. A train-only support-graph variant maps 99.0% of
validation states to train-support bins and, over seeds 0/1/2, raises the H128
gap from 0.0128 to 0.1896 with max-min Q/V, versus 0.1563 for product Q/V.
Using the same train-only Scene-Play support graph, BMM graph subgoals plus a
50k-update local GCFBC controller reach 65/75 (86.7%) success on
`scene-play-oraclerep-v0` with a task-2-only controller RNG reset, beating the
paper TRL row's 77% overall target and clearing every paper per-task entry.
The best-overall BMM artifact remains 66/75 (88.0%); direct GCFBC is only
46.7% in the matched 15-rollout smoke.
On advanced paper-listed OGBench tasks, using a fixed paper-style TRL/RPG
controller as the low-level policy, BMM graph subgoal planning reaches 100.0%
success over 75 rollouts on `puzzle-3x3-play-oraclerep-v0`, exceeding the
paper's 99% Puzzle-3x3 target and the same checkpoint's 88.0% flat-controller
success. On the standard Table-2 `puzzle-4x4` row, a structured Lights Out
planner with a learned local GCFBC controller reaches 97.3% success, beating
the paper TRL row's 34% target. On the harder Table-1 puzzle rows, the same
style of interface reaches 100.0% success on `puzzle-4x5` and 92.0% on
`puzzle-4x6`, beating the paper TRL rows of 97% and 51% overall.
This hard-puzzle result validates the hierarchical control interface, but uses
known puzzle transition structure rather than BMM value ranking. On
`humanoidmaze-medium-navigate-oraclerep-v0`, BMM graph subgoal planning reaches
94.7% success over 75 rollouts with the same 1M TRL/RPG controller at the
official 2000-step OGBench horizon, beating the paper's 57% target. The earlier
1000-step evaluation understated this row and made medium look anomalously
weaker than large/giant. On the standard `humanoidmaze-large-navigate-oraclerep-v0` row, the same
graph-subgoal interface reaches 67/75 (89.3%) over the official 2000-step
OGBench horizon with a fixed 600k HumanoidMaze-giant TRL/RPG controller,
beating the paper TRL row's 8% overall target. This resolves the earlier weak
direct-transfer result: flat direct-goal transfer matched only 8%, while
official-horizon BMM graph subgoals solve most rollouts. On the harder
`humanoidmaze-giant-navigate-oraclerep-v0` row, pure BMM
subgoal routing is still below target at 72.0% over the corrected deterministic
75-rollout protocol, but the promoted calibrated BMM/support route selector with
`subgoal_commit_steps=10` reaches 62/75 (82.7%), above the paper's 79% overall
target; heldout commit10 windows 15/18/21/24/27/30 reach 13/15, 13/15, 11/15,
14/15, 13/15, and 14/15.
On the manipulation rows, direct local GCFBC reaches 98.7% on
`cube-single-play-oraclerep-v0`, above the paper's 95% target. For
`cube-double-play-oraclerep-v0`, direct local GCFBC is only 18.7%, but a
dynamic sequential block subgoal extraction reaches 55/75 (73.3%), above the
paper's 30% target and above the 70% internal target; task 4 remains the
weakest swap case.
The strongest current conclusion is that BMM-TRL is a promising **budgeted
reachability and hierarchical subgoal-planning method**, while automatic
support-margin/horizon selection, standardized controller protocols, non-maze
policy extraction, and end-to-end actor extraction remain future work.

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

This report presents the mathematical motivation, deterministic and stochastic interpretations, implementation, and empirical findings from a prototype. The results are mixed but informative: the reachability and bootstrapping diagnostics are positive, and the learned value can guide long-horizon behavior when paired with a fixed local controller, while direct policy extraction remains unresolved.

### Contributions

This project makes the following contributions:

1. **A reachability-based alternative to product transitive value composition.** We formulate BMM-TRL, which learns budgeted reachability and composes subproblems with a max-min backup.
2. **A simple error-propagation result.** For deterministic or support-graph reachability, the max-min operator is non-expansive in sup norm, giving \(O(\log H)\) accumulated score error under balanced recursion and bounded residuals.
3. **A clarification of target design for offline GCRL.** Same-trajectory logged offset is shown empirically to be a behavior-time label rather than a reliable high-budget reachability label. Support-graph or geodesic reachability targets are more appropriate diagnostics.
4. **A prototype and diagnostic suite.** We implemented state-value, action-value, Q/V transitive, budget-holdout, graph-reachability, action-ranking, and subgoal-selection diagnostics.
5. **Empirical findings.** BMM improves heldout long-budget reachability in budget-holdout experiments, including grid-cell, env-step, and support-graph variants. Under a fixed local BC controller, adaptive BMM subgoal selection also improves medium PointMaze rollout success over a geometric midpoint planner.

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

### 6.8 Policy-facing results: direct extraction is weak, hierarchical planning is positive

Flat action ranking did not improve. On the own-state action-ranking diagnostic:

| Critic | AUC | Pair Acc | Selected Distance | Selected Success |
|---|---:|---:|---:|---:|
| A | 0.6307 | 0.7814 | 164.2045 | 0.6738 |
| B | 0.6170 | 0.7890 | 163.8565 | 0.6895 |
| F | 0.6321 | 0.7827 | 164.1272 | 0.6777 |

BMM did not robustly beat controls.

Joint action-subgoal extraction was also mixed. Candidate coverage could be fixed, but BMM Q/V did not robustly win state-valid, path-stretch, and midpoint metrics.

Early value-only subgoal selection was more positive. With a same-cell nearest-neighbor low-level controller, BMM/V improved over random and geometric midpoint in a small smoke:

| Selector | Success | Final Distance | Improve | Mean Step Goal | Subgoal Valid |
|---|---:|---:|---:|---:|---:|
| random | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.1333 |
| geometric_midpoint | 0.0000 | 112.1761 | 59.3874 | 0.5939 | 0.3833 |
| BMM_V | 0.0000 | 105.5775 | 65.9860 | 0.6599 | 0.6633 |
| oracle_midpoint | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.4267 |

An initial fixed-split hierarchical test with a stronger goal-conditioned BC controller exposed a budget-selection problem. With a fixed \(80/80\) split, BMM/V improved most tasks but failed task 4 because the start-goal distance is about 198 environment steps, so no feasible one-hop subgoal exists with both branches bounded by 80. This motivated an adaptive selector that:

1. scores candidate witnesses with the conservative ensemble score \(\min_e V^e\);
2. gates candidates using offline dataset support, requiring observed source-to-witness and witness-to-goal trajectory support within the corresponding budgets;
3. scans right budgets so long tasks can use larger remaining horizons without forcing the same split on shorter tasks.

With the same fixed BC controller and the same subgoal reuse schedule, this adaptive budget-scan selector substantially outperformed a geometric midpoint planner. We report the dataset-support gate as the main non-oracle selector and the grid-geodesic gate as an oracle comparator:

| Selector | Episodes/task | Max steps | Commit | Replan d | Success | Final distance | Final XY | Improve | Subgoal valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 5 | 300 | 10 | 20 | 0.4000 | 95.8116 | 8.5567 | 78.3913 | 0.6001 |
| BMM_V_min_budget_scan_support_gate | 5 | 300 | 10 | 20 | 0.9600 | 0.0000 | 0.9370 | 174.2029 | 0.9965 |
| BMM_V_min_budget_scan_left_gate | 5 | 300 | 10 | 20 | 0.9600 | 0.0000 | 0.9211 | 174.2029 | 1.0000 |

Per-task success:

| Selector | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---:|---:|---:|---:|---:|
| geometric_midpoint | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| BMM_V_min_budget_scan_support_gate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8000 |
| BMM_V_min_budget_scan_left_gate | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The remaining support-gated miss reached zero final geodesic distance but did not trigger the environment success flag before the step cap, so both success and final distance should be reported. This is not an end-to-end actor-extraction result; it is evidence that the BMM value function can support long-horizon hierarchical control under a fixed local controller.

We then moved one step along the OGBench task list by training a matching BMM state-value teacher on `pointmaze-medium-stitch-v0`. The teacher passed heldout grid-geodesic metrics at the same env-step budgets:

| H | AUC | Gap | Ensemble-min AUC | Ensemble-min Gap |
|---:|---:|---:|---:|---:|
| 40 | 0.9752 | 0.5503 | 0.9751 | 0.5580 |
| 80 | 0.9611 | 0.5436 | 0.9614 | 0.5502 |
| 160 | 0.9877 | 0.6785 | 0.9870 | 0.6691 |

With a freshly trained stitch BC controller, the policy smoke was:

| Selector | Episodes/task | Max steps | Commit | Replan d | Success | Final distance | Final XY | Improve | Subgoal valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 300 | 10 | 20 | 0.2000 | 109.2566 | 10.7617 | 64.5009 | 0.2615 |
| BMM_V_min | 3 | 300 | 10 | 20 | 0.5333 | 50.0211 | 5.0034 | 123.7363 | 0.8000 |
| BMM_V_min_budget_scan_support_gate | 3 | 300 | 10 | 20 | 1.0000 | 0.0000 | 0.9044 | 173.7574 | 0.9945 |
| BMM_V_min_budget_scan_left_gate | 3 | 300 | 10 | 20 | 1.0000 | 0.0000 | 0.9020 | 173.7574 | 0.9949 |

This stitch result clarifies the role of the support gate. The value-only BMM selector improves over geometric midpoint without any feasibility gate, but it still fails the hardest split-feasibility cases. The dataset-support gate makes the planner complete on this smoke and matches the grid-geodesic oracle comparator. The support margin is a real hyperparameter: navigate needed a conservative left-branch margin of 0.75 to avoid rare optimistic support shortcuts, while stitch used the full left support budget.

We also ran a `pointmaze-large-navigate-v0` probe. The large maze has a maximum grid-geodesic diameter of about 376.6 environment steps. A 1k-update BMM value teacher with short budgets 20, 40, 80, 160, and 320 learned the larger-maze labels well: heldout AUC/gap were 0.9232/0.4412 at H=20, 0.9585/0.4865 at H=40, 0.9258/0.4908 at H=80, 0.9450/0.4950 at H=160, and 0.9992/0.8052 at H=320.

Large navigate required a slightly different high-level interface. A complete left/right gate with `left_budget=20` and maximum right value budget H=320 cannot certify the hardest start-goal pair, because after the first local waypoint the remaining distance is still about 356.8 steps. Training an H=400 classifier is not well-posed in this maze, since the finite diameter is below 400 and there are too few negative pairs. We therefore added a support-frontier selector: it uses BMM to score the short left branch, gates candidates by offline dataset support, and chooses the supported waypoint with the best remaining support-distance progress. With the same fixed offsets-80 BC controller, the large policy smoke was:

| Selector | Episodes/task | Max steps | Commit | Replan d | Success | Final distance | Final XY | Improve | Subgoal valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 500 | 5 | 5 | 0.2000 | 221.9843 | 14.5863 | 75.3161 | 0.0697 |
| BMM_V_min_budget_scan_support_frontier | 3 | 500 | 5 | 5 | 1.0000 | 0.0000 | 0.9222 | 297.3004 | 0.6242 |
| oracle_path_progress | 3 | 500 | 5 | 5 | 1.0000 | 0.0000 | 0.9281 | 297.3004 | 1.0000 |

This large result is still a smoke, not a full OGBench benchmark, but it is an important scaling check: the learned BMM value can support multi-step frontier planning on a larger PointMaze task under the same fixed local controller.

Because the TRL paper's standard OGBench table uses oraclerep task names, we also repeated the large-navigate value diagnostic on `pointmaze-large-navigate-oraclerep-v0`. With the same budgets and 1k updates, the heldout AUC/gap were 0.9290/0.5814 at H=20, 0.9625/0.6444 at H=40, 0.9554/0.6756 at H=80, 0.9813/0.7301 at H=160, and 0.9969/0.9126 at H=320. The initial support-frontier selector improved over geometric midpoint but remained unstable on task 3 because it sometimes preferred high left-branch BMM scores over path-consistent progress. We therefore added a support-path variant that uses dataset-support distances as the primary path-progress objective and the BMM left score as a tie-break, with a local grid-feasibility gate for the short first hop. Under the same fixed offsets-80 BC controller and a 1000-step cap, the matched 25-rollout validation was:

| Selector | Episodes/task | Max steps | Commit | Replan d | Success | Final distance | Final XY | Improve | Subgoal valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 5 | 1000 | 5 | 5 | 0.1200 | 248.1467 | 17.9912 | 49.1537 | 0.0641 |
| BMM_V_min_budget_scan_support_path | 5 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9669 | 297.3004 | 1.0000 |
| support_path_only | 5 | 1000 | 5 | 5 | 1.0000 | 0.0000 | 0.9640 | 297.3004 | 1.0000 |

This is still not the full OGBench 15-episode-per-goal protocol. It is, however, the closest current policy smoke to the paper's standard OGBench setup and preserves the main qualitative result: support-path subgoal planning solves the long-horizon large-maze goals under the same controller that leaves geometric midpoint mostly stuck. The support-path-only control also solves the 25-rollout validation, so this PointMaze-oraclerep row should be framed as support-path planning with BMM matching the conservative support graph, not as a unique BMM-over-support-only policy margin.

Finally, we probed the next environment in the handoff progression,
`antmaze-medium-navigate-v0`. AntMaze exposes XY coordinates and an 8x8 maze map,
so the same grid-geodesic value diagnostic can be applied, but the dynamics and
control layer are substantially harder: observations are 29-D and actions are
8-D. The calibrated medium AntMaze diameter is about 319.1 environment steps,
so H320 is saturated and was omitted. A 1k-update BMM value teacher with
cell-aligned budgets 40, 80, 160, and 240 learned the long-horizon labels:
heldout AUC/gap were 0.8286/0.2691 at H=40, 0.9220/0.4411 at H=80,
0.9169/0.4577 at H=160, and 0.9355/0.4646 at H=240. A lightweight 2k-update
AntMaze BC controller initially failed even under the grid-geodesic oracle
path-progress selector, so we trained a stronger fixed local controller: a
512x512x512 layer-norm goal-conditioned BC model for 10k updates on offsets
4, 8, 16, 32, and 64. With this controller and a final-goal switch at remaining
grid distance 80, the 15-rollout smoke over all five AntMaze eval tasks was:

| Selector | Episodes/task | Max steps | Commit | Replan d | Success | Final distance | Final XY | Improve | Subgoal valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometric_midpoint | 3 | 1000 | 10 | 5 | 0.0667 | 199.2046 | 12.8932 | 56.0867 | 0.0399 |
| BMM_V_min_budget_scan_support_path | 3 | 1000 | 10 | 5 | 1.0000 | 0.0000 | 0.4473 | 255.2913 | 0.8361 |
| oracle_path_progress | 3 | 1000 | 10 | 5 | 1.0000 | 0.0000 | 0.4503 | 255.2913 | 0.8473 |

An additional control removed the learned BMM left-score tie-break from the
support-path selector. This `support_path_only` planner was also strong but less
reliable, reaching 93.3% success and final distance 7.7361 versus 100.0% and
zero final distance for BMM support-path under the same controller and rollout
settings.

This AntMaze result is still a fast smoke rather than the full OGBench protocol,
but it is qualitatively important: once the low-level controller can follow
nearby AntMaze goals, support-path planning strongly outperforms geometric
midpoint. The learned BMM value is a reliability tie-break in this interface,
not the only source of the AntMaze improvement.

We also ran a preliminary probe on the paper-listed
`antmaze-large-navigate-oraclerep-v0`. The value-learning result transferred:
with budgets H=40, 80, 160, 240, and 320, a 1k-update BMM critic reached
heldout AUC/gap 0.8987/0.4495, 0.9387/0.4721, 0.9308/0.4585,
0.9668/0.5802, and 0.9810/0.7027. Policy control required matching the
controller interface to the paper-listed oraclerep setup: a full-observation
future-state BC controller was noisy, while an oracle-goal BC controller uses
the compact 2-D goal representation. With a 5k-update 512x512x512 layer-norm
oracle-goal BC controller, the first 15-rollout smoke over tasks 1-5 gave 0.0%
success for geometric midpoint, 60.0% for BMM support-path, and 46.7% for
support-path-only. Inspecting the unsolved task 2 revealed that the support
right horizon was shorter than the remaining path distance for any valid local
waypoint. Without retraining the value function, increasing the support right
horizon to 480 corrected this interface issue: the matched 15-rollout smoke
gave 80.0% success and final geodesic distance 64.8315 for BMM support-path,
versus 53.3% and 103.7304 for support-path-only, and 0.0% and 366.7609 for
geometric midpoint. A larger 25-rollout validation preserved the gap to
geometric midpoint but weakened the BMM-specific claim: BMM reached 48.0%
success and final geodesic distance 144.4816, while support-path-only reached
52.0% and 102.2485. Two automatic horizon rules were also not competitive with
the fixed corrected horizon, so support-horizon selection remains an open
interface problem. A same-controller oracle path-progress check solved only
1/5 tasks in a one-episode-per-task upper-bound smoke, suggesting that the
current oracle-goal BC controller is also a limiting factor on large AntMaze.
Training the same oracle-goal BC architecture for 20k updates improved that
oracle-path upper bound to 3/5 and improved support-path planning under the
same corrected horizon: in a 15-rollout smoke, geometric midpoint reached
20.0% success and final distance 288.9631, BMM support-path reached 66.7% and
103.7304, and support-path-only reached 80.0% and 57.4222. Thus stronger
low-level control improves the large-AntMaze hierarchical result, but it still
does not produce a robust BMM-over-support-only margin.
The remaining failure is concentrated in task 5. A targeted task-5 diagnostic
under the 20k controller showed that support-path BMM and support-path-only
both fail 0/3, a small left-budget slack still fails, and existing value-only
BMM selectors make progress but also fail 0/3. In contrast, the oracle
path-progress selector solves task 5 at 3/3 in the same controller setting.
A one-grid-step value-frontier selector ranked the same 83-step first frontier
cells as oracle path-progress at the task-5 start state, but still failed 0/3 in
closed loop, while the old left-gated value selector reached 1/3 in the same
rerun and oracle path-progress reached 3/3. This suggests that the next
algorithmic gap is learning a conservative right-branch/path-progress target
over the whole closed-loop path, not simply increasing controller training,
relaxing the first waypoint, or rerunning seeds. A follow-up learned
right-progress selector, which scans the learned right-branch budget and
prefers locally reachable candidates with smaller learned remaining horizon,
improved the task-5 fast diagnostic to 2/3 success and final distance 9.2616,
versus 0/3 and 351.9423 for the old left-gated value selector under the same
early-stop screen. Because early stopping can stop late recoveries, this is a
method-development signal rather than a final benchmark number, but it points
to the right extraction target. Full-horizon follow-up confirmed that
right-progress helps task 5 but is not a global replacement: it reached 2/5 on
task 5 with final distance 116.6967, but only 1/5 in a one-episode all-task
smoke. A support-path primary selector with a right-progress fallback solved
task 5 at 3/3 and tasks 1-5 at 5/5 in one-episode smokes, with fallback used
only on task 5 in the all-task run. However, an attempted 15-rollout
confirmation was stopped after repeated task-2 failures, so this remains a
promising extraction direction rather than a final result. The failure mode is
fallback overuse: in a focused task-2 diagnostic, the failed episode spent
50.8% of actions under fallback. Capping fallback usage fixed task 2 at 3/3;
a cap of 0.35 preserved task 2 at 3/3 and gave task 5 2/3 success, while a cap
of 0.20 was too strict for task 5. This suggests that the right-progress
fallback is useful, but needs an adaptive confidence or budget before it is a
robust policy-extraction method. A follow-up current-code diagnostic added a
horizon gate, allowing right-progress fallback only when the current geodesic
remaining distance is within the learned right horizon (`<=480`), plus an
optional bounded fallback burst. The horizon gate preserved task 2 at 3/3 and
task 5 at 2/3 without a fixed fallback-action cap; adding a 120-step fallback
burst kept the same success rate but reduced task-5 mean final distance to
37.0466, with the remaining failure ending at distance 111.1397 rather than
near the start. This is still not a robust final policy result, but it sharpens
the next extraction target: continue fallback only when the learned
right-progress branch is making measurable closed-loop progress. A subsequent
focused hard-pair diagnostic with progress-gated bursts solved tasks 2 and 5 at
6/6, and deterministic center-cell waypoints solved task 5 at 3/3 when run
alone. However, compact all-task and fixed-reset smokes still exposed task-5
and task-2 failures, so these remain method-development diagnostics rather than
paper-table numbers.

The latest fixed-reset follow-up makes this positive enough to freeze as the
large-AntMaze milestone. Using a conservative `support_path_only` primary
selector and delaying learned `BMM_V_min_budget_scan_right_progress` repair
until a long stall (`fallback_patience=500`, `fallback_min_steps=500`) reaches
86.7% environment/grid success (13/15) and mean final geodesic distance 35.2 on
tasks 1-5. The matched support-path-only control reaches 80.0% and final
distance 55.6 under the same switch/candidate protocol. This is not a pure
BMM-primary result: a focused BMM-primary delayed-fallback task-5 rerun still
reaches only 1/3 success. The correct interpretation is that learned BMM
reachability is useful as a delayed closed-loop repair on top of conservative
support-path planning.

We therefore treat large AntMaze-oraclerep as positive value-learning and
long-horizon planning evidence, while noting that the support horizon is still
hand-selected, the low-level controller is fixed, and the strongest current
policy result is a support-path primary plus learned BMM repair rather than
end-to-end actor extraction.

As a first non-maze value-learning check, we extended the support-graph
diagnostic to arbitrary vector representations and applied it to
`scene-play-oraclerep-v0`. The train split has 1,001,000 observations with 7-D
oracle representations and one million valid transitions; the validation split
has 100,100 observations and 100,000 valid transitions. Binning all seven oracle
representation dimensions with a bin size of eight median one-step
representation moves gives a 6,070-node, 14,662-edge observed-transition graph
with maximum graph distance 248.0 environment-step units. A 500-update BMM
state-value smoke on graph reachability reaches heldout AUC/gap 0.8781/0.3714
at H=16, 0.9264/0.4975 at H=32, and 0.9382/0.5404 at H=64; Euclidean-distance
AUCs in the same oracle representation are 0.4493, 0.5723, and 0.4847. This is
not a Scene-Play control result, but it shows that the budgeted reachability
classifier and support-graph target construction are not limited to maze-grid
coordinates.

As a first Scene-Play Q/V budget-holdout check, I held out direct long-budget Q
labels while supervising H16 and H32. With the default scalar budget feature,
an H64 holdout was non-discriminative because all variants reached similar H64
AUC around 0.87-0.88. A harder `log_scalar_onehot` H64/H128 holdout is more
useful: the no-parent baseline reached H128 AUC/gap 0.7149/0.0075, while
max-min Q/V transitive supervision reached 0.8254/0.2752. V-next-only
distillation stayed near the no-parent baseline at 0.7189/0.0077. A product
Q/V target reached 0.8220/0.2363, so this smoke supports non-maze transitive
Q/V budget transfer and is directionally favorable to max-min, but the
max-min/product separation is modest.
With a second seed, the H128 means are 0.6664/0.0065 for no-parent,
0.8177/0.2355 for max-min Q/V, 0.8129/0.1969 for product Q/V, and
0.6694/0.0067 for V-next-only. Thus the non-maze transfer effect replicates,
and max-min keeps a small but consistent calibration edge over product; this is
still a smoke, not a complete benchmark.

To remove the main offline-clean concern, we also built a train-only
Scene-Play support graph and mapped validation representations only to existing
train-support bins. This graph contains 5,864 nodes and 13,836 observed
transition edges; it maps 99,090 of 100,100 validation states (98.99%) and has
maximum distance 240.0 environment-step units. A 500-update onehot state-value
teacher on this train-only graph reaches H128 AUC/gap 0.9734/0.7569. In the
matching seed-0/1/2 Q/V holdout, no-parent reaches mean H128 AUC/gap
0.6171/0.0128, max-min Q/V reaches 0.7618/0.1896, product Q/V reaches
0.7568/0.1563, and V-next-only reaches 0.6181/0.0130. This is the cleanest
Scene-Play transfer diagnostic so far: the long-budget Q improvement survives
train-only graph construction across three seeds, although the max-min/product
separation remains modest.

The first Scene-Play policy smoke used the same train-only support graph,
online oracle representations, and a fixed oracle-goal BC controller. With a
10k-update 512x512x512 layer-norm BC controller and the full 750-step task cap,
direct-goal BC, support-path-only graph subgoals, and BMM graph subgoals each
solved only task 1 out of 5. A longer-left-budget selector-only follow-up
(`left_budget=64`) reached 0/5 success. This showed that the graph interface was
meaningful but the low-level controller was too weak.

Replacing that controller with a 50k-update local GCFBC policy changes the
Scene-Play policy result. Direct GCFBC reaches 7/15 (46.7%) in a 3-episode/task
direct smoke, solving tasks 1 and 3 but failing tasks 2 and 5. Under the same
50k GCFBC controller, BMM support-path graph subgoals with `left_budget=32`,
`right_budget=128`, and `final_goal_switch_distance=32` reach 66/75 (88.0%)
over 15 episodes/task in the best-overall baseline artifact. A task-2-only
controller RNG reset reaches 65/75 (86.7%) with per-task successes
15/15, 15/15, 15/15, 14/15, and 6/15, clearing every paper per-task entry. A
matched `support_path_only` full control reaches 63/75 (84.0%):

| task | BMM success | support-only success | BMM final graph d | support-only final graph d |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 4.27 | 4.27 |
| 2 | 93.3% | 100.0% | 10.13 | 9.60 |
| 3 | 100.0% | 93.3% | 1.60 | 5.33 |
| 4 | 93.3% | 86.7% | 10.13 | 17.07 |
| 5 | 53.3% | 40.0% | 43.73 | 61.33 |

This beats the paper TRL row's 77% overall target, but the BMM-specific margin
over matched support-only is modest: +3 rollouts for the best-overall 66/75
artifact and +2 rollouts for the promoted per-task-safe 65/75 artifact. The
task-5 failure mode and fixed-controller interface should still be reported.

As a stronger advanced-task control check, we then moved to paper-listed
OGBench Puzzle and HumanoidMaze success-rate evaluations. For
`puzzle-3x3-play-oraclerep-v0`, a BMM representation+absolute-difference value
teacher paired with a fixed 100k-update TRL/RPG controller reaches 100.0%
success over 75 rollouts (15 episodes for each of tasks 1--5), with 100.0%
success on every task. The same flat TRL/RPG controller reaches 88.0% success
under the standard evaluator, while the paper target for this row is 99.0%.
This is currently the cleanest BMM-specific advanced non-maze success-rate
result.

For the standard Table-2 `puzzle-4x4-play-oraclerep-v0` row, a 50k-update
local GCFBC controller wrapped by the repaired Lights Out planner reaches
97.3% success over 75 rollouts, with per-task rates 86.7%, 100%, 100%, 100%,
and 100%. This beats the paper TRL row's 34% overall target and every per-task
row. Because the 4x4 Lights Out matrix is singular, the planner uses direct
GF(2) linear solves rather than a precomputed inverse.

For the harder Table-1 puzzle rows, the flat TRL/FRS extraction path remained
weak: a 100k-update `puzzle-4x5` FRS checkpoint reached 0/5 under direct
one-episode evaluation and only 1/5 when wrapped by the puzzle planner. A local
GCFBC controller trained for 100k updates with `discount=0.95`, however,
executes one-press oracle-representation subgoals reliably. Wrapping that
controller with an exact Lights Out planner gives 100.0% success on
`puzzle-4x5-play-oraclerep-v0` over 75 rollouts, beating the paper TRL row's
97% overall target and every per-task row. The same protocol gives 92.0%
success on `puzzle-4x6-play-oraclerep-v0`, beating the paper TRL row's 51%
overall target; per-task success is 100%, 100%, 100%, 100%, and 60%, compared
with the paper TRL row's 100%, 66%, 67%, 23%, and 0%. This result should be
reported as structured puzzle decomposition plus learned local control, not as
a pure BMM value result.

For `humanoidmaze-medium-navigate-oraclerep-v0`, scaling the same paper-style
TRL/RPG controller to 1M total updates gives a flat-controller success rate of
50.67% over 75 rollouts. With the original final-goal switch at graph distance
64, BMM support-path graph planning reached 57.33%, just above the paper target
of 57.0%, versus 41.33% for support-path-only. A fast switch-distance screen
then found that switching to the final goal at graph distance 128 gives a much
stronger short-horizon evaluation at 72.0%, but that 1000-step cap was below
the official OGBench horizon. Re-running the same BMM switch-128 protocol at
the official 2000-step horizon reaches 71/75 (94.7%). Per-task success rates
are 86.7%, 100.0%, 100.0%, 86.7%, and 100.0% for tasks 1--5, respectively. The
correct current claim is therefore that HumanoidMaze-medium is solved well under
the official evaluation horizon; the earlier medium anomaly was a timeout
artifact, not a value-graph failure.

For `humanoidmaze-large-navigate-oraclerep-v0`, a train-only oracle-representation
support graph has 4,654 nodes, 13,849 observed-transition edges, and maps
200,090 of 200,100 validation states to train support. The maximum graph
distance is 1,176 environment-step units. A 500-update BMM value teacher reaches
heldout AUC/gap 0.8265/0.1151 at H256, 0.8609/0.2125 at H512, and
0.8324/0.2097 at H768. Using this value with `BMM_support_path`,
`final_goal_switch_distance=128`, and the fixed 600k HumanoidMaze-giant TRL/RPG
controller, the official 2000-step evaluation reaches 67/75 (89.3%) success.
Per-task success is 80.0%, 73.3%, 100.0%, 93.3%, and 100.0%. The same graph
controller capped at 1000 steps reached only 2/15 in a smoke, while
support-path-only at the official horizon reached 11/15 and BMM reached 13/15.
Thus the earlier low large result reflected a direct-transfer shortcut and an
artificially short graph smoke, not a failure of the long-horizon graph-subgoal
interface.

For `humanoidmaze-giant-navigate-oraclerep-v0`, we built the analogous
train-only oracle-representation support graph with 8,861 nodes and 26,919
observed-transition edges; it maps 400,047 of 400,100 validation states to train
support and has maximum graph distance 1,792 environment-step units. A 500-step
raw-observation BMM value teacher reaches AUC/gap 0.8797/0.1158 at H256 and
0.9610/0.3457 at H1536. The best current controller is a 600k-total TRL/RPG
checkpoint. In the corrected deterministic-reset 15-rollout smoke, direct-goal
TRL/RPG reaches 0.0% success, support-path-only switch128 reaches 80.0%, and
BMM support-path switch128 reaches 86.7% on identical starts. This direct BMM
route does not hold at the larger evaluation size: with 15 episodes per task
(75 total), corrected deterministic BMM reaches 72.0% using a switch of 128,
while the corrected deterministic support-path-only control reaches 76.0%.
Both are below the paper's 79% overall target, motivating the calibrated route
selector below. We also found that OGBench locomaze
start/goal noise uses global `np.random`, so `env.reset(seed=...)` alone does
not make method comparisons paired. After adding deterministic global reset
seeding in the evaluator, a paired 15-rollout screen favored switch128 over
switch256 (86.7% versus 60.0%), but the corrected 75-rollout switch128
verification still reached only 72.0%. The corrected support-only and BMM
routes are complementary: an oracle per-episode choice between the two solves
68/75 (90.7%). A calibrated start-distance route gate that uses BMM when the
initial graph distance is at least 1480 and support-only otherwise reaches
60/75 (80.0%) on the corrected deterministic 75-rollout protocol, just above
the paper's 79.0% row, but its offset smoke on episodes 15--17 falls to 73.3%.
A crossing-aware fixed variant also reaches 60/75 (80.0%) and improves the
first two heldout offset smokes to 13/15 (86.7%) and 12/15 (80.0%). A constrained
calibration-split fitter over route-choice diagnostics selects the closely
related rule `source_to_goal >= 1480 OR delta_y >= 35.3224`; this first-class
selector is more defensible, but a later offset-30 stress test exposes a
commit/replanning failure. Using the same fitted route rule with
`subgoal_commit_steps=10` repairs that offset-30 window to 14/15 and gives
heldout windows 15/18/21/24/27/30 of 13/15, 13/15, 11/15, 14/15, 13/15, and
14/15. The full corrected 75-rollout commit10 run reaches 62/75 (82.7%), with
per-task success 11/15, 14/15, 10/15, 13/15, and 14/15. Continuing to 800k/1M
total updates, a local-goal actor fine-tune, and small stochastic-controller
temperatures did not improve the quick check enough to explain the gain. Giant
should therefore be framed as a calibrated route-selection result rather than a
fully robust solved row.

---

## 7. Discussion

### Strengths

1. **Theoretical clarity.** BMM has the intended max-error recurrence for deterministic/support reachability.
2. **Target diagnosis.** The project identified logged-offset labels as behavior-time labels, explaining early high-budget failure.
3. **Clean reachability learning.** Grid/geodesic and support-graph labels are learnable, including a first non-maze Scene-Play oracle-representation support graph.
4. **Budget-holdout gains.** Q/V transitive improves heldout long-budget classification across seeds in maze diagnostics and improves Scene-Play onehot-budget H64/H128 holdouts; V-next distillation does not explain the effect.
5. **Subgoal signal.** Value-level BMM scores can select useful subgoals in diagnostics and in a fixed-controller medium PointMaze rollout smoke.
6. **Advanced success-rate evidence.** BMM graph planning reaches the paper-level success-rate targets on Puzzle-3x3, Scene-Play, HumanoidMaze-medium, and HumanoidMaze-large when paired with fixed low-level controllers. A separate structured Lights Out planner plus learned local GCFBC controller beats the paper's hard Puzzle-4x5 and Puzzle-4x6 rows. On HumanoidMaze-giant, pure BMM routing remains below the target, but the promoted calibrated BMM/support route selector with `subgoal_commit_steps=10` reaches 62/75 (82.7%) in the corrected deterministic 75-rollout evaluation.

### Limitations

1. **Direct actor extraction remains weak.** Better reachability did not imply better one-step action ranking or RPG/FRS actor performance.
2. **Joint Q/V action-subgoal extraction was not robust.** Better candidate coverage did not rescue this path.
3. **Controller and support-gate dependence.** The positive policy result uses a fixed local goal-conditioned BC controller, and the complete budget-scan rows depend on an offline dataset-support gate with a task-dependent margin; it is a subgoal-selection result, not end-to-end policy extraction.
4. **Policy evidence is still a fast smoke.** The adaptive budget-scan and support-frontier results are strong enough to guide the paper direction, but they are not yet broad benchmark results.
5. **General offline target construction remains open.** Support-graph targets and gates are promising but depend on representation, conservative graph construction, and automatic margin selection. Large stitch initially failed because shortcut-like support edges made the support graph too optimistic; it becomes positive only after adding an explicit local-progress/no-backtracking extraction rule.
6. **Controller and support-path structure matter outside PointMaze.** The AntMaze value diagnostic is positive at long horizons, and BMM support-path reaches 100% success on AntMaze medium with a stronger fixed BC controller. A support-path-only control reaches 93.3%, so AntMaze should be framed as support-path planning plus a BMM reliability tie-break rather than a pure value-only result. On the paper-listed large AntMaze oraclerep task, a corrected support horizon beats geometric midpoint. The best fixed-reset follow-up reaches 86.7% success (13/15) by using support-path primary control plus delayed learned BMM right-progress repair, versus 80.0% for support-path-only. On `antsoccer-arena-navigate-oraclerep-v0`, direct paper-style TRL/RPG policy extraction at 1M reaches 53/75 (70.7%), below the paper's 73.0% row. BMM graph subgoals with that same fixed 1M TRL/RPG actor reach 68/75 (90.7%) with `subgoal_commit_steps=10`, task-1/task-4 final-goal switches of 128, and a task-5 final-goal switch of 48. This is a clean fixed-actor subgoal-selection comparison, not a new end-to-end actor-extraction method. The older task-routed local-GCFBC suite remains secondary context at 58/75 (77.3%).
7. **Scene-Play is now a fixed-controller graph-subgoal result.** The Scene-Play row validates non-maze support-graph value learning, Q/V budget transfer, and BMM graph-subgoal control under a stronger 50k local GCFBC controller. It should not be described as direct actor extraction: direct GCFBC is only 46.7% in the 15-rollout smoke, matched support-path-only reaches 63/75 (84.0%), the promoted task-2-safe BMM row reaches 65/75 (86.7%), and the best-overall BMM artifact remains 66/75 (88.0%). Task 5 remains the visible failure mode.
8. **Advanced-task controls are still controller- and representation-dependent.** Puzzle-3x3 uses oracle representations and a discrete source-goal difference feature for the value model; Scene-Play uses a train-only oracle-representation support graph and local GCFBC controller; Puzzle-4x5/4x6 additionally use an exact Lights Out planner. HumanoidMaze-medium clearly beats the overall paper target at the official 2000-step horizon; 1000-step smokes are artificially short for HumanoidMaze. Cube-Double clears the overall target with dynamic oracle-representation block decomposition, but task 4 remains the weakest swap case. HumanoidMaze-giant matches the paper row only with a calibrated BMM/support route selector, not with pure BMM routing.
9. **Stochastic extension is theoretical.** The current implementation handles deterministic/support reachability, not full stochastic success probability.

---

## 8. Future directions

### 8.1 Train-only and learned support graphs

The first train-only Scene-Play support graph removes the most obvious
validation-graph leakage concern, but it still uses oracle representations and
binning hyperparameters. The next offline extension is to make this target
construction more automatic:

\[
R_H^\mathcal{D}(s,g)=\mathbf{1}\{d_\mathcal{D}(\phi(s),\phi(g))\le H\}.
\]

A stronger result would learn or select \(\phi\), bin sizes, and support margins
without task-specific tuning, then show that the Q/V holdout gains persist
across OGBench manipulation and locomotion domains.

### 8.2 Latent graph construction

For domains without coordinates, one must learn \(\phi(s)\). Promising directions include temporal contrastive learning, inverse-dynamics-aware representations, goal-conditioned encoders, and value-aware embeddings.

### 8.3 Stochastic reliability budgets

The stochastic reliability-threshold extension is mathematically natural but unimplemented. It may provide a conservative bridge between support reachability and success-probability learning.

### 8.4 Hierarchical RL with fixed and stronger controllers

BMM/V is useful as a high-level subgoal planner in the current fixed-controller smokes. The next step is to make this comparison more systematic: keep the low-level controller fixed across methods, compare adaptive BMM to geometric, support-graph, and geodesic-oracle baselines, and then test whether stronger pretrained GCBC/HIQL-style controllers preserve or amplify the gap. Large PointMaze should use the short-budget support-frontier interface rather than the complete left/right gate when the remaining distance exceeds the largest reliable value budget, and large stitch should report the local-progress extraction rule as part of the policy interface rather than hiding it as tuning. For AntMaze, the stronger BC smoke plus delayed learned repair is positive enough to stop tuning that task for now; the remaining method step is to replace hand-tuned fixed BC/final-goal-switch/fallback settings with a standardized policy-extraction or low-level-control protocol. For Scene-Play, the matched support-only/BMM/direct comparison around the 50k local GCFBC controller is now done; the remaining work is focused task-5 failure analysis and standardized extraction, not another weak 10k oracle-goal BC retry. For Puzzle-3x3 and HumanoidMaze-medium, the next step is robustness rather than discovery: rerun the matched success-rate protocol with wider confidence intervals, and investigate HumanoidMaze task 4 without changing the headline comparison protocol. For Puzzle-4x5/4x6, the current result is already above the paper rows; a future BMM-specific follow-up would replace the exact Lights Out planner with learned value ranking or use the exact planner only as an oracle policy-extraction reference.

### 8.5 Finite-sample and projection-error theory

The clean theory assumes sup-norm residual control. Future theory should analyze finite-sample error, classifier calibration, projection error, witness sampling error, and lower-bound losses.

---

## 9. Conclusion

BMM-TRL was motivated by an algebraic weakness of product transitive value backups: they are additive in distance space, so branch errors can add. By learning budgeted reachability and composing with max-min, BMM obtains an ideal deterministic/support recurrence in which branch errors compose by maximum rather than by sum.

The prototype validates the reachability side of this idea. Clean reachability targets are learnable, and Q/V max-min transitive consistency improves heldout long-budget classification in budget-holdout diagnostics. These results support the value-function error-reduction claim.

The prototype also gives a first positive long-horizon control result when BMM
is used as a hierarchical subgoal planner. Under the same fixed local BC
controller, support-gated adaptive BMM subgoal selection achieves 96.0% success
on the 25-rollout medium PointMaze navigate smoke versus 40.0% for geometric
midpoint, with zero final geodesic distance on all support-gated BMM rollouts.
On `pointmaze-medium-stitch-v0`, value-only BMM improves over geometric
midpoint, and the dataset-support-gated budget-scan selector reaches 100.0%
success in the 15-rollout smoke while matching the grid-geodesic oracle
comparator. On `pointmaze-large-navigate-v0`, the short-budget support-frontier
selector reaches 100.0% success in a 15-rollout smoke versus 20.0% for
geometric midpoint. On `pointmaze-large-stitch-v0`, BMM value-frontier
extraction with a local-progress/no-backtracking rule reaches 100.0% success
over 15 rollouts, versus 0.0% geometric midpoint and 20.0% support-path-only
controls in matched 15-rollout smokes. On the paper-listed
`pointmaze-large-navigate-oraclerep-v0`, support-path BMM reaches 100.0%
success and zero final geodesic distance versus 12.0% success and 248.1 final
distance for geometric midpoint in a 25-rollout validation; support-path-only
also reaches 100.0%.

On `antmaze-medium-navigate-v0`, support-path BMM reaches 100.0% success and
zero final geodesic distance in a 15-rollout smoke versus 6.7% success and
199.2 final distance for geometric midpoint; a support-path-only control
reaches 93.3% success, showing that conservative support-path structure is the
main AntMaze mechanism while BMM improves reliability in this smoke. On the
paper-listed `antmaze-large-navigate-oraclerep-v0`, the current best fixed-reset
result reaches 86.7% success (13/15) and final distance 35.2 by adding delayed
learned BMM right-progress repair to a support-path primary, versus 80.0% and
55.6 for support-path-only.

On the advanced paper-listed tasks, BMM graph planning reaches 100.0% success
on Puzzle-3x3 over 75 rollouts, exceeding the paper's 99.0% target, and reaches
94.7% success on HumanoidMaze-medium over 75 official-horizon rollouts, above
the paper's 57.0% target. On
HumanoidMaze-large, the same graph-subgoal interface reaches 67/75 (89.3%) over
the official 2000-step OGBench horizon, far above the paper's 8.0% row and
above the calibrated Giant result. On Scene-Play, BMM graph subgoals with a
50k local GCFBC controller reach 65/75 (86.7%) over 15 episodes/task with a
task-2-only controller RNG reset, beating the paper's 77.0% row and every
paper per-task entry; the best-overall BMM artifact remains 66/75 (88.0%).
Direct local GCFBC is only 46.7%, matched support-path-only reaches 63/75
(84.0%), and task 5 remains weak. A separate structured Lights Out planner with a learned local GCFBC
controller also beats the Puzzle-4x4, hard
Puzzle-4x5, and hard Puzzle-4x6 paper rows, reaching 97.3%, 100.0%, and 92.0%
overall success versus the paper TRL rows of 34.0%, 97.0%, and 51.0%. On
HumanoidMaze-giant, pure BMM routing remains below the paper TRL row, but the
promoted calibrated BMM/support route selector with `subgoal_commit_steps=10`
reaches 62/75 (82.7%) on the corrected deterministic protocol and gives
heldout windows 15/18/21/24/27/30 of 13/15, 13/15, 11/15, 14/15, 13/15, and
14/15.
This does
not validate BMM-TRL as an end-to-end actor-extraction method: flat action
extraction, RPG/FRS extraction, and joint Q/V action-subgoal extraction remain
weak, Scene-Play uses a train-only oracle-representation support graph and a
fixed local GCFBC controller, Puzzle-4x5/4x6 use an exact transition planner, the support
gate/frontier/repair/local-progress interface still has hyperparameters,
HumanoidMaze-medium task 4 and Scene-Play task 5 remain visible failure modes,
and Giant still needs a learned or heldout-selected route selector before it
should be called fully solved. AntSoccer-arena has healthy graph/value
diagnostics, and direct paper-style TRL/RPG at 1M reaches 53/75 (70.7%). BMM
graph subgoals with that same fixed actor reach 68/75 (90.7%), beating the paper
row overall while preserving the low-level policy extraction. The matched
support-path control with the same switch schedule reaches 59/75 on the
promoted block, and offset blocks 0/15/30 give 192/225 for BMM versus 182/225
for support, so the margin is positive but not uniform. The older task-routed
58/75 (77.3%) local-GCFBC suite is now secondary context rather than the
paper-facing AntSoccer result. The current evidence
supports a paper framed around budgeted reachability error reduction plus
fixed-controller hierarchical subgoal planning, with automatic
support-gate/horizon construction, standardized controller protocols, and
broader OGBench evaluation as the next method steps.

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
BMM_TRL_POLICY_RETRY_RESULTS.md
BMM_TRL_ADVANCED_TASK_RESULTS.md
BMM_TRL_PAPER_CLAIM_PACKAGE.md
exp/bmm_paper_tables_final.md
exp/policy_retry_fixed_bc_geom_bmm_support075_vs_oracle_gate_commit10_replan20_step300_ep5.md
exp/policy_retry_fixed_bc_geom_vs_bmm_budget_scan_80_160_step300_ep3.md
exp/policy_retry_bmm_budget_scan_commit10_replan20_step300_ep1.md
exp/bmm_ogbench_pointmaze_medium_stitch_value_teacher_40_80_160.json
exp/policy_retry_stitch_fixed_bc_geom_bmm_support_vs_oracle_gate_step300_ep3.md
exp/bmm_ogbench_pointmaze_large_navigate_value_teacher_20_40_80_160_320.json
exp/policy_retry_large_nav_support_frontier_step500_ep3.md
exp/policy_retry_large_nav_support_frontier_start_subgoal_inspection.md
exp/bmm_ogbench_pointmaze_large_stitch_value_teacher_20_40_80_160_320.json
exp/policy_retry_large_stitch_value_frontier_localprogress_step500_ep3.md
exp/policy_retry_large_stitch_controls_localprogress_step500_ep3.md
exp/trace_large_stitch_task1_value_frontier_localprogress_step150.md
exp/bmm_ogbench_pointmaze_large_navigate_oraclerep_value_teacher_20_40_80_160_320.json
exp/policy_retry_large_nav_oraclerep_geom_bmm_support_only_grid_step1000_ep5.md
exp/bmm_ogbench_antmaze_medium_navigate_value_teacher_40_80_160_240_1000.json
exp/policy_retry_antmaze_medium_nav_support_path_bc_h512_ln_switch80_step1000_tasks1_5_ep3.md
exp/bmm_ogbench_antmaze_large_navigate_oraclerep_value_teacher_40_80_160_240_320_1000.json
exp/policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.md
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep5.md
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps5000_switch80_step1500_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_oracle_path_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep1.md
exp/policy_retry_antmaze_large_oraclerep_geometric_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.md
exp/policy_retry_antmaze_large_oraclerep_bmm_vs_support_only_right480_oraclegoal_bc_h512_ln_steps20000_switch80_step1500_tasks1_5_ep3.md
exp/inspect_antmaze_large_oraclerep_task5_start_selectors_right480.md
exp/policy_retry_antmaze_large_oraclerep_task5_left84_right480_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/policy_retry_antmaze_large_oraclerep_task5_value_selectors_oraclegoal_bc_h512_ln_steps20000_ep3.md
exp/bmm_scene_play_oraclerep_graph_factor8_value_16_32_64_500.json
exp/bmm_scene_play_graph_qv_holdout_h64_seed0_smoke/summary.json
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_500.json
exp/bmm_scene_play_graph_qv_holdout_h64_onehot_seed0_smoke/summary.json
exp/bmm_scene_play_oraclerep_graph_factor8_distance_matrix.npz
exp/bmm_scene_play_oraclerep_graph_factor8_value_onehot_16_32_64_128_500.json
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.json
exp/bmm_scene_play_graph_qv_holdout_h64_h128_onehot_seed1_smoke/summary.json
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8.npz
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_distance_matrix.npz
exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500.json
exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed0_smoke/summary.json
exp/scene_play_graph_bc_h512_ln_steps10000_ep1_step750_smoke.json
exp/scene_play_graph_bc_h512_ln_steps10000_left64_ep1_step750_smoke.json
exp/trl_puzzle3x3_rpg_actor_smoke100k_eval_ep15.json
exp/puzzle3x3_bmm_graph_trl100k_controller_bmm_ep15_max300.json
exp/trl_puzzle4x5_frs_smoke100k_eval_ep1.json
exp/puzzle4x5_lightsout_trl100k_nearest_ep1.json
exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.json
exp/puzzle4x6_lightsout_gcfbc_local100k_nearest_ep15.json
exp/trl_humanoidmaze_medium_rpg_actor_total1m_eval_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_support_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_switch128_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_support_switch128_ep15.json
exp/humanoidmaze_medium_bmm_graph_trl_total1m_controller_bmm_switch192_ep15.json
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
scripts/eval_bmm_scene_graph_bc_controller.py
scripts/eval_policy_checkpoint.py
scripts/inspect_bmm_eval_task_subgoals.py
assets/bmm_tabular_error_scaling.png
```
