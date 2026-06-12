# BMM-TRL: Budgeted Max-Min Transitive RL for Offline Goal-Conditioned RL

Date: 2026-06-11  
Status: archival technical report / preliminary research note

This report summarizes the motivation, theory, algorithm, and experimental findings from the BMM-TRL prototype in this repository. It is written as an archive-facing technical note rather than an advisor handoff. The goal is to make the research story clear: what problem we tried to solve, what mathematical property the proposed method targets, what worked empirically, what did not work, and which future directions remain plausible.

---

## Abstract

Transitive RL (TRL) uses a divide-and-conquer value backup for offline goal-conditioned reinforcement learning. In deterministic goal-reaching problems, discounted values can be written as `V*(s,g)=gamma^{d*(s,g)}`, and TRL composes two subproblems with a product backup. This gives a shallow recursive dependency graph, but in log-value or distance space the product backup is still additive. Thus, worst-case approximation bias in the distance estimate can compound as

```text
E(T) <= 2 E(T/2) + epsilon,
```

which remains linear in horizon under uniform residuals.

We investigated **Budgeted Max-Min Transitive RL (BMM-TRL)**, which replaces discounted distance values with finite-budget reachability predicates:

```text
R_H(s,g) = 1[d(s,g) <= H].
```

The associated transitive backup is

```text
R_H(s,g) <- max_w min(R_h(s,w), R_{H-h}(w,g)).
```

Because `max` and `min` are non-expansive in sup norm, the ideal balanced recursion has an error recurrence

```text
E(H) <= epsilon_H + max(E(h), E(H-h)),
```

which yields `O(epsilon log H)` accumulated score error for dyadic balanced budgets. This is the main theoretical motivation.

The project produced a clear empirical split. On clean deterministic or support-reachability targets, the BMM critic learns well and Q/V max-min transitive consistency improves heldout long-budget reachability in budget-holdout diagnostics. However, the current prototype did not translate these critic-level gains into robust policy improvement. Flat action ranking, Q/V joint action-subgoal extraction, and lightweight hierarchical control remained weak or inconclusive. The most defensible current conclusion is that BMM-TRL is promising as a **budgeted reachability and subgoal-planning diagnostic**, but not yet as an end-to-end long-horizon offline RL policy algorithm.

---

## 1. Background and related work

### 1.1 Goal-conditioned RL and offline GCRL

Goal-conditioned reinforcement learning (GCRL) studies policies and value functions conditioned on a goal. Universal Value Function Approximators (UVFAs) introduced the idea of learning value functions that generalize across goals and tasks by conditioning the approximator on the goal representation [Schaul et al., 2015]. Hindsight Experience Replay (HER) made sparse goal-reaching RL more practical by relabeling failed trajectories with goals that were actually achieved [Andrychowicz et al., 2017].

In offline GCRL, the agent must learn from a fixed reward-free dataset. This setting is attractive because unlabeled trajectories can in principle specify many goal-reaching tasks, but it is difficult because the agent must reason about long-horizon stitching and avoid out-of-support actions. OGBench was introduced to benchmark offline GCRL algorithms across many environments and specifically probe stitching, long-horizon reasoning, high-dimensional observations, and stochasticity [Park et al., 2024].

### 1.2 Hierarchical offline GCRL

Long-horizon GCRL naturally suggests subgoal decomposition. HIQL learns a hierarchy in which a high-level policy chooses latent subgoals and a low-level policy reaches those subgoals, motivated by the fact that accurate value estimation is easier for nearby goals than faraway goals [Park et al., 2023]. This perspective is closely related to the final empirical conclusion of this project: BMM's max-min structure appears more natural as a high-level reachability/subgoal-planning mechanism than as a flat one-step action scorer.

### 1.3 Transitive RL

Transitive RL (TRL) proposes a divide-and-conquer value learning rule for offline GCRL. In deterministic goal-reaching, if `d*(s,g)` is a shortest-path distance and `V*(s,g)=gamma^{d*(s,g)}`, then the triangle inequality induces a transitive product backup:

```text
V(s,g) <- max_w V(s,w) V(w,g).
```

TRL's central empirical claim is that divide-and-conquer recursions can help long-horizon value learning by reducing recursion depth compared with one-step temporal-difference backups [Park et al., 2025]. The starting observation for BMM-TRL is that the product backup corresponds to addition in distance/log-value space:

```text
d(s,g) <- min_w d(s,w) + d(w,g).
```

Thus, although the dependency depth is logarithmic under balanced decomposition, worst-case distance-bias accumulation is still additive across branches. BMM-TRL was designed to change the algebra so that branch errors compose by `max` rather than `sum`.

### 1.4 Offline RL constraints

Offline RL methods such as IQL emphasize that value learning from static datasets must avoid evaluating out-of-distribution actions or relying on unsupported policy improvement steps [Kostrikov et al., 2021]. This concern is important for BMM-TRL as well. A reachability target constructed from offline data should not claim true environment reachability outside the data support. The general offline target should instead be **support reachability** over a conservative graph induced by the dataset.

---

## 2. Motivation: from additive distance error to max-min reachability error

### 2.1 Product TRL is additive in distance space

Suppose TRL estimates discounted temporal-distance values

```text
V*(s,g) = gamma^{d*(s,g)}.
```

A product backup has the form

```text
V(s,g) = max_w V(s,w)V(w,g).
```

Taking negative log base `gamma` transforms this into the min-plus distance backup

```text
d(s,g) = min_w d(s,w) + d(w,g).
```

If the two branch distance estimates have worst-case errors `E(h)` and `E(H-h)`, then the parent distance error can be bounded as

```text
E(H) <= epsilon_H + E(h) + E(H-h).
```

For equal splits, this becomes

```text
E(H) <= epsilon_H + 2E(H/2),
```

which is linear in `H` under uniform residuals. This is the core concern that motivated BMM-TRL.

### 2.2 Desired recurrence

We wanted a learned object whose recursion satisfies

```text
E(H) <= epsilon_H + max(E(h), E(H-h)).
```

For dyadic balanced splits, this gives

```text
E(2^k) <= sum_{i=1}^k epsilon_i,
```

and if `epsilon_i <= epsilon`,

```text
E(H) <= epsilon log_2 H.
```

A budgeted reachability predicate has exactly the right algebra.

---

## 3. Deterministic BMM theory

Let the environment be a deterministic graph or deterministic MDP, and let `d*(s,g)` be the shortest-path distance from state `s` to goal `g`.

Define the state reachability predicate

```text
V_H*(s,g) = 1[d*(s,g) <= H].
```

For an action-conditioned critic with deterministic transition `s' = f(s,a)`, define

```text
Q_H*(s,a,g) = 1[d*(f(s,a),g) <= H - 1].
```

### Proposition 1: exact deterministic max-min identity

For any split `h in {0,...,H}`, allowing endpoint witnesses,

```text
V_H*(s,g) = max_w min(V_h*(s,w), V_{H-h}*(w,g)).
```

The action-conditioned identity is

```text
Q_H*(s,a,g) = max_w min(Q_h*(s,a,w), V_{H-h}*(w,g)).
```

#### Proof

If `V_H*(s,g)=1`, there exists a path from `s` to `g` of length `L <= H`. Choose a witness `w` on that path after at most `h` steps. The remaining suffix to `g` has length at most `H-h`, so

```text
d*(s,w) <= h,
d*(w,g) <= H-h.
```

Thus both branch predicates equal one and the max-min value is one.

Conversely, if the max-min value is one, then there exists `w` such that

```text
V_h*(s,w)=1,
V_{H-h}*(w,g)=1.
```

Therefore

```text
d*(s,w) <= h,
d*(w,g) <= H-h.
```

Concatenating the two paths gives a path from `s` to `g` of length at most `H`, so `V_H*(s,g)=1`. The Q identity follows by applying the same argument after the first action transition `f(s,a)`.

### Proposition 2: non-expansive error propagation

Let estimates satisfy

```text
||V_hat_h - V_h*||_infty <= E_h,
||V_hat_{H-h} - V_{H-h}*||_infty <= E_{H-h}.
```

Define

```text
T_hat_H(s,g) = max_w min(V_hat_h(s,w), V_hat_{H-h}(w,g)),
T*_H(s,g)   = max_w min(V_h*(s,w), V_{H-h}*(w,g)).
```

Then

```text
||T_hat_H - T*_H||_infty <= max(E_h, E_{H-h}).
```

#### Proof

For scalars, `min` is 1-Lipschitz under the max norm:

```text
|min(a,b) - min(a',b')| <= max(|a-a'|, |b-b'|).
```

The outer `max_w` is also 1-Lipschitz:

```text
|max_w z_w - max_w z'_w| <= sup_w |z_w - z'_w|.
```

Combining these two inequalities yields the bound.

If the function approximation or regression step introduces a projection residual `epsilon_H`, then

```text
E_H <= epsilon_H + max(E_h, E_{H-h}).
```

For balanced dyadic splits `H=2^k` and uniform residual `epsilon`,

```text
E_H <= k epsilon = O(epsilon log H).
```

This is the theoretical property BMM-TRL was designed to test.

---

## 4. Stochastic environments

The deterministic theory does not automatically extend to true finite-horizon success probability.

Let

```text
P_H*(s,g) = sup_pi Pr_pi[tau_g <= H | s_0=s].
```

In a stochastic environment, a two-stage policy that first reaches `w` with probability `p` and then reaches `g` from `w` with probability `q` has a success probability related to `p q`, not `min(p,q)`. Thus, the max-min operator is not an exact Bellman equation for arbitrary stochastic success probabilities.

There are two principled interpretations where BMM remains meaningful.

### 4.1 Support reachability

Define a support graph by adding an edge `s -> s'` when there exists a supported action `a` such that

```text
P(s' | s,a) > 0.
```

Then define support reachability:

```text
S_H(s,g) = 1[there exists a supported path from s to g of length <= H].
```

This is ordinary graph reachability. The deterministic max-min identity and non-expansive error recurrence apply exactly. This is the most relevant interpretation for offline RL, where unsupported transitions cannot be trusted.

In a dataset setting, we instantiate this as

```text
R_H^D(s,g) = 1[d_D(phi(s), phi(g)) <= H],
```

where `d_D` is the shortest-path distance in a conservative graph built from offline trajectories and `phi` is the state or learned representation.

### 4.2 Reliability-threshold reachability

For stochastic success probabilities, one conservative extension is to threshold probability at a reliability budget. Let `rho = exp(-b)` and define

```text
Z_{H,b}(s,g) = 1[P_H*(s,g) >= exp(-b)].
```

If a branch to `w` succeeds with probability at least `exp(-b1)` and a branch from `w` to `g` succeeds with probability at least `exp(-b2)`, then a two-stage policy has success probability at least `exp(-(b1+b2))` under the usual Markov policy-switching interpretation. Thus,

```text
Z_{H,b}(s,g) >= max_w min(Z_{h,b1}(s,w), Z_{H-h,b2}(w,g))
```

for `b=b1+b2`. This preserves the non-expansive max-min form but is a conservative lower-bound target, not an equality. It also requires tracking a reliability budget in addition to the time budget. This extension was not implemented in the current prototype.

### 4.3 Practical implication

The implemented method should be viewed as BMM for deterministic or support-reachability targets, not as an exact stochastic-probability Bellman method. This distinction is important for future work.

---

## 5. Algorithm

### 5.1 Parameterization

The prototype keeps the TRL-style action-conditioned critic:

```text
R_theta(s,a,g,H) = sigmoid(f_theta(s,a,g,H)).
```

The budget `H` is appended to the goal input through a normalized log-budget feature:

```text
budget_feature = log2(H) / log2(max_budget).
```

The state-only value version is obtained by dropping actions:

```text
V_theta(s,g,H).
```

### 5.2 Direct supervised labels

The clean labels used after the initial logged-offset failure were:

```text
V_H(s,g) = 1[d(s,g) <= H]
Q_H(s_t,a_t,g) = 1[d(s_{t+1},g) <= H-1].
```

The distance `d` was instantiated as:

1. layout/grid geodesic distance, used as an oracle diagnostic; and
2. dataset-position graph distance, used as an offline-support diagnostic.

### 5.3 Q/V transitive target

The main action-conditioned transitive target is

```text
y_QV = max_w min(Q_h(s,a,w), V_{H-h}(w,g)).
```

In the main experiments, the second branch `V_{H-h}` came from a frozen state-value teacher. This separated Q learning from simultaneous V instability.

Because sampled witnesses produce a lower-bound target rather than an exact equality target, later experiments used a lower-bound loss. The default later configuration was

```text
qv_trans_loss_type = bce_lower_bound
lambda_qv_trans = 0.01
num_trans_witnesses = 4
trans_witness_mode = slack_balanced
```

### 5.4 Budget-holdout protocol

The most important diagnostic protocol was budget holdout:

```text
Train direct supervised labels for shorter budgets.
Remove or heavily reduce labels for a longer parent budget.
Use Q/V transitive only for the parent budget.
Evaluate heldout parent-budget reachability.
```

This directly tests whether shorter-budget knowledge composes into longer-budget reachability.

---

## 6. Experiments

All experiments were on `pointmaze-medium-navigate-v0` unless otherwise stated. Detailed logs are in the `BMM_TRL_*.md` files in this repository.

### 6.1 Logged-offset labels failed

The first target was same-trajectory logged offset:

```text
label = 1[offset <= H].
```

This failed at high budgets. Diagnostics showed that the same BMM JAX critic learned deterministic chains, fixed-batch PointMaze overfit worked through `H=512`, but heldout PointMaze logged-offset labels failed at `H=256/512`. kNN on logged-offset labels was near chance: about `0.54` at `H=256` and `0.52` at `H=512`.

Conclusion: logged offset is behavior time, not clean reachability. It should be used as a source of positives, not hard high-budget negatives.

### 6.2 Grid/geodesic labels worked

PointMaze medium exposes its layout. The calibrated grid context was:

```text
median one-step xy displacement = 0.202063
maze unit = 4.0
steps per cell = 19.7958
max grid distance = 11 cells / 217.75 steps
```

Thus `H=256/512` are above the calibrated medium-maze diameter and are one-class under true geodesic reachability.

The state-only geodesic value critic trained on fresh supervised batches and passed heldout thresholds:

| H | AUC | Gap | Ensemble-min AUC | Ensemble-min Gap |
|---:|---:|---:|---:|---:|
| 32 | 0.9344 | 0.4309 | 0.9328 | 0.4266 |
| 64 | 0.9749 | 0.6086 | 0.9743 | 0.6064 |
| 96 | 0.9588 | 0.5916 | 0.9591 | 0.5903 |
| 128 | 0.9751 | 0.5476 | 0.9755 | 0.5402 |

Monotonicity violation was `0.0000`. The action-conditioned geodesic Q critic also passed clean supervised diagnostics using the next-state target.

### 6.3 Transitive consistency was stable but not decisive with abundant labels

State-only V transitive sweeps showed no catastrophic degradation. For example:

| lambda_trans | H | AUC | Gap |
|---:|---:|---:|---:|
| 0.000 | 64 | 0.9636 | 0.5017 |
| 0.000 | 128 | 0.9753 | 0.5479 |
| 0.010 | 64 | 0.9649 | 0.4910 |
| 0.010 | 128 | 0.9792 | 0.5739 |
| 0.025 | 64 | 0.9648 | 0.4914 |
| 0.025 | 128 | 0.9793 | 0.5719 |

Multi-witness targets were limited by witness geometry at small budgets. For `H=64`, only about `1.1` valid witness cells were available on average, so `K=4` mostly repeated the same witness.

### 6.4 Q/V transitive improved consistency but not abundant-label classification

Q/V transitive with a frozen V teacher passed the no-degradation gate and improved Q-V-next consistency:

| Mode | H40 AUC | H40 Gap | H80 AUC | H80 Gap | H160 AUC | H160 Gap | Q-V Next Abs Diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised only | 0.9563 | 0.5359 | 0.9786 | 0.5138 | 0.9875 | 0.6421 | 0.2036 |
| bce_lower_bound | 0.9564 | 0.5058 | 0.9765 | 0.4677 | 0.9857 | 0.6728 | 0.1327 |

Thus Q/V transitive was stable and improved consistency, but abundant direct labels already saturated classification performance.

### 6.5 Budget-holdout: strongest positive result

Budget holdout was the strongest BMM-specific result.

#### Grid-cell H8 holdout

Setup:

```text
supervised budgets = (2,4)
heldout parent = 8
```

Three-seed aggregate:

| Comparison | Seeds | Delta H8 AUC | Delta H8 Gap | Delta H8 BCE | Delta H8 ECE | Delta Q-V Abs | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0150 | +0.0751 | -0.3609 | -0.0575 | -0.0208 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0116 | +0.0507 | -0.1503 | -0.0758 | +0.0011 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0002 | +0.0017 | -0.0063 | -0.0010 | -0.0005 | V-next distill control |

#### Env-step H160 holdout

Setup:

```text
supervised budgets = (40,80)
heldout parent = 160
```

Three-seed aggregate:

| Comparison | Seeds | Delta H160 AUC | Delta H160 Gap | Delta H160 BCE | Delta H160 ECE | Delta Q-V Abs | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B-A | 0,1,2 | +0.0243 | +0.0647 | -0.5815 | -0.0380 | -0.0102 | no-parent BMM effect |
| D-C | 0,1,2 | +0.0041 | +0.0498 | -0.1815 | -0.0547 | -0.0293 | few-parent BMM effect |
| F-A | 0,1,2 | +0.0035 | +0.0030 | -0.0570 | -0.0016 | +0.0026 | V-next distill control |

Conclusion: shorter-budget Q/V knowledge helps heldout longer-budget parent classification. The V-next control is much smaller, suggesting the gain is BMM-specific rather than generic teacher smoothing.

### 6.6 Dataset-support graph labels

A dataset-position graph was constructed from observed transitions:

```text
nodes = 1698
edges = 5548
connected components = 1
graph diameter = 73 hops / 146 calibrated steps
```

Graph labels were clean and learnable for budgets below graph diameter. A graph-label budget-holdout diagnostic was mildly positive:

| Variant | H120 AUC | H120 Gap | Q-V Abs Diff | Q-V Rank Corr |
|---|---:|---:|---:|---:|
| A supervised-only | 0.9783 | 0.3459 | 0.3306 | 0.7802 |
| B Q/V transitive | 0.9849 | 0.3411 | 0.3277 | 0.8084 |
| F V-next distill | 0.9780 | 0.3491 | 0.3265 | 0.7802 |

This weakens the concern that BMM only works with grid-oracle labels, although the graph result is still diagnostic and modest.

### 6.7 Policy-facing results

#### Flat action ranking

Flat action ranking did not improve. On the credible own-state action-ranking mode:

| Critic | AUC | Pair Acc | Selected Distance | Selected Success |
|---|---:|---:|---:|---:|
| A | 0.6307 | 0.7814 | 164.2045 | 0.6738 |
| B | 0.6170 | 0.7890 | 163.8565 | 0.6895 |
| F | 0.6321 | 0.7827 | 164.1272 | 0.6777 |

BMM did not robustly beat A/F on action AUC.

#### Joint action-subgoal selection

Candidate coverage was improved with neighbor-cell, directional, and oracle-diverse candidate modes. Coverage was no longer the blocker. Even then, BMM Q/V did not robustly win the full joint objective. It often improved action-valid metrics but not state-valid, path stretch, or midpoint quality.

#### Value-only subgoal selection

Value-only subgoal selection was more promising:

```text
score(w) = min(V_h(s,w), V_{H-h}(w,g)).
```

With a same-cell nearest-neighbor controller:

| Selector | Success | Final Distance | Improve | Mean Step Goal | Subgoal Valid |
|---|---:|---:|---:|---:|---:|
| random | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.1333 |
| geometric_midpoint | 0.0000 | 112.1761 | 59.3874 | 0.5939 | 0.3833 |
| BMM_V | 0.0000 | 105.5775 | 65.9860 | 0.6599 | 0.6633 |
| oracle_midpoint | 0.0000 | 118.7747 | 52.7888 | 0.5279 | 0.4267 |

The graph-support subgoal diagnostic was also positive:

| Scorer | State Valid | Path Stretch | Midpoint Err |
|---|---:|---:|---:|
| random | 0.7422 | 29.0078 | 57.9219 |
| euclidean_midpoint | 0.9902 | 6.8711 | 68.5508 |
| oracle_graph_midpoint | 0.6387 | 65.6875 | 15.2422 |
| BMM_V_graph | 0.9902 | 4.7734 | 70.6406 |

However, the hierarchical pivot did not clear the final go/no-go criterion. With a stronger 5k-step, larger BC controller:

| Selector | Success | Final Distance | Improve | Subgoal Valid | Subgoal Reduce | Goal Reduce |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0000 | 134.6113 | 39.5916 | 0.1020 | 0.0400 | 0.0320 |
| geometric_midpoint | 0.0000 | 106.8972 | 67.3057 | 0.2820 | 0.0360 | 0.0360 |
| BMM_V | 0.0000 | 106.8972 | 67.3057 | 0.6100 | 0.0340 | 0.0340 |
| oracle_midpoint | 0.0000 | 122.7339 | 51.4690 | 0.1960 | 0.2800 | 0.1720 |

BMM/V tied geometric midpoint on final distance and improvement. Therefore the current hierarchical policy route did not pass its continue gate.

---

## 7. Discussion

### 7.1 Strengths found

1. **Clear theoretical property.** BMM's max-min reachability operator has the desired non-expansive error recurrence in deterministic/support settings.
2. **Target diagnosis.** The project identified a key failure mode: same-trajectory logged offsets are behavior-time labels, not high-budget reachability labels.
3. **Clean labels are learnable.** Geodesic and support-graph reachability labels are learnable with the current architecture.
4. **Budget-holdout gains.** Q/V transitive bootstrapping improves heldout long-budget reachability across seeds.
5. **BMM-specific control.** V-next distillation does not explain the budget-holdout gains.
6. **Subgoal signal.** Value-level BMM subgoal selection gives useful intermediate goals in diagnostics.

### 7.2 Limitations found

1. **No robust policy improvement.** The prototype did not produce a robust policy-improvement path.
2. **Flat Q extraction failed.** Better long-budget reachability did not translate into better one-step action ranking.
3. **Joint Q/V action-subgoal extraction was mixed.** Even with better candidates, it did not robustly beat controls.
4. **Controller bottleneck.** Hierarchical use depends strongly on the low-level controller.
5. **Grid labels are diagnostic, not general.** Dataset-support graph labels are the more general target, but still require good representation and graph construction.
6. **Stochastic extension remains theoretical.** The current implementation is deterministic/support reachability, not full stochastic success probability.

---

## 8. Future directions

### 8.1 Support-graph BMM as the main offline target

The most important future direction is to replace grid/geodesic oracles with train-only support graphs:

```text
nodes: offline states or learned latent clusters
edges: observed transitions and conservative support-preserving stitching edges
target: R_H^D(s,g)=1[d_D(phi(s),phi(g)) <= H]
```

A strong future result would show budget-holdout gains on train-only support graphs with heldout validation states mapped conservatively to graph nodes.

### 8.2 Latent support graphs

For more general offline domains, raw coordinates are not available. Future work should learn representations `phi(s)` for graph construction using temporal contrastive learning, inverse-dynamics-aware representation learning, or goal-conditioned value embeddings. BMM's target quality will likely depend heavily on representation quality.

### 8.3 Stochastic reliability budgets

The deterministic max-min identity can be extended conservatively to stochastic settings with an additional reliability budget. This direction would learn thresholded success-probability predicates rather than binary support reachability. It is mathematically promising but more complex than the current prototype.

### 8.4 Hierarchical planning with a strong low-level controller

The most plausible policy-facing direction is to use BMM/V for high-level subgoal planning and pair it with a separately trained low-level controller. A serious version would require comparison to HIQL/HIGL-style baselines and geometric/graph midpoint planners.

### 8.5 Better subgoal proposal

Current diagnostics often sample broad candidate sets. A deployable algorithm needs learned or graph-based proposal mechanisms that produce useful candidate subgoals without oracle grid knowledge.

### 8.6 Theory with projection error and finite samples

The clean `O(log H)` result assumes sup-norm residual control. Future theory should analyze projection error, finite-sample classification error, witness sampling error, and conservative lower-bound losses.

---

## 9. Conclusion

BMM-TRL was motivated by a simple but important algebraic observation: if value composition is additive in distance space, then branch errors can add. By learning budgeted reachability and composing with max-min, the ideal deterministic/support operator has logarithmic-depth sup-norm error accumulation.

The prototype validated the reachability side of this idea. Clean geodesic and support-graph targets are learnable, and Q/V max-min transitive consistency improves heldout long-budget reachability in budget-holdout diagnostics. This is the strongest scientific outcome of the project.

The prototype did not validate BMM as an end-to-end offline RL policy method. Flat action ranking, joint Q/V action-subgoal extraction, and lightweight hierarchical control did not produce robust policy gains. The current evidence supports pausing active policy-facing experimentation and preserving the work as a critic/reachability/subgoal-planning technical result, or formally pivoting to a new hierarchical RL project with a stronger low-level controller.

---

## References

- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., and Zaremba, W. **Hindsight Experience Replay.** NeurIPS, 2017. https://arxiv.org/abs/1707.01495
- Bellman, R. **Dynamic Programming.** Princeton University Press, 1957.
- Bertsekas, D. P. **Dynamic Programming and Optimal Control.** Athena Scientific.
- Kostrikov, I., Nair, A., and Levine, S. **Offline Reinforcement Learning with Implicit Q-Learning.** ICLR, 2022. https://arxiv.org/abs/2110.06169
- Park, S., Ghosh, D., Eysenbach, B., and Levine, S. **HIQL: Offline Goal-Conditioned RL with Latent States as Actions.** NeurIPS, 2023. https://arxiv.org/abs/2307.11949
- Park, S., Frans, K., Eysenbach, B., and Levine, S. **OGBench: Benchmarking Offline Goal-Conditioned RL.** 2024. https://arxiv.org/abs/2410.20092
- Park, S., Oberai, A., Atreya, P., and Levine, S. **Transitive RL: Value Learning via Divide and Conquer.** 2025. https://arxiv.org/abs/2510.22512
- Schaul, T., Horgan, D., Gregor, K., and Silver, D. **Universal Value Function Approximators.** ICML, 2015. https://proceedings.mlr.press/v37/schaul15.html

---

## Key repository artifacts

Result files:

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
```

Implementation files:

```text
agents/bmm_trl.py
utils/pointmaze_grid.py
utils/pointmaze_graph.py
scripts/train_bmm_geodesic_value.py
scripts/train_bmm_geodesic_q.py
scripts/run_bmm_qv_budget_holdout.py
scripts/eval_bmm_action_ranking.py
scripts/eval_bmm_joint_action_subgoal.py
scripts/eval_bmm_value_subgoal_controller.py
scripts/eval_bmm_value_subgoal_policy_smoke.py
scripts/eval_bmm_graph_value_subgoal.py
scripts/eval_bmm_subgoal_bc_controller.py
```
