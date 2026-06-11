"""Utilities for BMM-TRL reachability diagnostics."""

from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.datasets import get_bmm_budgets


def get_goal_vectors(gc_dataset, idxs):
    """Return goal vectors matching the representation used by GCDataset."""
    if "oracle_reps" in gc_dataset.dataset:
        return gc_dataset.dataset["oracle_reps"][idxs]
    return gc_dataset.get_observations(idxs)


def trajectory_remaining(gc_dataset, idxs):
    """Return the number of future steps available in each sampled trajectory."""
    final_idxs = gc_dataset.terminal_locs[
        np.searchsorted(gc_dataset.terminal_locs, idxs)
    ]
    return final_idxs - idxs


def sample_random_pairs(gc_dataset, num_pairs, max_offset, rng):
    """Sample same-trajectory pairs with uniformly sampled positive offsets."""
    valid_idxs = gc_dataset.dataset.valid_idxs
    remaining = trajectory_remaining(gc_dataset, valid_idxs)
    eligible = valid_idxs[remaining >= 1]
    if len(eligible) == 0:
        raise ValueError("No eligible non-terminal validation states found.")

    idxs = rng.choice(eligible, size=num_pairs, replace=True)
    pair_remaining = trajectory_remaining(gc_dataset, idxs)
    capped = np.minimum(pair_remaining, max_offset)
    offsets = 1 + np.floor(rng.random(num_pairs) * capped).astype(np.int32)
    goal_idxs = idxs + offsets
    return idxs.astype(np.int32), goal_idxs.astype(np.int32), offsets.astype(np.int32)


def sample_fixed_offset_pairs(gc_dataset, offset, num_pairs, rng):
    """Sample same-trajectory pairs with a fixed offset, if possible."""
    valid_idxs = gc_dataset.dataset.valid_idxs
    remaining = trajectory_remaining(gc_dataset, valid_idxs)
    eligible = valid_idxs[remaining >= offset]
    if len(eligible) == 0:
        return None

    idxs = rng.choice(eligible, size=num_pairs, replace=True)
    goal_idxs = idxs + offset
    offsets = np.full(num_pairs, offset, dtype=np.int32)
    return idxs.astype(np.int32), goal_idxs.astype(np.int32), offsets


def make_pair_batch(gc_dataset, idxs, goal_idxs, offsets, budget):
    """Build observations/actions/goals/labels for one diagnostic budget."""
    budgets = np.full(len(idxs), budget, dtype=np.int32)
    labels = (offsets <= budget).astype(np.float32)
    return dict(
        observations=gc_dataset.get_observations(idxs),
        actions=gc_dataset.dataset["actions"][idxs],
        goals=get_goal_vectors(gc_dataset, goal_idxs),
        budgets=budgets,
        offsets=offsets,
        labels=labels,
    )


def sample_balanced_budget_pairs(
    gc_dataset,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
):
    """Sample near-boundary positives and negatives for one budget."""
    valid_idxs = gc_dataset.dataset.valid_idxs
    remaining = trajectory_remaining(gc_dataset, valid_idxs)
    budget = int(budget)

    num_pos = num_pairs // 2
    num_neg = num_pairs - num_pos
    srcs = []
    goals = []
    offsets = []

    min_pos_remaining = max(1, int(np.ceil(pos_boundary_frac * budget)))
    pos_eligible = valid_idxs[remaining >= min_pos_remaining]
    if len(pos_eligible) == 0:
        pos_eligible = valid_idxs[remaining >= 1]

    for _ in range(num_pos):
        if len(pos_eligible) == 0:
            break
        src = int(rng.choice(pos_eligible))
        rem = int(trajectory_remaining(gc_dataset, np.asarray([src]))[0])
        hi = min(budget, rem)
        if hi < 1:
            continue
        lo = max(1, int(np.ceil(pos_boundary_frac * budget)))
        lo = min(lo, hi)
        offset = int(rng.integers(lo, hi + 1))
        srcs.append(src)
        goals.append(src + offset)
        offsets.append(offset)

    neg_eligible = valid_idxs[remaining >= budget + 1]
    for _ in range(num_neg):
        if len(neg_eligible) == 0:
            break
        src = int(rng.choice(neg_eligible))
        rem = int(trajectory_remaining(gc_dataset, np.asarray([src]))[0])
        lo = budget + 1
        hi = min(rem, int(np.floor(neg_max_factor * budget)))
        if lo > hi:
            hi = rem
        if lo > hi:
            continue
        offset = int(rng.integers(lo, hi + 1))
        srcs.append(src)
        goals.append(src + offset)
        offsets.append(offset)

    if len(srcs) == 0:
        return None

    return make_pair_batch(
        gc_dataset,
        np.asarray(srcs, dtype=np.int32),
        np.asarray(goals, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
        budget,
    )


def score_pair_batch(agent, pair_batch, batch_size=8192):
    """Score diagnostic pairs with mean and min ensemble sigmoid scores."""
    mean_scores = []
    min_scores = []
    num_pairs = len(pair_batch["offsets"])

    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        observations = pair_batch["observations"][start:end]
        actions = pair_batch["actions"][start:end]
        goals = pair_batch["goals"][start:end]
        budgets = pair_batch["budgets"][start:end]
        offsets = pair_batch["offsets"][start:end]
        logits = agent.critic_logits_for(
            observations,
            actions,
            goals,
            budgets,
            offsets=offsets,
        )
        scores = np.asarray(jax.nn.sigmoid(logits))
        mean_scores.append(scores.mean(axis=0))
        min_scores.append(scores.min(axis=0))

    return np.concatenate(mean_scores), np.concatenate(min_scores)


def rank_metrics(scores, labels):
    """Return AUC and class means for arbitrary rank scores."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos_mask = labels == 1.0
    neg_mask = ~pos_mask
    pos_mean = float(scores[pos_mask].mean()) if pos_mask.any() else np.nan
    neg_mean = float(scores[neg_mask].mean()) if neg_mask.any() else np.nan
    return dict(
        auc=float(binary_auc(scores, labels)),
        pos_mean=pos_mean,
        neg_mean=neg_mean,
        gap=float(pos_mean - neg_mean)
        if np.isfinite(pos_mean) and np.isfinite(neg_mean)
        else np.nan,
        pos_count=int(pos_mask.sum()),
        neg_count=int(neg_mask.sum()),
    )


def baseline_metrics(pair_batch):
    """Compute simple non-neural reachability ranking baselines."""
    labels = pair_batch["labels"]
    offsets = np.asarray(pair_batch["offsets"], dtype=np.float64)
    observations = np.asarray(pair_batch["observations"], dtype=np.float64)
    goals = np.asarray(pair_batch["goals"], dtype=np.float64)
    actions = np.asarray(pair_batch["actions"], dtype=np.float64)

    dim = min(2, observations.shape[-1], goals.shape[-1])
    deltas = goals[..., :dim] - observations[..., :dim]
    euclidean_scores = -np.linalg.norm(deltas, axis=-1)

    action_dim = min(dim, actions.shape[-1])
    action_goal_scores = np.sum(
        actions[..., :action_dim] * deltas[..., :action_dim], axis=-1
    )

    return dict(
        offset_oracle=rank_metrics(-offsets, labels),
        euclidean=rank_metrics(euclidean_scores, labels),
        action_goal=rank_metrics(action_goal_scores, labels),
    )


def binary_auc(scores, labels):
    """Compute ROC AUC with average ranks for ties; return nan if undefined."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    num_pos = int(labels.sum())
    num_neg = int((~labels).sum())
    if num_pos == 0 or num_neg == 0:
        return np.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels].sum()
    return (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)


def binary_metrics(scores, labels):
    """Return BCE, accuracy, score means, counts, and AUC."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    pos_mask = labels == 1.0
    neg_mask = ~pos_mask
    preds = scores >= 0.5
    return dict(
        bce=float(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)).mean()),
        accuracy=float((preds == pos_mask).mean()),
        pos_mean=float(scores[pos_mask].mean()) if pos_mask.any() else np.nan,
        neg_mean=float(scores[neg_mask].mean()) if neg_mask.any() else np.nan,
        auc=float(binary_auc(scores, labels)),
        pos_count=int(pos_mask.sum()),
        neg_count=int(neg_mask.sum()),
    )


def evaluate_reachability(
    agent,
    gc_dataset,
    num_pairs=4096,
    balanced_pairs=4096,
    score_batch_size=8192,
    max_offset_factor=4,
    bucket_samples=512,
    balanced_pos_boundary_frac=0.5,
    balanced_neg_max_factor=2.0,
    rng=None,
):
    """Evaluate budget-conditioned reachability on validation trajectory pairs."""
    rng = np.random.default_rng(0) if rng is None else rng
    budgets = get_bmm_budgets(agent.config)
    max_remaining = int(trajectory_remaining(gc_dataset, gc_dataset.dataset.valid_idxs).max())
    max_offset = max(1, min(max_remaining, int(budgets[-1] * max_offset_factor)))
    idxs, goal_idxs, offsets = sample_random_pairs(
        gc_dataset, num_pairs, max_offset, rng
    )

    budget_rows = []
    balanced_budget_rows = []
    mean_score_by_budget = []
    min_score_by_budget = []
    for budget in budgets:
        pair_batch = make_pair_batch(gc_dataset, idxs, goal_idxs, offsets, int(budget))
        mean_scores, min_scores = score_pair_batch(agent, pair_batch, score_batch_size)
        labels = pair_batch["labels"]
        mean_metrics = binary_metrics(mean_scores, labels)
        min_metrics = binary_metrics(min_scores, labels)
        budget_rows.append(
            dict(
                budget=int(budget),
                mean=mean_metrics,
                ensemble_min=min_metrics,
                baselines=baseline_metrics(pair_batch),
            )
        )
        mean_score_by_budget.append(mean_scores)
        min_score_by_budget.append(min_scores)

        balanced_batch = sample_balanced_budget_pairs(
            gc_dataset,
            int(budget),
            balanced_pairs,
            rng,
            pos_boundary_frac=balanced_pos_boundary_frac,
            neg_max_factor=balanced_neg_max_factor,
        )
        if balanced_batch is not None:
            balanced_mean_scores, balanced_min_scores = score_pair_batch(
                agent, balanced_batch, score_batch_size
            )
            balanced_labels = balanced_batch["labels"]
            balanced_budget_rows.append(
                dict(
                    budget=int(budget),
                    mean=binary_metrics(balanced_mean_scores, balanced_labels),
                    ensemble_min=binary_metrics(
                        balanced_min_scores, balanced_labels
                    ),
                    baselines=baseline_metrics(balanced_batch),
                )
            )

    mean_matrix = np.stack(mean_score_by_budget, axis=0)
    min_matrix = np.stack(min_score_by_budget, axis=0)
    if len(budgets) > 1:
        mean_mono = float((mean_matrix[:-1] > mean_matrix[1:] + 1e-6).mean())
        min_mono = float((min_matrix[:-1] > min_matrix[1:] + 1e-6).mean())
    else:
        mean_mono = 0.0
        min_mono = 0.0

    bucket_rows = []
    multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    for budget in budgets:
        for multiplier in multipliers:
            offset = max(1, int(round(int(budget) * multiplier)))
            fixed = sample_fixed_offset_pairs(gc_dataset, offset, bucket_samples, rng)
            if fixed is None:
                continue
            bucket_batch = make_pair_batch(gc_dataset, *fixed, budget=int(budget))
            mean_scores, min_scores = score_pair_batch(
                agent, bucket_batch, score_batch_size
            )
            labels = bucket_batch["labels"]
            bucket_rows.append(
                dict(
                    budget=int(budget),
                    offset=int(offset),
                    multiplier=float(multiplier),
                    label_mean=float(labels.mean()),
                    count=int(len(labels)),
                    mean_score=float(mean_scores.mean()),
                    min_score=float(min_scores.mean()),
                )
            )

    return dict(
        budgets=[int(x) for x in budgets],
        max_offset=max_offset,
        random_pair_count=int(num_pairs),
        balanced_pair_count=int(balanced_pairs),
        mean_monotonicity_violation=mean_mono,
        min_monotonicity_violation=min_mono,
        budget_rows=budget_rows,
        balanced_budget_rows=balanced_budget_rows,
        bucket_rows=bucket_rows,
    )


def format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"
