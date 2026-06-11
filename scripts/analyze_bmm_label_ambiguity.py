#!/usr/bin/env python
"""Analyze whether BMM logged-offset labels are locally identifiable.

For each budget, this script samples balanced same-trajectory pairs, builds
feature vectors, and predicts heldout labels with k-nearest-neighbor label
averages. If kNN is weak at high budgets, the logged-offset label is probably
not a clean heldout function of the critic inputs.
"""

import json
from pathlib import Path
import random
import sys

import numpy as np
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import (
    binary_auc,
    format_metric,
    sample_balanced_budget_pairs,
)
from utils.datasets import Dataset, GCDataset


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional directory of OGBench npz files.")
flags.DEFINE_string("budgets", "64,128,256,512", "Comma-separated budgets.")
flags.DEFINE_integer("num_train_pairs", 100000, "Training pairs per budget.")
flags.DEFINE_integer("num_eval_pairs", 20000, "Evaluation pairs per budget.")
flags.DEFINE_integer("k", 32, "Number of nearest neighbors.")
flags.DEFINE_string(
    "features",
    "xy_pair,xy_delta,full_pair,full_pair_action",
    "Comma-separated feature modes.",
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_chunk_size", 256, "Eval chunk size for kNN.")
flags.DEFINE_integer("train_chunk_size", 50000, "Train chunk size for near-duplicate counts.")
flags.DEFINE_float("near_xy_radius", 0.05, "Raw xy-pair radius for near duplicates.")
flags.DEFINE_float(
    "balanced_pos_boundary_frac",
    0.5,
    "Positive offset lower bound as a fraction of each budget.",
)
flags.DEFINE_float(
    "balanced_neg_max_factor",
    2.0,
    "Negative offset upper bound as a factor of each budget.",
)
flags.DEFINE_string("output_json", None, "Optional JSON output path.")

config_flags.DEFINE_config_file("agent", "agents/bmm_trl.py", lock_config=False)


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def make_gc_datasets(config):
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    return (
        dataset_class(Dataset.create(**train_dataset), config),
        dataset_class(Dataset.create(**val_dataset), config),
    )


def sample_pairs(gc_dataset, budget, num_pairs, rng):
    pair_batch = sample_balanced_budget_pairs(
        gc_dataset,
        int(budget),
        int(num_pairs),
        rng,
        pos_boundary_frac=FLAGS.balanced_pos_boundary_frac,
        neg_max_factor=FLAGS.balanced_neg_max_factor,
    )
    if pair_batch is None:
        raise ValueError(f"No balanced pairs available for H={budget}.")
    return pair_batch


def xy_pair(pair_batch):
    observations = np.asarray(pair_batch["observations"], dtype=np.float32)
    goals = np.asarray(pair_batch["goals"], dtype=np.float32)
    if observations.shape[-1] < 2 or goals.shape[-1] < 2:
        raise ValueError("xy features require observation and goal dims >= 2.")
    return np.concatenate([observations[:, :2], goals[:, :2]], axis=-1)


def features_for(pair_batch, mode):
    observations = np.asarray(pair_batch["observations"], dtype=np.float32)
    goals = np.asarray(pair_batch["goals"], dtype=np.float32)
    actions = np.asarray(pair_batch["actions"], dtype=np.float32)

    if mode == "xy_pair":
        return xy_pair(pair_batch)
    if mode == "xy_delta":
        obs_xy = observations[:, :2]
        goal_xy = goals[:, :2]
        return np.concatenate([obs_xy, goal_xy, goal_xy - obs_xy], axis=-1)
    if mode == "full_pair":
        return np.concatenate([observations, goals], axis=-1)
    if mode == "full_pair_action":
        return np.concatenate([observations, actions, goals], axis=-1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def standardize(train_features, eval_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((eval_features - mean) / std).astype(np.float32),
    )


def entropy(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def knn_label_probs(train_features, train_labels, eval_features, k, eval_chunk_size):
    """Return kNN positive-label probabilities using chunked NumPy distances."""
    train_features = np.asarray(train_features, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    k = min(int(k), len(train_features))
    train_norms = np.sum(train_features * train_features, axis=1)
    probs = np.empty(len(eval_features), dtype=np.float32)

    for start in range(0, len(eval_features), eval_chunk_size):
        end = min(start + eval_chunk_size, len(eval_features))
        chunk = eval_features[start:end]
        dists = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + train_norms[None, :]
            - 2.0 * chunk @ train_features.T
        )
        neighbor_idxs = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        probs[start:end] = train_labels[neighbor_idxs].mean(axis=1)

    return probs


def near_duplicate_stats(train_pair_xy, train_labels, eval_pair_xy, radius, train_chunk_size):
    """Measure mixed labels among raw xy-pair near duplicates."""
    train_pair_xy = np.asarray(train_pair_xy, dtype=np.float32)
    eval_pair_xy = np.asarray(eval_pair_xy, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    radius_sq = float(radius) ** 2
    counts = np.zeros(len(eval_pair_xy), dtype=np.int32)
    pos_counts = np.zeros(len(eval_pair_xy), dtype=np.float32)

    for start in range(0, len(train_pair_xy), train_chunk_size):
        end = min(start + train_chunk_size, len(train_pair_xy))
        train_chunk = train_pair_xy[start:end]
        dists = (
            np.sum(eval_pair_xy * eval_pair_xy, axis=1, keepdims=True)
            + np.sum(train_chunk * train_chunk, axis=1)[None, :]
            - 2.0 * eval_pair_xy @ train_chunk.T
        )
        near = dists <= radius_sq
        counts += near.sum(axis=1).astype(np.int32)
        pos_counts += near @ train_labels[start:end]

    has_near = counts > 0
    if not has_near.any():
        return dict(
            near_duplicate_count=0,
            near_duplicate_frac=0.0,
            near_duplicate_mixed_frac=np.nan,
            near_duplicate_mean_count=0.0,
        )
    near_probs = pos_counts[has_near] / counts[has_near]
    mixed = (near_probs >= 0.25) & (near_probs <= 0.75)
    return dict(
        near_duplicate_count=int(has_near.sum()),
        near_duplicate_frac=float(has_near.mean()),
        near_duplicate_mixed_frac=float(mixed.mean()),
        near_duplicate_mean_count=float(counts[has_near].mean()),
    )


def metrics_from_probs(probs, labels):
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    pos_mask = labels == 1.0
    neg_mask = ~pos_mask
    pos_mean = float(probs[pos_mask].mean()) if pos_mask.any() else np.nan
    neg_mean = float(probs[neg_mask].mean()) if neg_mask.any() else np.nan
    ent = entropy(probs)
    return dict(
        knn_auc=float(binary_auc(probs, labels)),
        knn_gap=float(pos_mean - neg_mean)
        if np.isfinite(pos_mean) and np.isfinite(neg_mean)
        else np.nan,
        knn_pos_mean=pos_mean,
        knn_neg_mean=neg_mean,
        mean_neighbor_entropy=float(ent.mean()),
        median_neighbor_entropy=float(np.median(ent)),
        contradiction_rate_25_75=float(((probs >= 0.25) & (probs <= 0.75)).mean()),
        contradiction_rate_10_90=float(((probs >= 0.10) & (probs <= 0.90)).mean()),
        pos_count=int(pos_mask.sum()),
        neg_count=int(neg_mask.sum()),
    )


def print_budget_report(budget, rows):
    print(f"\nH={budget}")
    print(
        "feature | auc | gap | entropy | c25_75 | c10_90 | near_n | near_mixed"
    )
    print(
        "--------|-----|-----|---------|--------|--------|--------|-----------"
    )
    for row in rows:
        near = row["near_duplicates"]
        print(
            f"{row['feature']:16s} | {format_metric(row['knn_auc'])} | "
            f"{format_metric(row['knn_gap'])} | "
            f"{format_metric(row['mean_neighbor_entropy'])} | "
            f"{format_metric(row['contradiction_rate_25_75'])} | "
            f"{format_metric(row['contradiction_rate_10_90'])} | "
            f"{near['near_duplicate_count']:6d} | "
            f"{format_metric(near['near_duplicate_mixed_frac'])}"
        )


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed)
    config = FLAGS.agent
    if config["agent_name"] != "bmm_trl":
        raise ValueError("analyze_bmm_label_ambiguity.py requires bmm_trl.")

    budgets = parse_int_list(FLAGS.budgets)
    feature_modes = parse_str_list(FLAGS.features)
    train_dataset, val_dataset = make_gc_datasets(config)

    report = dict(
        env_name=FLAGS.env_name,
        budgets=budgets,
        num_train_pairs=int(FLAGS.num_train_pairs),
        num_eval_pairs=int(FLAGS.num_eval_pairs),
        k=int(FLAGS.k),
        features=feature_modes,
        near_xy_radius=float(FLAGS.near_xy_radius),
        rows=[],
    )

    for budget in budgets:
        train_pairs = sample_pairs(train_dataset, budget, FLAGS.num_train_pairs, rng)
        eval_pairs = sample_pairs(val_dataset, budget, FLAGS.num_eval_pairs, rng)
        train_labels = np.asarray(train_pairs["labels"], dtype=np.float32)
        eval_labels = np.asarray(eval_pairs["labels"], dtype=np.float32)
        train_xy = xy_pair(train_pairs)
        eval_xy = xy_pair(eval_pairs)
        budget_rows = []

        for mode in feature_modes:
            train_features = features_for(train_pairs, mode)
            eval_features = features_for(eval_pairs, mode)
            train_features, eval_features = standardize(train_features, eval_features)
            probs = knn_label_probs(
                train_features,
                train_labels,
                eval_features,
                FLAGS.k,
                FLAGS.eval_chunk_size,
            )
            row = dict(
                budget=int(budget),
                feature=mode,
                **metrics_from_probs(probs, eval_labels),
                near_duplicates=near_duplicate_stats(
                    train_xy,
                    train_labels,
                    eval_xy,
                    FLAGS.near_xy_radius,
                    FLAGS.train_chunk_size,
                ),
            )
            report["rows"].append(row)
            budget_rows.append(row)

        print_budget_report(budget, budget_rows)

    if FLAGS.output_json is not None:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote label ambiguity report to {output_path}")


if __name__ == "__main__":
    app.run(main)
