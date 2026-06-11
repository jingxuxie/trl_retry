#!/usr/bin/env python
"""Evaluate a BMM-TRL checkpoint as a budgeted reachability classifier."""

import csv
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

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import evaluate_reachability, format_metric
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional directory of OGBench npz files.")
flags.DEFINE_string("restore_path", None, "Directory or glob containing saved params.")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint epoch to restore.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_pairs", 4096, "Random validation pairs per budget.")
flags.DEFINE_integer(
    "balanced_pairs", 4096, "Balanced near-boundary validation pairs per budget."
)
flags.DEFINE_integer("score_batch_size", 8192, "Batch size for critic scoring.")
flags.DEFINE_integer("bucket_samples", 512, "Pairs per offset bucket.")
flags.DEFINE_integer("max_offset_factor", 4, "Max random offset as factor of max budget.")
flags.DEFINE_string("output_json", None, "Optional path for JSON diagnostics.")
flags.DEFINE_string("output_csv", None, "Optional path for flat CSV diagnostics.")

config_flags.DEFINE_config_file("agent", "agents/bmm_trl.py", lock_config=False)


def print_report(report):
    print("\nBMM reachability diagnostics")
    print(f"Random pairs: {report['random_pair_count']}")
    print(f"Balanced pairs: {report.get('balanced_pair_count', 0)}")
    print(f"Max sampled offset: {report['max_offset']}")
    print(
        "Monotonicity violation: "
        f"mean={format_metric(report['mean_monotonicity_violation'])}, "
        f"ensemble_min={format_metric(report['min_monotonicity_violation'])}"
    )

    print("\nBy budget")
    print(
        "H | bce | acc | pos | neg | auc | min_bce | min_acc | min_pos | min_neg | min_auc | pos_n | neg_n"
    )
    print(
        "--|-----|-----|-----|-----|-----|---------|---------|---------|---------|---------|-------|------"
    )
    for row in report["budget_rows"]:
        mean = row["mean"]
        ens_min = row["ensemble_min"]
        print(
            f"{row['budget']:4d} | {format_metric(mean['bce'])} | "
            f"{format_metric(mean['accuracy'])} | {format_metric(mean['pos_mean'])} | "
            f"{format_metric(mean['neg_mean'])} | {format_metric(mean['auc'])} | "
            f"{format_metric(ens_min['bce'])} | {format_metric(ens_min['accuracy'])} | "
            f"{format_metric(ens_min['pos_mean'])} | {format_metric(ens_min['neg_mean'])} | "
            f"{format_metric(ens_min['auc'])} | {mean['pos_count']:5d} | {mean['neg_count']:5d}"
        )

    if report.get("balanced_budget_rows"):
        print("\nBalanced near-boundary by budget")
        print(
            "H | bce | acc | pos | neg | auc | min_bce | min_acc | min_pos | min_neg | min_auc | pos_n | neg_n"
        )
        print(
            "--|-----|-----|-----|-----|-----|---------|---------|---------|---------|---------|-------|------"
        )
        for row in report["balanced_budget_rows"]:
            mean = row["mean"]
            ens_min = row["ensemble_min"]
            print(
                f"{row['budget']:4d} | {format_metric(mean['bce'])} | "
                f"{format_metric(mean['accuracy'])} | {format_metric(mean['pos_mean'])} | "
                f"{format_metric(mean['neg_mean'])} | {format_metric(mean['auc'])} | "
                f"{format_metric(ens_min['bce'])} | {format_metric(ens_min['accuracy'])} | "
                f"{format_metric(ens_min['pos_mean'])} | {format_metric(ens_min['neg_mean'])} | "
                f"{format_metric(ens_min['auc'])} | {mean['pos_count']:5d} | {mean['neg_count']:5d}"
            )

        print("\nBalanced baseline AUCs")
        print("H | offset_oracle | euclidean | action_goal")
        print("--|---------------|-----------|------------")
        for row in report["balanced_budget_rows"]:
            baselines = row.get("baselines", {})
            print(
                f"{row['budget']:4d} | "
                f"{format_metric(baselines.get('offset_oracle', {}).get('auc'))} | "
                f"{format_metric(baselines.get('euclidean', {}).get('auc'))} | "
                f"{format_metric(baselines.get('action_goal', {}).get('auc'))}"
            )

    print("\nOffset buckets")
    print("H | offset | offset/H | label | mean_score | min_score | n")
    print("--|--------|----------|-------|------------|-----------|---")
    for row in report["bucket_rows"]:
        print(
            f"{row['budget']:4d} | {row['offset']:6d} | {row['multiplier']:8.2f} | "
            f"{format_metric(row['label_mean'])} | {format_metric(row['mean_score'])} | "
            f"{format_metric(row['min_score'])} | {row['count']:4d}"
        )


def write_outputs(report):
    if FLAGS.output_json is not None:
        with open(FLAGS.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON diagnostics to {FLAGS.output_json}")

    if FLAGS.output_csv is not None:
        rows = []
        for row in report["budget_rows"]:
            flat = {"kind": "budget", "budget": row["budget"]}
            for prefix, metrics in (
                ("mean", row["mean"]),
                ("ensemble_min", row["ensemble_min"]),
            ):
                for key, value in metrics.items():
                    flat[f"{prefix}_{key}"] = value
            for baseline_name, metrics in row.get("baselines", {}).items():
                for key, value in metrics.items():
                    flat[f"baseline_{baseline_name}_{key}"] = value
            rows.append(flat)
        for row in report.get("balanced_budget_rows", []):
            flat = {"kind": "balanced_budget", "budget": row["budget"]}
            for prefix, metrics in (
                ("mean", row["mean"]),
                ("ensemble_min", row["ensemble_min"]),
            ):
                for key, value in metrics.items():
                    flat[f"{prefix}_{key}"] = value
            for baseline_name, metrics in row.get("baselines", {}).items():
                for key, value in metrics.items():
                    flat[f"baseline_{baseline_name}_{key}"] = value
            rows.append(flat)
        for row in report["bucket_rows"]:
            flat = {"kind": "bucket"}
            flat.update(row)
            rows.append(flat)

        keys = sorted({key for row in rows for key in row})
        with open(FLAGS.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV diagnostics to {FLAGS.output_csv}")


def main(_):
    if FLAGS.restore_path is None or FLAGS.restore_epoch is None:
        raise ValueError("--restore_path and --restore_epoch are required.")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent

    if config["agent_name"] != "bmm_trl":
        raise ValueError("scripts/eval_bmm_reachability.py requires agent_name='bmm_trl'.")

    dataset_path = None
    if FLAGS.dataset_dir is not None:
        candidates = [
            str(path)
            for path in sorted(Path(FLAGS.dataset_dir).glob("*.npz"))
            if "-val.npz" not in path.name
        ]
        dataset_path = candidates[0] if candidates else None

    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    example_batch = train_dataset.sample(1)
    agent = agents[config["agent_name"]].create(FLAGS.seed, example_batch, config)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    report = evaluate_reachability(
        agent,
        val_dataset,
        num_pairs=FLAGS.num_pairs,
        balanced_pairs=FLAGS.balanced_pairs,
        score_batch_size=FLAGS.score_batch_size,
        max_offset_factor=FLAGS.max_offset_factor,
        bucket_samples=FLAGS.bucket_samples,
        rng=np.random.default_rng(FLAGS.seed),
    )
    print_report(report)
    write_outputs(report)


if __name__ == "__main__":
    app.run(main)
