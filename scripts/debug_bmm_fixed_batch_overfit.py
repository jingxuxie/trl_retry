#!/usr/bin/env python
"""Overfit a frozen BMM supervised reachability batch.

This is a diagnostic-only script. It intentionally skips actor loss and all
non-supervised critic losses so a failure is easier to interpret.
"""

import ast
import json
from pathlib import Path
import random
import sys

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_metrics, format_metric
from utils.datasets import Dataset, GCDataset


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional directory of OGBench npz files.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("steps", 1000, "Maximum fixed-batch update steps.")
flags.DEFINE_integer("eval_interval", 100, "Evaluate every N update steps.")
flags.DEFINE_string("budgets", "(256, 512)", "Budgets to train/evaluate.")
flags.DEFINE_integer("batch_size", 256, "Frozen transition batch size.")
flags.DEFINE_string(
    "diagnostic_critic_mode", "state", "Diagnostic critic mode: state or action."
)
flags.DEFINE_float("target_auc", 0.98, "Early-stop AUC target for every valid budget.")
flags.DEFINE_float("target_gap", 0.5, "Early-stop score-gap target for every valid budget.")
flags.DEFINE_string("output_json", None, "Optional path to write final metrics.")

config_flags.DEFINE_config_file("agent", "agents/bmm_trl.py", lock_config=False)


def parse_budgets(value):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        parsed = (parsed,)
    budgets = tuple(int(x) for x in parsed)
    if not budgets:
        raise ValueError("--budgets must contain at least one budget.")
    return budgets


def configure_agent(config):
    budgets = parse_budgets(FLAGS.budgets)
    config.budgets = budgets
    config.max_budget = max(budgets)
    config.batch_size = FLAGS.batch_size
    config.diagnostic_critic_mode = FLAGS.diagnostic_critic_mode
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_rank_pairs = 0
    return budgets


def load_dataset(config):
    dataset_path = None
    if FLAGS.dataset_dir is not None:
        candidates = [
            str(path)
            for path in sorted(Path(FLAGS.dataset_dir).glob("*.npz"))
            if "-val.npz" not in path.name
        ]
        dataset_path = candidates[0] if candidates else None

    _, train_dataset, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=dataset_path)
    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    return dataset_class(Dataset.create(**train_dataset), config)


def score_frozen_batch(agent, batch, budgets):
    logits = agent.critic_logits_for_pair_grid(
        batch["value_sup_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    scores = np.asarray(jax.nn.sigmoid(logits))
    mean_scores = scores.mean(axis=0)
    min_scores = scores.min(axis=0)
    labels = np.asarray(batch["value_sup_labels"])
    valids = np.asarray(batch["value_sup_valids"]) > 0
    sup_budgets = np.asarray(batch["value_sup_budgets"])

    report = {
        "mean": binary_metrics(mean_scores[valids], labels[valids]),
        "ensemble_min": binary_metrics(min_scores[valids], labels[valids]),
        "budget_rows": [],
    }
    for budget in budgets:
        mask = valids & (sup_budgets == int(budget))
        if not mask.any():
            continue
        mean_metrics = binary_metrics(mean_scores[mask], labels[mask])
        min_metrics = binary_metrics(min_scores[mask], labels[mask])
        report["budget_rows"].append(
            {
                "budget": int(budget),
                "mean": mean_metrics,
                "ensemble_min": min_metrics,
            }
        )
    return report


def passed(report):
    rows = report["budget_rows"]
    if not rows:
        return False
    for row in rows:
        for key in ("mean", "ensemble_min"):
            metrics = row[key]
            if metrics["pos_count"] == 0 or metrics["neg_count"] == 0:
                return False
            gap = metrics["pos_mean"] - metrics["neg_mean"]
            if metrics["auc"] < FLAGS.target_auc or gap < FLAGS.target_gap:
                return False
    return True


def print_report(step, report, loss=None):
    loss_text = "" if loss is None else f" loss={format_metric(loss)}"
    print(f"\nstep={step}{loss_text}")
    print("H | auc | gap | pos | neg | min_auc | min_gap | pos_n | neg_n")
    print("--|-----|-----|-----|-----|---------|---------|-------|------")
    for row in report["budget_rows"]:
        mean = row["mean"]
        ens_min = row["ensemble_min"]
        gap = mean["pos_mean"] - mean["neg_mean"]
        min_gap = ens_min["pos_mean"] - ens_min["neg_mean"]
        print(
            f"{row['budget']:4d} | {format_metric(mean['auc'])} | "
            f"{format_metric(gap)} | {format_metric(mean['pos_mean'])} | "
            f"{format_metric(mean['neg_mean'])} | {format_metric(ens_min['auc'])} | "
            f"{format_metric(min_gap)} | {mean['pos_count']:5d} | {mean['neg_count']:5d}"
        )


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent
    if config["agent_name"] != "bmm_trl":
        raise ValueError("debug_bmm_fixed_batch_overfit.py requires bmm_trl.")
    budgets = configure_agent(config)

    train_dataset = load_dataset(config)
    example_batch = train_dataset.sample(1)
    frozen_batch = train_dataset.sample(config.batch_size)
    coverage = score_frozen_batch(
        agents[config["agent_name"]].create(FLAGS.seed, example_batch, config),
        frozen_batch,
        budgets,
    )
    print_report(0, coverage)

    agent = agents[config["agent_name"]].create(FLAGS.seed, example_batch, config)
    final_report = coverage
    final_loss = None
    for step in range(1, FLAGS.steps + 1):
        agent, info = agent.update(frozen_batch)
        final_loss = float(info["critic/loss_sup"])
        if step % FLAGS.eval_interval == 0 or step == FLAGS.steps:
            final_report = score_frozen_batch(agent, frozen_batch, budgets)
            print_report(step, final_report, loss=final_loss)
            if passed(final_report):
                print(f"\nPassed fixed-batch target at step {step}.")
                break

    final_report["config"] = {
        "env_name": FLAGS.env_name,
        "budgets": [int(x) for x in budgets],
        "diagnostic_critic_mode": FLAGS.diagnostic_critic_mode,
        "steps": int(step),
        "target_auc": FLAGS.target_auc,
        "target_gap": FLAGS.target_gap,
        "final_loss_sup": final_loss,
    }
    if FLAGS.output_json is not None:
        with open(FLAGS.output_json, "w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nWrote fixed-batch report to {FLAGS.output_json}")

    if not passed(final_report):
        raise SystemExit("Fixed-batch overfit target was not reached.")


if __name__ == "__main__":
    app.run(main)
