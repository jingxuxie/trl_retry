#!/usr/bin/env python
"""Neural BMM reachability check on a deterministic chain.

This complements the tabular test by training the actual JAX critic and BMM
supervised batch path on labels where offset equals graph distance.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import BMMTRLAgent, config_budgets, get_config
from scripts.bmm_reachability_utils import binary_metrics, format_metric
from utils.datasets import Dataset, GCDataset


def make_chain_dataset(num_trajs=3, traj_len=1200):
    size = num_trajs * traj_len
    positions = np.tile(np.arange(traj_len, dtype=np.float32), num_trajs)
    traj_ids = np.repeat(np.arange(num_trajs, dtype=np.float32), traj_len)
    observations = np.stack(
        [
            positions / float(traj_len - 1),
            traj_ids / max(float(num_trajs - 1), 1.0),
        ],
        axis=-1,
    ).astype(np.float32)
    actions = np.ones((size, 1), dtype=np.float32)
    terminals = np.zeros(size, dtype=np.float32)
    terminals[traj_len - 1 :: traj_len] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def make_config():
    config = get_config()
    config.batch_size = 128
    config.lr = 1e-3
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (128, 128)
    config.layer_norm = False
    config.budgets = (1, 128, 256, 512)
    config.max_budget = 512
    config.num_sup_pairs = 4
    config.num_rank_pairs = 0
    config.diagnostic_critic_mode = "state"
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.frs.flow_steps = 1
    config.frs.num_samples = 1
    return config


def evaluate_batch(agent, batch, budgets):
    logits = agent.critic_logits_for_pair_grid(
        batch["value_sup_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    scores = np.asarray(jax.nn.sigmoid(logits).mean(axis=0))
    labels = np.asarray(batch["value_sup_labels"])
    valids = np.asarray(batch["value_sup_valids"]) > 0
    sup_budgets = np.asarray(batch["value_sup_budgets"])
    rows = []
    for budget in budgets:
        if int(budget) == 1:
            continue
        mask = valids & (sup_budgets == int(budget))
        if not mask.any():
            continue
        metrics = binary_metrics(scores[mask], labels[mask])
        rows.append((int(budget), metrics))
    return rows


def print_rows(step, rows, loss):
    print(f"\nstep={step} loss_sup={format_metric(loss)}")
    print("H | auc | gap | pos | neg | pos_n | neg_n")
    print("--|-----|-----|-----|-----|-------|------")
    for budget, metrics in rows:
        gap = metrics["pos_mean"] - metrics["neg_mean"]
        print(
            f"{budget:4d} | {format_metric(metrics['auc'])} | {format_metric(gap)} | "
            f"{format_metric(metrics['pos_mean'])} | {format_metric(metrics['neg_mean'])} | "
            f"{metrics['pos_count']:5d} | {metrics['neg_count']:5d}"
        )


def main():
    np.random.seed(0)
    config = make_config()
    dataset = GCDataset(make_chain_dataset(), config)
    example_batch = dataset.sample(1)
    frozen_batch = dataset.sample(config.batch_size)
    budgets = config_budgets(config)
    agent = BMMTRLAgent.create(0, example_batch, config)

    final_rows = evaluate_batch(agent, frozen_batch, budgets)
    final_loss = np.nan
    print_rows(0, final_rows, final_loss)
    for step in range(1, 801):
        agent, info = agent.update(frozen_batch)
        final_loss = float(info["critic/loss_sup"])
        if step % 100 == 0 or step == 800:
            jax.block_until_ready(info["critic/loss_sup"])
            final_rows = evaluate_batch(agent, frozen_batch, budgets)
            print_rows(step, final_rows, final_loss)
            if all(
                metrics["auc"] >= 0.98
                and metrics["pos_mean"] - metrics["neg_mean"] >= 0.5
                for _, metrics in final_rows
            ):
                break

    failures = []
    for budget, metrics in final_rows:
        gap = metrics["pos_mean"] - metrics["neg_mean"]
        if metrics["pos_count"] == 0 or metrics["neg_count"] == 0:
            failures.append(f"H={budget}: missing one class")
        elif metrics["auc"] < 0.98 or gap < 0.5:
            failures.append(
                f"H={budget}: auc={metrics['auc']:.4f}, gap={gap:.4f}"
            )

    assert not failures, "Neural chain BMM overfit failed: " + "; ".join(failures)
    assert bool(jnp.isfinite(final_loss))
    print("\nBMM neural chain fixed-batch check passed.")


if __name__ == "__main__":
    main()
