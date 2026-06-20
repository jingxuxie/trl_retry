#!/usr/bin/env python
"""Tiny BMM-TRL agent update and action-sampling smoke test."""

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

from agents.bmm_trl import BMMTRLAgent, get_config
from utils.datasets import Dataset, GCDataset


def make_fake_dataset(num_trajs=4, traj_len=12, obs_dim=5, action_dim=2):
    size = num_trajs * traj_len
    observations = np.arange(size * obs_dim, dtype=np.float32).reshape(size, obs_dim)
    observations = observations / observations.max()
    actions = np.linspace(-1.0, 1.0, size * action_dim, dtype=np.float32).reshape(
        size, action_dim
    )
    terminals = np.zeros(size, dtype=np.float32)
    terminals[traj_len - 1 :: traj_len] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def make_config():
    config = get_config()
    config.batch_size = 8
    config.discount = 0.9
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.layer_norm = False
    config.budgets = (1, 2, 4, 8)
    config.max_budget = 8
    config.frs.flow_steps = 2
    config.frs.num_samples = 4
    return config


def main():
    np.random.seed(0)
    config = make_config()
    dataset = GCDataset(make_fake_dataset(), config)
    example_batch = dataset.sample(1)
    batch = dataset.sample(config.batch_size)

    agent = BMMTRLAgent.create(0, example_batch, config)
    agent, info = agent.update(batch)
    required_metrics = {
        "critic/total_loss",
        "critic/loss_trans",
        "critic/loss_pos",
        "critic/loss_budget_neg",
        "critic/loss_hard_neg",
        "critic/loss_rand_hinge",
        "critic/loss_mono",
        "critic/loss_sup",
        "critic/loss_rank",
        "critic/sup_valid_frac",
        "critic/rank_valid_frac",
        "critic/sup_pos_count_H=1",
        "critic/sup_neg_count_H=1",
        "actor/actor_loss",
    }
    missing = required_metrics - set(info.keys())
    assert not missing, f"Missing update metrics: {sorted(missing)}"
    for key, value in info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    actions = agent.sample_actions(
        batch["observations"],
        goals=batch["actor_goals"],
        seed=jax.random.PRNGKey(1),
    )
    assert actions.shape == batch["actions"].shape, (actions.shape, batch["actions"].shape)
    assert bool(jnp.all(jnp.isfinite(actions)))

    single_action = agent.sample_actions(
        batch["observations"][0],
        goals=batch["actor_goals"][0],
        seed=jax.random.PRNGKey(2),
    )
    assert single_action.shape == batch["actions"][0].shape
    assert bool(jnp.all(jnp.isfinite(single_action)))

    one_batch_action = agent.sample_actions(
        batch["observations"][:1],
        goals=batch["actor_goals"][:1],
        seed=jax.random.PRNGKey(3),
    )
    assert one_batch_action.shape == batch["actions"][:1].shape
    assert bool(jnp.all(jnp.isfinite(one_batch_action)))

    scan_config = make_config()
    scan_config.actor_budget_mode = "scan"
    scan_dataset = GCDataset(make_fake_dataset(), scan_config)
    scan_example_batch = scan_dataset.sample(1)
    scan_batch = scan_dataset.sample(scan_config.batch_size)
    scan_agent = BMMTRLAgent.create(9, scan_example_batch, scan_config)
    scan_agent, _ = scan_agent.update(scan_batch)
    scan_actions = scan_agent.sample_actions(
        scan_batch["observations"],
        goals=scan_batch["actor_goals"],
        seed=jax.random.PRNGKey(9),
    )
    assert scan_actions.shape == scan_batch["actions"].shape
    assert bool(jnp.all(jnp.isfinite(scan_actions)))
    scan_single_action = scan_agent.sample_actions(
        scan_batch["observations"][0],
        goals=scan_batch["actor_goals"][0],
        seed=jax.random.PRNGKey(10),
    )
    assert scan_single_action.shape == scan_batch["actions"][0].shape
    assert bool(jnp.all(jnp.isfinite(scan_single_action)))

    zero_config = make_config()
    zero_config.lambda_hard_neg = 0.0
    zero_config.lambda_rank = 0.0
    zero_dataset = GCDataset(make_fake_dataset(), zero_config)
    zero_example_batch = zero_dataset.sample(1)
    zero_batch = zero_dataset.sample(zero_config.batch_size)
    zero_agent = BMMTRLAgent.create(4, zero_example_batch, zero_config)
    _, zero_info = zero_agent.update(zero_batch)
    assert "critic/loss_hard_neg" in zero_info
    for key, value in zero_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    onehot_config = make_config()
    onehot_config.budget_feature = "log_scalar_onehot"
    onehot_dataset = GCDataset(make_fake_dataset(), onehot_config)
    onehot_example_batch = onehot_dataset.sample(1)
    onehot_batch = onehot_dataset.sample(onehot_config.batch_size)
    onehot_agent = BMMTRLAgent.create(5, onehot_example_batch, onehot_config)
    _, onehot_info = onehot_agent.update(onehot_batch)
    assert "critic/loss_sup" in onehot_info
    for key, value in onehot_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    absdiff_config = make_config()
    absdiff_config.diagnostic_critic_mode = "state"
    absdiff_config.value_only = True
    absdiff_config.lambda_rank = 0.0
    absdiff_config.num_rank_pairs = 0
    absdiff_config.critic_absdiff_goal_feature = True
    absdiff_dataset = GCDataset(make_fake_dataset(), absdiff_config)
    absdiff_example_batch = absdiff_dataset.sample(1)
    absdiff_batch = absdiff_dataset.sample(absdiff_config.batch_size)
    absdiff_agent = BMMTRLAgent.create(11, absdiff_example_batch, absdiff_config)
    _, absdiff_info = absdiff_agent.update(absdiff_batch)
    assert "critic/loss_sup" in absdiff_info
    for key, value in absdiff_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    state_config = make_config()
    state_config.diagnostic_critic_mode = "state"
    state_config.value_only = True
    state_config.lambda_mono = 0.0
    state_config.lambda_rank = 0.0
    state_config.num_rank_pairs = 0
    state_dataset = GCDataset(make_fake_dataset(), state_config)
    state_example_batch = state_dataset.sample(1)
    state_batch = state_dataset.sample(state_config.batch_size)
    state_agent = BMMTRLAgent.create(6, state_example_batch, state_config)
    _, state_info = state_agent.update(state_batch)
    assert "critic/loss_sup" in state_info
    assert "actor/actor_loss" not in state_info
    for key, value in state_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    multi_witness_batch = dict(state_batch)
    for key in (
        "value_midpoint_observations",
        "value_midpoint_actions",
        "value_midpoint_goals",
        "value_midpoint_offsets",
        "value_left_budgets",
        "value_right_budgets",
        "trans_valids",
    ):
        multi_witness_batch[key] = np.stack([state_batch[key], state_batch[key]], axis=0)
    multi_witness_batch["trans_parent_oracle_labels"] = np.ones(
        state_config.batch_size, dtype=np.float32
    )
    multi_witness_batch["trans_branch_oracle_valids"] = np.ones(
        (2, state_config.batch_size), dtype=np.float32
    )
    multi_witness_agent = BMMTRLAgent.create(8, state_example_batch, state_config)
    _, multi_witness_info = multi_witness_agent.update(multi_witness_batch)
    for key in (
        "critic/loss_trans_over_sup",
        "critic/y_trans_mean_H=1",
        "critic/first_r_mean_H=1",
        "critic/second_r_mean_H=1",
        "critic/parent_r_mean_H=1",
        "critic/trans_parent_oracle_label_mean",
        "critic/trans_branch_oracle_valid_mean",
    ):
        assert key in multi_witness_info, f"Missing multi-witness metric {key}"
    for key, value in multi_witness_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    oracle_config = make_config()
    oracle_config.oracle_offset_feature = True
    oracle_config.value_only = True
    oracle_config.lambda_rank = 0.0
    oracle_config.num_rank_pairs = 0
    oracle_dataset = GCDataset(make_fake_dataset(), oracle_config)
    oracle_example_batch = oracle_dataset.sample(1)
    oracle_batch = oracle_dataset.sample(oracle_config.batch_size)
    oracle_agent = BMMTRLAgent.create(7, oracle_example_batch, oracle_config)
    _, oracle_info = oracle_agent.update(oracle_batch)
    assert "critic/loss_sup" in oracle_info
    for key, value in oracle_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    print("BMM agent update and sample_actions smoke checks passed.")


if __name__ == "__main__":
    main()
