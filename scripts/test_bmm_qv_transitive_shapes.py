#!/usr/bin/env python
"""Synthetic smoke checks for Q/V transitive diagnostic training."""

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import BMMTRLAgent, get_config
from scripts import train_bmm_geodesic_q as qdiag
from utils.datasets import Dataset, GCDataset


def make_line_dataset(num_cells=16, obs_dim=4, action_dim=2):
    positions = np.arange(num_cells, dtype=np.float32)
    observations = np.zeros((num_cells, obs_dim), dtype=np.float32)
    observations[:, 0] = positions
    observations[:, 1] = positions / max(num_cells - 1, 1)
    actions = np.zeros((num_cells, action_dim), dtype=np.float32)
    actions[:, 0] = np.linspace(-1.0, 1.0, num_cells, dtype=np.float32)
    terminals = np.zeros(num_cells, dtype=np.float32)
    terminals[-1] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def make_line_context(num_cells=16):
    cells = np.arange(num_cells, dtype=np.int32)
    cell_distances = np.abs(cells[:, None] - cells[None, :]).astype(np.int32)
    goal_by_cell = [np.asarray([idx], dtype=np.int32) for idx in cells]
    return dict(
        kind="grid_geodesic",
        geodesic_budget_unit="grid_cells",
        cell_distances=cell_distances,
        label_distance_scale=1.0,
        train_state_to_cell=cells.copy(),
        train_goal_by_cell=goal_by_cell,
    )


def make_config(critic_mode):
    config = get_config()
    config.batch_size = 8
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.layer_norm = False
    config.budgets = (1, 2, 4)
    config.max_budget = 4
    config.diagnostic_critic_mode = critic_mode
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_sup_pairs = 0
    config.num_rank_pairs = 0
    return config


def main():
    if not qdiag.FLAGS.is_parsed():
        qdiag.FLAGS([sys.argv[0]])
    qdiag.FLAGS.num_trans_witnesses = 2
    qdiag.FLAGS.trans_witness_mode = "avoid_endpoints"
    qdiag.FLAGS.trans_pos_boundary_frac = 0.5
    qdiag.FLAGS.trans_endpoint_epsilon = 1e-6
    qdiag.FLAGS.trans_boundary_beta = 0.25

    rng = np.random.default_rng(0)
    raw_dataset = make_line_dataset()
    context = make_line_context()
    gc_dataset = GCDataset(raw_dataset, make_config("action"))

    example_batch = gc_dataset.sample(1)
    action_config = make_config("action")
    value_config = make_config("state")
    agent = BMMTRLAgent.create(0, example_batch, action_config)
    value_agent = BMMTRLAgent.create(1, example_batch, value_config)

    batch = gc_dataset.sample(action_config.batch_size)
    batch.update(
        qdiag.make_sup_fields(
            raw_dataset,
            context,
            "train",
            budgets=(4,),
            pairs_per_budget=action_config.batch_size,
            rng=rng,
        )
    )
    sparse_sup = qdiag.make_sup_fields(
        raw_dataset,
        context,
        "train",
        budgets=(2, 4),
        pairs_per_budget=action_config.batch_size,
        rng=rng,
        valid_counts={2: action_config.batch_size, 4: 2},
    )
    assert sparse_sup["value_sup_observations"].shape[:2] == (
        2,
        action_config.batch_size,
    )
    h4_mask = sparse_sup["value_sup_budgets"] == 4
    h4_valids = sparse_sup["value_sup_valids"][h4_mask]
    h4_labels = sparse_sup["value_sup_labels"][h4_mask]
    assert int(h4_valids.sum()) == 2
    assert int((h4_labels[h4_valids > 0] == 1.0).sum()) == 1
    assert int((h4_labels[h4_valids > 0] == 0.0).sum()) == 1

    qv_fields = qdiag.sample_grid_qv_transitive_pairs(
        raw_dataset,
        context,
        "train",
        budgets=(4,),
        batch_size=action_config.batch_size,
        rng=rng,
    )
    batch.update(qv_fields)

    assert batch["qv_valids"].shape == (2, action_config.batch_size)
    assert np.all(batch["qv_left_distances"][batch["qv_valids"] > 0] <= 1.0)
    assert np.all(batch["qv_right_distances"][batch["qv_valids"] > 0] <= 2.0)
    assert np.all(batch["qv_effective_unique_witness_counts"] >= 1.0)
    assert "qv_parent_oracle_labels" in batch
    assert "qv_left_oracle_labels" in batch
    assert "qv_right_oracle_labels" in batch
    assert batch["qv_left_oracle_labels"].shape == batch["qv_valids"].shape
    assert np.all(batch["qv_left_oracle_labels"][batch["qv_valids"] > 0] == 1.0)
    assert np.all(batch["qv_right_oracle_labels"][batch["qv_valids"] > 0] == 1.0)

    branch_modes = (
        "learned_q_frozen_v",
        "oracle_q_frozen_v",
        "oracle_q_oracle_v",
    )
    for loss_type in ("bce_equal", "prob_hinge", "bce_lower_bound"):
        branch_mode = branch_modes[0]
        agent, info = qdiag.update_with_qv_trans(
            agent,
            batch,
            value_agent,
            1.0,
            qv_trans_loss_type=loss_type,
            qv_trans_bce_margin=0.0,
            qv_branch_mode=branch_mode,
            trans_budgets=(4,),
            budgets=(1, 2, 4),
        )
        for key in (
            "critic/loss_sup",
            "critic/loss_qv_trans",
            "critic/qv_valid_frac",
            "critic/qv_y_trans_mean",
            "critic/qv_parent_r_mean",
            "critic/qv_target_minus_parent_mean",
            "critic/qv_frac_y_trans_gt_parent",
            "critic/qv_frac_y_trans_lt_parent",
            "critic/qv_min_candidate_mean",
            "critic/qv_product_candidate_mean",
            "critic/qv_first_q_mean",
            "critic/qv_second_v_mean",
            "critic/qv_parent_oracle_label_mean",
            "critic/qv_left_oracle_label_mean",
            "critic/qv_right_oracle_label_mean",
            "critic/loss_qv_trans_by_budget/H4",
            "critic/qv_y_trans_mean_by_budget/H4",
            "critic/qv_parent_r_mean_by_budget/H4",
            "critic/qv_first_q_mean_by_budget/H4",
            "critic/qv_second_v_mean_by_budget/H4",
            "critic/qv_target_minus_parent_mean_by_budget/H4",
            "critic/qv_frac_y_trans_gt_parent_by_budget/H4",
            "critic/qv_frac_y_trans_lt_parent_by_budget/H4",
            "critic/qv_min_candidate_mean_by_budget/H4",
            "critic/qv_product_candidate_mean_by_budget/H4",
            "critic/total_loss_with_qv",
            "critic/total_loss_with_aux",
        ):
            assert key in info, f"Missing {loss_type} Q/V metric {key}"
            assert bool(jnp.all(jnp.isfinite(info[key]))), (key, info[key])

    for branch_mode in branch_modes[1:]:
        agent, info = qdiag.update_with_qv_trans(
            agent,
            batch,
            value_agent if branch_mode != "oracle_q_oracle_v" else None,
            1.0,
            qv_trans_loss_type="bce_lower_bound",
            qv_trans_bce_margin=0.0,
            qv_branch_mode=branch_mode,
            trans_budgets=(4,),
            budgets=(1, 2, 4),
        )
        assert "critic/loss_qv_trans" in info
        assert bool(jnp.all(jnp.isfinite(info["critic/loss_qv_trans"])))

    agent, info = qdiag.update_with_qv_trans(
        agent,
        batch,
        value_agent,
        1.0,
        qv_trans_loss_type="bce_lower_bound",
        qv_trans_target_type="product",
        qv_trans_bce_margin=0.0,
        qv_branch_mode="learned_q_frozen_v",
        trans_budgets=(4,),
        budgets=(1, 2, 4),
    )
    assert "critic/loss_qv_trans" in info
    assert bool(jnp.all(jnp.isfinite(info["critic/loss_qv_trans"])))
    assert bool(jnp.all(info["critic/qv_product_candidate_mean"] <= 1.0))

    vnext_batch = gc_dataset.sample(action_config.batch_size)
    vnext_batch.update(
        qdiag.make_sup_fields(
            raw_dataset,
            context,
            "train",
            budgets=(4,),
            pairs_per_budget=action_config.batch_size,
            rng=rng,
        )
    )
    agent, info = qdiag.update_with_qv_trans(
        agent,
        vnext_batch,
        value_agent,
        0.0,
        lambda_vnext_distill=1.0,
        vnext_distill_loss_type="bce_lower_bound",
        vnext_distill_bce_margin=0.0,
        trans_budgets=(4,),
        budgets=(1, 2, 4),
        use_qv_trans=False,
        use_vnext_distill=True,
    )
    for key in (
        "critic/loss_vnext_distill",
        "critic/vnext_valid_frac",
        "critic/vnext_target_r_mean",
        "critic/vnext_parent_r_mean",
        "critic/vnext_target_minus_parent_mean",
        "critic/loss_vnext_distill_by_budget/H4",
        "critic/vnext_target_r_mean_by_budget/H4",
        "critic/total_loss_with_aux",
    ):
        assert key in info, f"Missing V-next metric {key}"
        assert bool(jnp.all(jnp.isfinite(info[key]))), (key, info[key])

    summary = qdiag.summarize_qv_transitive_batch(qv_fields, budgets=(4,))
    assert summary["budget_rows"][0]["effective_unique_witness_count_mean"] >= 1.0
    assert summary["budget_rows"][0]["zero_left_frac"] == 0.0
    assert summary["budget_rows"][0]["zero_right_frac"] == 0.0
    assert summary["budget_rows"][0]["left_oracle_label_mean"] == 1.0
    assert summary["budget_rows"][0]["right_oracle_label_mean"] == 1.0

    del agent
    print("BMM Q/V transitive shape checks passed.")


if __name__ == "__main__":
    main()
