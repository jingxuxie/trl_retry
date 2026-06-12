#!/usr/bin/env python
"""Train a tiny goal-conditioned BC controller and evaluate BMM subgoals."""

import argparse
import json
from pathlib import Path
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts import eval_bmm_value_subgoal_policy_smoke as smoke
from utils.flax_utils import restore_agent
from utils.networks import MLP
from utils.pointmaze_graph import valid_transition_indices
from utils.pointmaze_grid import unwrap_maze_env


class SubgoalBCActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        return jnp.tanh(nn.Dense(self.action_dim)(x))


def parse_int_list(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_tuple(value):
    parsed = str(value).strip()
    if parsed.startswith("("):
        return tuple(int(part.strip()) for part in parsed.strip("()").split(",") if part.strip())
    return tuple(parse_int_list(parsed))


def source_indices_for_offsets(dataset, offsets):
    valid = np.zeros(len(dataset["observations"]), dtype=bool)
    valid[valid_transition_indices(dataset)] = True
    max_offset = max(int(x) for x in offsets)
    candidates = np.arange(0, len(valid) - max_offset, dtype=np.int32)
    keep = []
    for idx in candidates:
        idx = int(idx)
        ok = True
        for delta in range(max_offset):
            if not valid[idx + delta]:
                ok = False
                break
        if ok:
            keep.append(idx)
    if not keep:
        raise ValueError("No valid source indices for requested BC offsets.")
    return np.asarray(keep, dtype=np.int32)


def sample_bc_batch(dataset, source_idxs, offsets, batch_size, rng):
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    idxs = rng.choice(source_idxs, size=int(batch_size), replace=True)
    ks = rng.choice(np.asarray(offsets, dtype=np.int32), size=int(batch_size), replace=True)
    return dict(
        observations=observations[idxs],
        goals=observations[idxs + ks],
        actions=actions[idxs],
        offsets=ks.astype(np.int32),
    )


def batch_loss(params, model, batch):
    pred = model.apply(
        {"params": params},
        jnp.asarray(batch["observations"]),
        jnp.asarray(batch["goals"]),
    )
    target = jnp.asarray(batch["actions"])
    return jnp.mean((pred - target) ** 2), {
        "mse": jnp.mean((pred - target) ** 2),
        "action_norm": jnp.mean(jnp.linalg.norm(pred, axis=-1)),
    }


def make_update_bc(model, optimizer):
    @jax.jit
    def update_bc(params, opt_state, batch):
        (loss, info), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, model, batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        info = dict(info)
        info["loss"] = loss
        return params, opt_state, info

    return update_bc


def train_bc(train_dataset, args, rng):
    offsets = parse_int_list(args.bc_offsets)
    source_idxs = source_indices_for_offsets(train_dataset, offsets)
    action_dim = np.asarray(train_dataset["actions"]).shape[-1]
    obs_dim = np.asarray(train_dataset["observations"]).shape[-1]
    model = SubgoalBCActor(
        hidden_dims=parse_tuple(args.bc_hidden_dims),
        action_dim=action_dim,
        layer_norm=args.bc_layer_norm,
    )
    init_batch = dict(
        observations=np.zeros((1, obs_dim), dtype=np.float32),
        goals=np.zeros((1, obs_dim), dtype=np.float32),
    )
    params = model.init(
        jax.random.PRNGKey(int(args.seed)),
        init_batch["observations"],
        init_batch["goals"],
    )["params"]
    optimizer = optax.adam(float(args.bc_lr))
    opt_state = optimizer.init(params)
    update_bc = make_update_bc(model, optimizer)

    last_info = {}
    for step in range(1, int(args.bc_steps) + 1):
        batch = sample_bc_batch(
            train_dataset, source_idxs, offsets, args.bc_batch_size, rng
        )
        params, opt_state, info = update_bc(params, opt_state, batch)
        if step == 1 or step == int(args.bc_steps) or step % int(args.bc_log_interval) == 0:
            last_info = {key: float(np.asarray(value)) for key, value in info.items()}
            print(
                f"bc_step={step} loss={last_info['loss']:.6f} "
                f"mse={last_info['mse']:.6f} action_norm={last_info['action_norm']:.4f}"
            )
    return model, params, last_info


class ValueSubgoalBCPolicy(smoke.ValueSubgoalNNPolicy):
    def __init__(self, *args, bc_model, bc_params, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_model = bc_model
        self.bc_params = bc_params

    def action_for_choice(self, observation, goal, choice):
        del goal
        obs = np.asarray(observation, dtype=np.float32)[None, :]
        subgoal = np.asarray(choice["subgoal_observation"], dtype=np.float32)[None, :]
        action = self.bc_model.apply({"params": self.bc_params}, obs, subgoal)
        return np.asarray(action)[0].astype(np.float32)


def aggregate_selector_smoke(env, value_agent, bc_model, bc_params, train_dataset, context, args):
    task_ids = parse_int_list(args.task_ids)
    selectors = [smoke.normalize_selector(name) for name in smoke.parse_str_list(args.selectors)]
    selector_results = []
    for selector_idx, selector in enumerate(selectors):
        policy = ValueSubgoalBCPolicy(
            value_agent,
            train_dataset,
            context,
            unwrap_maze_env(env),
            args.left_budget,
            args.right_budget,
            args.controller_hops,
            args.num_subgoal_candidates,
            np.random.default_rng(int(args.seed) + 1009 * selector_idx),
            args.score_batch_size,
            selector=selector,
            bc_model=bc_model,
            bc_params=bc_params,
        )
        rows = smoke.run_policy_smoke(
            env,
            policy,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
        )
        for row in rows:
            row["selector"] = selector
        selector_results.append(
            dict(name=selector, aggregate=smoke.aggregate(rows), episodes=rows)
        )
    return task_ids, selector_results


def markdown(result):
    lines = [
        "# BMM subgoal BC-controller smoke",
        "",
        f"env: `{result['env_name']}`",
        f"bc steps: `{result['bc_steps']}`, offsets: `{result['bc_offsets']}`",
        f"task ids: `{result['task_ids']}`, max steps: `{result['max_steps']}`",
        "",
        "| selector | success | final_d | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce | selected_src_d | selected_right_d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {success:.4f} | {final:.4f} | {improve:.4f} | {mean_goal:.4f} | {valid:.4f} | {subgoal_reduce:.4f} | {goal_reduce:.4f} | {src_d:.4f} | {right_d:.4f} |".format(
                name=row["name"],
                success=m["success"],
                final=m["final_goal_distance"],
                improve=m["goal_distance_improvement"],
                mean_goal=m["mean_step_goal_improvement"],
                valid=m["subgoal_valid_frac"],
                subgoal_reduce=m["subgoal_reduce_frac"],
                goal_reduce=m["goal_reduce_frac"],
                src_d=m["selected_source_to_subgoal"],
                right_d=m["selected_subgoal_to_goal"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--geodesic_budget_unit", default="env_steps")
    parser.add_argument("--xy_dims", default="0,1")
    parser.add_argument("--budgets", default="40,80,160")
    parser.add_argument("--left_budget", type=int, default=80)
    parser.add_argument("--right_budget", type=int, default=80)
    parser.add_argument("--controller_hops", type=int, default=0)
    parser.add_argument("--num_subgoal_candidates", type=int, default=64)
    parser.add_argument("--selectors", default="random,geometric_midpoint,BMM_V,oracle_midpoint")
    parser.add_argument("--task_ids", default="1,2,3")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--bc_offsets", default="1,2,4,8")
    parser.add_argument("--bc_steps", type=int, default=1000)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_hidden_dims", default="(256, 256)")
    parser.add_argument("--bc_lr", type=float, default=3e-4)
    parser.add_argument("--bc_layer_norm", action="store_true")
    parser.add_argument("--bc_log_interval", type=int, default=500)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rng = np.random.default_rng(args.seed)
    dataset_path = ar.dataset_path_from_dir(args.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    context = ar.make_grid_context(
        env,
        train_dataset,
        val_dataset,
        ar.parse_xy_dims(args.xy_dims),
        args.geodesic_budget_unit,
    )
    value_agent = ar.configure_restore_agent(
        args, train_dataset, ar.parse_int_list(args.budgets), critic_mode="state"
    )
    value_agent = restore_agent(
        value_agent, args.value_restore_path, args.value_restore_epoch
    )
    bc_model, bc_params, bc_info = train_bc(train_dataset, args, rng)
    task_ids, selector_results = aggregate_selector_smoke(
        env, value_agent, bc_model, bc_params, train_dataset, context, args
    )
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        max_steps=int(args.max_steps),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        controller_hops=int(args.controller_hops),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        bc_steps=int(args.bc_steps),
        bc_offsets=args.bc_offsets,
        bc_final_info=bc_info,
        selectors=selector_results,
    )
    text = markdown(result)
    print(text)
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
