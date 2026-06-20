#!/usr/bin/env python
"""Train a tiny goal-conditioned BC controller and evaluate BMM subgoals."""

import argparse
import json
import pickle
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

from agents import agents
from agents.gcfbc import get_config as get_gcfbc_config
from agents.gciql import get_config as get_gciql_config
from agents.trl import get_config as get_trl_config
from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts import eval_bmm_value_subgoal_policy_smoke as smoke
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from utils.networks import MLP
from utils.pointmaze_graph import valid_transition_indices
from utils.pointmaze_grid import unwrap_maze_env

AGENT_CONFIGS = {
    "gcfbc": get_gcfbc_config,
    "gciql": get_gciql_config,
    "trl": get_trl_config,
}


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


def maybe_tuple(key, value):
    if key in {"actor_hidden_dims", "value_hidden_dims", "budgets"}:
        return tuple(value)
    return value


def merge_config(config, values):
    for key, value in values.items():
        if isinstance(value, dict) and key in config:
            merge_config(config[key], value)
        else:
            config[key] = maybe_tuple(key, value)


def load_agent_config(restore_path):
    flags_path = Path(restore_path) / "flags.json"
    if not flags_path.exists():
        raise FileNotFoundError(f"Missing controller flags.json: {flags_path}")
    flags = json.loads(flags_path.read_text())
    agent_flags = flags.get("agent", {})
    agent_name = str(agent_flags.get("agent_name", ""))
    if agent_name not in AGENT_CONFIGS:
        raise ValueError(
            f"Unsupported controller agent {agent_name!r}. "
            f"Expected one of {sorted(AGENT_CONFIGS)}."
        )
    config = AGENT_CONFIGS[agent_name]()
    merge_config(config, agent_flags)
    return agent_name, config, flags


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


def bc_goal_rep_mode(train_dataset, args):
    mode = str(getattr(args, "bc_goal_rep", "observation"))
    if mode == "oracle" and "oracle_reps" not in train_dataset:
        raise ValueError("--bc_goal_rep=oracle requires dataset['oracle_reps'].")
    return mode


def bc_goal_vectors(dataset, idxs, args):
    idxs = np.asarray(idxs, dtype=np.int32)
    if bc_goal_rep_mode(dataset, args) == "oracle":
        return np.asarray(dataset["oracle_reps"], dtype=np.float32)[idxs]
    return np.asarray(dataset["observations"], dtype=np.float32)[idxs]


def bc_goal_dim(train_dataset, args):
    if bc_goal_rep_mode(train_dataset, args) == "oracle":
        return int(np.asarray(train_dataset["oracle_reps"]).shape[-1])
    return int(np.asarray(train_dataset["observations"]).shape[-1])


def sample_bc_batch(dataset, source_idxs, offsets, batch_size, rng, args):
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    idxs = rng.choice(source_idxs, size=int(batch_size), replace=True)
    ks = rng.choice(np.asarray(offsets, dtype=np.int32), size=int(batch_size), replace=True)
    return dict(
        observations=observations[idxs],
        goals=bc_goal_vectors(dataset, idxs + ks, args),
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


def build_bc_model(train_dataset, args):
    action_dim = np.asarray(train_dataset["actions"]).shape[-1]
    obs_dim = np.asarray(train_dataset["observations"]).shape[-1]
    goal_dim = bc_goal_dim(train_dataset, args)
    model = SubgoalBCActor(
        hidden_dims=parse_tuple(args.bc_hidden_dims),
        action_dim=action_dim,
        layer_norm=args.bc_layer_norm,
    )
    return model, obs_dim, goal_dim, action_dim


def make_bc_apply(model):
    @jax.jit
    def apply_bc(params, observations, goals):
        return model.apply({"params": params}, observations, goals)

    return apply_bc


def gelu_numpy(x):
    x = np.asarray(x, dtype=np.float32)
    return (
        0.5
        * x
        * (
            1.0
            + np.tanh(
                np.float32(np.sqrt(2.0 / np.pi))
                * (x + np.float32(0.044715) * np.power(x, 3))
            )
        )
    ).astype(np.float32)


def layer_norm_numpy(x, params, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + float(eps))
    return (
        y * np.asarray(params["scale"], dtype=np.float32)
        + np.asarray(params["bias"], dtype=np.float32)
    ).astype(np.float32)


def dense_numpy(x, params):
    return (
        np.asarray(x, dtype=np.float32) @ np.asarray(params["kernel"], dtype=np.float32)
        + np.asarray(params["bias"], dtype=np.float32)
    ).astype(np.float32)


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32), tree)


def bc_apply_numpy(params, observations, goals, layer_norm=False):
    """NumPy inference for SubgoalBCActor; avoids one JAX dispatch per env step."""
    params = tree_to_numpy(params)
    x = np.concatenate(
        [
            np.asarray(observations, dtype=np.float32),
            np.asarray(goals, dtype=np.float32),
        ],
        axis=-1,
    )
    mlp_params = params["MLP_0"]
    dense_names = sorted(
        [name for name in mlp_params if name.startswith("Dense_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    for idx, dense_name in enumerate(dense_names):
        x = dense_numpy(x, mlp_params[dense_name])
        x = gelu_numpy(x)
        if layer_norm:
            x = layer_norm_numpy(x, mlp_params[f"LayerNorm_{idx}"])
    return np.tanh(dense_numpy(x, params["Dense_0"])).astype(np.float32)


def train_bc(train_dataset, args, rng):
    offsets = parse_int_list(args.bc_offsets)
    source_idxs = source_indices_for_offsets(train_dataset, offsets)
    model, obs_dim, goal_dim, _ = build_bc_model(train_dataset, args)
    init_batch = dict(
        observations=np.zeros((1, obs_dim), dtype=np.float32),
        goals=np.zeros((1, goal_dim), dtype=np.float32),
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
            train_dataset, source_idxs, offsets, args.bc_batch_size, rng, args
        )
        params, opt_state, info = update_bc(params, opt_state, batch)
        if step == 1 or step == int(args.bc_steps) or step % int(args.bc_log_interval) == 0:
            last_info = {key: float(np.asarray(value)) for key, value in info.items()}
            print(
                f"bc_step={step} loss={last_info['loss']:.6f} "
                f"mse={last_info['mse']:.6f} action_norm={last_info['action_norm']:.4f}"
            )
    last_info = dict(last_info)
    last_info["bc_steps"] = int(args.bc_steps)
    last_info["bc_offsets"] = str(args.bc_offsets)
    last_info["bc_hidden_dims"] = tuple(parse_tuple(args.bc_hidden_dims))
    last_info["bc_layer_norm"] = bool(args.bc_layer_norm)
    last_info["bc_goal_rep"] = bc_goal_rep_mode(train_dataset, args)
    return model, params, last_info


def save_bc(path, params, args, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=jax.tree_util.tree_map(lambda x: np.asarray(x), params),
        bc_hidden_dims=tuple(parse_tuple(args.bc_hidden_dims)),
        bc_layer_norm=bool(args.bc_layer_norm),
        bc_offsets=str(args.bc_offsets),
        bc_steps=int(args.bc_steps),
        bc_goal_rep=str(getattr(args, "bc_goal_rep", "observation")),
        bc_final_info=info,
    )
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_bc(path, train_dataset, args):
    model, _, _, _ = build_bc_model(train_dataset, args)
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    expected_hidden = tuple(parse_tuple(args.bc_hidden_dims))
    saved_hidden = tuple(payload.get("bc_hidden_dims", expected_hidden))
    if saved_hidden != expected_hidden:
        raise ValueError(
            f"BC hidden dims mismatch: checkpoint {saved_hidden}, args {expected_hidden}"
        )
    saved_layer_norm = bool(payload.get("bc_layer_norm", bool(args.bc_layer_norm)))
    if saved_layer_norm != bool(args.bc_layer_norm):
        raise ValueError(
            "BC layer-norm mismatch: checkpoint "
            f"{saved_layer_norm}, args {bool(args.bc_layer_norm)}"
        )
    expected_goal_rep = str(getattr(args, "bc_goal_rep", "observation"))
    saved_goal_rep = str(payload.get("bc_goal_rep", "observation"))
    if saved_goal_rep != expected_goal_rep:
        raise ValueError(
            f"BC goal-representation mismatch: checkpoint {saved_goal_rep!r}, "
            f"args {expected_goal_rep!r}"
        )
    info = dict(payload.get("bc_final_info", {}))
    info["bc_steps"] = int(payload.get("bc_steps", args.bc_steps))
    info["bc_offsets"] = str(payload.get("bc_offsets", args.bc_offsets))
    info["bc_hidden_dims"] = tuple(payload.get("bc_hidden_dims", expected_hidden))
    info["bc_layer_norm"] = bool(payload.get("bc_layer_norm", bool(args.bc_layer_norm)))
    info["bc_goal_rep"] = saved_goal_rep
    return model, payload["params"], info


class ValueSubgoalBCPolicy(smoke.ValueSubgoalNNPolicy):
    def __init__(
        self,
        *args,
        bc_model,
        bc_params,
        bc_goal_rep="observation",
        bc_layer_norm=False,
        bc_inference="numpy",
        final_goal_switch_distance=-1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_model = bc_model
        self.bc_params = bc_params
        self.bc_goal_rep = str(bc_goal_rep)
        self.bc_layer_norm = bool(bc_layer_norm)
        self.bc_inference = str(bc_inference)
        if self.bc_inference not in {"jax", "numpy"}:
            raise ValueError("--bc_inference must be one of jax, numpy.")
        self.final_goal_switch_distance = float(final_goal_switch_distance)
        self.train_observations = np.asarray(
            self.train_dataset["observations"], dtype=np.float32
        )
        self.train_xy = self.train_observations[:, :2]
        self.oracle_goal_dim = (
            int(np.asarray(self.train_dataset["oracle_reps"]).shape[-1])
            if "oracle_reps" in self.train_dataset
            else None
        )
        self._controller_goal_cache = {}
        self._bc_apply = None
        self._bc_numpy_params = None
        if self.bc_inference == "jax":
            self._bc_apply = make_bc_apply(self.bc_model)
        else:
            self._bc_numpy_params = tree_to_numpy(self.bc_params)

    def controller_goal_vector(self, goal):
        goal = np.asarray(goal, dtype=np.float32)
        if self.bc_goal_rep == "oracle":
            if self.oracle_goal_dim is None:
                raise ValueError("Oracle-goal BC requires dataset['oracle_reps'].")
            if goal.shape[-1] == self.oracle_goal_dim:
                return goal
            if goal.shape[-1] < self.oracle_goal_dim:
                raise ValueError(
                    f"Cannot map controller goal with shape {goal.shape} to "
                    f"oracle dim {self.oracle_goal_dim}."
                )
            return goal[..., : self.oracle_goal_dim]

        obs_dim = self.train_observations.shape[-1]
        if goal.shape[-1] == obs_dim:
            return goal
        if goal.shape[-1] < 2:
            raise ValueError(
                f"Cannot map controller goal with shape {goal.shape} to observation dim {obs_dim}."
            )
        key = tuple(np.round(goal[:2], decimals=4).tolist())
        if key not in self._controller_goal_cache:
            distances = np.linalg.norm(self.train_xy - goal[None, :2], axis=1)
            self._controller_goal_cache[key] = int(np.argmin(distances))
        return self.train_observations[self._controller_goal_cache[key]]

    def action_for_choice(self, observation, goal, choice):
        obs = np.asarray(observation, dtype=np.float32)[None, :]
        if (
            self.final_goal_switch_distance >= 0.0
            and choice["source_to_goal"] <= self.final_goal_switch_distance
        ):
            target = self.controller_goal_vector(goal)
        else:
            target = np.asarray(choice["subgoal_observation"], dtype=np.float32)
            target = self.controller_goal_vector(target)
        target = target[None, :]
        if self.bc_inference == "numpy":
            action = bc_apply_numpy(
                self._bc_numpy_params,
                obs,
                target,
                layer_norm=self.bc_layer_norm,
            )
        else:
            action = self._bc_apply(self.bc_params, obs, target)
        return np.asarray(action)[0].astype(np.float32)


class ValueSubgoalAgentPolicy(smoke.ValueSubgoalNNPolicy):
    def __init__(
        self,
        *args,
        controller_agent,
        controller_temperature=0.0,
        final_goal_switch_distance=-1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_agent = controller_agent
        self.controller_temperature = float(controller_temperature)
        self.final_goal_switch_distance = float(final_goal_switch_distance)
        self.controller_rng = jax.random.PRNGKey(0)

    def action_for_choice(self, observation, goal, choice):
        self.controller_rng, key = jax.random.split(self.controller_rng)
        obs = np.asarray(observation, dtype=np.float32)[None, :]
        if (
            self.final_goal_switch_distance >= 0.0
            and choice["source_to_goal"] <= self.final_goal_switch_distance
        ):
            target = np.asarray(goal, dtype=np.float32)[None, :]
        else:
            target = np.asarray(choice["subgoal_observation"], dtype=np.float32)[None, :]
        action = self.controller_agent.sample_actions(
            obs,
            goals=target,
            seed=key,
            temperature=self.controller_temperature,
        )
        return np.asarray(action)[0].astype(np.float32)


class ValueSubgoalXYPolicy(smoke.ValueSubgoalNNPolicy):
    def __init__(
        self,
        *args,
        action_gain=0.5,
        final_goal_switch_distance=-1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_gain = float(action_gain)
        self.final_goal_switch_distance = float(final_goal_switch_distance)

    def action_for_choice(self, observation, goal, choice):
        if (
            self.final_goal_switch_distance >= 0.0
            and choice["source_to_goal"] <= self.final_goal_switch_distance
        ):
            target = np.asarray(goal, dtype=np.float32)
        else:
            target = np.asarray(choice["subgoal_observation"], dtype=np.float32)
        obs = np.asarray(observation, dtype=np.float32)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:2] = self.action_gain * (target[:2] - obs[:2])
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def aggregate_selector_smoke(env, value_agent, bc_model, bc_params, train_dataset, context, args):
    task_ids = parse_int_list(args.task_ids)
    selectors = [smoke.normalize_selector(name) for name in smoke.parse_str_list(args.selectors)]
    selector_results = []

    def make_policy(eval_env, selector_idx, selector, task_seed_offset=0):
        return ValueSubgoalBCPolicy(
            value_agent,
            train_dataset,
            context,
            unwrap_maze_env(eval_env),
            args.left_budget,
            args.right_budget,
            args.controller_hops,
            args.num_subgoal_candidates,
            np.random.default_rng(
                int(args.seed) + 1009 * selector_idx + 9176 * int(task_seed_offset)
            ),
            args.score_batch_size,
            selector=selector,
            budgets=ar.parse_int_list(args.budgets),
            value_gate_threshold=args.value_gate_threshold,
            support_gate_left_frac=args.support_gate_left_frac,
            support_frontier_left_gate=args.support_frontier_left_gate,
            support_frontier_min_progress_frac=args.support_frontier_min_progress_frac,
            support_frontier_max_xy_factor=args.support_frontier_max_xy_factor,
            support_path_horizon_mode=args.support_path_horizon_mode,
            subgoal_sample_mode=args.subgoal_sample_mode,
            candidate_sample_mode=args.candidate_sample_mode,
            choice_cache_mode=args.choice_cache_mode,
            require_goal_progress=args.require_goal_progress,
            bc_model=bc_model,
            bc_params=bc_params,
            bc_goal_rep=args.bc_goal_rep,
            bc_layer_norm=args.bc_layer_norm,
            bc_inference=args.bc_inference,
            final_goal_switch_distance=args.final_goal_switch_distance,
        )

    def eval_rows(eval_env, policy, eval_task_ids):
        return smoke.run_policy_smoke(
            eval_env,
            policy,
            task_ids=eval_task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            subgoal_commit_steps=args.subgoal_commit_steps,
            subgoal_replan_distance=args.subgoal_replan_distance,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_min_delta=args.early_stop_min_delta,
            fallback_selector=args.fallback_selector,
            fallback_patience=args.fallback_patience,
            fallback_min_steps=args.fallback_min_steps,
            fallback_min_delta=args.fallback_min_delta,
            fallback_max_action_frac=args.fallback_max_action_frac,
            fallback_burst_steps=args.fallback_burst_steps,
            fallback_cooldown_steps=args.fallback_cooldown_steps,
            fallback_max_goal_distance=args.fallback_max_goal_distance,
            fallback_burst_min_delta=args.fallback_burst_min_delta,
            fallback_min_active_subgoal_to_goal=args.fallback_min_active_subgoal_to_goal,
            reset_seed_base=args.reset_seed_base,
            stop_on_grid_goal_distance=args.stop_on_grid_goal_distance,
            progress_prefix=f"selector={policy.selector}",
        )

    for selector_idx, selector in enumerate(selectors):
        print(f"Starting selector={selector}", flush=True)
        if args.fresh_env_per_task:
            rows = []
            dataset_path = ar.dataset_path_from_dir(args.dataset_dir)
            for task_id in task_ids:
                task_env, _, _ = make_env_and_datasets(
                    args.env_name, dataset_path=dataset_path
                )
                try:
                    task_policy = make_policy(
                        task_env, selector_idx, selector, task_seed_offset=task_id
                    )
                    rows.extend(eval_rows(task_env, task_policy, [task_id]))
                finally:
                    if hasattr(task_env, "close"):
                        task_env.close()
        else:
            policy = make_policy(env, selector_idx, selector)
            rows = eval_rows(env, policy, task_ids)
        for row in rows:
            row["selector"] = selector
        selector_aggregate = smoke.aggregate(rows)
        print(
            "Finished selector={selector} success={success:.4f} final_d={final_d:.4f} "
            "improve={improve:.4f}".format(
                selector=selector,
                success=selector_aggregate["success"],
                final_d=selector_aggregate["final_goal_distance"],
                improve=selector_aggregate["goal_distance_improvement"],
            ),
            flush=True,
        )
        selector_results.append(
            dict(name=selector, aggregate=selector_aggregate, episodes=rows)
        )
    return task_ids, selector_results


def restore_controller_agent(train_dataset, args):
    if args.controller_agent_restore_path is None:
        raise ValueError("--controller_agent_restore_path is required for agent controller.")
    agent_name, config, flags = load_agent_config(args.controller_agent_restore_path)
    controller_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = controller_dataset.sample(1)
    agent = agents[agent_name].create(int(args.seed), example_batch, config)
    agent = restore_agent(
        agent,
        args.controller_agent_restore_path,
        int(args.controller_agent_restore_epoch),
    )
    return agent, agent_name, flags


def aggregate_agent_selector_smoke(
    env, value_agent, controller_agent, train_dataset, context, args
):
    task_ids = parse_int_list(args.task_ids)
    selectors = [smoke.normalize_selector(name) for name in smoke.parse_str_list(args.selectors)]
    selector_results = []
    for selector_idx, selector in enumerate(selectors):
        print(f"Starting selector={selector}", flush=True)
        policy = ValueSubgoalAgentPolicy(
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
            budgets=ar.parse_int_list(args.budgets),
            value_gate_threshold=args.value_gate_threshold,
            support_gate_left_frac=args.support_gate_left_frac,
            support_frontier_left_gate=args.support_frontier_left_gate,
            support_frontier_min_progress_frac=args.support_frontier_min_progress_frac,
            support_frontier_max_xy_factor=args.support_frontier_max_xy_factor,
            support_path_horizon_mode=args.support_path_horizon_mode,
            subgoal_sample_mode=args.subgoal_sample_mode,
            candidate_sample_mode=args.candidate_sample_mode,
            choice_cache_mode=args.choice_cache_mode,
            require_goal_progress=args.require_goal_progress,
            final_goal_switch_distance=args.final_goal_switch_distance,
            controller_agent=controller_agent,
            controller_temperature=args.controller_temperature,
        )
        rows = smoke.run_policy_smoke(
            env,
            policy,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            subgoal_commit_steps=args.subgoal_commit_steps,
            subgoal_replan_distance=args.subgoal_replan_distance,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_min_delta=args.early_stop_min_delta,
            fallback_selector=args.fallback_selector,
            fallback_patience=args.fallback_patience,
            fallback_min_steps=args.fallback_min_steps,
            fallback_min_delta=args.fallback_min_delta,
            fallback_max_action_frac=args.fallback_max_action_frac,
            fallback_burst_steps=args.fallback_burst_steps,
            fallback_cooldown_steps=args.fallback_cooldown_steps,
            fallback_max_goal_distance=args.fallback_max_goal_distance,
            fallback_burst_min_delta=args.fallback_burst_min_delta,
            fallback_min_active_subgoal_to_goal=args.fallback_min_active_subgoal_to_goal,
            reset_seed_base=args.reset_seed_base,
            stop_on_grid_goal_distance=args.stop_on_grid_goal_distance,
            progress_prefix=f"selector={selector}",
        )
        for row in rows:
            row["selector"] = selector
        selector_aggregate = smoke.aggregate(rows)
        print(
            "Finished selector={selector} success={success:.4f} final_d={final_d:.4f} "
            "improve={improve:.4f}".format(
                selector=selector,
                success=selector_aggregate["success"],
                final_d=selector_aggregate["final_goal_distance"],
                improve=selector_aggregate["goal_distance_improvement"],
            ),
            flush=True,
        )
        selector_results.append(
            dict(name=selector, aggregate=selector_aggregate, episodes=rows)
        )
    return task_ids, selector_results


def aggregate_xy_selector_smoke(env, value_agent, train_dataset, context, args):
    task_ids = parse_int_list(args.task_ids)
    selectors = [smoke.normalize_selector(name) for name in smoke.parse_str_list(args.selectors)]
    selector_results = []
    for selector_idx, selector in enumerate(selectors):
        print(f"Starting selector={selector}", flush=True)
        policy = ValueSubgoalXYPolicy(
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
            budgets=ar.parse_int_list(args.budgets),
            value_gate_threshold=args.value_gate_threshold,
            support_gate_left_frac=args.support_gate_left_frac,
            support_frontier_left_gate=args.support_frontier_left_gate,
            support_frontier_min_progress_frac=args.support_frontier_min_progress_frac,
            support_frontier_max_xy_factor=args.support_frontier_max_xy_factor,
            support_path_horizon_mode=args.support_path_horizon_mode,
            subgoal_sample_mode=args.subgoal_sample_mode,
            candidate_sample_mode=args.candidate_sample_mode,
            choice_cache_mode=args.choice_cache_mode,
            require_goal_progress=args.require_goal_progress,
            final_goal_switch_distance=args.final_goal_switch_distance,
            action_gain=args.xy_controller_gain,
        )
        rows = smoke.run_policy_smoke(
            env,
            policy,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            subgoal_commit_steps=args.subgoal_commit_steps,
            subgoal_replan_distance=args.subgoal_replan_distance,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_min_delta=args.early_stop_min_delta,
            fallback_selector=args.fallback_selector,
            fallback_patience=args.fallback_patience,
            fallback_min_steps=args.fallback_min_steps,
            fallback_min_delta=args.fallback_min_delta,
            fallback_max_action_frac=args.fallback_max_action_frac,
            fallback_burst_steps=args.fallback_burst_steps,
            fallback_cooldown_steps=args.fallback_cooldown_steps,
            fallback_max_goal_distance=args.fallback_max_goal_distance,
            fallback_burst_min_delta=args.fallback_burst_min_delta,
            fallback_min_active_subgoal_to_goal=args.fallback_min_active_subgoal_to_goal,
            reset_seed_base=args.reset_seed_base,
            stop_on_grid_goal_distance=args.stop_on_grid_goal_distance,
            progress_prefix=f"selector={selector}",
        )
        for row in rows:
            row["selector"] = selector
        selector_aggregate = smoke.aggregate(rows)
        print(
            "Finished selector={selector} success={success:.4f} final_d={final_d:.4f} "
            "improve={improve:.4f}".format(
                selector=selector,
                success=selector_aggregate["success"],
                final_d=selector_aggregate["final_goal_distance"],
                improve=selector_aggregate["goal_distance_improvement"],
            ),
            flush=True,
        )
        selector_results.append(
            dict(name=selector, aggregate=selector_aggregate, episodes=rows)
        )
    return task_ids, selector_results


def markdown(result):
    lines = [
        "# BMM subgoal BC-controller smoke",
        "",
        f"env: `{result['env_name']}`",
        f"bc steps: `{result['bc_steps']}`, offsets: `{result['bc_offsets']}`",
        f"bc goal representation: `{result.get('bc_goal_rep', 'observation')}`",
        f"bc inference: `{result.get('bc_inference')}`",
        f"bc restore: `{result.get('bc_restore_path')}`",
        f"bc save: `{result.get('bc_save_path')}`",
        f"controller type: `{result.get('controller_type')}`",
        f"controller agent: `{result.get('controller_agent_name')}`",
        f"controller restore: `{result.get('controller_agent_restore_path')}`",
        f"final goal switch distance: `{result['final_goal_switch_distance']}`",
        f"support frontier left gate: `{result.get('support_frontier_left_gate', 'support')}`",
        f"support frontier min progress frac: `{result.get('support_frontier_min_progress_frac', 0.0)}`",
        f"support frontier max xy factor: `{result.get('support_frontier_max_xy_factor', 0.0)}`",
        f"support path horizon mode: `{result.get('support_path_horizon_mode', 'fixed')}`",
        f"subgoal sample mode: `{result.get('subgoal_sample_mode', 'random')}`",
        f"candidate sample mode: `{result.get('candidate_sample_mode', 'random')}`",
        f"fresh env per task: `{result.get('fresh_env_per_task', False)}`",
        f"require goal progress: `{result.get('require_goal_progress', False)}`",
        f"task ids: `{result['task_ids']}`, max steps: `{result['max_steps']}`",
        f"subgoal commit steps: `{result.get('subgoal_commit_steps', 1)}`",
        f"subgoal replan distance: `{result.get('subgoal_replan_distance', -1.0)}`",
        f"early stop patience: `{result.get('early_stop_patience', 0)}`",
        f"fallback selector: `{result.get('fallback_selector')}`",
        f"fallback patience: `{result.get('fallback_patience', 0)}`",
        f"fallback max action frac: `{result.get('fallback_max_action_frac', 0.0)}`",
        f"fallback burst steps: `{result.get('fallback_burst_steps', 0)}`",
        f"fallback cooldown steps: `{result.get('fallback_cooldown_steps', 0)}`",
        f"fallback max goal distance: `{result.get('fallback_max_goal_distance', 0.0)}`",
        f"fallback burst min delta: `{result.get('fallback_burst_min_delta', 0.0)}`",
        f"fallback min active subgoal-to-goal: `{result.get('fallback_min_active_subgoal_to_goal', 0.0)}`",
        f"reset seed base: `{result.get('reset_seed_base', -1)}`",
        f"stop on grid goal distance: `{result.get('stop_on_grid_goal_distance', -1.0)}`",
        f"choice cache mode: `{result.get('choice_cache_mode', 'none')}`",
        "",
        "| selector | success | grid_success | final_d | final_xy | improve | mean_step_goal | subgoal_valid | subgoal_reduce | goal_reduce | selected_src_d | selected_right_d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {success:.4f} | {grid_success:.4f} | {final:.4f} | {final_xy:.4f} | {improve:.4f} | {mean_goal:.4f} | {valid:.4f} | {subgoal_reduce:.4f} | {goal_reduce:.4f} | {src_d:.4f} | {right_d:.4f} |".format(
                name=row["name"],
                success=m["success"],
                grid_success=m.get("grid_success", float("nan")),
                final=m["final_goal_distance"],
                final_xy=m["final_goal_xy_distance"],
                improve=m["goal_distance_improvement"],
                mean_goal=m["mean_step_goal_improvement"],
                valid=m["subgoal_valid_frac"],
                subgoal_reduce=m["subgoal_reduce_frac"],
                goal_reduce=m["goal_reduce_frac"],
                src_d=m["selected_source_to_subgoal"],
                right_d=m["selected_subgoal_to_goal"],
            )
        )
    lines.extend(
        [
            "",
            "Timing:",
            "",
            "| selector | wall_s/ep | steps/s | policy_s/ep | select_s/ep | action_s/ep | env_s/ep | other_s/ep | cache_hit | grid_stop |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["selectors"]:
        m = row["aggregate"]
        lines.append(
            "| {name} | {wall:.4f} | {steps_s:.4f} | {policy:.4f} | {select:.4f} | {action:.4f} | {env:.4f} | {other:.4f} | {cache_hit:.4f} | {grid_stop:.4f} |".format(
                name=row["name"],
                wall=m.get("wall_time_sec", float("nan")),
                steps_s=m.get("steps_per_sec", float("nan")),
                policy=m.get("policy_time_sec", float("nan")),
                select=m.get("select_time_sec", float("nan")),
                action=m.get("action_time_sec", float("nan")),
                env=m.get("env_step_time_sec", float("nan")),
                other=m.get("other_time_sec", float("nan")),
                cache_hit=m.get("choice_cache_hit_frac", float("nan")),
                grid_stop=m.get("grid_goal_stopped", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Per-task:",
            "",
            "| selector | task | success | grid_success | final_d | final_xy | improve | subgoal_valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["selectors"]:
        for task_row in smoke.per_task_aggregate(row["episodes"]):
            lines.append(
                "| {name} | {task} | {success:.4f} | {grid_success:.4f} | {final:.4f} | {final_xy:.4f} | {improve:.4f} | {valid:.4f} |".format(
                    name=row["name"],
                    task=int(task_row["task_id"]),
                    success=task_row["success"],
                    grid_success=task_row.get("grid_success", float("nan")),
                    final=task_row["final_goal_distance"],
                    final_xy=task_row["final_goal_xy_distance"],
                    improve=task_row["goal_distance_improvement"],
                    valid=task_row["subgoal_valid_frac"],
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
    parser.add_argument(
        "--selectors",
        default="random,geometric_midpoint,BMM_V,oracle_midpoint",
        help=(
            "Comma-separated selectors: random, geometric_midpoint, BMM_V, "
            "BMM_V_min, BMM_V_left_gate, BMM_V_min_left_gate, "
            "BMM_V_min_budget_scan_left_gate, "
            "BMM_V_min_budget_scan_value_gate, "
            "BMM_V_min_budget_scan_value_frontier, "
            "BMM_V_min_budget_scan_right_progress, "
            "BMM_V_min_budget_scan_support_gate, "
            "BMM_V_min_budget_scan_support_frontier, "
            "BMM_V_min_budget_scan_support_path, support_path_only, oracle_midpoint, "
            "oracle_path_progress."
        ),
    )
    parser.add_argument("--task_ids", default="1,2,3")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--subgoal_commit_steps", type=int, default=1)
    parser.add_argument("--subgoal_replan_distance", type=float, default=-1.0)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Diagnostic speedup. If positive, stop a rollout after this many "
            "environment steps without improving the best geodesic goal "
            "distance. Disabled by default."
        ),
    )
    parser.add_argument(
        "--early_stop_min_steps",
        type=int,
        default=0,
        help="Minimum rollout length before --early_stop_patience can trigger.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum geodesic-distance improvement counted by early stopping.",
    )
    parser.add_argument(
        "--fallback_selector",
        default=None,
        help=(
            "Optional selector to use after --fallback_patience steps without "
            "geodesic goal-distance improvement. Disabled by default."
        ),
    )
    parser.add_argument(
        "--fallback_patience",
        type=int,
        default=0,
        help="Steps without improvement before switching to --fallback_selector.",
    )
    parser.add_argument(
        "--fallback_min_steps",
        type=int,
        default=0,
        help="Minimum rollout length before selector fallback can trigger.",
    )
    parser.add_argument(
        "--fallback_min_delta",
        type=float,
        default=0.0,
        help="Minimum geodesic-distance improvement counted by selector fallback.",
    )
    parser.add_argument(
        "--fallback_max_action_frac",
        type=float,
        default=0.0,
        help=(
            "If positive, disable fallback after fallback actions reach this "
            "fraction of elapsed episode actions."
        ),
    )
    parser.add_argument(
        "--fallback_burst_steps",
        type=int,
        default=0,
        help=(
            "If positive, use fallback only in bounded bursts of this many "
            "environment steps, then reset the fallback trigger timer. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--fallback_cooldown_steps",
        type=int,
        default=0,
        help=(
            "Optional minimum cooldown after a fallback burst expires. The "
            "patience timer is also reset at burst expiry."
        ),
    )
    parser.add_argument(
        "--fallback_max_goal_distance",
        type=float,
        default=0.0,
        help=(
            "If positive, only allow fallback when the current geodesic "
            "source-goal distance is at most this value."
        ),
    )
    parser.add_argument(
        "--fallback_burst_min_delta",
        type=float,
        default=0.0,
        help=(
            "If positive, extend a fallback burst for another "
            "--fallback_burst_steps only when geodesic goal distance improved "
            "by at least this amount during the previous burst."
        ),
    )
    parser.add_argument(
        "--fallback_min_active_subgoal_to_goal",
        type=float,
        default=0.0,
        help=(
            "If positive, only allow selector fallback when the current "
            "base subgoal's grid/geodesic subgoal-to-goal distance is at "
            "least this value. Disabled by default."
        ),
    )
    parser.add_argument("--value_gate_threshold", type=float, default=0.5)
    parser.add_argument("--support_gate_left_frac", type=float, default=1.0)
    parser.add_argument(
        "--support_frontier_left_gate",
        choices=("support", "xy", "grid", "grid_xy", "support_grid_xy"),
        default="support",
    )
    parser.add_argument("--support_frontier_min_progress_frac", type=float, default=0.0)
    parser.add_argument("--support_frontier_max_xy_factor", type=float, default=0.0)
    parser.add_argument(
        "--subgoal_sample_mode",
        choices=("random", "center", "first"),
        default="random",
        help=(
            "How to choose a representative observation for each candidate "
            "subgoal cell. 'random' matches the original stochastic evaluator; "
            "'center' and 'first' make policy extraction deterministic."
        ),
    )
    parser.add_argument(
        "--candidate_sample_mode",
        choices=("random", "topk", "stratified"),
        default="random",
        help=(
            "How to choose candidate subgoal cells when there are more cells "
            "than --num_subgoal_candidates. 'random' matches the original "
            "evaluator; 'topk' and 'stratified' make the candidate set "
            "deterministic."
        ),
    )
    parser.add_argument(
        "--support_path_horizon_mode",
        choices=("fixed", "source_goal_grid", "local_grid_min_right"),
        default="fixed",
        help=(
            "For support-frontier/path selectors, choose the support-distance "
            "cache horizon. 'fixed' uses --right_budget; 'source_goal_grid' "
            "uses max(--right_budget, ceil(grid source-goal distance)); "
            "'local_grid_min_right' uses the smallest grid right distance "
            "among local-progress candidates."
        ),
    )
    parser.add_argument(
        "--choice_cache_mode",
        choices=("none", "cell"),
        default="none",
        help=(
            "Diagnostic speedup. 'cell' caches selected subgoals by "
            "(selector, source cell, goal cell), avoiding repeated value "
            "rescoring from the same maze cell. Disabled by default."
        ),
    )
    parser.add_argument(
        "--require_goal_progress",
        action="store_true",
        help=(
            "Diagnostic extraction gate. If set, candidate subgoals must reduce "
            "grid/geodesic distance to the goal when any such candidate exists."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset_seed_base",
        type=int,
        default=-1,
        help=(
            "If nonnegative, seed each env reset with "
            "base + 1009 * task_id + episode so task resets are independent "
            "of evaluation order."
        ),
    )
    parser.add_argument(
        "--stop_on_grid_goal_distance",
        type=float,
        default=-1.0,
        help=(
            "Diagnostic speedup. If nonnegative, stop a rollout once the "
            "grid/geodesic goal distance is at most this threshold. Disabled "
            "by default; official env success remains the env's success flag."
        ),
    )
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
    parser.add_argument(
        "--bc_goal_rep",
        choices=("observation", "oracle"),
        default="observation",
        help=(
            "Goal representation for the scripted BC controller. "
            "'oracle' uses dataset['oracle_reps'] when available."
        ),
    )
    parser.add_argument("--bc_log_interval", type=int, default=500)
    parser.add_argument("--bc_save_path", default=None)
    parser.add_argument("--bc_restore_path", default=None)
    parser.add_argument(
        "--bc_inference",
        choices=("numpy", "jax"),
        default="numpy",
        help=(
            "Inference backend for the custom BC controller. 'numpy' avoids "
            "one JAX dispatch per environment step and is faster for "
            "sequential evaluation; 'jax' preserves the original backend."
        ),
    )
    parser.add_argument(
        "--final_goal_switch_distance",
        type=float,
        default=-1.0,
        help=(
            "If nonnegative, command the low-level BC controller toward the "
            "actual task goal whenever the grid/geodesic source-to-goal "
            "distance is at most this value."
        ),
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    parser.add_argument("--controller_type", choices=("bc", "agent", "xy"), default="bc")
    parser.add_argument("--controller_agent_restore_path", default=None)
    parser.add_argument("--controller_agent_restore_epoch", type=int, default=None)
    parser.add_argument("--controller_temperature", type=float, default=0.0)
    parser.add_argument("--xy_controller_gain", type=float, default=0.5)
    parser.add_argument(
        "--fresh_env_per_task",
        action="store_true",
        help=(
            "For BC-controller evaluation, create a fresh environment instance "
            "for each task to avoid cross-task reset/order effects."
        ),
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.fresh_env_per_task and args.controller_type != "bc":
        raise ValueError("--fresh_env_per_task is currently implemented for --controller_type=bc.")

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
    controller_agent_name = None
    controller_agent_flags = None
    if args.controller_type == "agent":
        controller_agent, controller_agent_name, controller_agent_flags = (
            restore_controller_agent(train_dataset, args)
        )
        bc_info = {}
        task_ids, selector_results = aggregate_agent_selector_smoke(
            env, value_agent, controller_agent, train_dataset, context, args
        )
    elif args.controller_type == "xy":
        bc_info = {}
        task_ids, selector_results = aggregate_xy_selector_smoke(
            env, value_agent, train_dataset, context, args
        )
    else:
        if args.bc_restore_path is not None:
            bc_model, bc_params, bc_info = load_bc(args.bc_restore_path, train_dataset, args)
            print(f"Restored BC controller from {args.bc_restore_path}")
        else:
            bc_model, bc_params, bc_info = train_bc(train_dataset, args, rng)
            if args.bc_save_path is not None:
                save_bc(args.bc_save_path, bc_params, args, bc_info)
                print(f"Saved BC controller to {args.bc_save_path}")
        task_ids, selector_results = aggregate_selector_smoke(
            env, value_agent, bc_model, bc_params, train_dataset, context, args
        )
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        task_ids=task_ids,
        episodes_per_task=int(args.episodes_per_task),
        max_steps=int(args.max_steps),
        subgoal_commit_steps=int(args.subgoal_commit_steps),
        subgoal_replan_distance=float(args.subgoal_replan_distance),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_steps=int(args.early_stop_min_steps),
        early_stop_min_delta=float(args.early_stop_min_delta),
        fallback_selector=(
            None
            if args.fallback_selector in (None, "")
            else smoke.normalize_selector(args.fallback_selector)
        ),
        fallback_patience=int(args.fallback_patience),
        fallback_min_steps=int(args.fallback_min_steps),
        fallback_min_delta=float(args.fallback_min_delta),
        fallback_max_action_frac=float(args.fallback_max_action_frac),
        fallback_burst_steps=int(args.fallback_burst_steps),
        fallback_cooldown_steps=int(args.fallback_cooldown_steps),
        fallback_max_goal_distance=float(args.fallback_max_goal_distance),
        fallback_burst_min_delta=float(args.fallback_burst_min_delta),
        fallback_min_active_subgoal_to_goal=float(args.fallback_min_active_subgoal_to_goal),
        reset_seed_base=int(args.reset_seed_base),
        stop_on_grid_goal_distance=float(args.stop_on_grid_goal_distance),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        controller_hops=int(args.controller_hops),
        num_subgoal_candidates=int(args.num_subgoal_candidates),
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        bc_steps=int(bc_info.get("bc_steps", args.bc_steps)),
        bc_offsets=str(bc_info.get("bc_offsets", args.bc_offsets)),
        bc_goal_rep=str(bc_info.get("bc_goal_rep", args.bc_goal_rep)),
        bc_inference=args.bc_inference if args.controller_type == "bc" else None,
        bc_save_path=args.bc_save_path,
        bc_restore_path=args.bc_restore_path,
        controller_type=args.controller_type,
        controller_agent_name=controller_agent_name,
        controller_agent_restore_path=args.controller_agent_restore_path,
        controller_agent_restore_epoch=args.controller_agent_restore_epoch,
        controller_temperature=float(args.controller_temperature),
        xy_controller_gain=float(args.xy_controller_gain),
        controller_agent_env_name=(
            None if controller_agent_flags is None else controller_agent_flags.get("env_name")
        ),
        final_goal_switch_distance=float(args.final_goal_switch_distance),
        value_gate_threshold=float(args.value_gate_threshold),
        support_gate_left_frac=float(args.support_gate_left_frac),
        support_frontier_left_gate=args.support_frontier_left_gate,
        support_frontier_min_progress_frac=float(args.support_frontier_min_progress_frac),
        support_frontier_max_xy_factor=float(args.support_frontier_max_xy_factor),
        support_path_horizon_mode=args.support_path_horizon_mode,
        subgoal_sample_mode=args.subgoal_sample_mode,
        candidate_sample_mode=args.candidate_sample_mode,
        choice_cache_mode=args.choice_cache_mode,
        require_goal_progress=bool(args.require_goal_progress),
        fresh_env_per_task=bool(args.fresh_env_per_task),
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
