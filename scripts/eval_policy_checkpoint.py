#!/usr/bin/env python
"""Evaluate a saved goal-conditioned policy checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from agents.bmm_trl import get_config as get_bmm_config
from agents.gcfbc import get_config as get_gcfbc_config
from agents.gciql import get_config as get_gciql_config
from agents.trl import get_config as get_trl_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

AGENT_CONFIGS = {
    "bmm_trl": get_bmm_config,
    "gcfbc": get_gcfbc_config,
    "gciql": get_gciql_config,
    "trl": get_trl_config,
}


def parse_int_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


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


def load_config(restore_path):
    flags_path = Path(restore_path) / "flags.json"
    if not flags_path.exists():
        raise FileNotFoundError(f"Missing flags.json next to checkpoint: {flags_path}")
    flags = json.loads(flags_path.read_text())
    agent_flags = flags.get("agent", {})
    agent_name = agent_flags.get("agent_name")
    if agent_name not in AGENT_CONFIGS:
        raise ValueError(
            f"Unsupported checkpoint agent {agent_name!r}. "
            f"Expected one of {sorted(AGENT_CONFIGS)}."
        )
    config = AGENT_CONFIGS[agent_name]()
    merge_config(config, agent_flags)
    return flags, config


def metric(stats, *names):
    for name in names:
        if name in stats:
            return float(stats[name])
    return float("nan")


def markdown(result):
    lines = [
        "# Policy checkpoint evaluation",
        "",
        f"checkpoint: `{result['restore_path']}:{result['restore_epoch']}`",
        f"env: `{result['env_name']}`",
        f"agent: `{result['agent_name']}`",
        f"eval episodes/task: `{result['eval_episodes']}`",
        "",
        "| task | success | return | length |",
        "|---:|---:|---:|---:|",
    ]
    if result.get("actor_budget_mode") is not None:
        lines.insert(5, f"actor budget mode: `{result['actor_budget_mode']}`")
    for row in result["tasks"]:
        lines.append(
            "| {task_id} | {success:.4f} | {ret:.4f} | {length:.4f} |".format(
                task_id=row["task_id"],
                success=row["success"],
                ret=row["episode_return"],
                length=row["length"],
            )
        )
    lines.extend(
        [
            "",
            "| overall success | overall return |",
            "|---:|---:|",
            "| {success:.4f} | {ret:.4f} |".format(
                success=result["overall"]["success"],
                ret=result["overall"]["episode_return"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_path", required=True)
    parser.add_argument("--restore_epoch", type=int, required=True)
    parser.add_argument("--env_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--task_ids", default="1,2,3")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--actor_budget_mode", default=None, choices=("max", "scan"))
    parser.add_argument("--actor_budget_threshold", type=float, default=None)
    parser.add_argument("--frs_num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    flags, config = load_config(args.restore_path)
    env_name = args.env_name or flags["env_name"]
    if args.actor_budget_mode is not None:
        if config.agent_name != "bmm_trl":
            raise ValueError("--actor_budget_mode is only valid for bmm_trl checkpoints.")
        config.actor_budget_mode = args.actor_budget_mode
    if args.actor_budget_threshold is not None:
        if config.agent_name != "bmm_trl":
            raise ValueError(
                "--actor_budget_threshold is only valid for bmm_trl checkpoints."
            )
        config.actor_budget_threshold = float(args.actor_budget_threshold)
    if args.frs_num_samples is not None:
        if "frs" not in config:
            raise ValueError("--frs_num_samples requires a checkpoint config with frs.")
        config.frs.num_samples = int(args.frs_num_samples)

    np.random.seed(int(args.seed))
    env, train_dataset, _ = make_env_and_datasets(env_name, dataset_path=None)
    train_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_dataset.sample(1)
    agent = agents[config.agent_name].create(int(args.seed), example_batch, config)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    rows = []
    for task_id in parse_int_list(args.task_ids):
        stats, _, _ = evaluate(
            agent=agent,
            env=env,
            env_name=env_name,
            goal_conditioned=True,
            task_id=int(task_id),
            config=config,
            num_eval_episodes=int(args.eval_episodes),
            num_video_episodes=0,
            eval_temperature=float(args.eval_temperature),
        )
        rows.append(
            dict(
                task_id=int(task_id),
                success=metric(stats, "success", "episode.success"),
                episode_return=metric(stats, "episode.return"),
                length=metric(stats, "episode.length"),
                raw=stats,
            )
        )

    result = dict(
        restore_path=str(args.restore_path),
        restore_epoch=int(args.restore_epoch),
        env_name=env_name,
        agent_name=str(config.agent_name),
        actor_budget_mode=(
            str(config.actor_budget_mode) if config.agent_name == "bmm_trl" else None
        ),
        actor_budget_threshold=(
            float(config.actor_budget_threshold)
            if config.agent_name == "bmm_trl"
            else None
        ),
        frs_num_samples=(int(config.frs.num_samples) if "frs" in config else None),
        eval_episodes=int(args.eval_episodes),
        task_ids=parse_int_list(args.task_ids),
        tasks=rows,
        overall=dict(
            success=float(np.nanmean([row["success"] for row in rows])),
            episode_return=float(np.nanmean([row["episode_return"] for row in rows])),
        ),
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
