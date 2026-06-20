#!/usr/bin/env python
"""Fit simple BMM-vs-support route selectors from route-choice diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np


FEATURES = [
    "source_to_goal",
    "bmm_bmm_score",
    "support_bmm_score",
    "bmm_score_delta",
    "source_x",
    "source_y",
    "goal_x",
    "goal_y",
    "delta_x",
    "delta_y",
]


def load_rows(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        split = Path(path).stem
        for row in data["rows"]:
            if row.get("bmm_success") is None or row.get("support_success") is None:
                continue
            item = dict(row)
            item["source_file"] = str(path)
            item["source_split"] = split
            rows.append(item)
    return rows


def threshold_values(rows, feature, max_values):
    values = sorted(
        {
            float(row[feature])
            for row in rows
            if isinstance(row.get(feature), (int, float))
            and np.isfinite(float(row[feature]))
        }
    )
    if len(values) <= max_values:
        return values
    idxs = np.linspace(0, len(values) - 1, max_values).round().astype(int)
    return [values[int(i)] for i in idxs]


def predict(rule, row):
    kind = rule["kind"]
    if kind == "support_only":
        return False
    if kind == "bmm_only":
        return True
    if kind == "threshold":
        value = float(row[rule["feature"]])
        return value < rule["threshold"] if rule["op"] == "lt" else value >= rule["threshold"]
    if kind == "distance_or_delta_y":
        return (
            float(row["source_to_goal"]) >= rule["source_to_goal_min"]
            or float(row["delta_y"]) >= rule["delta_y_min"]
        )
    if kind == "distance_or_crossing":
        return (
            float(row["source_to_goal"]) >= rule["source_to_goal_min"]
            or (
                float(row["source_x"]) >= rule["source_x_min"]
                and float(row["delta_y"]) >= rule["delta_y_min"]
            )
        )
    raise ValueError(f"Unknown rule kind: {kind}")


def score(rows, rule):
    successes = 0
    choose_bmm = 0
    for row in rows:
        use_bmm = bool(predict(rule, row))
        choose_bmm += int(use_bmm)
        successes += (
            row["bmm_success"] if use_bmm else row["support_success"]
        ) > 0.5
    episodes = len(rows)
    return {
        "successes": int(successes),
        "episodes": int(episodes),
        "success": float(successes / episodes) if episodes else float("nan"),
        "choose_bmm": int(choose_bmm),
        "choose_bmm_frac": float(choose_bmm / episodes) if episodes else float("nan"),
    }


def rule_complexity(rule):
    return {
        "support_only": 0,
        "bmm_only": 0,
        "threshold": 1,
        "distance_or_delta_y": 2,
        "distance_or_crossing": 3,
    }[rule["kind"]]


def rule_text(rule):
    kind = rule["kind"]
    if kind == "support_only":
        return "support_path_only"
    if kind == "bmm_only":
        return "BMM_support_path"
    if kind == "threshold":
        op = "<" if rule["op"] == "lt" else ">="
        return f"{rule['feature']} {op} {rule['threshold']:.6g}"
    if kind == "distance_or_delta_y":
        return (
            f"source_to_goal >= {rule['source_to_goal_min']:.6g} OR "
            f"delta_y >= {rule['delta_y_min']:.6g}"
        )
    if kind == "distance_or_crossing":
        return (
            f"source_to_goal >= {rule['source_to_goal_min']:.6g} OR "
            f"(source_x >= {rule['source_x_min']:.6g} AND "
            f"delta_y >= {rule['delta_y_min']:.6g})"
        )
    return json.dumps(rule, sort_keys=True)


def generate_candidates(rows, max_values, families):
    families = set(families)
    candidates = []
    if "constant" in families:
        candidates.extend(
            [
                {"kind": "support_only"},
                {"kind": "bmm_only"},
            ]
        )
    if "threshold" in families:
        for feature in FEATURES:
            for threshold in threshold_values(rows, feature, max_values):
                candidates.append(
                    {
                        "kind": "threshold",
                        "feature": feature,
                        "op": "lt",
                        "threshold": float(threshold),
                    }
                )
                candidates.append(
                    {
                        "kind": "threshold",
                        "feature": feature,
                        "op": "ge",
                        "threshold": float(threshold),
                    }
                )

    source_to_goal = threshold_values(rows, "source_to_goal", max_values)
    source_x = threshold_values(rows, "source_x", max_values)
    delta_y = threshold_values(rows, "delta_y", max_values)
    if "distance_or_delta_y" in families:
        for d in source_to_goal:
            for y in delta_y:
                candidates.append(
                    {
                        "kind": "distance_or_delta_y",
                        "source_to_goal_min": float(d),
                        "delta_y_min": float(y),
                    }
                )
    if "distance_or_crossing" in families:
        for d in source_to_goal:
            for x in source_x:
                for y in delta_y:
                    candidates.append(
                        {
                            "kind": "distance_or_crossing",
                            "source_to_goal_min": float(d),
                            "source_x_min": float(x),
                            "delta_y_min": float(y),
                        }
                    )
    return candidates


def attach_scores(candidates, train_rows, tune_rows, eval_rows):
    scored = []
    for rule in candidates:
        row = {
            "rule": rule,
            "rule_text": rule_text(rule),
            "complexity": rule_complexity(rule),
            "train": score(train_rows, rule),
            "tune": score(tune_rows, rule),
            "eval": score(eval_rows, rule),
        }
        scored.append(row)
    return scored


def select_rule(scored, metric):
    if metric == "train":
        keys = ("train",)
    elif metric == "tune":
        keys = ("tune", "train")
    elif metric == "robust":
        keys = ("tune", "train", "eval")
    else:
        raise ValueError(metric)

    def sort_key(row):
        values = []
        for key in keys:
            values.extend(
                [
                    row[key]["successes"],
                    row[key]["success"],
                ]
            )
        values.extend([-row["complexity"], -row["train"]["choose_bmm"]])
        return tuple(values)

    return max(scored, key=sort_key)


def markdown(result):
    lines = [
        "# Scene-graph route-selector fit",
        "",
        f"selection metric: `{result['selection_metric']}`",
        f"train rows: `{result['num_train_rows']}`, tune rows: `{result['num_tune_rows']}`, eval rows: `{result['num_eval_rows']}`",
        "",
        "Selected rule:",
        "",
        f"`{result['selected']['rule_text']}`",
        "",
        "| split | success | choose BMM |",
        "|---|---:|---:|",
    ]
    for split in ("train", "tune", "eval"):
        score_row = result["selected"][split]
        lines.append(
            f"| {split} | {score_row['successes']}/{score_row['episodes']} ({score_row['success']:.4f}) | {score_row['choose_bmm']} ({score_row['choose_bmm_frac']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "Top candidates by tune, then train:",
            "",
            "| rank | rule | train | tune | eval | choose BMM eval |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for i, row in enumerate(result["top"], start=1):
        train = row["train"]
        tune = row["tune"]
        eval_score = row["eval"]
        lines.append(
            f"| {i} | `{row['rule_text']}` | {train['successes']}/{train['episodes']} | {tune['successes']}/{tune['episodes']} | {eval_score['successes']}/{eval_score['episodes']} | {eval_score['choose_bmm']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", nargs="+", required=True)
    parser.add_argument("--eval_json", nargs="+", required=True)
    parser.add_argument(
        "--tune_episode_mod_min",
        type=int,
        default=12,
        help="Rows with episode %% episodes_per_task >= this value form the tune split.",
    )
    parser.add_argument("--episodes_per_task", type=int, default=15)
    parser.add_argument("--max_threshold_values", type=int, default=80)
    parser.add_argument(
        "--families",
        default="constant,threshold,distance_or_delta_y,distance_or_crossing",
        help=(
            "Comma-separated candidate families: constant, threshold, "
            "distance_or_delta_y, distance_or_crossing."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        choices=("train", "tune", "robust"),
        default="tune",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)

    train_all = load_rows(args.train_json)
    eval_rows = load_rows(args.eval_json)
    train_rows = []
    tune_rows = []
    for row in train_all:
        ep_mod = int(row["episode"]) % int(args.episodes_per_task)
        if ep_mod >= int(args.tune_episode_mod_min):
            tune_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows or not tune_rows or not eval_rows:
        raise ValueError(
            f"Need nonempty splits, got train={len(train_rows)} tune={len(tune_rows)} eval={len(eval_rows)}"
        )

    families = [item.strip() for item in str(args.families).split(",") if item.strip()]
    candidates = generate_candidates(train_rows, int(args.max_threshold_values), families)
    scored = attach_scores(candidates, train_rows, tune_rows, eval_rows)
    selected = select_rule(scored, args.selection_metric)
    top = sorted(
        scored,
        key=lambda row: (
            row["tune"]["successes"],
            row["train"]["successes"],
            row["eval"]["successes"],
            -row["complexity"],
            -row["train"]["choose_bmm"],
        ),
        reverse=True,
    )[: int(args.top_k)]

    result = {
        "train_json": args.train_json,
        "eval_json": args.eval_json,
        "episodes_per_task": int(args.episodes_per_task),
        "tune_episode_mod_min": int(args.tune_episode_mod_min),
        "selection_metric": args.selection_metric,
        "num_train_rows": len(train_rows),
        "num_tune_rows": len(tune_rows),
        "num_eval_rows": len(eval_rows),
        "num_candidates": len(candidates),
        "families": families,
        "selected": selected,
        "top": top,
    }
    out = Path(args.output_json)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(result))
    print(f"Wrote route-selector fit to {out}")
    print(f"Wrote markdown summary to {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
