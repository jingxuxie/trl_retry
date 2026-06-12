#!/usr/bin/env python
"""Tabular sanity checks and error-scaling artifacts for BMM reachability."""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def shortest_paths(num_states, edges):
    neighbors = [[] for _ in range(num_states)]
    for src, dst in edges:
        neighbors[src].append(dst)

    dist = np.full((num_states, num_states), np.inf, dtype=np.float32)
    for start in range(num_states):
        dist[start, start] = 0
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for next_state in neighbors[state]:
                if np.isinf(dist[start, next_state]):
                    dist[start, next_state] = dist[start, state] + 1
                    queue.append(next_state)
    return dist


def compose_reachability(left, right):
    return np.max(np.minimum(left[:, :, None], right[None, :, :]), axis=1)


def max_min_reachability(dist, budgets):
    reach = {1: (dist <= 1).astype(np.float32)}
    for horizon in budgets[1:]:
        left_horizon = horizon // 2
        right_horizon = horizon - left_horizon
        reach[horizon] = compose_reachability(
            reach[left_horizon], reach[right_horizon]
        )
    return reach


def chain_edges(num_states):
    return [(idx, idx + 1) for idx in range(num_states - 1)]


def grid_edges(width, height):
    edges = []
    for y in range(height):
        for x in range(width):
            state = y * width + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    edges.append((state, ny * width + nx))
    return edges


def assert_exact_backup(name, num_states, edges, budgets):
    dist = shortest_paths(num_states, edges)
    reach = max_min_reachability(dist, budgets)
    print(f"\n{name}")
    print("H | positives | exact_match")
    print("--|-----------|------------")
    rows = []
    for horizon in budgets:
        exact = (dist <= horizon).astype(np.float32)
        matches = np.array_equal(reach[horizon], exact)
        print(f"{horizon:2d} | {int(exact.sum()):9d} | {matches}")
        assert matches, f"{name}: max-min backup mismatch at H={horizon}"
        rows.append(
            dict(
                graph=name,
                horizon=int(horizon),
                positives=int(exact.sum()),
                exact_match=bool(matches),
            )
        )
    return rows


def residual_error_rows(max_horizon=1024, epsilon=0.02):
    horizons = [1]
    while horizons[-1] < max_horizon:
        horizons.append(horizons[-1] * 2)

    balanced = {1: epsilon}
    for horizon in horizons[1:]:
        balanced[horizon] = epsilon + max(
            balanced[horizon // 2], balanced[horizon - horizon // 2]
        )

    unbalanced = {1: epsilon}
    for horizon in range(2, max_horizon + 1):
        unbalanced[horizon] = epsilon + max(unbalanced[horizon - 1], unbalanced[1])

    additive = {horizon: epsilon * horizon for horizon in horizons}
    rows = []
    for horizon in horizons:
        rows.append(
            dict(
                horizon=int(horizon),
                bmm_balanced_sup_error=float(balanced[horizon]),
                bmm_unbalanced_sup_error=float(unbalanced[horizon]),
                additive_distance_sup_error=float(additive[horizon]),
                product_style_sup_error=float(additive[horizon]),
                td_unbalanced_sup_error=float(unbalanced[horizon]),
                bmm_balanced_mean_abs_error=float(balanced[horizon]),
                additive_distance_mean_abs_error=float(additive[horizon]),
                bmm_balanced_classification_error_bound=float(
                    min(0.5, balanced[horizon])
                ),
                additive_distance_classification_error_bound=float(
                    min(0.5, additive[horizon])
                ),
            )
        )
    return rows


def residual_error_table(max_horizon=1024, epsilon=0.02, output_dir=None):
    rows = residual_error_rows(max_horizon=max_horizon, epsilon=epsilon)

    print("\nInjected per-level residual error")
    print("H    | balanced max-min | unbalanced max-min | additive-distance")
    print("-----|------------------|--------------------|------------------")
    for row in rows:
        print(
            f"{row['horizon']:4d} | {row['bmm_balanced_sup_error']:16.4f} | "
            f"{row['bmm_unbalanced_sup_error']:18.4f} | "
            f"{row['additive_distance_sup_error']:16.4f}"
        )

    if output_dir is None:
        return rows

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return rows

    output_dir = Path(output_dir)
    plot_path = output_dir / "bmm_tabular_error_scaling.png"
    horizons = [row["horizon"] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(
        horizons,
        [row["bmm_balanced_sup_error"] for row in rows],
        marker="o",
        label="BMM balanced",
    )
    plt.plot(
        horizons,
        [row["bmm_unbalanced_sup_error"] for row in rows],
        marker="o",
        label="BMM unbalanced",
    )
    plt.plot(
        horizons,
        [row["additive_distance_sup_error"] for row in rows],
        marker="o",
        label="additive/product",
    )
    plt.xscale("log", base=2)
    plt.xlabel("Budget H")
    plt.ylabel("Sup-norm error bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved optional plot to {plot_path}")
    return rows


def graph_metadata():
    chain_sizes = (32, 64, 128, 256, 512, 1024)
    grid_sides = (4, 8, 16, 32)
    rows = []
    for num_states in chain_sizes:
        rows.append(
            dict(
                graph="directed_chain",
                num_states=int(num_states),
                diameter=int(num_states - 1),
            )
        )
    for side in grid_sides:
        rows.append(
            dict(
                graph="undirected_grid",
                side=int(side),
                num_states=int(side * side),
                diameter=int(2 * (side - 1)),
            )
        )
    return rows


def markdown_report(report):
    lines = [
        "# BMM tabular error scaling",
        "",
        (
            "Controlled residual recurrence with per-level residual "
            f"epsilon={report['epsilon']}."
        ),
        "",
        "| H | BMM balanced sup error | BMM unbalanced sup error | additive/product sup error |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["error_scaling"]:
        lines.append(
            "| {horizon} | {bmm:.4f} | {unbalanced:.4f} | {additive:.4f} |".format(
                horizon=row["horizon"],
                bmm=row["bmm_balanced_sup_error"],
                unbalanced=row["bmm_unbalanced_sup_error"],
                additive=row["additive_distance_sup_error"],
            )
        )
    lines.extend(
        [
            "",
            "Exact max-min backup checks:",
            "",
            "| graph | H | positives | exact match |",
            "|---|---:|---:|---|",
        ]
    )
    for row in report["exact_backup_checks"]:
        lines.append(
            f"| {row['graph']} | {row['horizon']} | {row['positives']} | {row['exact_match']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_horizon", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[1] / "exp"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = (1, 2, 4, 8)
    exact_rows = []
    exact_rows.extend(assert_exact_backup("Directed chain", 12, chain_edges(12), budgets))
    exact_rows.extend(assert_exact_backup("Undirected 4x4 grid", 16, grid_edges(4, 4), budgets))
    error_rows = residual_error_table(
        max_horizon=args.max_horizon,
        epsilon=args.epsilon,
        output_dir=output_dir,
    )
    report = dict(
        epsilon=float(args.epsilon),
        max_horizon=int(args.max_horizon),
        exact_backup_checks=exact_rows,
        graph_metadata=graph_metadata(),
        error_scaling=error_rows,
    )
    json_path = output_dir / "bmm_tabular_error_scaling.json"
    md_path = output_dir / "bmm_tabular_error_scaling.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(markdown_report(report))
    print(f"Saved tabular error-scaling JSON to {json_path}")
    print(f"Saved tabular error-scaling markdown to {md_path}")


if __name__ == "__main__":
    main()
