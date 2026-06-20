#!/usr/bin/env python
"""Inspect OGBench dataset shapes and representation ranges."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def finite_range(array, max_rows):
    array = np.asarray(array)
    if array.ndim == 0:
        return None
    if len(array) > int(max_rows):
        idxs = np.linspace(0, len(array) - 1, int(max_rows)).astype(np.int64)
        array = array[idxs]
    if not np.issubdtype(array.dtype, np.number):
        return None
    flat = array.reshape(len(array), -1)
    return np.stack([np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)], axis=-1)


def summarize_array(name, value, max_rows, max_dims):
    array = np.asarray(value)
    print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
    ranges = finite_range(array, max_rows)
    if ranges is None:
        return
    dims = min(len(ranges), int(max_dims))
    for dim in range(dims):
        print(f"    dim {dim}: [{ranges[dim, 0]:.6g}, {ranges[dim, 1]:.6g}]")
    if len(ranges) > dims:
        print(f"    ... {len(ranges) - dims} more flattened dims")


def summarize_dataset(name, dataset, args):
    print(f"\n{name} dataset")
    keys = sorted(dataset._dict.keys() if hasattr(dataset, "_dict") else dataset.keys())
    print(f"  keys: {keys}")
    for key in keys:
        summarize_array(key, dataset[key], args.range_sample_size, args.max_range_dims)

    if "valids" in dataset:
        valids = np.asarray(dataset["valids"]) > 0
        print(f"  valid transitions: {int(valids.sum())} / {len(valids)}")
    if "terminals" in dataset:
        terminals = np.asarray(dataset["terminals"]) > 0
        print(f"  terminals: {int(terminals.sum())} / {len(terminals)}")
    if "oracle_reps" in dataset and "valids" in dataset:
        reps = np.asarray(dataset["oracle_reps"], dtype=np.float32)
        valid_idxs = np.nonzero(np.asarray(dataset["valids"]) > 0)[0]
        valid_idxs = valid_idxs[valid_idxs < len(reps) - 1]
        if len(valid_idxs) > 0:
            if len(valid_idxs) > args.range_sample_size:
                valid_idxs = valid_idxs[
                    np.linspace(0, len(valid_idxs) - 1, args.range_sample_size).astype(
                        np.int64
                    )
                ]
            step = np.linalg.norm(reps[valid_idxs + 1] - reps[valid_idxs], axis=-1)
            step = step[np.isfinite(step) & (step > 1e-8)]
            if len(step) > 0:
                print(
                    "  oracle_rep step norm: "
                    f"median={np.median(step):.6g}, p90={np.percentile(step, 90):.6g}"
                )


def unwrap_chain(env):
    chain = []
    cur = env
    for _ in range(16):
        chain.append(type(cur).__module__ + "." + type(cur).__name__)
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return chain


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--range_sample_size", type=int, default=200000)
    parser.add_argument("--max_range_dims", type=int, default=32)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_path = dataset_path_from_dir(args.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    print(f"env_name: {args.env_name}")
    print(f"env type chain: {unwrap_chain(env)}")
    summarize_dataset("train", train_dataset, args)
    summarize_dataset("validation", val_dataset, args)


if __name__ == "__main__":
    main()
