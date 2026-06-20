"""Conservative observed-transition graph utilities for reachability diagnostics."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_xy_dims(value):
    """Parse an xy dim flag such as '0,1'."""
    if isinstance(value, str):
        dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        dims = tuple(int(part) for part in value)
    if len(dims) != 2:
        raise ValueError(f"Expected exactly two xy dims, got {dims}.")
    return dims


def dataset_xy(dataset, xy_dims=(0, 1)):
    """Return xy positions from a Dataset-like object."""
    xy_dims = parse_xy_dims(xy_dims)
    observations = np.asarray(dataset["observations"])
    if observations.ndim != 2:
        raise ValueError(f"Expected 2D observations, got shape {observations.shape}.")
    if max(xy_dims) >= observations.shape[-1]:
        raise ValueError(
            f"xy dims {xy_dims} exceed observation dim {observations.shape[-1]}."
        )
    return observations[:, xy_dims].astype(np.float32)


def parse_feature_dims(value, feature_dim):
    """Parse feature dims, allowing 'all' for every dimension."""
    if value is None:
        return tuple(range(int(feature_dim)))
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "all":
            return tuple(range(int(feature_dim)))
        dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        dims = tuple(int(part) for part in value)
    if not dims:
        raise ValueError("Expected at least one feature dim.")
    if min(dims) < 0 or max(dims) >= int(feature_dim):
        raise ValueError(f"dims {dims} exceed feature dim {feature_dim}.")
    return dims


def dataset_features(dataset, rep_key="observations", dims=None):
    """Return selected continuous features from a Dataset-like object."""
    if rep_key not in dataset:
        keys = sorted(dataset._dict.keys() if hasattr(dataset, "_dict") else dataset.keys())
        raise KeyError(f"Dataset has no key {rep_key!r}; available keys: {keys}.")
    features = np.asarray(dataset[rep_key])
    if features.ndim != 2:
        raise ValueError(f"Expected 2D {rep_key}, got shape {features.shape}.")
    dims = parse_feature_dims(dims, features.shape[-1])
    return features[:, dims].astype(np.float32)


def valid_transition_indices(dataset):
    """Return indices whose next observation belongs to the same trajectory."""
    size = int(dataset.size if hasattr(dataset, "size") else len(dataset["observations"]))
    if "valids" in dataset:
        valid = np.asarray(dataset["valids"]) > 0
        idxs = np.nonzero(valid)[0]
    elif "terminals" in dataset:
        terminals = np.asarray(dataset["terminals"]) > 0
        idxs = np.nonzero(~terminals)[0]
    else:
        idxs = np.arange(size - 1)
    return idxs[idxs < size - 1].astype(np.int32)


def source_indices(dataset):
    """Return valid source-state indices for diagnostic pair sampling."""
    if hasattr(dataset, "valid_idxs"):
        idxs = np.asarray(dataset.valid_idxs, dtype=np.int32)
    else:
        idxs = valid_transition_indices(dataset)
    return idxs[idxs < len(dataset["observations"])].astype(np.int32)


def median_step_xy(datasets, xy_dims=(0, 1), max_samples=200000):
    """Estimate the median nonzero one-step xy displacement."""
    return median_step_features(
        datasets, rep_key="observations", dims=parse_xy_dims(xy_dims), max_samples=max_samples
    )


def median_step_features(datasets, rep_key="observations", dims=None, max_samples=200000):
    """Estimate the median nonzero one-step displacement in representation space."""
    deltas = []
    for dataset in datasets:
        features = dataset_features(dataset, rep_key=rep_key, dims=dims)
        idxs = valid_transition_indices(dataset)
        if len(idxs) == 0:
            continue
        if len(idxs) > max_samples:
            idxs = np.linspace(0, len(idxs) - 1, max_samples).astype(np.int64)
            idxs = valid_transition_indices(dataset)[idxs]
        step = np.linalg.norm(features[idxs + 1] - features[idxs], axis=-1)
        step = step[step > 1e-8]
        if len(step) > 0:
            deltas.append(step)
    if not deltas:
        raise ValueError(f"No nonzero valid {rep_key} transitions found.")
    return float(np.median(np.concatenate(deltas)))


def _bin_xy(xy, origin, bin_size):
    return np.floor((xy - origin[None, :]) / float(bin_size)).astype(np.int32)


def _edge_pairs_for_dataset(dataset, state_to_bin):
    idxs = valid_transition_indices(dataset)
    src = state_to_bin[idxs]
    dst = state_to_bin[idxs + 1]
    mask = (src >= 0) & (dst >= 0) & (src != dst)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    src = src[mask]
    dst = dst[mask]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.stack([lo, hi], axis=-1).astype(np.int32)


def _map_coords_to_existing_bins(coords, unique_coords):
    coord_to_bin = {
        tuple(np.asarray(coord, dtype=np.int32).tolist()): idx
        for idx, coord in enumerate(np.asarray(unique_coords, dtype=np.int32))
    }
    result = np.full(len(coords), -1, dtype=np.int32)
    for idx, coord in enumerate(np.asarray(coords, dtype=np.int32)):
        result[idx] = coord_to_bin.get(tuple(coord.tolist()), -1)
    return result


def build_dataset_position_graph(
    train_dataset,
    val_dataset,
    xy_dims=(0, 1),
    bin_size=None,
    bin_size_factor=2.0,
):
    """Build an undirected graph from observed dataset transitions.

    Nodes are occupied xy bins from train+validation observations. Edges are
    only consecutive observed transitions, so this avoids geometric shortcuts
    through walls.
    """
    xy_dims = parse_xy_dims(xy_dims)
    train_xy = dataset_xy(train_dataset, xy_dims)
    val_xy = dataset_xy(val_dataset, xy_dims)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    if bin_size is None:
        bin_size = max(1e-6, float(bin_size_factor) * median_step)
    bin_size = float(bin_size)

    all_xy = np.concatenate([train_xy, val_xy], axis=0)
    origin = np.floor(all_xy.min(axis=0) / bin_size) * bin_size
    all_coords = _bin_xy(all_xy, origin, bin_size)
    unique_coords, inverse = np.unique(all_coords, axis=0, return_inverse=True)
    train_state_to_bin = inverse[: len(train_xy)].astype(np.int32)
    val_state_to_bin = inverse[len(train_xy) :].astype(np.int32)
    bin_centers = origin[None, :] + (unique_coords.astype(np.float32) + 0.5) * bin_size

    edge_pairs = np.concatenate(
        [
            _edge_pairs_for_dataset(train_dataset, train_state_to_bin),
            _edge_pairs_for_dataset(val_dataset, val_state_to_bin),
        ],
        axis=0,
    )
    if len(edge_pairs) > 0:
        edge_pairs = np.unique(edge_pairs, axis=0)

    metadata = dict(
        xy_dims=list(xy_dims),
        bin_size=bin_size,
        median_step_xy=median_step,
        bin_size_factor=float(bin_size_factor),
        env_steps_per_graph_edge=max(1.0, bin_size / max(median_step, 1e-6)),
        graph_kind="dataset_position_observed_transition",
    )
    return dict(
        bin_centers=bin_centers.astype(np.float32),
        bin_coords=unique_coords.astype(np.int32),
        edge_src=edge_pairs[:, 0].astype(np.int32),
        edge_dst=edge_pairs[:, 1].astype(np.int32),
        train_state_to_bin=train_state_to_bin,
        val_state_to_bin=val_state_to_bin,
        metadata=metadata,
    )


def build_train_support_position_graph(
    train_dataset,
    val_dataset,
    xy_dims=(0, 1),
    bin_size=None,
    bin_size_factor=2.0,
):
    """Build a graph from train transitions and map validation states to train bins."""
    xy_dims = parse_xy_dims(xy_dims)
    train_xy = dataset_xy(train_dataset, xy_dims)
    val_xy = dataset_xy(val_dataset, xy_dims)
    median_step = median_step_xy((train_dataset,), xy_dims)
    if bin_size is None:
        bin_size = max(1e-6, float(bin_size_factor) * median_step)
    bin_size = float(bin_size)

    origin = np.floor(train_xy.min(axis=0) / bin_size) * bin_size
    train_coords = _bin_xy(train_xy, origin, bin_size)
    unique_coords, train_inverse = np.unique(
        train_coords, axis=0, return_inverse=True
    )
    train_state_to_bin = train_inverse.astype(np.int32)
    val_coords = _bin_xy(val_xy, origin, bin_size)
    val_state_to_bin = _map_coords_to_existing_bins(val_coords, unique_coords)
    bin_centers = origin[None, :] + (unique_coords.astype(np.float32) + 0.5) * bin_size

    edge_pairs = _edge_pairs_for_dataset(train_dataset, train_state_to_bin)
    if len(edge_pairs) > 0:
        edge_pairs = np.unique(edge_pairs, axis=0)

    metadata = dict(
        xy_dims=list(xy_dims),
        bin_size=bin_size,
        median_step_xy=median_step,
        bin_size_factor=float(bin_size_factor),
        env_steps_per_graph_edge=max(1.0, bin_size / max(median_step, 1e-6)),
        graph_kind="train_support_position_observed_transition",
        val_mapped_count=int(np.sum(val_state_to_bin >= 0)),
        val_unmapped_count=int(np.sum(val_state_to_bin < 0)),
        val_mapped_frac=float(np.mean(val_state_to_bin >= 0)),
    )
    return dict(
        bin_centers=bin_centers.astype(np.float32),
        bin_coords=unique_coords.astype(np.int32),
        edge_src=edge_pairs[:, 0].astype(np.int32),
        edge_dst=edge_pairs[:, 1].astype(np.int32),
        train_state_to_bin=train_state_to_bin,
        val_state_to_bin=val_state_to_bin,
        metadata=metadata,
    )


def build_dataset_representation_graph(
    train_dataset,
    val_dataset,
    rep_key="observations",
    dims=None,
    bin_size=None,
    bin_size_factor=2.0,
):
    """Build an undirected graph from observed transitions in a vector representation.

    Nodes are occupied bins in the selected representation. Edges are only
    consecutive observed transitions. This generalizes the PointMaze xy graph to
    oracle-representation datasets such as scene-play-oraclerep.
    """
    sample = np.asarray(train_dataset[rep_key])
    dims = parse_feature_dims(dims, sample.shape[-1])
    train_features = dataset_features(train_dataset, rep_key=rep_key, dims=dims)
    val_features = dataset_features(val_dataset, rep_key=rep_key, dims=dims)
    median_step = median_step_features(
        (train_dataset, val_dataset), rep_key=rep_key, dims=dims
    )
    if bin_size is None:
        bin_size = max(1e-6, float(bin_size_factor) * median_step)
    bin_size = float(bin_size)

    all_features = np.concatenate([train_features, val_features], axis=0)
    origin = np.floor(all_features.min(axis=0) / bin_size) * bin_size
    all_coords = _bin_xy(all_features, origin, bin_size)
    unique_coords, inverse = np.unique(all_coords, axis=0, return_inverse=True)
    train_state_to_bin = inverse[: len(train_features)].astype(np.int32)
    val_state_to_bin = inverse[len(train_features) :].astype(np.int32)
    bin_centers = (
        origin[None, :] + (unique_coords.astype(np.float32) + 0.5) * bin_size
    )

    edge_pairs = np.concatenate(
        [
            _edge_pairs_for_dataset(train_dataset, train_state_to_bin),
            _edge_pairs_for_dataset(val_dataset, val_state_to_bin),
        ],
        axis=0,
    )
    if len(edge_pairs) > 0:
        edge_pairs = np.unique(edge_pairs, axis=0)

    metadata = dict(
        rep_key=str(rep_key),
        rep_dims=[int(x) for x in dims],
        bin_size=bin_size,
        median_step_rep=median_step,
        bin_size_factor=float(bin_size_factor),
        env_steps_per_graph_edge=max(1.0, bin_size / max(median_step, 1e-6)),
        graph_kind="dataset_representation_observed_transition",
    )
    return dict(
        bin_centers=bin_centers.astype(np.float32),
        bin_coords=unique_coords.astype(np.int32),
        edge_src=edge_pairs[:, 0].astype(np.int32),
        edge_dst=edge_pairs[:, 1].astype(np.int32),
        train_state_to_bin=train_state_to_bin,
        val_state_to_bin=val_state_to_bin,
        metadata=metadata,
    )


def build_train_support_representation_graph(
    train_dataset,
    val_dataset,
    rep_key="observations",
    dims=None,
    bin_size=None,
    bin_size_factor=2.0,
):
    """Build a graph from train transitions in a representation space.

    Validation states are assigned to an existing train bin when possible and
    marked as unsupported (-1) otherwise. No validation transitions or
    validation-only nodes are added to the graph.
    """
    sample = np.asarray(train_dataset[rep_key])
    dims = parse_feature_dims(dims, sample.shape[-1])
    train_features = dataset_features(train_dataset, rep_key=rep_key, dims=dims)
    val_features = dataset_features(val_dataset, rep_key=rep_key, dims=dims)
    median_step = median_step_features(
        (train_dataset,), rep_key=rep_key, dims=dims
    )
    if bin_size is None:
        bin_size = max(1e-6, float(bin_size_factor) * median_step)
    bin_size = float(bin_size)

    origin = np.floor(train_features.min(axis=0) / bin_size) * bin_size
    train_coords = _bin_xy(train_features, origin, bin_size)
    unique_coords, train_inverse = np.unique(
        train_coords, axis=0, return_inverse=True
    )
    train_state_to_bin = train_inverse.astype(np.int32)
    val_coords = _bin_xy(val_features, origin, bin_size)
    val_state_to_bin = _map_coords_to_existing_bins(val_coords, unique_coords)
    bin_centers = (
        origin[None, :] + (unique_coords.astype(np.float32) + 0.5) * bin_size
    )

    edge_pairs = _edge_pairs_for_dataset(train_dataset, train_state_to_bin)
    if len(edge_pairs) > 0:
        edge_pairs = np.unique(edge_pairs, axis=0)

    metadata = dict(
        rep_key=str(rep_key),
        rep_dims=[int(x) for x in dims],
        bin_size=bin_size,
        median_step_rep=median_step,
        bin_size_factor=float(bin_size_factor),
        env_steps_per_graph_edge=max(1.0, bin_size / max(median_step, 1e-6)),
        graph_kind="train_support_representation_observed_transition",
        val_mapped_count=int(np.sum(val_state_to_bin >= 0)),
        val_unmapped_count=int(np.sum(val_state_to_bin < 0)),
        val_mapped_frac=float(np.mean(val_state_to_bin >= 0)),
    )
    return dict(
        bin_centers=bin_centers.astype(np.float32),
        bin_coords=unique_coords.astype(np.int32),
        edge_src=edge_pairs[:, 0].astype(np.int32),
        edge_dst=edge_pairs[:, 1].astype(np.int32),
        train_state_to_bin=train_state_to_bin,
        val_state_to_bin=val_state_to_bin,
        metadata=metadata,
    )


def save_graph_npz(path, graph):
    """Save a graph dictionary to an npz file."""
    np.savez_compressed(
        path,
        bin_centers=graph["bin_centers"],
        bin_coords=graph["bin_coords"],
        edge_src=graph["edge_src"],
        edge_dst=graph["edge_dst"],
        train_state_to_bin=graph["train_state_to_bin"],
        val_state_to_bin=graph["val_state_to_bin"],
        metadata_json=np.asarray(json.dumps(graph["metadata"])),
    )


def load_graph_npz(path):
    """Load a graph dictionary saved by save_graph_npz."""
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return dict(
        bin_centers=data["bin_centers"].astype(np.float32),
        bin_coords=data["bin_coords"].astype(np.int32),
        edge_src=data["edge_src"].astype(np.int32),
        edge_dst=data["edge_dst"].astype(np.int32),
        train_state_to_bin=data["train_state_to_bin"].astype(np.int32),
        val_state_to_bin=data["val_state_to_bin"].astype(np.int32),
        metadata=metadata,
    )


def graph_fingerprint(graph):
    """Return a stable fingerprint for graph-distance cache validation."""
    digest = hashlib.sha1()
    for key in ("bin_centers", "edge_src", "edge_dst"):
        arr = np.ascontiguousarray(graph[key])
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.view(np.uint8))
    metadata_json = json.dumps(graph["metadata"], sort_keys=True)
    digest.update(metadata_json.encode("utf-8"))
    return digest.hexdigest()


def graph_distance_matrix_cache_path(graph_path):
    """Return the default all-pairs distance-matrix cache path for a graph npz."""
    graph_path = Path(graph_path)
    return graph_path.with_name(f"{graph_path.stem}_distance_matrix.npz")


def save_graph_distance_matrix_npz(path, distance_matrix, graph):
    """Save an all-pairs graph distance matrix with graph metadata validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        distance_matrix=np.asarray(distance_matrix, dtype=np.float32),
        graph_fingerprint=np.asarray(graph_fingerprint(graph)),
        node_count=np.asarray(len(graph["bin_centers"]), dtype=np.int32),
    )


def load_graph_distance_matrix_npz(path, graph):
    """Load a cached all-pairs graph distance matrix if it matches the graph."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        expected = graph_fingerprint(graph)
        if str(data["graph_fingerprint"].item()) != expected:
            return None
        matrix = data["distance_matrix"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        return None
    if matrix.shape != (len(graph["bin_centers"]), len(graph["bin_centers"])):
        return None
    return matrix


def adjacency_lists(num_nodes, edge_src, edge_dst):
    """Create undirected adjacency lists from edge arrays."""
    neighbors = [[] for _ in range(int(num_nodes))]
    for src, dst in zip(np.asarray(edge_src), np.asarray(edge_dst)):
        src = int(src)
        dst = int(dst)
        neighbors[src].append(dst)
        neighbors[dst].append(src)
    return [np.asarray(sorted(set(items)), dtype=np.int32) for items in neighbors]


def shortest_hop_distances(adjacency, source):
    """Return BFS hop distances from source; unreachable nodes are -1."""
    distances = np.full(len(adjacency), -1, dtype=np.int32)
    source = int(source)
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            neighbor = int(neighbor)
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def connected_component_sizes(adjacency):
    """Return connected component sizes for a graph."""
    seen = np.zeros(len(adjacency), dtype=bool)
    sizes = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                neighbor = int(neighbor)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    return np.asarray(sorted(sizes, reverse=True), dtype=np.int32)


def graph_distance_statistics(adjacency, graph=None, max_sources=None, rng=None):
    """Return BFS distance statistics, optionally sampling sources."""
    num_nodes = len(adjacency)
    if max_sources is not None and int(max_sources) > 0 and int(max_sources) < num_nodes:
        rng = np.random.default_rng(0) if rng is None else rng
        sources = rng.choice(num_nodes, size=int(max_sources), replace=False)
        sampled = True
    else:
        sources = np.arange(num_nodes, dtype=np.int32)
        sampled = False
    max_hops = 0
    finite_pair_count = 0
    hop_sum = 0.0
    for source in sources:
        distances = shortest_hop_distances(adjacency, source)
        finite = distances >= 0
        if finite.any():
            max_hops = max(max_hops, int(distances[finite].max()))
            finite_pair_count += int(finite.sum())
            hop_sum += float(distances[finite].sum())
    mean_hops = hop_sum / finite_pair_count if finite_pair_count else np.nan
    scale = 1.0
    if graph is not None:
        scale = float(graph["metadata"].get("env_steps_per_graph_edge", 1.0))
    return dict(
        max_hops=int(max_hops),
        max_steps=float(max_hops * scale),
        mean_hops=float(mean_hops),
        mean_steps=float(mean_hops * scale) if np.isfinite(mean_hops) else np.nan,
        finite_pair_count=int(finite_pair_count),
        sampled_source_count=int(len(sources)),
        sampled=bool(sampled),
    )


def graph_distance_matrix_statistics(distance_matrix, graph=None):
    """Return distance statistics from an all-pairs step-distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=np.float32)
    finite = np.isfinite(matrix)
    if finite.any():
        max_steps = float(matrix[finite].max())
        mean_steps = float(matrix[finite].mean())
        finite_pair_count = int(finite.sum())
    else:
        max_steps = 0.0
        mean_steps = np.nan
        finite_pair_count = 0
    scale = 1.0
    if graph is not None:
        scale = float(graph["metadata"].get("env_steps_per_graph_edge", 1.0))
    return dict(
        max_hops=int(round(max_steps / max(scale, 1e-6))),
        max_steps=max_steps,
        mean_hops=float(mean_steps / scale) if np.isfinite(mean_steps) else np.nan,
        mean_steps=mean_steps,
        finite_pair_count=finite_pair_count,
        sampled_source_count=int(matrix.shape[0]),
        sampled=False,
    )


def bin_to_state_indices(state_to_bin, num_bins, valid_idxs=None):
    """Return state-index arrays for every bin."""
    state_to_bin = np.asarray(state_to_bin, dtype=np.int32)
    if valid_idxs is None:
        idxs = np.arange(len(state_to_bin), dtype=np.int32)
    else:
        idxs = np.asarray(valid_idxs, dtype=np.int32)
    bins = state_to_bin[idxs]
    result = [[] for _ in range(int(num_bins))]
    for idx, bin_idx in zip(idxs, bins):
        if int(bin_idx) < 0 or int(bin_idx) >= int(num_bins):
            continue
        result[int(bin_idx)].append(int(idx))
    return [np.asarray(items, dtype=np.int32) for items in result]


def graph_step_distances(hop_distances, graph):
    """Convert graph hop distances to calibrated environment-step distances."""
    scale = float(graph["metadata"].get("env_steps_per_graph_edge", 1.0))
    hop_distances = np.asarray(hop_distances)
    distances = hop_distances.astype(np.float32) * scale
    distances[hop_distances < 0] = np.inf
    return distances


def graph_step_distance_matrix(adjacency, graph):
    """Return all-pairs graph distances in calibrated environment-step units."""
    rows = [
        graph_step_distances(shortest_hop_distances(adjacency, source), graph)
        for source in range(len(adjacency))
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def sample_graph_budget_pairs(
    dataset,
    state_to_bin,
    graph,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
    adjacency=None,
    distance_matrix=None,
    goal_by_bin=None,
    src_idxs=None,
):
    """Sample balanced graph-distance positive/negative pairs for one budget."""
    budget = int(budget)
    rng = np.random.default_rng() if rng is None else rng
    adjacency = (
        adjacency
        if adjacency is not None
        else adjacency_lists(len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"])
    )
    state_to_bin = np.asarray(state_to_bin, dtype=np.int32)
    src_idxs = source_indices(dataset) if src_idxs is None else np.asarray(src_idxs, dtype=np.int32)
    src_idxs = src_idxs[state_to_bin[src_idxs] >= 0]
    if goal_by_bin is None:
        goal_by_bin = bin_to_state_indices(state_to_bin, len(graph["bin_centers"]))
    has_goal = np.asarray([len(items) > 0 for items in goal_by_bin])
    distance_cache = {}

    observations = []
    actions = []
    goals = []
    budgets = []
    labels = []
    graph_distances = []
    source_bins = []
    goal_bins = []
    source_idxs_out = []
    goal_idxs_out = []

    def distances_for_bin(bin_idx):
        bin_idx = int(bin_idx)
        if distance_matrix is not None:
            return np.asarray(distance_matrix[bin_idx], dtype=np.float32)
        if bin_idx not in distance_cache:
            hops = shortest_hop_distances(adjacency, bin_idx)
            distance_cache[bin_idx] = graph_step_distances(hops, graph)
        return distance_cache[bin_idx]

    def add_pairs(target_label, target_count):
        attempts = 0
        max_attempts = max(1000, target_count * 100)
        while target_count > 0 and attempts < max_attempts:
            attempts += 1
            src_idx = int(rng.choice(src_idxs))
            src_bin = int(state_to_bin[src_idx])
            distances = distances_for_bin(src_bin)
            finite = np.isfinite(distances) & has_goal
            if target_label == 1.0:
                lo = max(0.0, float(pos_boundary_frac) * budget)
                hi = float(budget)
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any() and lo > 0.0:
                    candidate_mask = finite & (distances <= hi)
            else:
                lo = np.nextafter(float(budget), np.inf)
                hi = float(neg_max_factor) * budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any():
                    candidate_mask = finite & (distances > float(budget))

            candidate_bins = np.nonzero(candidate_mask)[0]
            if len(candidate_bins) == 0:
                continue
            goal_bin = int(rng.choice(candidate_bins))
            goal_idx = int(rng.choice(goal_by_bin[goal_bin]))
            observations.append(np.asarray(dataset["observations"])[src_idx])
            actions.append(np.asarray(dataset["actions"])[src_idx])
            goals.append(np.asarray(dataset["observations"])[goal_idx])
            budgets.append(budget)
            labels.append(target_label)
            graph_distances.append(float(distances[goal_bin]))
            source_bins.append(src_bin)
            goal_bins.append(goal_bin)
            source_idxs_out.append(src_idx)
            goal_idxs_out.append(goal_idx)
            target_count -= 1

    num_pos = int(num_pairs) // 2
    num_neg = int(num_pairs) - num_pos
    add_pairs(1.0, num_pos)
    add_pairs(0.0, num_neg)

    if len(labels) == 0:
        return None
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        budgets=np.asarray(budgets, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.float32),
        graph_distances=np.asarray(graph_distances, dtype=np.float32),
        source_bins=np.asarray(source_bins, dtype=np.int32),
        goal_bins=np.asarray(goal_bins, dtype=np.int32),
        source_idxs=np.asarray(source_idxs_out, dtype=np.int32),
        goal_idxs=np.asarray(goal_idxs_out, dtype=np.int32),
    )
