"""Layout/grid BFS utilities for PointMaze reachability diagnostics."""

from collections import deque

import numpy as np

from utils.pointmaze_graph import dataset_xy, source_indices, valid_transition_indices


def unwrap_maze_env(env):
    """Return the innermost env object exposing maze_map."""
    cur = env
    for _ in range(32):
        if hasattr(cur, "maze_map"):
            return cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    raise ValueError("Could not find an unwrapped env with maze_map.")


def xy_to_ij(xy, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    """Vectorized version of OGBench MazeEnv.xy_to_ij."""
    xy = np.asarray(xy, dtype=np.float32)
    i = ((xy[..., 1] + offset_y + 0.5 * maze_unit) / maze_unit).astype(np.int32)
    j = ((xy[..., 0] + offset_x + 0.5 * maze_unit) / maze_unit).astype(np.int32)
    return np.stack([i, j], axis=-1)


def ij_to_xy(ij, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    """Vectorized version of OGBench MazeEnv.ij_to_xy."""
    ij = np.asarray(ij, dtype=np.float32)
    x = ij[..., 1] * maze_unit - offset_x
    y = ij[..., 0] * maze_unit - offset_y
    return np.stack([x, y], axis=-1)


def free_cell_distance_matrix(maze_map):
    """Return all-pairs BFS distances between free cells."""
    maze_map = np.asarray(maze_map)
    free_cells = np.argwhere(maze_map == 0).astype(np.int32)
    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(free_cells)}
    distances = np.full((len(free_cells), len(free_cells)), -1, dtype=np.int32)

    for src_idx, src_cell in enumerate(free_cells):
        src = tuple(int(x) for x in src_cell)
        distances[src_idx, src_idx] = 0
        queue = deque([src])
        while queue:
            i, j = queue.popleft()
            cur_idx = cell_to_idx[(i, j)]
            next_distance = distances[src_idx, cur_idx] + 1
            for di, dj in ((-1, 0), (0, -1), (1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                neighbor = (ni, nj)
                if (
                    0 <= ni < maze_map.shape[0]
                    and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and neighbor in cell_to_idx
                ):
                    neighbor_idx = cell_to_idx[neighbor]
                    if distances[src_idx, neighbor_idx] < 0:
                        distances[src_idx, neighbor_idx] = next_distance
                        queue.append(neighbor)
    return free_cells, distances


def xy_pair_grid_distances(
    start_xy,
    goal_xy,
    maze_map,
    free_cells,
    cell_distances,
    steps_per_cell,
    maze_unit=4.0,
    offset_x=4.0,
    offset_y=4.0,
):
    """Return calibrated grid distances for xy pairs; invalid pairs are inf."""
    start_ij = xy_to_ij(start_xy, maze_unit, offset_x, offset_y)
    goal_ij = xy_to_ij(goal_xy, maze_unit, offset_x, offset_y)
    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(np.asarray(free_cells))}
    result = np.full(start_ij.shape[:-1], np.inf, dtype=np.float32)
    flat_result = result.reshape(-1)
    flat_start = start_ij.reshape(-1, 2)
    flat_goal = goal_ij.reshape(-1, 2)
    maze_map = np.asarray(maze_map)

    for idx, (src, dst) in enumerate(zip(flat_start, flat_goal)):
        src = tuple(int(x) for x in src)
        dst = tuple(int(x) for x in dst)
        if (
            src not in cell_to_idx
            or dst not in cell_to_idx
            or maze_map[src] != 0
            or maze_map[dst] != 0
        ):
            continue
        cell_distance = cell_distances[cell_to_idx[src], cell_to_idx[dst]]
        if cell_distance >= 0:
            flat_result[idx] = float(cell_distance) * float(steps_per_cell)
    return result


def state_to_free_cell_indices(
    dataset,
    maze_map,
    free_cells,
    xy_dims=(0, 1),
    maze_unit=4.0,
    offset_x=4.0,
    offset_y=4.0,
):
    """Map each dataset state to a free-cell index; invalid states get -1."""
    xy = dataset_xy(dataset, xy_dims)
    ij = xy_to_ij(xy, maze_unit, offset_x, offset_y)
    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(np.asarray(free_cells))}
    result = np.full(len(xy), -1, dtype=np.int32)
    maze_map = np.asarray(maze_map)
    in_bounds = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < maze_map.shape[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < maze_map.shape[1])
    )
    valid_idxs = np.nonzero(in_bounds)[0]
    for idx in valid_idxs:
        cell = tuple(int(x) for x in ij[idx])
        if cell in cell_to_idx and maze_map[cell] == 0:
            result[idx] = int(cell_to_idx[cell])
    return result


def free_cell_to_state_indices(state_to_cell, num_cells, valid_idxs=None):
    """Return state-index arrays for every free cell."""
    state_to_cell = np.asarray(state_to_cell, dtype=np.int32)
    if valid_idxs is None:
        idxs = np.arange(len(state_to_cell), dtype=np.int32)
    else:
        idxs = np.asarray(valid_idxs, dtype=np.int32)
    result = [[] for _ in range(int(num_cells))]
    for idx in idxs:
        cell_idx = int(state_to_cell[idx])
        if cell_idx >= 0:
            result[cell_idx].append(int(idx))
    return [np.asarray(items, dtype=np.int32) for items in result]


def sample_grid_budget_pairs(
    dataset,
    state_to_cell,
    goal_by_cell,
    cell_distances,
    steps_per_cell,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
    source_idxs=None,
):
    """Sample balanced layout-grid geodesic pairs for one budget."""
    budget = int(budget)
    rng = np.random.default_rng() if rng is None else rng
    state_to_cell = np.asarray(state_to_cell, dtype=np.int32)
    if source_idxs is None:
        src_idxs = source_indices(dataset)
    else:
        src_idxs = np.asarray(source_idxs, dtype=np.int32)
    src_idxs = src_idxs[state_to_cell[src_idxs] >= 0]
    has_goal = np.asarray([len(items) > 0 for items in goal_by_cell])
    step_distances = np.asarray(cell_distances, dtype=np.float32) * float(
        steps_per_cell
    )

    observations = []
    actions = []
    goals = []
    budgets = []
    labels = []
    grid_distances = []
    source_cells = []
    goal_cells = []
    source_idxs_out = []
    goal_idxs_out = []
    goal_idxs_out = []

    def add_pairs(target_label, target_count):
        attempts = 0
        max_attempts = max(1000, int(target_count) * 100)
        while target_count > 0 and attempts < max_attempts:
            attempts += 1
            src_idx = int(rng.choice(src_idxs))
            src_cell = int(state_to_cell[src_idx])
            distances = step_distances[src_cell]
            finite = (cell_distances[src_cell] >= 0) & has_goal
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

            candidate_cells = np.nonzero(candidate_mask)[0]
            if len(candidate_cells) == 0:
                continue
            goal_cell = int(rng.choice(candidate_cells))
            goal_idx = int(rng.choice(goal_by_cell[goal_cell]))
            observations.append(np.asarray(dataset["observations"])[src_idx])
            actions.append(np.asarray(dataset["actions"])[src_idx])
            goals.append(np.asarray(dataset["observations"])[goal_idx])
            budgets.append(budget)
            labels.append(target_label)
            grid_distances.append(float(distances[goal_cell]))
            source_cells.append(src_cell)
            goal_cells.append(goal_cell)
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
        grid_distances=np.asarray(grid_distances, dtype=np.float32),
        source_cells=np.asarray(source_cells, dtype=np.int32),
        goal_cells=np.asarray(goal_cells, dtype=np.int32),
        source_idxs=np.asarray(source_idxs_out, dtype=np.int32),
        goal_idxs=np.asarray(goal_idxs_out, dtype=np.int32),
    )


def sample_grid_budget_q_pairs(
    dataset,
    state_to_cell,
    goal_by_cell,
    cell_distances,
    steps_per_cell,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
    source_idxs=None,
):
    """Sample balanced geodesic Q pairs labeled from the next state.

    The returned observations/actions are from ``s_t, a_t``. Distances and
    labels are computed from ``s_{t+1}`` to the sampled goal:

        Q_H(s_t, a_t, g) = 1[d_grid(s_{t+1}, g) <= H - 1].
    """
    budget = int(budget)
    rng = np.random.default_rng() if rng is None else rng
    state_to_cell = np.asarray(state_to_cell, dtype=np.int32)
    if source_idxs is None:
        src_idxs = valid_transition_indices(dataset)
    else:
        src_idxs = np.asarray(source_idxs, dtype=np.int32)
    src_idxs = src_idxs[src_idxs + 1 < len(state_to_cell)]
    src_idxs = src_idxs[
        (state_to_cell[src_idxs] >= 0) & (state_to_cell[src_idxs + 1] >= 0)
    ]
    has_goal = np.asarray([len(items) > 0 for items in goal_by_cell])
    step_distances = np.asarray(cell_distances, dtype=np.float32) * float(
        steps_per_cell
    )
    remaining_budget = max(float(budget - 1), 1.0)

    observations = []
    actions = []
    next_observations = []
    goals = []
    budgets = []
    remaining_budgets = []
    labels = []
    grid_distances = []
    source_cells = []
    next_cells = []
    goal_cells = []
    source_idxs_out = []

    def add_pairs(target_label, target_count):
        attempts = 0
        max_attempts = max(1000, int(target_count) * 100)
        while target_count > 0 and attempts < max_attempts:
            attempts += 1
            src_idx = int(rng.choice(src_idxs))
            src_cell = int(state_to_cell[src_idx])
            next_cell = int(state_to_cell[src_idx + 1])
            distances = step_distances[next_cell]
            finite = (cell_distances[next_cell] >= 0) & has_goal
            if target_label == 1.0:
                lo = max(0.0, float(pos_boundary_frac) * remaining_budget)
                hi = remaining_budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any() and lo > 0.0:
                    candidate_mask = finite & (distances <= hi)
            else:
                lo = np.nextafter(remaining_budget, np.inf)
                hi = float(neg_max_factor) * remaining_budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any():
                    candidate_mask = finite & (distances > remaining_budget)

            candidate_cells = np.nonzero(candidate_mask)[0]
            if len(candidate_cells) == 0:
                continue
            goal_cell = int(rng.choice(candidate_cells))
            goal_idx = int(rng.choice(goal_by_cell[goal_cell]))
            distance = float(distances[goal_cell])
            observations.append(np.asarray(dataset["observations"])[src_idx])
            actions.append(np.asarray(dataset["actions"])[src_idx])
            next_observations.append(np.asarray(dataset["observations"])[src_idx + 1])
            goals.append(np.asarray(dataset["observations"])[goal_idx])
            budgets.append(budget)
            remaining_budgets.append(remaining_budget)
            labels.append(float(distance <= remaining_budget))
            grid_distances.append(distance)
            source_cells.append(src_cell)
            next_cells.append(next_cell)
            goal_cells.append(goal_cell)
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
        next_observations=np.asarray(next_observations, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        budgets=np.asarray(budgets, dtype=np.int32),
        remaining_budgets=np.asarray(remaining_budgets, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        grid_distances=np.asarray(grid_distances, dtype=np.float32),
        source_cells=np.asarray(source_cells, dtype=np.int32),
        next_cells=np.asarray(next_cells, dtype=np.int32),
        goal_cells=np.asarray(goal_cells, dtype=np.int32),
        source_idxs=np.asarray(source_idxs_out, dtype=np.int32),
        goal_idxs=np.asarray(goal_idxs_out, dtype=np.int32),
    )


def grid_distance_statistics(cell_distances, steps_per_cell):
    """Summarize finite grid distances."""
    finite = cell_distances >= 0
    max_cells = int(cell_distances[finite].max()) if finite.any() else 0
    mean_cells = float(cell_distances[finite].mean()) if finite.any() else np.nan
    return dict(
        max_cells=max_cells,
        max_steps=float(max_cells * steps_per_cell),
        mean_cells=mean_cells,
        mean_steps=float(mean_cells * steps_per_cell)
        if np.isfinite(mean_cells)
        else np.nan,
        finite_pair_count=int(finite.sum()),
    )
