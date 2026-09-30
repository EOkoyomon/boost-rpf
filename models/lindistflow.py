import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import dijkstra


class _DistFlowModule(nn.Module):
    """Shared plumbing for the (Lin)DistFlow sweeps.

    Args:
        use_fast (bool): Use the vectorized sweep (`calculate_distflow_fast`). The
            node-at-a-time version is kept as a reference; the two agree to machine
            precision.
        cache_topology (bool): Reuse the extracted radial tree across samples of the
            same grid. The topology is a property of the network, not of the loading,
            so this is exact, but it moves topology extraction out of the measured
            per-sample path, so it is off by default to keep inference timings
            comparable to the other models.
    """

    def __init__(self, linear, use_fast=True, cache_topology=False):
        super().__init__()
        self.linear = linear
        self.use_fast = use_fast
        self.cache_topology = cache_topology
        self._topo_cache = {}

    def is_analytical(self):
        return True

    def forward(self, data):
        # `data.slack_info` is a float32 tensor, and letting it into the recursion
        # makes numpy promote every step to a float32 torch scalar. This costs
        # a few ms in torch dispatch and silently computes the whole angle sweep
        # in single precision, adding a small amount of error.
        slack_vm_pu = float(data.slack_info[0])
        slack_va_degree = float(data.slack_info[1])

        if self.use_fast:
            topology = None
            key = getattr(data, 'grid_name', None)
            if self.cache_topology and key is not None:
                topology = self._topo_cache.get(key)
            vm, va, topology = calculate_distflow_fast(
                data, slack_index=0, slack_vm_pu=slack_vm_pu,
                slack_va_degree=slack_va_degree, linear=self.linear, topology=topology,
            )
            if self.cache_topology and key is not None:
                self._topo_cache[key] = topology
        else:
            vm, va = calculate_distflow_iterative(
                data, slack_index=0, slack_vm_pu=slack_vm_pu,
                slack_va_degree=slack_va_degree, linear=self.linear,
            )

        return torch.stack([torch.from_numpy(vm), torch.from_numpy(va)], dim=1)


class LinDistFlow(_DistFlowModule):
    def __init__(self, use_fast=True, cache_topology=False):
        super().__init__(linear=True, use_fast=use_fast, cache_topology=cache_topology)


class DistFlow(_DistFlowModule):
    def __init__(self, use_fast=True, cache_topology=False):
        super().__init__(linear=False, use_fast=use_fast, cache_topology=cache_topology)

# ---------------------------------------------------------------------------
# Vectorized forward-backward sweep
#
# The vectorized version below replaces the slow node-at-a-time sweep with:
#   1. Array topology extraction. Branch r/x/tap/shift are derived from Ybus with
#      whole-array operations, and the radial tree comes from a single C-level
#      BFS (`scipy.sparse.csgraph.dijkstra`, unweighted) that returns depths and
#      predecessors at once, in place of networkx.
#   2. Depth-batched sweeps. Both sweeps are sequential only in hop distance from
#      the slack bus, so each depth level is processed as one vectorized step:
#      D array operations instead of N scalar iterations.
# ---------------------------------------------------------------------------

def _build_distflow_topology(data, slack_index=0):
    """Extract the radial tree and per-branch impedances as flat arrays.

    Depends only on the network (Ybus, edge_index), not on the loading, so the
    result can be reused across samples of the same grid.

    Returns a dict with, indexed by child bus: `parent`, `r`, `x`, `tap_sq`,
    `shift_rad`; plus `levels` (bus indices grouped by depth, shallowest first)
    and `parents_by_level`.
    """
    Ybus = data.ppci["Ybus"]
    num_nodes = data.ppci["Sbus"].shape[0]

    rows, cols, vals = find(Ybus)
    off_diagonal = rows != cols
    rows, cols, vals = rows[off_diagonal], cols[off_diagonal], vals[off_diagonal]

    # Keep only the Ybus off-diagonals that correspond to a real branch, and pick
    # up each one's trafo flag, via a sorted-key lookup instead of a dict.
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    edge_keys = edge_index[0].astype(np.int64) * num_nodes + edge_index[1].astype(np.int64)
    order = np.argsort(edge_keys, kind='stable')
    edge_keys_sorted = edge_keys[order]
    is_trafo_sorted = edge_attr[order, 0].astype(bool)

    ybus_keys = rows.astype(np.int64) * num_nodes + cols.astype(np.int64)
    pos = np.clip(np.searchsorted(edge_keys_sorted, ybus_keys), 0, max(len(edge_keys_sorted) - 1, 0))
    keep = edge_keys_sorted[pos] == ybus_keys

    src, dst, y_ij = rows[keep], cols[keep], vals[keep]
    is_trafo = is_trafo_sorted[pos[keep]]

    # Same impedance recovery as the reference sweep (see the transformer note
    # below), applied to whole arrays. For a plain line z = -1/Y_ij; for a
    # transformer edge (src side carrying the tap) t = -Y_ij/Y_ss and
    # z = 1/(Y_ss |t|^2).
    diag = Ybus.diagonal()
    tap_ratio = np.ones(len(src))
    shift_deg = np.zeros(len(src))
    z = np.empty(len(src), dtype=np.complex128)

    line = ~is_trafo
    z[line] = -1.0 / y_ij[line]
    if is_trafo.any():
        t = -y_ij[is_trafo] / diag[src[is_trafo]]
        tap_ratio[is_trafo] = np.abs(t)
        shift_deg[is_trafo] = np.degrees(np.angle(t))
        z[is_trafo] = 1.0 / (diag[src[is_trafo]] * tap_ratio[is_trafo] ** 2)

    # Radial tree rooted at the slack bus: one C-level unweighted BFS gives both
    # the depth and the parent of every bus.
    adjacency = csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
    distance, predecessor = dijkstra(adjacency, directed=False, indices=slack_index,
                                    unweighted=True, return_predecessors=True)
    reachable = np.isfinite(distance)
    depth = np.full(num_nodes, -1, dtype=np.int64)
    depth[reachable] = distance[reachable].astype(np.int64)
    parent = predecessor.astype(np.int64)

    # Per-child branch attributes: the edge (parent[j] -> j).
    child = np.flatnonzero(reachable & (np.arange(num_nodes) != slack_index))
    kept_keys = src.astype(np.int64) * num_nodes + dst.astype(np.int64)
    key_order = np.argsort(kept_keys, kind='stable')
    tree_edge = key_order[np.searchsorted(kept_keys[key_order],
                                         parent[child] * num_nodes + child)]

    r = np.zeros(num_nodes)
    x = np.zeros(num_nodes)
    tap_sq = np.ones(num_nodes)
    shift_rad = np.zeros(num_nodes)
    r[child] = z.real[tree_edge]
    x[child] = z.imag[tree_edge]
    tap_sq[child] = tap_ratio[tree_edge] ** 2
    shift_rad[child] = np.deg2rad(shift_deg[tree_edge])

    # Group buses by depth so each level is one vectorized step.
    by_depth = child[np.argsort(depth[child], kind='stable')]
    level_depths = depth[by_depth]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(level_depths)) + 1, [len(by_depth)]])
    levels = [by_depth[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]

    return {
        "num_nodes": num_nodes,
        "parent": parent,
        "r": r,
        "x": x,
        "tap_sq": tap_sq,
        "shift_rad": shift_rad,
        "levels": levels,
        "parents_by_level": [parent[level] for level in levels],
    }


def calculate_distflow_fast(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=0.0,
                            linear=True, topology=None):
    """Vectorized equivalent of `calculate_distflow_iterative`.

    Args:
        topology: Optional pre-built topology from `_build_distflow_topology`. Pass
            one to reuse the tree across samples of the same grid; omit to rebuild
            it (the default, so the cost stays inside the measured call).

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
        dict: the topology used, so callers can cache it.
    """
    if topology is None:
        topology = _build_distflow_topology(data, slack_index)

    num_nodes = topology["num_nodes"]
    parent = topology["parent"]
    r, x = topology["r"], topology["x"]
    levels, parents_by_level = topology["levels"], topology["parents_by_level"]

    # ppci Sbus is Net Injection; we need Net Load.
    Sbus = -data.ppci["Sbus"]
    P_load, Q_load = Sbus.real, Sbus.imag

    slack_vm_pu = float(slack_vm_pu)
    slack_va_degree = float(slack_va_degree)

    # Same transformed-schema sanity check as the reference sweep, vectorized.
    # (One-sided, as in the original -- it checks difference < 1e-6, not |difference|.)
    num_x = len(data.x)
    x_np = data.x.numpy()
    assert np.all((x_np[1:, 0] - P_load[1:num_x]) < 1e-6)
    assert np.all((x_np[1:, 1] - Q_load[1:num_x]) < 1e-6)

    ## Backward Sweep (Summing Power), deepest level first
    P_flow = P_load.copy()
    Q_flow = Q_load.copy()
    for level, level_parents in zip(reversed(levels), reversed(parents_by_level)):
        p_node = P_flow[level]
        q_node = Q_flow[level]
        if not linear:
            # As in the reference: the loss term is added to what flows up to the
            # parent, but P_flow/Q_flow at this bus stay loss-free.
            loss = p_node ** 2 + q_node ** 2
            p_node = p_node + r[level] * loss
            q_node = q_node + x[level] * loss
        # bincount, not `+=`, so siblings sharing a parent all accumulate.
        P_flow += np.bincount(level_parents, weights=p_node, minlength=num_nodes)
        Q_flow += np.bincount(level_parents, weights=q_node, minlength=num_nodes)

    ## Forward Sweep (Calculating Voltage), shallowest level first
    V_sq = np.full(num_nodes, slack_vm_pu ** 2)
    Va_rad = np.full(num_nodes, np.deg2rad(slack_va_degree))
    for level, level_parents in zip(levels, parents_by_level):
        p_line = P_flow[level]
        q_line = Q_flow[level]

        V_sq_before_tap = V_sq[level_parents] - 2.0 * (r[level] * p_line + x[level] * q_line)
        if not linear:
            V_sq_before_tap = V_sq_before_tap + (
                (r[level] ** 2 + x[level] ** 2) * (p_line ** 2 + q_line ** 2) / V_sq[level_parents]
            )
        V_sq[level] = V_sq_before_tap / topology["tap_sq"][level]

        Va_rad[level] = (
            Va_rad[level_parents]
            - topology["shift_rad"][level]
            - (x[level] * p_line - r[level] * q_line) / slack_vm_pu
        )

    vm_full = np.sqrt(np.maximum(V_sq, 0))
    va_full = np.rad2deg(Va_rad)

    return vm_full[:num_x], va_full[:num_x], topology

# ---------------------------------------------------------------------------
# Iterative forward-backward sweep
# ---------------------------------------------------------------------------

def calculate_lindistflow_iterative(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=0.0, return_internals=False):
    """
    Iterative Forward-Backward Sweep implementation of LinDistFlow.

    Args:
        data: PyTorch Geometric Data object grid info and ppci attribute.
        slack_index (int): Index of the slack bus (usually 0).
        slack_vm_pu (float): Voltage magnitude at slack bus in p.u.
        slack_va_degree (float): Voltage angle at slack bus in degrees (the true ext_grid angle,
            e.g. `data.slack_info[1]` — typically 0). Any transformer phase shift (e.g. simbench's
            usual ~150deg Dyn5 vector group) is applied per-edge in the sweep itself, not baked
            into this starting value — see the trafo handling in the topology pre-processing below.

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
    """
    return calculate_distflow_iterative(data, slack_index=slack_index, slack_vm_pu=slack_vm_pu, slack_va_degree=slack_va_degree, linear=True, return_internals=return_internals)

def calculate_distflow_iterative(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=0.0, linear=False, return_internals=False):
    """
    Iterative Forward-Backward Sweep implementation of DistFlow.
    https://doi.org/10.1109/61.25627.

    Args:
        data: PyTorch Geometric Data object grid info and ppci attribute.
        slack_index (int): Index of the slack bus (usually 0).
        slack_vm_pu (float): Voltage magnitude at slack bus in p.u.
        slack_va_degree (float): Voltage angle at slack bus in degrees (the true ext_grid angle,
            e.g. `data.slack_info[1]` — typically 0). See note in calculate_lindistflow_iterative
            about transformer phase shift.
        linear (bool): Whether to use linearized DistFlow equations.
        return_internals (bool): If True, also return a dict of intermediate quantities from the
            forward/backward sweep (tree paths, per-edge r/x/tap/shift, aggregated P/Q, and the
            full-length LDF voltages). Lets callers reuse this single LDF implementation to build
            downstream features instead of re-deriving it. Callers passing raw (untransformed) data
            should set this True — it also skips the transformed-schema injection assertion below.

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
        dict (only if return_internals=True): intermediate sweep quantities keyed by
            paths, parents, edge_r, edge_x, edge_tap_ratio, edge_shift_deg,
            P_load, Q_load, P_flow, Q_flow, vm_full, va_full.
    """
    ## Extract Data from Source
    Ybus = data.ppci["Ybus"].copy()

    # ppci Sbus is Net Injection. We need Net Load.
    # This handles both P (Real) and Q (Imag) simultaneously.
    Sbus = -1 * data.ppci["Sbus"].copy()
    num_nodes = Sbus.shape[0]
    P_load = Sbus.real
    Q_load = Sbus.imag

    # Pandapower adds extra buses for pypower modeling. Luckily, based on how pandapower does it, when we only have
    # slack and PQ nodes, we know the first N would be the predictions for the buses we are interested in.
    # This sanity check assumes the *transformed* node schema where data.x[:, 0:2] == [p_mw, q_mvar]. Callers that
    # pass raw (untransformed) data — where those columns are the Slack?/PV? one-hot flags — request internals and
    # skip it; P/Q are read from ppci's Sbus regardless, so the sweep does not depend on data.x layout.
    if not return_internals:
        assert all((data.x[1:, 0] - P_load[1:len(data.x)]) < 1e-6)
        assert all((data.x[1:, 1] - Q_load[1:len(data.x)]) < 1e-6)

    ## Pre-processing Topology

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    is_trafo_edge = {}  # (i, j) -> bool, from edge_attr's trafo flag (0/1 marker, not corrupted)
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        is_trafo_edge[(i, j)] = bool(edge_attr[k, 0])

    rows, cols, vals = find(Ybus) # find() returns (row_indices, col_indices, values)

    # NOTE ON TRANSFORMERS: for a plain line, r,x = -1/Ybus[i,j] recovers the physical series
    # impedance directly (verified exact). For a transformer edge that's wrong: pandapower encodes
    # both an off-nominal tap ratio and (for simbench-style Dyn* windings) a large vector-group
    # phase shift as a complex tap `t` in the branch admittance model, so `-1/Ybus[i,j]` recovers
    # a `t`-rotated, `t`-scaled quantity, not the physical impedance.
    # Thus, we do NOT read the true impedance from `data.edge_attr` for transformer edges.
    # Instead we derive tap ratio, phase shift, AND the true impedance purely from Ybus, using the
    # tap-side bus's own diagonal entry.
    # 
    # For a transformer with tap `t` on bus i (bus i touching no
    # other branch — true for every MV/LV substation bus in this dataset):
    #     Ybus[i,i] = y/t*conj(t) = y/|t|^2        Ybus[i,j] = -y/conj(t)
    #     =>  t = -Ybus[i,j] / Ybus[i,i]           y = Ybus[i,i] * |t|^2
    # (Verified exact against the true vk_percent/vkr_percent/tap_pos-derived impedance, tap
    # ratio, and shift_degree for rural1/2/3.)

    for r, c, val in zip(rows, cols, vals):
        # We only care about off-diagonals (lines/transformers)
        if r != c and (r, c) in is_trafo_edge:
            tap_ratio, shift_deg = 1.0, 0.0
            if is_trafo_edge[(r, c)]:
                t = -val / Ybus[r, r]
                tap_ratio, shift_deg = abs(t), np.degrees(np.angle(t))
                z = 1.0 / (Ybus[r, r] * tap_ratio**2)
                r_pu, x_pu = z.real, z.imag
            else:
                z_pu = -1.0 / val
                r_pu, x_pu = z_pu.real, z_pu.imag

            # Add to graph (undirected for now, we direct it later using paths)
            G.add_edge(r, c, r=r_pu, x=x_pu, tap_ratio=tap_ratio, shift_deg=shift_deg)

    ## Build Path Matrix (BFS Tree)
    # Create a directed tree rooted at slack to determine paths
    try:
        paths = nx.shortest_path(G, source=slack_index)
    except nx.NetworkXNoPath:
        print("Creating tree ourselves.")
        # Fallback if directionality is ambiguous in meshed elements,
        # force a tree via BFS
        bfs_tree = nx.bfs_tree(G, source=slack_index)
        paths = nx.shortest_path(bfs_tree, source=slack_index)

    # Pre-fetch edge attributes to speed up loop
    edge_r = nx.get_edge_attributes(G, 'r')
    edge_x = nx.get_edge_attributes(G, 'x')
    edge_tap_ratio = nx.get_edge_attributes(G, 'tap_ratio')
    edge_shift_deg = nx.get_edge_attributes(G, 'shift_deg')

    ## Backward Sweep (Summing Power)
    # Sort by distance to slack (leaves last)
    sorted_nodes = sorted(paths.keys(), key=lambda n: len(paths[n]))
    P_flow = P_load.copy()
    Q_flow = Q_load.copy()

    # Map each node to its parent for fast lookup
    # paths[node] = [slack, ..., parent, node]
    parents = {}
    for node in paths:
        if node != slack_index:
            # Node is at last index (-1), so parent is -2.
            parents[node] = paths[node][-2]

    # Iterate from leaves up to slack
    for node in sorted_nodes[::-1]:
        if node == slack_index:
            continue

        parent = parents[node]
        # Accumulate this node's total required power into the parent
        P_node = P_flow[node]
        Q_node = Q_flow[node]

        if not linear:
            r = edge_r[(parent, node)]
            x = edge_x[(parent, node)]
            # Here, we usually need to divide this value by v^2, but this is unknown.
            # Like in the original paper, we assume v^2 ≈ 1 p.u.
            P_loss_line = (P_node**2 + Q_node**2)
            P_node += (r*P_loss_line)
            Q_node += (x*P_loss_line)

        P_flow[parent] += P_node
        Q_flow[parent] += Q_node

    ## Forward Sweep (Calculating Voltage)
    # Initialize voltages with slack voltage
    V_sq = np.zeros(num_nodes)
    V_sq[:] = slack_vm_pu**2 # Set all to slack initially (will be overwritten)

    Va_rad = np.zeros(num_nodes)
    Va_rad[:] = np.deg2rad(slack_va_degree)

    # Iterate from slack down to leaves
    for node in sorted_nodes:
        if node == slack_index:
            continue

        parent = parents[node]
        r = edge_r[(parent, node)]
        x = edge_x[(parent, node)]
        tap_ratio = edge_tap_ratio[(parent, node)]  # 1.0 for plain lines
        shift_deg = edge_shift_deg[(parent, node)]  # 0.0 for plain lines

        # The flow on the line connecting parent -> node
        # is exactly the accumulated flow we calculated for 'node'
        p_line = P_flow[node]
        q_line = Q_flow[node]

        # LinDistFlow equation: V_node^2 = V_parent^2 - 2(rP + xQ)
        # Positive Load (P_line) causes Voltage Drop (Subtraction)
        V_sq_before_tap = V_sq[parent] - 2 * (r * p_line + x * q_line)

        if not linear:
            # DistFlow equation: V_node^2 = V_parent^2 - 2(rP + xQ) + (r^2 + x^2)(P^2 + Q^2)/(V_parent^2)
            # Positive Load (P_line) causes Voltage Drop (Subtraction)
            V_sq_before_tap += ((r**2 + x**2)*(p_line**2 + q_line**2)/V_sq[parent])

        # Off-nominal tap ratio (1.0 for plain lines) rescales the far-side voltage on top of the
        # impedance-drop term above — this is the piece that's missing if you only fix r,x.
        V_sq[node] = V_sq_before_tap / tap_ratio**2

        # Angle Drop = (X * P - R * Q) / V_nom, plus the transformer's fixed vector-group phase
        # shift (0 for plain lines) — this is NOT loading-dependent, it's a constant per edge.
        Va_rad[node] = Va_rad[parent] - np.deg2rad(shift_deg) - ((x * p_line - r * q_line) / slack_vm_pu)

    ## Full-length results across all ppci nodes (before truncation to the original data)
    vm_full = np.sqrt(np.maximum(V_sq, 0))
    va_full = np.rad2deg(Va_rad)

    ## Return the same length as the original data
    vm = vm_full[:len(data.x)]
    va = va_full[:len(data.x)]

    if return_internals:
        # Expose the intermediate quantities so callers (e.g. path/feature extraction in
        # utils/data_utils.py) can reuse this single LDF implementation instead of duplicating it.
        internals = {
            "paths": paths,
            "parents": parents,
            "edge_r": edge_r,
            "edge_x": edge_x,
            "edge_tap_ratio": edge_tap_ratio,
            "edge_shift_deg": edge_shift_deg,
            "P_load": P_load,
            "Q_load": Q_load,
            "P_flow": P_flow,  # backward-swept (aggregated) active power per node
            "Q_flow": Q_flow,  # backward-swept (aggregated) reactive power per node
            "vm_full": vm_full,  # LDF voltage magnitude (p.u.) for all ppci nodes
            "va_full": va_full,  # LDF voltage angle (degrees) for all ppci nodes
        }
        return vm, va, internals

    return vm, va

def verify_lindistflow_calculation(data, true_voltages, slack_index=0, slack_vm_pu=1.025):
    """
    Debugs the LinDistFlow calculation by inspecting the worst prediction.
    """
    # 1. Check both calculations (one with Numpy vectorization vs the standard iterative approach)
    start = time.time()
    pred_voltages_matrix, _ = calculate_lindistflow(data, slack_index, slack_vm_pu)
    middle = time.time()
    pred_voltages_iter, _ = calculate_lindistflow_iterative(data, slack_index, slack_vm_pu)
    end = time.time()
    rmse = lambda x,y: np.sqrt(np.mean((x - y)**2))
    print("\n--- VERIFYING IMPLEMENTATIONS MATCH ---")
    print(f"Pred rmse matrix: {rmse(pred_voltages_matrix, true_voltages):.4f} p.u.")
    print(f"Pred rmse iter:   {rmse(pred_voltages_iter, true_voltages):.4f} p.u.")
    print(f"Predictions are the same: {all((pred_voltages_matrix - pred_voltages_iter)) < 1e-10}")

    pred_speed_matrix = (middle-start)
    pred_speed_iter = (end-middle)
    print("\n--- COMPARING IMPLEMENTATION SPEEDS ---")
    print(f"Pred speed(s) matrix: {pred_speed_matrix:.4f}s.")
    print(f"Pred speed(s) iter:   {pred_speed_iter:.4f}s.")

    if pred_speed_matrix < pred_speed_iter:
        pred_voltages = pred_voltages_matrix
    else:
        pred_voltages = pred_voltages_iter
    
    # 2. Find the node with the biggest mismatch
    # (Skip slack index 0)
    diff = pred_voltages - true_voltages
    worst_node = np.argmax(np.abs(diff)[1:]) + 1 # +1 because we skipped index 0
    
    print(f"\n--- DEBUGGING NODE {worst_node} (Worst Node) ---")
    print(f"Pred: {pred_voltages[worst_node]:.4f} p.u.")
    print(f"True: {true_voltages[worst_node]:.4f} p.u.")
    print(f"Slack V: {ppci['bus'][slack_index, 7]:.4f} p.u.")
