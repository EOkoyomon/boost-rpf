import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import find


class LinDistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_lindistflow_iterative(
            data, slack_index=0, slack_vm_pu=data.slack_info[0], slack_va_degree=data.slack_info[1]
        )
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out

class DistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_distflow_iterative(
            data, slack_index=0, slack_vm_pu=data.slack_info[0], slack_va_degree=data.slack_info[1], linear=False
        )
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out

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
