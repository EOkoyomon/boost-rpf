import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import find


class LinDistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_lindistflow_iterative(data, slack_index=0, slack_vm_pu=data.slack_info[0])
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out

class DistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_distflow_iterative(data, slack_index=0, slack_vm_pu=data.slack_info[0])
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out

def calculate_lindistflow(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=-150.0):
    """
    Calculates LinDistFlow voltages using ONLY the internal Ybus and Sbus.
    This bypasses manual feature extraction and guarantees correct units.

    Args:
        data: PyTorch Geometric Data object grid info and ppci attribute.
        slack_index (int): Index of the slack bus (usually 0).
        slack_vm_pu (float): Voltage magnitude at slack bus in p.u.
        slack_va_degree (float): Voltage angle at slack bus in degrees. Default -150deg due to typical simbench
            trafo Winding Connection (Vector Group), which introduces a 150deg phase shift (ex. Dyn5).

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
    """
    
    # 1. Extract Data from Source
    Ybus = data.ppci["Ybus"].copy()
    
    # ppci Sbus is Net Injection. We need Net Load.
    # This handles both P (Real) and Q (Imag) simultaneously.
    Sbus = -1 * data.ppci["Sbus"].copy()
    num_nodes = Sbus.shape[0]

    # Pandapower adds extra buses for pypower modeling. Luckily, based on how pandapower does it, when we only have
    # slack and PQ nodes, we know the first N would be the predictions for the buses we are interested in.
    assert all((data.x[1:, 0] - Sbus.real[1:len(data.x)]) < 1e-6)
    assert all((data.x[1:, 1] - Sbus.imag[1:len(data.x)]) < 1e-6)

    # 2. Construct Impedance Topology from Ybus
    # Y_ij = -1/z_ij  -> z_ij = -1/Y_ij
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    
    # find() returns (row_indices, col_indices, values)
    rows, cols, vals = find(Ybus)

    for r, c, val in zip(rows, cols, vals):
        # We only care about off-diagonals (lines/transformers)
        if r != c:
            # Calculate Series Impedance (Z = R + jX)
            # Y_bus_element = -1/Z  => Z = -1/Y_bus_element
            z_pu = -1.0 / val

            # Add to graph (undirected for now, we direct it later using paths)
            G.add_edge(r, c, r=z_pu.real, x=z_pu.imag)

    # 3. Build Path Matrix (BFS Tree)
    # Create a directed tree rooted at slack to determine paths
    try:
        paths = nx.shortest_path(G, source=slack_index)
    except nx.NetworkXNoPath:
        print("Creating tree ourselves.")
        # Fallback if directionality is ambiguous in meshed elements,
        # force a tree via BFS
        bfs_tree = nx.bfs_tree(G, source=slack_index)
        paths = nx.shortest_path(bfs_tree, source=slack_index)

    # (R_path, X_path)
    R_matrix = np.zeros((num_nodes, num_nodes))
    X_matrix = np.zeros((num_nodes, num_nodes))
    
    # Pre-fetch edge attributes to speed up loop
    edge_r = nx.get_edge_attributes(G, 'r')
    edge_x = nx.get_edge_attributes(G, 'x')
    
    # Iterate over every node 'j' (The Source of Power Flow/Injection)
    for j in range(num_nodes):
        if j == slack_index: continue

        path_j = paths[j]
        
        # Iterate over every node 'i' (The Observer of Voltage)
        # For every node i, does it share the path to node j?
        for i in range(num_nodes):
            if i == slack_index: continue

            path_i = paths[i]
            
            # Find shared edges / common paths
            # Walk from slack downwards. Stop when paths diverge.
            common_r = 0.0
            common_x = 0.0
            
            # Iterate through edges efficently to find common prefix
            min_len = min(len(path_i), len(path_j))
            for k in range(min_len - 1):
                u, v = path_i[k], path_i[k+1]
                
                # If the path to J also goes through u->v, then I sees the voltage drop caused by J
                if path_j[k+1] == v:
                    # Handle potential undirected storage in NetworkX
                    if (u, v) in edge_r:
                        common_r += edge_r[(u, v)]
                        common_x += edge_x[(u, v)]
                    elif (v, u) in edge_r:
                        common_r += edge_r[(v, u)]
                        common_x += edge_x[(v, u)]
                else:
                    break
            
            R_matrix[i, j] = common_r
            X_matrix[i, j] = common_x

    # 4. Prepare Power Vector
    P_inj = Sbus.real
    Q_inj = Sbus.imag

    # Zero out the slack power. It does not flow down the lines.
    P_inj[slack_index] = 0.0
    Q_inj[slack_index] = 0.0
    
    # 5. Calculate Voltages
    # LinDistFlow for Injection S:
    # V^2 = V_slack^2 - 2 * (R * P_inj + X * Q_inj)
    v_drop_sq = 2 * (R_matrix @ P_inj + X_matrix @ Q_inj)

    v_sq_pred = (slack_vm_pu ** 2) - v_drop_sq

    # --- 6. Calculate Angles (The New Part) ---
    # Formula: theta_j = theta_slack - (1/V_nom)*(X*P - R*Q)
    # Angle Drop = (X * P - R * Q) / V_nom
    # X is correctly paired with P, and R is paired with Q
    ang_drop_rad = (X_matrix @ P_inj - R_matrix @ Q_inj) / slack_vm_pu
    theta_rad = np.deg2rad(slack_va_degree) - ang_drop_rad
    v_ang_pred_deg = np.rad2deg(theta_rad)
    
    # Return the same length as the original data
    return np.sqrt(np.maximum(v_sq_pred, 0.0))[:len(data.x)], v_ang_pred_deg[:len(data.x)]

def calculate_lindistflow_iterative(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=-150.0):
    """
    Iterative Forward-Backward Sweep implementation of LinDistFlow.

    Args:
        data: PyTorch Geometric Data object grid info and ppci attribute.
        slack_index (int): Index of the slack bus (usually 0).
        slack_vm_pu (float): Voltage magnitude at slack bus in p.u.
        slack_va_degree (float): Voltage angle at slack bus in degrees. Default -150deg due to typical simbench
            trafo Winding Connection (Vector Group), which introduces a 150deg phase shift (ex. Dyn5).

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
    """
    return calculate_distflow_iterative(data, slack_index=slack_index, slack_vm_pu=slack_vm_pu, slack_va_degree=slack_va_degree, linear=True)

def calculate_distflow_iterative(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=-150.0, linear=False):
    """
    Iterative Forward-Backward Sweep implementation of DistFlow.
    https://doi.org/10.1109/61.25627.

    Args:
        data: PyTorch Geometric Data object grid info and ppci attribute.
        slack_index (int): Index of the slack bus (usually 0).
        slack_vm_pu (float): Voltage magnitude at slack bus in p.u.
        slack_va_degree (float): Voltage angle at slack bus in degrees. Default -150deg due to typical simbench
            trafo Winding Connection (Vector Group), which introduces a 150deg phase shift (ex. Dyn5).
        linear (bool): Whether to use linearized DistFlow equations.

    Returns:
        np.array: Predicted Voltage Magnitudes (p.u.)
        np.array: Predicted Voltage Angles (degrees)
    """
    # 1. Extract Data from Source
    Ybus = data.ppci["Ybus"].copy()

    # ppci Sbus is Net Injection. We need Net Load.
    # This handles both P (Real) and Q (Imag) simultaneously.
    Sbus = -1 * data.ppci["Sbus"].copy()
    num_nodes = Sbus.shape[0]
    P_load = Sbus.real
    Q_load = Sbus.imag

    # Pandapower adds extra buses for pypower modeling. Luckily, based on how pandapower does it, when we only have
    # slack and PQ nodes, we know the first N would be the predictions for the buses we are interested in.
    assert all((data.x[1:, 0] - P_load[1:len(data.x)]) < 1e-6)
    assert all((data.x[1:, 1] - Q_load[1:len(data.x)]) < 1e-6)

    # 2. Pre-processing Topology
    # We need to map nodes to their parents and the connecting edge parameters
        # parent_map[node_id] = (parent_id, r_line, x_line)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    rows, cols, vals = find(Ybus) # find() returns (row_indices, col_indices, values)

    for r, c, val in zip(rows, cols, vals):
        # We only care about off-diagonals (lines/transformers)
        if r != c:
            # Calculate Series Impedance (Z = R + jX)
            # Y_bus_element = -1/Z  => Z = -1/Y_bus_element
            z_pu = -1.0 / val

            # Add to graph (undirected for now, we direct it later using paths)
            G.add_edge(r, c, r=z_pu.real, x=z_pu.imag)

    # 3. Build Path Matrix (BFS Tree)
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

    # 4. Backward Sweep (Summing Power)
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

    # 4. Forward Sweep (Calculating Voltage)
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

        # The flow on the line connecting parent -> node
        # is exactly the accumulated flow we calculated for 'node'
        p_line = P_flow[node]
        q_line = Q_flow[node]

        # LinDistFlow equation: V_node^2 = V_parent^2 - 2(rP + xQ)
        # Positive Load (P_line) causes Voltage Drop (Subtraction)
        V_sq[node] = V_sq[parent] - 2 * (r * p_line + x * q_line)
        # Angle Drop = (X * P - R * Q) / V_nom
        Va_rad[node] = Va_rad[parent] - ((x * p_line - r * q_line) / slack_vm_pu)

        if not linear:
            # DistFlow equation: V_node^2 = V_parent^2 - 2(rP + xQ) + (r^2 + x^2)(P^2 + Q^2)/(V_parent^2)
            # Positive Load (P_line) causes Voltage Drop (Subtraction)
            V_sq[node] += ((r**2 + x**2)*(p_line**2 + q_line**2)/V_sq[parent])

    # Return the same length as the original data
    return np.sqrt(np.maximum(V_sq, 0))[:len(data.x)], np.rad2deg(Va_rad[:len(data.x)])

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
