import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, TensorDataset
from torch.utils.data import DataLoader as TabularDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from models.lindistflow import calculate_lindistflow_iterative

DATASET_CACHE = {}

# Bumped when the layout of dataset_sequential.pkl changes. Version 2 stores one row
# per bus plus the radial tree, superseding the slack-to-every-bus path enumeration.
SEQUENTIAL_FORMAT_VERSION = 2

def get_networkx_graph(data, include_features=False):
    """
    Convert a PyTorch Geometric Data object to a NetworkX graph.
    Args:
        data (torch_geometric.data.Data): The PyTorch Geometric Data object.
        include_features (bool): Whether to include node and edge features in the NetworkX graph.
    
    Returns:
        networkx.Graph: The converted NetworkX graph.
    """
    if include_features:
        return to_networkx(data, node_attrs=['x', 'y'], edge_attrs=['edge_attr'], to_undirected='upper')
    else:
        return to_networkx(data, to_undirected='upper')
    
def get_path_lengths_to_slack(nx_graph, slack_bus):
    """
    Get the shortest path lengths from all nodes to the slack bus in the NetworkX graph.
    Args:
        nx_graph (networkx.Graph): The NetworkX graph.
        slack_bus (int): The index of the slack bus node.
    
    Returns:
        list: List of shortest path lengths from each node to the slack bus.
    """
    paths = [len(path) - 1 for _, path in 
             sorted(nx.shortest_path(nx_graph, target=slack_bus).items())]
    return paths

def add_path_length_to_slack_bus(dataset):
    """
    Add the shortest path length to the slack bus as a feature to each node in the dataset.
    Args:
        dataset (list of torch_geometric.data.Data): List of PyTorch Geometric Data objects.
    
    Returns:
        list of torch_geometric.data.Data: The dataset with added path length to slack bus as a feature.
    """
    for data in dataset:
        nx_graph = get_networkx_graph(data)

        # Find the slack bus
        slack_bus = -1
        for i, node in enumerate(data.x):
            if node[0] == 1:
                slack_bus = i
                break
        assert slack_bus != -1
        path_lengths = get_path_lengths_to_slack(nx_graph, slack_bus)
        path_lengths = np.array(path_lengths).reshape(-1, 1)
        data.x = torch.tensor(np.hstack([data.x, path_lengths]),
                              dtype=torch.float32)
    return dataset

def transform_dataset(dataset, add_hops=True, grid_name=None):
    """
    Transform the dataset to:
    1. Store slack bus info globally (vm_pu, va_degree, connection impedances)
    2. Remove bus type encodings (Slack?, PV?, PQ?)
    3. Remove p_mw and q_mvar from labels since they are not predicted for PQ buses.
    
    Args:
        dataset: List of PyTorch Geometric Data objects
        add_hops: Whether to add hops to slack bus as a feature before transformation
        grid_name: Name of the grid type for batching optimization
    
    Returns:
        List of transformed Data objects with only PQ buses and global slack info
    """

    # Pre-process dataset by adding hops to slack bus
    if add_hops:
        dataset = add_path_length_to_slack_bus(dataset)

    transformed_dataset = []

    for data in dataset:
        # Find the slack bus
        slack_bus_idx = None
        for i, node in enumerate(data.x):
            if node[0] == 1:  # Slack? feature
                slack_bus_idx = i
                break
        
        if slack_bus_idx is None:
            raise ValueError("No slack bus found in the data")
        
        # Extract slack bus information
        slack_vm_pu = data.x[slack_bus_idx, 5].item()  # vm_pu from node features
        slack_va_degree = data.x[slack_bus_idx, 6].item()  # va_degree from node features
        
        # Find the edge connected to slack bus to get impedance parameters
        slack_r_pu = 0.01  # Default value
        slack_x_pu = 0.005  # Default value

        edge_mask = (data.edge_index[0] == slack_bus_idx) | (data.edge_index[1] == slack_bus_idx)
        slack_edge_attrs = data.edge_attr[edge_mask]
        
        if len(slack_edge_attrs) > 0:
            # Use the first edge connected to slack bus for impedance parameters
            # There should typically be only one such edge in our datasets
            first_slack_edge = slack_edge_attrs[0]
            slack_r_pu = first_slack_edge[1].item()  # r_pu
            slack_x_pu = first_slack_edge[2].item()  # x_pu

        new_x = data.x
        new_y = data.y
        
        # For y labels, keep only vm_pu and va_degree (remove p_mw and q_mvar)
        # Original y: [p_mw, q_mvar, vm_pu, va_degree]
        # New y: [vm_pu, va_degree]
        new_y = new_y[:, 2:4]  # Keep only vm_pu and va_degree
        
        # Remove bus type encodings. Also remove vm_pu and va_degree from inputs 
        # since they are unknowns for PQ buses.
        # Original: [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree, hops_to_slack] (where hops_to_slack is added if add_hops=True)
        # New: [p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]
        new_x_transformed = torch.cat([new_x[:, 3:7], new_x[:, 7:]], dim=1)  # [p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]

        new_edge_index = data.edge_index
        new_edge_attr = data.edge_attr
        
        # Simplify edge attributes to [r_pu, x_pu] (remove trafo? and sc_voltage)
        # Original: [trafo?, r_pu, x_pu, sc_voltage]
        # # New: [r_pu, x_pu]
        # new_edge_attr_simplified = new_edge_attr[:, 1:3]  # Keep only r_pu and x_pu
        
        # Create new Data object with slack connection info as global attribute
        transformed_data = Data(
            x=new_x_transformed, # [p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]
            edge_index=new_edge_index, 
            edge_attr=new_edge_attr, # [trafo?, r_pu, x_pu, sc_voltage]
            y=new_y, # [vm_pu, va_degree]
            dc_pf=data.dc_pf[:, 2:4], # [vm_pu, va_degree]
            slack_info=torch.tensor([slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]),  # Global slack connection info
            ppci=data.ppci,
            grid_name=grid_name  # For batching optimization
        )
        
        transformed_dataset.append(transformed_data)
    
    return transformed_dataset

def get_pyg_graphs(data_dir, grid_type):
    """
    Load PyTorch Geometric graphs from the specified directory and grid type.
    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of sb grid (e.g., '1-LV-rural1--0-no_sw', '1-MV-urban--1-no_sw', etc.)
    
    Returns:
        list of torch_geometric.data.Data: List of PyTorch Geometric Data objects.
    """
    dataset_path = os.path.join(data_dir, grid_type, 'train', 'dataset_with_ppci.pt')
    pyg_dataset = torch.load(dataset_path, weights_only=False)
    pyg_dataset = transform_dataset(pyg_dataset, add_hops=True, grid_name=grid_type)
    return pyg_dataset

def _extract_nodes_from_sample(data, slack_index=0):
    """
    Extract the per-bus sequential learning data from a single PyG data sample.

    This function implements the following methodology:
    1. Backward Power Accumulation: Compute P_agg and Q_agg for each node
    2. LinDistFlow Baseline: Compute V_LDF and theta_LDF sequentially
    3. Graph-to-Path Conversion: store the radial tree (parent/depth per bus)

    The sequential models are 1-step Markov: the training row for bus j is built from
    (parent(j), j) only. Because the grid is radial, every bus has exactly one parent, so
    one row per bus holds all the information that enumerating every slack-to-bus path
    does. We therefore store per-bus arrays plus the tree structure, and reproduce that
    implicit weighting explicitly at training time via `compute_sample_weights`.

    Note: V_i and theta_i (parent voltage) are NOT included in the covariates. The model
    reads the previous voltage through the parent row instead, which keeps training (true
    parent voltage) separate from testing (predicted parent voltage).

    Args:
        data: PyTorch Geometric Data object with ppci attribute (raw, untransformed)
        slack_index (int): Index of the slack bus (usually 0)

    Returns:
        dict with (N = number of buses):
            - 'features': np.array (N, 8) [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
                where r_ij, x_ij belong to the branch parent(j) -> j (zero at the slack)
            - 'targets': np.array (N, 2) with the true [V_j, theta_j] (slack row = true slack state)
            - 'parent': np.array (N,) int32 parent bus index, -1 at the slack
            - 'depth': np.array (N,) int32 hops from the slack bus
    """
    # 1. Ground truth voltages from y labels
    # y format: [p_mw, q_mvar, vm_pu, va_degree]
    V_true = data.y[:, 2].numpy()  # vm_pu
    theta_true = data.y[:, 3].numpy()  # va_degree

    num_nodes = len(data.x)

    # True slack (ext_grid) state.
    slack_vm_pu = data.y[slack_index, 2].item()
    slack_va_degree = data.y[slack_index, 3].item()

    # 2. Compute the LinDistFlow baseline via the implementation in
    # `models/lindistflow.py` instead of duplicating the sweep here.
    # `return_internals=True` gives us the intermediate quantities (tree paths,
    # per-edge r/x, aggregated P/Q, LDF V/theta) needed to build the branch
    # features below.
    _, _, internals = calculate_lindistflow_iterative(
        data,
        slack_index=slack_index,
        slack_vm_pu=slack_vm_pu,
        slack_va_degree=slack_va_degree,
        return_internals=True,
    )

    paths = internals["paths"]
    parents = internals["parents"]
    edge_r = internals["edge_r"]
    edge_x = internals["edge_x"]
    P_load = internals["P_load"]
    Q_load = internals["Q_load"]
    P_agg = internals["P_flow"]  # backward-swept (aggregated) active power per node
    Q_agg = internals["Q_flow"]  # backward-swept (aggregated) reactive power per node
    V_LDF = internals["vm_full"]  # LDF voltage magnitude (p.u.), all ppci nodes
    theta_LDF_deg = internals["va_full"]  # LDF voltage angle (degrees), all ppci nodes

    # 3. Build one row per bus
    features = np.zeros((num_nodes, 8))
    targets = np.zeros((num_nodes, 2))
    parent = np.full(num_nodes, -1, dtype=np.int32)
    depth = np.zeros(num_nodes, dtype=np.int32)

    for j in range(num_nodes):
        if j == slack_index:
            # Slack bus: no branch leading to it, just its own properties
            r_ij = 0.0
            x_ij = 0.0
        else:
            if j not in parents:
                raise ValueError(f'Bus {j} is not connected to the slack bus {slack_index}.')
            i = parents[j]
            parent[j] = i
            depth[j] = len(paths[j]) - 1
            # Get branch impedance (i -> j)
            if (i, j) in edge_r:
                r_ij = edge_r[(i, j)]
                x_ij = edge_x[(i, j)]
            elif (j, i) in edge_r:
                r_ij = edge_r[(j, i)]
                x_ij = edge_x[(j, i)]
            else:
                r_ij = 0.0
                x_ij = 0.0

        features[j, 0] = r_ij  # Branch resistance
        features[j, 1] = x_ij  # Branch reactance
        features[j, 2] = P_load[j]  # Local P injection
        features[j, 3] = Q_load[j]  # Local Q injection
        features[j, 4] = P_agg[j]  # Aggregated P
        features[j, 5] = Q_agg[j]  # Aggregated Q
        features[j, 6] = V_LDF[j]  # LinDistFlow V estimate
        features[j, 7] = theta_LDF_deg[j]  # LinDistFlow theta estimate

        targets[j, 0] = V_true[j]
        targets[j, 1] = theta_true[j]

    return {
        'features': features,
        'targets': targets,
        'parent': parent,
        'depth': depth,
    }

def compute_sample_weights(parent, depth, scheme='subtree'):
    """
    Training weights for the per-bus rows, reproducing the weighting that path
    enumeration applies implicitly through duplicated rows.

    Args:
        parent (np.array): (N,) parent bus index, -1 at the slack.
        depth (np.array): (N,) hops from the slack bus.
        scheme (str): One of
            - 'subtree': number of buses at or below j. Identical to the number of times
              slack-to-every-bus paths duplicated the row for j, i.e. this reproduces the
              original training behavior. Also the number of final predictions that a
              prediction error at j propagates into.
            - 'leaves': number of leaves at or below j, i.e. what slack-to-leaf-only paths
              would have implied.
            - 'uniform': every branch transition counted once.

    Returns:
        np.array: (N-1,) weights aligned with buses 1..N-1 (the slack has no row).
    """
    num_nodes = len(parent)
    # Deepest first, so a bus is always visited before its parent.
    order = np.argsort(depth)[::-1]

    if scheme == 'uniform':
        weights = np.ones(num_nodes)
    elif scheme == 'subtree':
        weights = np.ones(num_nodes)
        for node in order:
            if parent[node] >= 0:
                weights[parent[node]] += weights[node]
    elif scheme == 'leaves':
        is_parent = np.zeros(num_nodes, dtype=bool)
        is_parent[parent[parent >= 0]] = True
        weights = (~is_parent).astype(float)  # leaves start at 1, internal buses at 0
        for node in order:
            if parent[node] >= 0:
                weights[parent[node]] += weights[node]
    else:
        raise ValueError(f"Unknown weight scheme '{scheme}'. Choose from 'subtree', 'leaves', 'uniform'.")

    return weights[1:]  # The slack bus is never a prediction target

def get_tabular_data(data_dir, grid_type):
    graph_dataset = get_pyg_graphs(data_dir, grid_type)

    MAX_NODES = 129 # LV Rural3
    MAX_EDGES = 129 # LV Rural3
    DIM_NODE_FEATURES = graph_dataset[0].x.shape[1] # 5
    DIM_EDGE_FEATURES = graph_dataset[0].edge_attr.shape[1] # 4
    DIM_OUTPUT_FEATURES = graph_dataset[0].y.shape[1] # 2

    # tabular_dataset = []
    x_data = []
    y_data = []

    for data in graph_dataset:
        # Build fixed-size input feature vector
        inputs = np.zeros(MAX_NODES * DIM_NODE_FEATURES + MAX_EDGES * DIM_EDGE_FEATURES)

        # Add node features
        flattened_node_features = data.x.numpy().flatten()
        inputs[:len(flattened_node_features)] = flattened_node_features

        # Add edge features
        flattened_edge_features = data.edge_attr[::2, :].numpy().flatten()
        inputs[MAX_NODES * DIM_NODE_FEATURES:MAX_NODES * DIM_NODE_FEATURES + len(flattened_edge_features)] = flattened_edge_features

        # Create fixed-size output vector
        outputs = np.zeros(MAX_NODES * DIM_OUTPUT_FEATURES)

        # Add output targets
        flattened_targets = data.y.numpy().flatten()
        outputs[:len(flattened_targets)] = flattened_targets

        # Append to grid lists
        # tabular_dataset.append((grid_features, grid_targets))
        x_data.append(inputs)
        y_data.append(outputs)

    return TensorDataset(torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32))

def get_grid_paths(data_dir, grid_type, slack_vm_pu=1.025, slack_va_degree=0.0):
    """
    Load grid data and convert to the per-bus sequential format used by the XGB models.

    This function transforms graph-based power flow data into one feature/target row per
    bus, plus the radial tree (parent and depth per bus) that the models walk at training
    and prediction time. See `_extract_nodes_from_sample` for why one row per bus is
    sufficient, and `compute_sample_weights` for the weighting that path enumeration uses
    to apply implicitly.

    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of sb grid (e.g., '1-LV-rural1--1-no_sw', '1-MV-urban--1-no_sw', etc.)
        slack_vm_pu (float): Deprecated/ignored — the true slack magnitude is read per-sample from
            the labels. Kept for backward-compatible call signatures.
        slack_va_degree (float): Deprecated/ignored — the true slack (ext_grid) angle is read
            per-sample from the labels and the transformer phase shift is applied per-edge in the
            LDF sweep. Must not be pre-seeded to -150. Kept for signature compatibility.

    Returns:
        list of dict: Each dict represents one network sample and contains:
            - 'grid_type': str, the grid type identifier
            - 'sample_idx': int, index of this sample in the original dataset
            - 'num_nodes': int, number of buses N
            - 'features': np.array (N, 8), covariates per bus
            - 'targets': np.array (N, 2), true [V_j, theta_j] per bus
            - 'parent': np.array (N,), parent bus index (-1 at the slack)
            - 'depth': np.array (N,), hops from the slack bus
            - 'true_voltages': np.array (N, 2), ground truth for all buses

    Feature vector (covariates) for each bus j:
        [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]

    Note: V_i, theta_i (parent voltage) are NOT in the covariates - they are read from the
    parent's target row.

    Target vector for each bus j:
        [V_j, theta_j]
    """
    # Load raw dataset (without transformation)
    dataset_path = os.path.join(data_dir, grid_type, 'train', 'dataset_with_ppci.pt')
    raw_dataset = torch.load(dataset_path, weights_only=False)

    # Process each sample in the dataset
    all_samples = []

    for sample_idx, data in enumerate(tqdm(raw_dataset, desc=f"Processing {grid_type}", leave=False)):
        # Extract the per-bus rows and tree structure from this sample
        sample_nodes = _extract_nodes_from_sample(data, slack_index=0)

        all_samples.append({
            'grid_type': grid_type,
            'sample_idx': sample_idx,
            'num_nodes': len(data.x),
            'features': sample_nodes['features'],
            'targets': sample_nodes['targets'],
            'parent': sample_nodes['parent'],
            'depth': sample_nodes['depth'],
            'true_voltages': data.y[:, 2:4].numpy(),  # Ground truth for all nodes
        })

    return all_samples


def load_precomputed_paths(data_dir, grid_type):
    """
    Load pre-computed per-bus sequential data from disk.

    This is much faster than get_grid_paths() because the expensive path extraction
    and feature computation is already done.

    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of grid (e.g., '1-LV-rural1--1-no_sw')

    Returns:
        list of dict: Same format as get_grid_paths() output.

    Raises:
        FileNotFoundError: If pre-computed data doesn't exist, or was written in another
            path-enumeration format. Run precompute_paths.py first.
    """
    # Load pre-computed data
    precomputed_path = os.path.join(data_dir, grid_type, 'train', 'dataset_sequential.pkl')

    if not os.path.exists(precomputed_path):
        raise FileNotFoundError(
            f"Pre-computed path data not found at {precomputed_path}. "
            f"Run 'python scripts/precompute_paths.py --data_dir {data_dir}' first."
        )

    with open(precomputed_path, 'rb') as f:
        save_data = pickle.load(f)

    if save_data.get('format_version') != SEQUENTIAL_FORMAT_VERSION:
        raise FileNotFoundError(
            f"{precomputed_path} was written in the superseded slack-to-every-bus path format "
            f"(found format_version={save_data.get('format_version')}, "
            f"expected {SEQUENTIAL_FORMAT_VERSION}). "
            f"Run 'python scripts/precompute_paths.py --data_dir {data_dir}' to regenerate it."
        )

    return save_data['samples']


def get_dataset(data_dir, grid_types, paths=False, tabular=False):
    """
    Load and cache datasets for the specified grid types.
    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_types (list of str): List of grid types to load.
        paths (bool): Whether to load path-based datasets.
        tabular (bool): Whether to load tabular datasets.

    Returns:
        list of torch_geometric.data.Data: Combined list of PyTorch Geometric Data objects from all specified grid types.
    """
    complete_dataset = []
    for grid in grid_types:
        pyg_dataset = None
        id = (grid, "real", "paths" if paths else "tabular" if tabular else "graphs")
        if id in DATASET_CACHE:
            pyg_dataset = DATASET_CACHE[id]
        else:
            print('Cache miss:', id, '... fetching')
            if paths:
                # Try to load pre-computed paths first (fast), fall back to computing (slow)
                try:
                    pyg_dataset = load_precomputed_paths(data_dir, grid)
                    print(f'  Loaded pre-computed paths for grid {grid}.')
                except FileNotFoundError:
                    print(f'  Pre-computed paths not found, computing (slow)...')
                    print(f'  Hint: Run "python scripts/precompute_paths.py --data_dir {data_dir}" to speed up future loads.')
                    pyg_dataset = get_grid_paths(data_dir, grid)
                DATASET_CACHE[(grid, "real", "paths")] = pyg_dataset
            elif tabular:
                pyg_dataset = get_tabular_data(data_dir, grid)
                DATASET_CACHE[(grid, "real", "tabular")] = pyg_dataset
            else:
                pyg_dataset = get_pyg_graphs(data_dir, grid) # Fetch real dataset
                DATASET_CACHE[(grid, "real", "graphs")] = pyg_dataset # Cache real dataset
        complete_dataset.extend(pyg_dataset)

    return complete_dataset

def get_dataloaders(data_dir,
                    training_grids,
                    testing_grid=None,
                    batch_size=16,
                    paths=False,
                    tabular=False):
    """
    Get PyTorch DataLoaders for training, validation, and testing.
    Args:
        data_dir (str): Base directory where datasets are stored.
        training_grids (list of str): List of grid types to use for training.
        testing_grid (str or None): Grid type to use for testing. If None, a portion of training data is used for testing.
        batch_size (int): Batch size for the DataLoaders.
        paths (bool): Whether to load path-based datasets.
        tabular (bool): Whether to load tabular datasets.
    Returns:
        tuple: (loader_train, loader_val, loader_test) DataLoaders or Numpy Arrays.
    """
    train_dataset = get_dataset(data_dir, training_grids, paths=paths, tabular=tabular)

    if testing_grid:
        # Out of distribution test on left over grid
        train_val_split = [4/5, 1/5]
        train_val_split = [x / sum(train_val_split) for x in train_val_split] # Redistribute to sum to 1
        train_split, val_split = random_split(train_dataset, train_val_split)
        test_split = get_dataset(data_dir, [testing_grid], paths=paths, tabular=tabular)
    else:
        train_val_test_split = [4/6, 1/6, 1/6]
        train_split, val_split, test_split = random_split(train_dataset, train_val_test_split)

    if paths:
        return train_split, val_split, test_split
    elif tabular:
        loader_train = TabularDataLoader(train_split,
                                  batch_size=batch_size,
                                  shuffle=True)
        loader_val = TabularDataLoader(val_split,
                                batch_size=batch_size,
                                shuffle=True)
        loader_test = TabularDataLoader(test_split,
                                 batch_size=batch_size,
                                 shuffle=True)
        return loader_train, loader_val, loader_test
    else:
        loader_train = DataLoader(train_split,
                                batch_size=batch_size,
                                shuffle=True)
        loader_val = DataLoader(val_split,
                                batch_size=batch_size,
                                shuffle=True)
        loader_test = DataLoader(test_split,
                                batch_size=batch_size,
                                shuffle=True)
        return loader_train, loader_val, loader_test
