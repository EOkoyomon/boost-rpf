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

def _extract_paths_from_sample(data, slack_index=0, slack_vm_pu=1.025, slack_va_degree=0.0):
    """
    Extract all root-to-node paths from a single PyG data sample and compute features.
    
    This function implements the following methodology:
    1. Backward Power Accumulation: Compute P_agg and Q_agg for each node
    2. LinDistFlow Baseline: Compute V_LDF and theta_LDF sequentially
    3. Graph-to-Path Conversion: Extract paths from slack to every non-slack node
    
    Note: V_i and theta_i (parent voltage) are NOT included in covariates.
    Instead, the model uses lags on the target series to access previous voltage.
    This ensures clean separation between training (uses true lags) and testing
    (uses predicted lags via recursive prediction).
    
    Args:
        data: PyTorch Geometric Data object with ppci attribute (raw, untransformed)
        slack_index (int): Index of the slack bus (usually 0)
        slack_vm_pu (float): Deprecated/ignored. The true slack magnitude is read from
            data.y[slack_index, 2]. Kept for backward-compatible call signatures.
        slack_va_degree (float): Deprecated/ignored. The true slack (ext_grid) angle is read from
            data.y[slack_index, 3]; the ~150deg transformer phase shift is applied per-edge inside
            the LDF sweep, so this must NOT be pre-seeded to -150. Kept for signature compatibility.

    Returns:
        list of dict: Each dict contains:
            - 'path': list of node indices from slack to target node
            - 'features': np.array of shape (path_length, 8) with feature vectors
            - 'targets': np.array of shape (path_length, 2) with [V_j, theta_j]
            - 'target_node': the final node in the path
    """
    # 1. Ground truth voltages from y labels
    # y format: [p_mw, q_mvar, vm_pu, va_degree]
    V_true = data.y[:, 2].numpy()  # vm_pu
    theta_true = data.y[:, 3].numpy()  # va_degree

    num_pyg_nodes = len(data.x)

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
    edge_r = internals["edge_r"]
    edge_x = internals["edge_x"]
    P_load = internals["P_load"]
    Q_load = internals["Q_load"]
    P_agg = internals["P_flow"]  # backward-swept (aggregated) active power per node
    Q_agg = internals["Q_flow"]  # backward-swept (aggregated) reactive power per node
    V_LDF = internals["vm_full"]  # LDF voltage magnitude (p.u.), all ppci nodes
    theta_LDF_deg = internals["va_full"]  # LDF voltage angle (degrees), all ppci nodes

    # 3. Extract paths for each non-slack node (up to num_pyg_nodes)
    path_data_list = []
    
    for target_node in range(1, num_pyg_nodes):  # Skip slack (node 0)
        if target_node not in paths:
            continue
            
        path = paths[target_node]  # [slack, ..., parent, target_node]
        
        # Include slack in the sequence so we have a "previous" value for the first child
        # The sequence is: slack -> child1 -> child2 -> ... -> target_node
        # Target at step 0 is slack voltage, target at step 1 is child1 voltage, etc.
        path_length = len(path)
        
        # Build feature vectors for each step in the path (including slack)
        # Feature vector: [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
        # Note: V_i, theta_i are NOT included - they come from target lags
        features = np.zeros((path_length, 8))
        # features = np.zeros((path_length, 9)) # Added one more feature for node degree
        targets = np.zeros((path_length, 2))  # [V_j, theta_j]
        
        for step_idx, j in enumerate(path):
            if step_idx == 0:
                # Slack bus: no branch to it, just its properties
                r_ij = 0.0
                x_ij = 0.0
            else:
                i = path[step_idx - 1]  # Parent node
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

            # Build feature vector (no parent voltage - that comes from lags)
            features[step_idx, 0] = r_ij  # Branch resistance
            features[step_idx, 1] = x_ij  # Branch reactance
            features[step_idx, 2] = P_load[j]  # Local P injection
            features[step_idx, 3] = Q_load[j]  # Local Q injection
            features[step_idx, 4] = P_agg[j]  # Aggregated P
            features[step_idx, 5] = Q_agg[j]  # Aggregated Q
            features[step_idx, 6] = V_LDF[j]  # LinDistFlow V estimate
            features[step_idx, 7] = theta_LDF_deg[j]  # LinDistFlow theta estimate
            # features[step_idx, 8] = degrees_dict[j]  # Node degree
            
            # Target: true voltage at node j
            if j < num_pyg_nodes:
                targets[step_idx, 0] = V_true[j]
                targets[step_idx, 1] = theta_true[j]
            else:
                # For extra ppci nodes (shouldn't happen)
                targets[step_idx, 0] = V_LDF[j]
                targets[step_idx, 1] = theta_LDF_deg[j]
        
        path_data_list.append({
            'path': path,
            'features': features,
            'targets': targets,
            'target_node': target_node
        })
    
    return path_data_list

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
    Load grid data and convert to sequential path format for darts time series training.
    
    This function transforms graph-based power flow data into sequences of feature vectors
    along paths from the slack bus to each node, following the methodology for recursive
    voltage prediction.
    
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
            - 'paths': list of dict, each containing:
                - 'targets': numpy array with targets [V_j, theta_j]
                - 'features': numpy array with features (past_covariates)
                - 'path': list of node indices
                - 'target_node': int, the final node in the path
    
    Feature vector (covariates) for each step j in path:
        [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
    
    Note: V_i, theta_i (parent voltage) are NOT in covariates - they come from target lags.
    
    Target vector for each step j in path:
        [V_j, theta_j]
    """
    # Load raw dataset (without transformation)
    dataset_path = os.path.join(data_dir, grid_type, 'train', 'dataset_with_ppci.pt')
    raw_dataset = torch.load(dataset_path, weights_only=False)
    
    # Feature and target column names
    # Note: V_i, theta_i are NOT included - they come from target lags
    feature_names = [
        'r_ij', 'x_ij', 
        'P_j', 'Q_j', 'P_agg_j', 'Q_agg_j',
        'V_LDF_j', 'theta_LDF_j'
    ]
    target_names = ['V_j', 'theta_j']
    
    # Process each sample in the dataset
    all_samples = []

    for sample_idx, data in enumerate(tqdm(raw_dataset, desc=f"Processing {grid_type}", leave=False)):
        # Extract paths from this sample
        sample_paths = _extract_paths_from_sample(
            data, 
            slack_index=0,
            slack_vm_pu=slack_vm_pu,
            slack_va_degree=slack_va_degree
        )

        all_samples.append({
            'grid_type': grid_type,
            'sample_idx': sample_idx,
            'num_nodes': len(data.x),
            'paths': sample_paths,
            'true_voltages': data.y[:, 2:4].numpy(),  # Ground truth for all nodes
        })

    return all_samples


def load_precomputed_paths(data_dir, grid_type):
    """
    Load pre-computed path data from disk.

    This is much faster than get_grid_paths() because the expensive path extraction
    and feature computation is already done.

    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of grid (e.g., '1-LV-rural1--1-no_sw')

    Returns:
        list of dict: Same format as get_grid_paths() output.

    Raises:
        FileNotFoundError: If pre-computed data doesn't exist. Run precompute_paths.py first.
    """
    # Load pre-computed data
    precomputed_path = os.path.join(data_dir, grid_type, 'train', 'dataset_sequential.pkl')
    print(precomputed_path)

    if not os.path.exists(precomputed_path):
        raise FileNotFoundError(
            f"Pre-computed path data not found at {precomputed_path}. "
            f"Run 'python scripts/precompute_paths.py --data_dir {data_dir}' first."
        )

    with open(precomputed_path, 'rb') as f:
        save_data = pickle.load(f)

    feature_names = save_data['feature_names']
    target_names = save_data['target_names']
    samples = save_data['samples']

    # Convert numpy arrays to desired format
    all_samples = []

    for sample in tqdm(samples):
        sample_paths = []

        for path_data in sample['paths']:
            target_series = path_data['targets']
            covariate_series = path_data['features']

            sample_paths.append({
                'targets': target_series,
                'features': covariate_series,
                'path': path_data['path'],
                'target_node': path_data['target_node'],
            })

        all_samples.append({
            'grid_type': sample['grid_type'],
            'sample_idx': sample['sample_idx'],
            'num_nodes': sample['num_nodes'],
            'paths': sample_paths,
            'true_voltages': sample['true_voltages'],
        })

    return all_samples

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
