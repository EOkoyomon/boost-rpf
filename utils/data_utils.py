import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import networkx as nx
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

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
    3. Remove vm_pu and va_degree from input features since they are unknowns for PQ buses.
    4. Remove p_mw and q_mvar from labels since they are not predicted for PQ buses.
    5. Simplify edge attributes to [r_pu, x_pu]
    
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
        # Original: [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]
        # New: [p_mw, q_mvar, hops_to_slack]
        new_x_transformed = torch.cat([new_x[:, 3:5], new_x[:, 7:]], dim=1)  # [p_mw, q_mvar, hops_to_slack]

        new_edge_index = data.edge_index
        new_edge_attr = data.edge_attr
        
        # Simplify edge attributes to [r_pu, x_pu] (remove trafo? and sc_voltage)
        # Original: [trafo?, r_pu, x_pu, sc_voltage]
        # New: [r_pu, x_pu]
        new_edge_attr_simplified = new_edge_attr[:, 1:3]  # Keep only r_pu and x_pu
        
        # Create new Data object with slack connection info as global attribute
        transformed_data = Data(
            x=new_x_transformed, # [p_mw, q_mvar, hops_to_slack]
            edge_index=new_edge_index, 
            edge_attr=new_edge_attr_simplified, # [r_pu, x_pu]
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

def create_complex_features(dataset):
    """
    Convert real-valued features in the dataset to complex-valued features.
    Args:
        dataset (list of torch_geometric.data.Data): List of PyTorch Geometric Data objects with real-valued features.

    Returns:
        list of torch_geometric.data.Data: List of PyTorch Geometric Data objects with complex-valued features.
    """
    complex_dataset = []

    for data in dataset:
        # Original data format:
        #   x features: [p_mw, q_mvar, hops_to_slack]
        #   edge_attr features: [r_pu, x_pu]
        #   y labels: [vm_pu, va_degree]
        #   slack_info (global): [slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]

        # New data format:
        #   x features: # [complex_power, hops_to_slack]
        #   edge_attr features: [complex_impedance]
        #   y labels: [complex_voltage]
        #   slack_info (global): [slack_complex_voltage, slack_complex_impedance]

        # Make x into complex: p_mw + 1j * q_mvar
        complex_power = data.x[:, 0] + 1j * data.x[:, 1]
        hops_to_slack = data.x[:, 2]
        complex_x = torch.cat([complex_power.view(-1, 1), hops_to_slack.unsqueeze(1).type(torch.complex64)], dim=1)

        # Make edge_attr into complex: r_pu + 1j * x_pu
        complex_impedance = data.edge_attr[:, 0] + 1j * data.edge_attr[:, 1]
        complex_edge_attr = complex_impedance.view(-1, 1).type(torch.complex64)

        # Make y into complex: [vm_pu * exp(1j * va_radian)]
        phase_angle_rad = data.y[:, 1] * torch.pi / 180.0  # Convert to radians
        complex_voltage = torch.polar(data.y[:, 0], phase_angle_rad)  # Convert to complex form
        complex_y = complex_voltage.view(-1, 1).type(torch.complex64)

        # Make slack_info into complex: [slack_vm_pu * exp(1j * slack_va_radian), slack_r_pu + 1j * slack_x_pu]
        slack_complex_impedance = data.slack_info[2] + 1j * data.slack_info[3]
        slack_complex_impedance = slack_complex_impedance.type(torch.complex64)
        slack_phase_angle_rad = data.slack_info[1] * torch.pi / 180.0
        slack_complex_voltage = torch.polar(data.slack_info[0], slack_phase_angle_rad)
        slack_complex_voltage = slack_complex_voltage.type(torch.complex64)

        slack_info = torch.tensor([slack_complex_voltage, slack_complex_impedance], dtype=torch.complex64)

        complex_data = Data(
            x=complex_x,
            edge_index=data.edge_index, # Stays the same
            edge_attr=complex_edge_attr,
            y=complex_y,
            slack_info=slack_info,
            ppci=data.ppci,
            grid_name=data.grid_name
        )
        complex_dataset.append(complex_data)

    return complex_dataset

def get_dataset(data_dir, grid_types, complex=False):
    """
    Load and cache datasets for the specified grid types.
    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_types (list of str): List of grid types to load.
        complex (bool): Whether to load complex datasets.

    Returns:
        list of torch_geometric.data.Data: Combined list of PyTorch Geometric Data objects from all specified grid types.
    """
    complete_dataset = []
    for grid in grid_types:
        pyg_dataset = None
        id = (grid, "complex" if complex else "real")
        if id in DATASET_CACHE:
            pyg_dataset = DATASET_CACHE[id]
        else:
            print('Cache miss:', id, '... fetching')
            pyg_dataset = get_pyg_graphs(data_dir, grid) # Fetch real dataset
            DATASET_CACHE[(grid, "real")] = pyg_dataset # Cache real dataset
            if complex:
                pyg_dataset = create_complex_features(pyg_dataset) # Convert to complex dataset
                DATASET_CACHE[id] = pyg_dataset # Cache complex dataset
        complete_dataset.extend(pyg_dataset)

    return complete_dataset

def get_dataloaders(data_dir,
                    training_grids,
                    testing_grid=None,
                    batch_size=16,
                    complex=False):
    """
    Get PyTorch DataLoaders for training, validation, and testing.
    Args:
        data_dir (str): Base directory where datasets are stored.
        training_grids (list of str): List of grid types to use for training.
        testing_grid (str or None): Grid type to use for testing. If None, a portion of training data is used for testing.
        batch_size (int): Batch size for the DataLoaders.
        complex (bool): Whether to load complex datasets.

    Returns:
        tuple: (loader_train, loader_val, loader_test) DataLoaders for training, validation, and testing.
    """
    train_dataset = get_dataset(data_dir, training_grids, complex=complex)

    if testing_grid:
        # Out of distribution test on left over grid
        train_val_split = [0.75, 0.15]
        train_val_split = [x / sum(train_val_split) for x in train_val_split] # Redistribute to sum to 1
        train_split, val_split = random_split(train_dataset, train_val_split)
        test_split = get_dataset(data_dir, [testing_grid], complex=complex)
    else:
        train_val_test_split = [0.75, 0.15, 0.10]
        train_split, val_split, test_split = random_split(train_dataset, train_val_test_split)

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
