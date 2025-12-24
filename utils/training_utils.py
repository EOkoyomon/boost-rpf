import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_scatter import scatter_add
import matplotlib.pyplot as plt
import simbench as sb
from tqdm import tqdm

import time
import os

from utils.physics_informed_loss_optimized import create_batch_physics_loss

def create_log_dir():
    """
    Create a logging directory for the current run.
    The directory structure is: out/<timestamp>/
        
    Returns:
        str: Path to the created logging directory.
    """
    log_dir = os.path.join('out', time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_model_save_path(log_dir, model_id='0'):
    """
    Generate a file path for saving model weights.
    The file is named 'model_weights_<model_id>.pt' and is located in the logging directory.
    Args:
        log_dir (str): Path to the logging directory.
        model_id (str, optional): Identifier for the model version. Defaults to '0'.
    Returns:
        str: Full path to the model weights file.
    """
    return os.path.join(log_dir, f'model_weights_{model_id}')

def setup_pytorch():
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(12)
    return

def get_device():
    """
    Get the device to be used for PyTorch operations.
    
    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    device = (
        "cuda:1"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def get_dist_grid_codes(scenario=1):
    """
    Get distribution grid codes for a specific scenario.
    Args:
        scenario (int, optional): Scenario number for Simbench. Defaults to 1.
    Returns:
        list: Sorted list of distribution grid codes.
    """
    # Create the codes for the distribution grid cases of Simbench (LV and MV and any combination of the two)
    codes = sb.collect_all_simbench_codes(scenario=scenario)
    dist_grid_codes = list(filter(lambda x: "no_sw" in x and ("-MV-" in x or "-LV-" in x), codes))
    return sorted(dist_grid_codes)

def plot_loss(log_dir,
              model_classname,
              train_loss_vec,
              val_loss_vec,
              fig_id='0'):
    """
    Plot training and validation loss curves and save the figure.
    Args:
        log_dir (str): Directory to save the plot.
        model_classname (str): Name of the model class (for title).
        train_loss_vec (list or np.array): Training loss values over epochs.
        val_loss_vec (list or np.array): Validation loss values over epochs.
        fig_id (str, optional): Identifier for the figure file. Defaults to '0'.
    """
    filename = os.path.join(log_dir, f'fig_{fig_id}.png')
    _, ax = plt.subplots()
    start = (len(train_loss_vec) // 5)*4 # Plot only last 20% of epochs
    ax.plot(train_loss_vec[start:], label = 'train loss')
    ax.plot(val_loss_vec[start:], label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    title = f"{model_classname}"
    ax.set_title(title)
    plt.savefig(filename)
    print(f'Figure saved to: {filename}')

def normalized_mse_loss(pred, target, eps=1e-8):
    """
    Compute the normalized mean squared error loss.
    Args:
        pred (torch.Tensor): Predicted values of shape (N, D).
        target (torch.Tensor): True values of shape (N, D).
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-8.
    Returns:
        torch.Tensor: The computed normalized MSE loss (scalar).
    """
    # To give equal importance to smaller and larger features, we weigh the loss
    # by the inverse of the true vector’s norm.

    # Compute the L2 norm across the dimensions of the true vectors
    target_norm = torch.norm(target, dim=0, keepdim=True) + eps  # Shape: (1, D)
    weights = 1.0 / target_norm  # Shape: (1, D)
    # Compute the element-wise MSE
    mse = nn.functional.mse_loss(pred, target, reduction='none')  # Shape: (N, D)
    # Apply weights and compute the mean
    weighted_mse = weights * mse  # Broadcasting over (N, D)
    # Return the mean loss across all elements
    return weighted_mse.mean()

def physics_loss_vectorized(data, predictions):
    """
    Compute physics-informed loss based on AC power flow equations.

    Fully vectorized physics-informed loss. Instead of using for-loops, this is 
    optimized using torch_scatter for all aggregation operations instead of for-loops,
    making it much faster for batched training.

    This function computes AC power flow physics loss for a dataset where:
    - The slack bus has been removed from the graph structure
    - Only PQ buses remain as nodes
    - Only PQ-PQ edges remain
    - Slack bus information is embedded in global attributes
    - Missing PQ-slack connections must be reconstructed using global slack impedance
    
    Uses the nodal power injection convention: P_injection = P_generation - P_load
    Implements correct AC power flow equations with self-admittance terms (π-model):
    P_i = -V_i² * G_ii + Σ_j V_i * V_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j))
    
    Args:
        data: PyTorch Geometric Data object (transformed) containing:
            - x: Node features [p_mw, q_mvar, hops_to_slack] (3 features after transformation)
                Shape: [num_pq_buses, 3]
            - edge_index: Edge connectivity for PQ-PQ connections [2, num_pq_edges]  
            - edge_attr: Simplified edge attributes [r_pu, x_pu] for PQ-PQ connections
                Shape: [num_pq_edges, 2]
            - slack_info: Global slack connection info [slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]
                Shape: [4] or [batch_size*4] for batched data
            - net: Original (solved) pandapower network object for reference
        predictions: Predicted voltages [vm_pu, va_degree] for PQ buses only
            Shape: [num_pq_buses, 2]
    
    Returns:
        torch.Tensor: Physics-informed loss scalar
        
    Note:
        - Only adds slack interactions to buses with hops_to_slack == 1 (directly connected)
        - Uses correct sign convention for nodal power injections
        - Uses torch_scatter for vectorized aggregation operations
    """
    
    # Extract predicted voltages for PQ buses
    V_pred = predictions[:, 0]  # Voltage magnitudes (p.u.)
    theta_pred_deg = predictions[:, 1]  # Voltage angles (degrees)
    theta_pred_rad = theta_pred_deg * torch.pi / 180.0  # Convert to radians
    
    num_pq_buses = len(V_pred)
    
    # Extract slack bus reference values from global attributes
    # Handle batched data: reshape slack_info if it contains multiple graphs
    if hasattr(data, 'batch') and data.batch is not None:
        # Batched data: slack_info contains info for each graph in the batch
        batch_size = data.batch.max().item() + 1
        slack_info_per_graph = data.slack_info.view(batch_size, 4)
        # Get slack info for each node based on which graph it belongs to
        node_slack_info = slack_info_per_graph[data.batch]  # [num_nodes, 4]
        slack_vm_pu = node_slack_info[:, 0]  # Per-node slack voltage magnitude
        slack_va_rad = node_slack_info[:, 1] * torch.pi / 180.0  # Per-node slack angle
        slack_r_pu = node_slack_info[:, 2]  # Per-node slack resistance
        slack_x_pu = node_slack_info[:, 3]  # Per-node slack reactance
    else:
        # Single graph: broadcast slack_info to all nodes
        slack_vm_pu = data.slack_info[0].expand(num_pq_buses)
        slack_va_rad = (data.slack_info[1] * torch.pi / 180.0).expand(num_pq_buses)
        slack_r_pu = data.slack_info[2].expand(num_pq_buses)
        slack_x_pu = data.slack_info[3].expand(num_pq_buses)
    
    # Extract edge data (contains only PQ-PQ connections after transformation)
    edge_index = data.edge_index
    r_pu = data.edge_attr[:, 0]  # Resistance per unit
    x_pu = data.edge_attr[:, 1]  # Reactance per unit
    
    # Compute edge admittances for PQ-PQ connections: Y = 1/Z = 1/(R + jX)
    impedance_sq = r_pu**2 + x_pu**2
    G_edges = r_pu / impedance_sq  # Conductance G = R/(R² + X²)
    B_edges = -x_pu / impedance_sq  # Susceptance B = -X/(R² + X²)
    
    # Build nodal admittance matrix diagonal elements using vectorized scatter operations
    # Sum admittances from PQ-PQ connections only from from_bus to match original implementation
    G_diag_pq = scatter_add(G_edges, edge_index[0], dim=0, dim_size=num_pq_buses)
    
    B_diag_pq = scatter_add(B_edges, edge_index[0], dim=0, dim_size=num_pq_buses)
    
    # Add admittances from missing PQ-slack connections using global slack info
    # Compute slack connection admittances
    slack_impedance_sq = slack_r_pu**2 + slack_x_pu**2
    G_slack = slack_r_pu / slack_impedance_sq
    B_slack = -slack_x_pu / slack_impedance_sq
    
    # Identify which PQ buses were originally connected to slack bus
    # These buses have hops_to_slack == 1 in the node features
    hops_to_slack = data.x[:, 2]  # 3rd feature contains hops to slack
    directly_connected_to_slack = (hops_to_slack == 1).float()  # Binary mask
    
    # Add slack connection admittances only to directly connected PQ buses
    G_diag_pq += directly_connected_to_slack * G_slack
    B_diag_pq += directly_connected_to_slack * B_slack
    
    # Calculate self-admittance terms (π-model diagonal contribution)
    # Use correct sign convention for nodal power injection
    P_self_pq = -V_pred**2 * G_diag_pq  # Negative for injection convention
    Q_self_pq = -V_pred**2 * B_diag_pq  # Self-reactive power contribution
    
    # Calculate PQ-PQ interaction terms (off-diagonal contributions)
    i_nodes = edge_index[0]  # From bus indices
    j_nodes = edge_index[1]  # To bus indices
    
    # Compute angle differences and voltage products for all PQ-PQ connections
    theta_diff_pq = theta_pred_rad[i_nodes] - theta_pred_rad[j_nodes]
    V_products_pq = V_pred[i_nodes] * V_pred[j_nodes]
    
    # AC power flow interaction equations
    P_interactions_pq = V_products_pq * (G_edges * torch.cos(theta_diff_pq) + B_edges * torch.sin(theta_diff_pq))
    Q_interactions_pq = V_products_pq * (G_edges * torch.sin(theta_diff_pq) - B_edges * torch.cos(theta_diff_pq))
    
    # Sum all PQ-PQ interactions at each bus using vectorized scatter operations
    # Apply negative sign for injection convention during aggregation
    # Only sum from from_bus (edge_index[0]) to match original implementation
    P_interaction_sum_pq = -scatter_add(P_interactions_pq, edge_index[0], dim=0, dim_size=num_pq_buses)
    Q_interaction_sum_pq = scatter_add(Q_interactions_pq, edge_index[0], dim=0, dim_size=num_pq_buses)
    
    # Add PQ-slack interactions for directly connected buses
    # These represent the missing edges that were removed during transformation
    theta_diff_slack = theta_pred_rad - slack_va_rad  # PQ angles - slack angle
    V_products_slack = V_pred * slack_vm_pu  # PQ voltages * slack voltage
    
    # Compute power flows to slack bus (same equations as PQ-PQ but with slack voltage)
    P_interactions_slack = V_products_slack * (G_slack * torch.cos(theta_diff_slack) + B_slack * torch.sin(theta_diff_slack))
    Q_interactions_slack = V_products_slack * (G_slack * torch.sin(theta_diff_slack) - B_slack * torch.cos(theta_diff_slack))
    
    # Add slack interactions only for directly connected PQ buses
    P_interaction_sum_pq += -directly_connected_to_slack * P_interactions_slack
    Q_interaction_sum_pq += directly_connected_to_slack * Q_interactions_slack
    
    # Total nodal power injections using AC power flow equations
    # P_i = -V_i² * G_ii + Σ_j interactions_ij (with correct signs)
    P_calculated = P_self_pq - P_interaction_sum_pq
    Q_calculated = Q_self_pq - Q_interaction_sum_pq
    
    # Extract true power injections from node features
    # Transformed dataset stores actual injections (generation - load) in first two features
    P_true = data.x[:, 0]  # Active power injection (MW) 
    Q_true = data.x[:, 1]  # Reactive power injection (Mvar)
    
    # Physics loss: calculated power injections should match true values
    # This enforces Kirchhoff's laws at each bus
    P_error = P_true - P_calculated
    Q_error = Q_true - Q_calculated

    epsilon = 1e-8  # Small value to prevent division by zero
    normalized_p_imbalance = P_error / (torch.abs(P_true) + epsilon)
    normalized_q_imbalance = Q_error / (torch.abs(Q_true) + epsilon)
    loss = torch.mean(normalized_p_imbalance**2) + torch.mean(normalized_q_imbalance**2)

    return loss
    
    # Return sum of squared errors across all buses
    physics_loss = torch.sum(P_error**2 + Q_error**2)
    
    return physics_loss


def complex_mse_loss(y_pred, y_true):
    """
    Compute mean squared error loss for complex-valued predictions.
    Args:
        y_pred (torch.Tensor): Predicted complex values.
        y_true (torch.Tensor): True complex values.

    Returns:
        torch.Tensor: The computed mean squared error loss.
    """
    return torch.mean(torch.abs(y_pred - y_true)**2)

def train(model,
          device,
          loader_train,
          loader_val,
          epochs=100,
          learning_rate=1e-3,
          early_stopping=True,
          patience=100,
          best_val_weights=True,
          save_model_to='',
          log_epochs=False):
    """
    Train a PyTorch model with early stopping and optional best weights saving.
    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        device (torch.device): The device to run the training on (CPU or GPU).
        loader_train (torch.utils.data.DataLoader): DataLoader for the training dataset.
        loader_val (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epochs (int, optional): Maximum number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 100.
        best_val_weights (bool, optional): Whether to save the best model weights. Defaults to True.
        save_model_to (str, optional): Path to save the final model weights. If empty, model is not saved. Defaults to ''.
        log_epochs (bool, optional): Whether to log loss every 100 epochs. Defaults to False.
    Returns:
        tuple: (train_loss_vec, val_loss_vec, best_val_loss, corresponding_train_loss, total_epochs, train_time)
            - train_loss_vec (np.array): Training loss values over epochs.
            - val_loss_vec (np.array): Validation loss values over epochs.
            - best_val_loss (float): Best validation loss achieved.
            - corresponding_train_loss (float): Training loss corresponding to the best validation loss.
            - total_epochs (int): Total number of epochs run (may be less than max epochs due to early stopping).
            - train_time (float): Total training time in seconds.
    """
    # Configure hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler for physics-informed training
    # Start high to escape poor local minima, then reduce for fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200,
        min_lr=0.00001, verbose=True
    )
    
    loss_fn = normalized_mse_loss
    
    # This helps balance supervised vs physics loss contributions
    lambda_phys = 0.00001  # Weight for physics loss if used
    # Standard MSE loss (not normalized)
    # loss_fn = nn.MSELoss(reduction='mean') # Average over all elements

    # Variables to track best model
    best_val_loss = np.Inf
    best_weights = None
    corresponding_train_loss = np.Inf
    wait = 0

    # Setup arrays to track training performance
    train_loss_vec = np.empty(epochs)
    train_loss_vec[:] = np.nan 
    val_loss_vec = np.empty(epochs)
    val_loss_vec[:] = np.nan

    # Run timed train-eval loop
    start = time.time()
    total_epochs = -1

    assert model.is_supervised() or model.use_physics_loss() or model.is_analytical(), (
        "Model must be supervised, use physics loss, or be analytical."
    )

    if model.is_analytical():
        # Does not need to be trained
        return train_loss_vec, val_loss_vec, best_val_loss, corresponding_train_loss, total_epochs, 0.0

    if model.is_complex():
        loss_fn = complex_mse_loss # Use MSE loss for complex models. Only one output (complex voltage).

    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        loss_train = 0
        
        for batch_train in loader_train:
            optimizer.zero_grad()
            batch_train = batch_train.to(device)
            pred = model(batch_train)
            hops_to_slack = batch_train.x.shape[-1] - 1
            pq_mask = (batch_train.x[:, hops_to_slack] != 0)
            loss = loss_fn(pred[pq_mask], batch_train.y[pq_mask])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()*batch_train.num_graphs
        loss_train /= len(loader_train.dataset)

        # Validate
        model.eval()
        loss_val = 0

        # Disable gradient tracking during validation for efficiency
        with torch.no_grad():
            for batch_val in loader_val:
                batch_val = batch_val.to(device)
                pred = model(batch_val)
                hops_to_slack = batch_val.x.shape[-1] - 1
                pq_mask = (batch_val.x[:, hops_to_slack] != 0)
                loss = loss_fn(pred[pq_mask], batch_val.y[pq_mask])
                loss_val += loss.item()*batch_val.num_graphs
        loss_val /= len(loader_val.dataset)

        # Early stopping and update of best model
        if early_stopping or best_val_weights:
            if loss_val < best_val_loss:
                wait = 0
                best_weights = model.state_dict()
                best_val_loss = loss_val
                corresponding_train_loss = loss_train
            elif wait >= patience and early_stopping:
                total_epochs = epoch
                break
            else:
                wait += 1

        # Step learning rate scheduler
        scheduler.step(loss_val)
        
        # Track model performance
        train_loss_vec[epoch] = loss_train
        val_loss_vec[epoch] = loss_val
        if log_epochs and epoch % 10 == 9:
            print('Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f} LR: {:.2e}'
                    .format(epoch + 1, loss_train, loss_val, optimizer.param_groups[0]['lr']), flush=True)

    if model.use_physics_loss():
        physics_loss = create_batch_physics_loss(device=device, is_complex=model.is_complex()) # Create physics loss function for batches
       # Variables to track best model
        best_val_loss_physics = best_val_loss
        best_weights_physics = best_weights
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        physics_fine_tuning_epochs = 20
        for epoch in tqdm(range(physics_fine_tuning_epochs)):
            # Train for some more epochs to fine-tune with physics loss
            model.train()
            loss_train = 0

            for batch_train in loader_train:
                optimizer.zero_grad()
                batch_train = batch_train.to(device)
                pred = model(batch_train)
                hops_to_slack = batch_train.x.shape[-1] - 1
                pq_mask = (batch_train.x[:, hops_to_slack] != 0)

                # Separate the batch into individual graphs for physics loss calculation
                # We use boolean mask (batch_train.batch == i) to select the rows, i.e. nodes, of pred that belong to graph i.
                batch_predictions = [pred[(batch_train.batch == i)] for i in range(batch_train.num_graphs)]
                loss = physics_loss(batch_predictions, batch_train.to_data_list())

                loss.backward()
                optimizer.step()
                loss_train += loss.item()*batch_train.num_graphs
            loss_train /= len(loader_train.dataset)

            # Validate
            model.eval()
            loss_val = 0

            # Disable gradient tracking during validation for efficiency
            with torch.no_grad():
                for batch_val in loader_val:
                    batch_val = batch_val.to(device)
                    pred = model(batch_val)
                    hops_to_slack = batch_val.x.shape[-1] - 1
                    pq_mask = (batch_val.x[:, hops_to_slack] != 0)

                    # Separate the batch into individual graphs for physics loss calculation
                    batch_predictions = [pred[(batch_val.batch == i)] for i in range(batch_val.num_graphs)]
                    loss = physics_loss(batch_predictions, batch_val.to_data_list())

                    loss_val += loss.item()*batch_val.num_graphs
            loss_val /= len(loader_val.dataset)

            # Early stopping and update of best model
            if best_val_weights and loss_val < best_val_loss_physics:
                    best_weights_physics = model.state_dict()
                    best_val_loss_physics = loss_val
                    corresponding_train_loss = loss_train

            if log_epochs:
                print('Physics Fine-tuning Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f} LR: {:.2e}'
                        .format(epoch + 1, loss_train, loss_val, optimizer.param_groups[0]['lr']), flush=True)

    # Total training time
    train_time = time.time() - start

    # Total num epochs (if stopped early)
    total_epochs = epochs if total_epochs == -1 else total_epochs

    if best_val_weights:
        model.load_state_dict(best_weights if not model.use_physics_loss() else best_weights_physics)

    if  model.use_physics_loss():
        best_val_loss = best_val_loss_physics
        total_epochs += physics_fine_tuning_epochs

    if save_model_to:
        torch.save(model.state_dict(), save_model_to)
        print(f'Model weights saved to: {save_model_to}')

    return train_loss_vec, val_loss_vec, best_val_loss, corresponding_train_loss, total_epochs, train_time

def rmse(y_pred, y_true):
    """
    Compute the root mean squared error (RMSE) between true and predicted values.
    Args:
        y_pred (torch.Tensor): Predicted values of shape (N, D).
        y_true (torch.Tensor): True values of shape (N, D).
    Returns:
        torch.Tensor: RMSE for each dimension (D,).
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=0))

def rmse_wrapped_va(y_pred, y_true, period=360.0):
    """
    Compute the root mean squared error (RMSE) between true and predicted values.
    Args:
        y_pred (torch.Tensor): Predicted values of shape (N, D), where second column is va_degree.
        y_true (torch.Tensor): True values of shape (N, D), where second column is va_degree.
    Returns:
        torch.Tensor: RMSE for each dimension (D,).
    """
    # Calculate simple difference
    diff = y_true - y_pred

    # Wrap the difference to the interval [-period/2, period/2]
    # This solves an issue where 359 is seen as far from 0.1 degree
    diff[:, 1] = torch.remainder(diff[:, 1] + (period / 2), period) - (period / 2)

    return torch.sqrt(torch.mean(diff ** 2, dim=0))

def mape(y_pred, y_true):
    """
    Compute the mean absolute percentage error (MAPE) between true and predicted values.
    Args:
        y_pred (torch.Tensor): Predicted values of shape (N, D).
        y_true (torch.Tensor): True values of shape (N, D).
    Returns:
        torch.Tensor: MAPE for each dimension (D,).
    """
    epsilon = 1e-8
    return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)), dim=0)*100

def mae_wrapped_va(y_pred, y_true, period=360.0):
    """
    Compute the mean absolute error (MAE) between true and predicted values, primarily for voltage angle (va_degree).
    Args:
        y_pred (torch.Tensor): Predicted values of shape (N,2), where second column is va_degree.
        y_true (torch.Tensor): True values of shape (N,2), where second column is va_degree.
    Returns:
        torch.Tensor: MAE for voltage angle (va_degree).
    """
    # Calculate simple difference
    diff = y_true - y_pred

    # Wrap the difference to the interval [-period/2, period/2]
    # This solves an issue where 359 is seen as far from 0.1 degree
    diff[:, 1] = torch.remainder(diff[:, 1] + (period / 2), period) - (period / 2)

    return torch.mean(torch.abs(diff), dim=0)

def test(model,
         device,
         loader_test):
    """
    Evaluate a PyTorch model on a test dataset and compute RMSE and MAPE for voltage magnitude and angle.
    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        loader_test (torch.utils.data.DataLoader): DataLoader for the test dataset.
    Returns:
        tuple: (rmse_vm, rmse_va, mape_vm, mape_va)
            - rmse_vm (float): RMSE for voltage magnitude (vm_pu).
            - rmse_va (float): RMSE for voltage angle (va_degree).
            - mape_vm (float): MAPE for voltage magnitude (vm_pu) in percentage.
            - mape_va (float): MAPE for voltage angle (va_degree) in percentage.
    """
    model.eval()
    # TODO: MAPE doesnt give that much physically meaningful info, since voltage
    #       already in p.u. and ref degree angle from slack is somewhat arbitrary.
    #       Consider changing to MAE (implemented above).
    rmse_vm = rmse_va = mape_vm = mape_va = 0

    if model.is_analytical():
        # Do not need to do this on GPU. Run sequentially on cpu for now.
        device = 'cpu'
        model = model.to(device)
        with torch.no_grad():
            for data in loader_test.dataset:
                data = data.to(device)
                pred = model(data)
                hops_to_slack = data.x.shape[-1] - 1
                pq_mask = (data.x[:, hops_to_slack] != 0)
                pred = pred[pq_mask]
                true_y = data.y[pq_mask]
                loss_rmse = rmse_wrapped_va(pred, true_y) # [vm_pu, va_degree]
                loss_mape = mape(pred, true_y) # [vm_pu, va_degree]
                rmse_vm += loss_rmse[0].item()
                rmse_va += loss_rmse[1].item()
                mape_vm += loss_mape[0].item()
                mape_va += loss_mape[1].item()
    else:
        # Disable gradient tracking during testing for efficiency and correctness
        with torch.no_grad():
            for batch_test in loader_test:
                batch_test = batch_test.to(device)
                pred = model(batch_test)
                hops_to_slack = batch_test.x.shape[-1] - 1
                pq_mask = (batch_test.x[:, hops_to_slack] != 0)
                pred = pred[pq_mask]
                true_y = batch_test.y[pq_mask]
                if model.is_complex():
                    # For complex models, convert complex voltage to [vm_pu, va_degree]
                    pred = torch.cat([pred.abs(), pred.angle()], dim=1)
                    true_y = torch.cat([true_y.abs(), true_y.angle()], dim=1)
                loss_rmse = rmse_wrapped_va(pred, true_y) # [vm_pu, va_degree]
                loss_mape = mape(pred, true_y) # [vm_pu, va_degree]
                rmse_vm += loss_rmse[0].item()*batch_test.num_graphs
                rmse_va += loss_rmse[1].item()*batch_test.num_graphs
                mape_vm += loss_mape[0].item()*batch_test.num_graphs
                mape_va += loss_mape[1].item()*batch_test.num_graphs

    rmse_vm /= len(loader_test.dataset)
    rmse_va /= len(loader_test.dataset)
    mape_vm /= len(loader_test.dataset)
    mape_va /= len(loader_test.dataset)

    return rmse_vm, rmse_va, mape_vm, mape_va
