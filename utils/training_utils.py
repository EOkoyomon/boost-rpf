import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import simbench as sb
from tqdm import tqdm

import time
import os

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

def get_model_save_path(log_dir, model_id='0', sequential=False):
    """
    Generate a file path for saving model weights.
    The file is named 'model_weights_<model_id>.<pth|joblib>' and is located in the logging directory.
    Args:
        log_dir (str): Path to the logging directory.
        model_id (str, optional): Identifier for the model version. Defaults to '0'.
    Returns:
        str: Full path to the model weights file.
    """
    return os.path.join(log_dir, f'model_weights_{model_id}{".joblib" if sequential else ".pth"}')

def setup_seeds(seed=12):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def get_device():
    """
    Get the device to be used for PyTorch operations.
    
    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def get_lv_grid_codes(scenario=1):
    """
    Get lv distribution grid codes for a specific scenario.
    Args:
        scenario (int, optional): Scenario number for Simbench. Defaults to 1.
    Returns:
        list: Sorted list of distribution grid codes.
    """
    # Create the codes for the distribution grid cases of Simbench (LV)
    codes = sb.collect_all_simbench_codes(scenario=scenario)
    dist_grid_codes = list(filter(lambda x: "no_sw" in x and "-LV-" in x, codes))
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
    _, ax = plt.subplots()
    start = (len(train_loss_vec) // 5)*4 # Plot only last 20% of epochs
    ax.plot(train_loss_vec[start:], label = 'train loss')
    ax.plot(val_loss_vec[start:], label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    title = f"{model_classname}"
    ax.set_title(title)
    if log_dir:
        filename = os.path.join(log_dir, f'fig_{fig_id}.png')
        plt.savefig(filename)
        print(f'Figure saved to: {filename}')
    else:
        plt.show()

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

def train(model,
          device,
          loader_train,
          loader_val,
          epochs=100,
          learning_rate=0.001,
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
    
    loss_fn = normalized_mse_loss

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

    if model.is_analytical():
        # Does not need to be trained
        return train_loss_vec, val_loss_vec, best_val_loss, corresponding_train_loss, total_epochs, 0.0

    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        loss_train = 0
        
        for batch_train in loader_train:
            optimizer.zero_grad()
            if hasattr(model, 'is_tabular') and model.is_tabular():
                # Tabular model input
                inputs, outputs = batch_train
                x_train = inputs.to(device)
                y_train = outputs.to(device)
                predictions = model(x_train)[2:].view(-1,2) # Skip slack bus
                ground_truth = y_train[2:].view(-1,2) # Skip slack bus
                num_graphs = inputs.shape[0] # Batch size
            else:
                # Graph model input
                batch_train = batch_train.to(device)
                pred = model(batch_train)
                hops_to_slack = batch_train.x.shape[-1] - 1
                pq_mask = (batch_train.x[:, hops_to_slack] != 0)
                predictions = pred[pq_mask]
                ground_truth = batch_train.y[pq_mask]
                num_graphs = batch_train.num_graphs
            loss = loss_fn(predictions, ground_truth)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()*num_graphs
        loss_train /= len(loader_train.dataset)

        # Validate
        model.eval()
        loss_val = 0

        # Disable gradient tracking during validation for efficiency
        with torch.no_grad():
            for batch_val in loader_val:
                if hasattr(model, 'is_tabular') and model.is_tabular():
                    inputs, outputs = batch_val
                    x_val = inputs.to(device)
                    y_val = outputs.to(device)
                    predictions = model(x_val)[2:].view(-1,2) # Skip slack bus
                    ground_truth = y_val[2:].view(-1,2) # Skip slack bus
                    num_graphs = inputs.shape[0] # Batch size
                else:
                    batch_val = batch_val.to(device)
                    pred = model(batch_val)
                    hops_to_slack = batch_val.x.shape[-1] - 1
                    pq_mask = (batch_val.x[:, hops_to_slack] != 0)
                    predictions = pred[pq_mask]
                    ground_truth = batch_val.y[pq_mask]
                    num_graphs = batch_val.num_graphs
                loss = loss_fn(predictions, ground_truth)
                loss_val += loss.item()*num_graphs
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
        
        # Track model performance
        train_loss_vec[epoch] = loss_train
        val_loss_vec[epoch] = loss_val
        if log_epochs and epoch % 10 == 9:
            print('Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f} LR: {:.2e}'
                    .format(epoch + 1, loss_train, loss_val, optimizer.param_groups[0]['lr']), flush=True)

    # Total training time
    train_time = time.time() - start

    # Total num epochs (if stopped early)
    total_epochs = epochs if total_epochs == -1 else total_epochs

    if best_val_weights:
        model.load_state_dict(best_weights)

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
         loader_test,
         plot=False):
    """
    Evaluate a PyTorch model on a test dataset and compute RMSE and inference time for voltage magnitude and angle.
    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        loader_test (torch.utils.data.DataLoader): DataLoader for the test dataset.
    Returns:
        tuple: (rmse_vm, rmse_va, avg_inference_time_ms)
            - rmse_vm (float): RMSE for voltage magnitude (vm_pu).
            - rmse_va (float): RMSE for voltage angle (va_degree).
            - avg_inference_time_ms (float): Average inference time per sample in milliseconds.
    """
    model = model.to(device)
    model.eval()

    # For plotting
    largest_error = 0
    largest_error_pred = None
    largest_error_true = None
    smallest_error = np.Inf
    smallest_error_pred = None
    smallest_error_true = None

    # For metrics
    rmse_vm = rmse_va = 0
    inference_time = 0

    if model.is_analytical():
        # Do not need to do this on GPU. Run sequentially on cpu for now.
        device = 'cpu'
        model = model.to(device)
        with torch.no_grad():
            for data in loader_test.dataset:
                data = data.to(device)
                start = time.time()
                pred = model(data)
                inference_time += (time.time() - start)
                hops_to_slack = data.x.shape[-1] - 1
                pq_mask = (data.x[:, hops_to_slack] != 0)
                pred = pred[pq_mask]
                true_y = data.y[pq_mask]
                loss_rmse = rmse_wrapped_va(pred, true_y) # [vm_pu, va_degree]
                rmse_vm += loss_rmse[0].item()
                rmse_va += loss_rmse[1].item()

                if loss_rmse[0].item() > largest_error:
                    largest_error = loss_rmse[0].item()
                    largest_error_pred = pred
                    largest_error_true = true_y

                if loss_rmse[0].item() < smallest_error:
                    smallest_error = loss_rmse[0].item()
                    smallest_error_pred = pred
                    smallest_error_true = true_y
    else:
        # Disable gradient tracking during testing for efficiency and correctness
        with torch.no_grad():
            for batch_test in loader_test:
                if hasattr(model, 'is_tabular') and model.is_tabular():
                    inputs, outputs = batch_test
                    x_test = inputs.to(device)
                    y_test = outputs.to(device)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.time()

                    pred = model(x_test)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time += (time.time() - start)

                    pred = pred[2:].view(-1,2) # Skip slack bus
                    true_y = y_test[2:].view(-1,2) # Skip slack bus
                    num_graphs = inputs.shape[0] # Batch size
                else:
                    batch_test = batch_test.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.time()

                    pred = model(batch_test)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time += (time.time() - start)

                    hops_to_slack = batch_test.x.shape[-1] - 1
                    pq_mask = (batch_test.x[:, hops_to_slack] != 0)
                    pred = pred[pq_mask]
                    true_y = batch_test.y[pq_mask]
                    num_graphs = batch_test.num_graphs
                loss_rmse = rmse_wrapped_va(pred, true_y) # [vm_pu, va_degree]
                rmse_vm += loss_rmse[0].item()*num_graphs
                rmse_va += loss_rmse[1].item()*num_graphs

                # Figuring out smallest and largest is too complicated in batch mode,
                # so just do the first and second graphs in the batch for simplicity.
                if plot and largest_error == 0:
                    largest_error = 1
                    num_points = min(100, len(pred)) // 2 # Pick first 100 points for plotting. Arbitrary.
                    largest_error_pred = pred[:num_points]
                    largest_error_true = true_y[:num_points]
                    smallest_error_pred = pred[num_points:num_points*2]
                    smallest_error_true = true_y[num_points:num_points*2]

    rmse_vm /= len(loader_test.dataset)
    rmse_va /= len(loader_test.dataset)
    if model.is_analytical():
        avg_inference_time_ms = inference_time * 1000 / len(loader_test.dataset)
    else:
        avg_inference_time_ms = inference_time * 1000 / len(loader_test)

    if plot:
        largest_error_pred = largest_error_pred.cpu().numpy()
        largest_error_true = largest_error_true.cpu().numpy()
        smallest_error_pred = smallest_error_pred.cpu().numpy()
        smallest_error_true = smallest_error_true.cpu().numpy()
        # Saved the predictions without slack bus for plotting
        plot_predictions(smallest_error_pred, smallest_error_true, largest_error_pred, largest_error_true) # Skip slack

    return rmse_vm, rmse_va, avg_inference_time_ms

def train_sequential(model,
                     loader_train,
                     loader_val):
    """
    Train a sequential model with early stopping and optional best weights saving.
    Args:
        model (darts Model): The PyTorch model to be trained.
        loader_train (list[Object]): DataLoader for the training dataset.
        loader_val (list[Object]): DataLoader for the validation dataset.
    Returns:
        tuple: (train_time, validation_error)
            - train_time (float): Total training time in seconds.
            - validation_error (float): Validation error corresponding to the best model.
    """
    # Train the model
    start_time = time.time()
    model.fit(loader_train, loader_val)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds", flush=True)
    validation_error = model.get_validation_error()

    return train_time, validation_error

def rmse_sequential_wrapped_va(y_pred, y_true, period=360.0):
    """
    Compute the root mean squared error (RMSE) between true and predicted values.
    Args:
        y_pred (np.array): Predicted values of shape (N, D), where second column is va_degree.
        y_true (np.array): True values of shape (N, D), where second column is va_degree.
    Returns:
        np.array: RMSE for each dimension (D,).
    """
    # Calculate simple difference
    diff = y_true - y_pred

    # Wrap the difference to the interval [-period/2, period/2]
    # This solves an issue where 359 is seen as far from 0.1 degree
    diff[:, 1] = np.remainder(diff[:, 1] + (period / 2), period) - (period / 2)

    return np.sqrt(np.mean(diff ** 2, axis=0))

def plot_predictions(pred_smallest, true_smallest, pred_largest, true_largest):
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.plot(true_smallest[:,0], label='True vm_pu')
    plt.plot(pred_smallest[:,0], label='Predicted vm_pu')
    plt.title('Voltage Magnitude Prediction with Smallest Error')
    plt.xlabel('Time Step')
    plt.ylabel('vm_pu')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(true_largest[:,0], label='True vm_pu')
    plt.plot(pred_largest[:,0], label='Predicted vm_pu')
    plt.title('Voltage Magnitude Prediction with Largest Error')
    plt.xlabel('Time Step')
    plt.ylabel('vm_pu')
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(true_smallest[:,1], label='True va_degree')
    plt.plot(pred_smallest[:,1], label='Predicted va_degree')
    plt.title('Voltage Angle Prediction with Smallest Error')
    plt.xlabel('Time Step')
    plt.ylabel('va_degree')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(true_largest[:,1], label='True va_degree')
    plt.plot(pred_largest[:,1], label='Predicted va_degree')
    plt.title('Voltage Angle Prediction with Largest Error')
    plt.xlabel('Time Step')
    plt.ylabel('va_degree')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_sequential(model,
                    loader_test,
                    plot=False):
    """
    Evaluate a sequential model on a test dataset and compute RMSE and inference time for voltage magnitude and angle.
    Args:
        model (darts Model): The PyTorch model to be evaluated.
        loader_test (list[Object]): DataLoader for the test dataset.
    Returns:
        tuple: (rmse_vm, rmse_va, avg_inference_time_ms)
            - rmse_vm (float): RMSE for voltage magnitude (vm_pu).
            - rmse_va (float): RMSE for voltage angle (va_degree).
            - avg_inference_time_ms (float): Average inference time per sample in milliseconds.
    """
    # For plotting
    largest_error = 0
    largest_error_pred = None
    largest_error_true = None
    smallest_error = np.Inf
    smallest_error_pred = None
    smallest_error_true = None

    # Test the model
    rmse_vm = 0
    rmse_va = 0
    inference_time = 0

    for sample in tqdm(loader_test):
        start = time.time()
        predictions = model.predict(sample)
        inference_time += (time.time() - start)
        loss_rmse = rmse_sequential_wrapped_va(predictions[1:],
                                               sample['true_voltages'][1:]) # Skip slack
        if loss_rmse[0] > largest_error:
            largest_error = loss_rmse[0]
            largest_error_pred = predictions
            largest_error_true = sample['true_voltages']

        if loss_rmse[0] < smallest_error:
            smallest_error = loss_rmse[0]
            smallest_error_pred = predictions
            smallest_error_true = sample['true_voltages']

        rmse_vm += loss_rmse[0]
        rmse_va += loss_rmse[1]

    rmse_vm /= len(loader_test)
    rmse_va /= len(loader_test)
    avg_inference_time_ms = inference_time * 1000 / len(loader_test)
    # print(f"Average inference time per sample: {avg_inference_time_ms:.4f} milliseconds", flush=True)

    if plot:
        plot_predictions(smallest_error_pred[1:], smallest_error_true[1:], largest_error_pred[1:], largest_error_true[1:]) # Skip slack
        plot_error_accumulation(model=model, loader_test=loader_test)

    return rmse_vm, rmse_va, avg_inference_time_ms

def plot_error_accumulation(model, loader_test):
    """
    Plots the accumulation of errors in voltage magnitude and angle predictions as a function of hops from the slack bus.

    Args:
        model_class: The sequential model class (e.g., XGBModelWrapper)
        loader_test: List of sample dicts from get_grid_paths for testing

    Returns:
        Tuple of (errors_vm, errors_va)
    """

    sample = loader_test[0]
    hops_to_slack = {p['target_node']: len(p['path'])-1 for p in sample['paths']} # -1 because path includes the target node itself
    max_dist = max(hops_to_slack.values())
    errors_vm = {i: [] for i in range(1, max_dist+1)}
    errors_va = {i: [] for i in range(1, max_dist+1)}

    for sample in tqdm(loader_test):
        predictions = model.predict(sample)

        for node, hops in hops_to_slack.items():
            loss_rmse = rmse_sequential_wrapped_va(predictions[node:node+1],
                                               sample['true_voltages'][node:node+1])
            errors_vm[hops].append(loss_rmse[0])
            errors_va[hops].append(loss_rmse[1])

    def plot_errors(errors_vm, errors_va):
        plt.figure(figsize=(5, 5)) # Taller for better spacing when stacked vertically

        def apply_styling(ax, data, title, ylabel, is_vm=False):
            # Calculate data range for a clean buffer
            ymin, ymax = min(data), max(data)
            margin = (ymax - ymin) * 0.15 if ymax != ymin else 0.001

            ax.plot(sorted(errors_vm.keys()), data, '-o', markersize=4, linewidth=1.2, color='#1f77b4', clip_on=False)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xticks(sorted(errors_vm.keys()))

            # Grid: Horizontal lines only
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Apply Y-axis buffer
            ax.set_ylim(ymin - margin, ymax + margin)

            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            if is_vm:
                # Force scientific notation for Voltage Magnitude
                formatter.set_powerlimits((-3, -3)) # forces 10^-3
            else:
                formatter.set_powerlimits((-2, -2)) # forces 10^-3
            ax.yaxis.set_major_formatter(formatter)

            # Shrink tick label size slightly
            ax.tick_params(axis='both', which='major', labelsize=9)

        # Subplot 1: Voltage Magnitude
        ax1 = plt.subplot(2, 1, 1)
        means_vm = [np.mean(errors_vm[i]) for i in sorted(errors_vm.keys())]
        apply_styling(ax1, means_vm, 'Voltage Magnitude RMSE', 'RMSE (pu)', is_vm=True)
        ax1.set_title('Prediction Stability', fontsize=14, pad=10)

        # Subplot 2: Voltage Angle
        ax2 = plt.subplot(2, 1, 2)
        means_va = [np.mean(errors_va[i]) for i in sorted(errors_va.keys())]
        apply_styling(ax2, means_va, 'Voltage Angle RMSE', 'RMSE (degree)')
        ax2.set_xlabel('Distance from Slack', fontsize=12)

        plt.tight_layout()
        # plt.savefig('error_accumulation.pdf', format="pdf", dpi=300)
        plt.show()

    plot_errors(errors_vm, errors_va)

    return errors_vm, errors_va