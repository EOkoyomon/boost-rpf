import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from darts import TimeSeries

# from complexPyTorch.complexFunctions import complex_relu
# from complexPyTorch.complexLayers import ComplexLinear
from darts.models import XGBModel
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from models.lindistflow import (
    calculate_distflow_iterative,
    calculate_lindistflow_iterative,
)


class GNNLayer(MessagePassing):
    def __init__(self, in_c, out_c):
        super().__init__(aggr="add")
        # Linear applied after aggregation
        self.lin = nn.Linear(in_c, out_c)
        # Linear for the root/self contribution
        self.lin_root = nn.Linear(in_c, out_c)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * self.lin(x_j)

    def update(self, inputs, x):
        return self.lin_root(x) + inputs


class NormedGNN(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=2,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
        complex=False,
    ):
        super().__init__()

        self.residuals = residuals
        self.physics_loss = physics_loss
        self.supervised = supervised
        self.complex = complex
        self.hidden_dim = 128
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyReLU = nn.functional.relu

        # Node feature processing layers
        self.predense1 = nn.Linear(self.input_dim, self.hidden_dim * 2)
        self.prenorm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.predense2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.prenorm2 = nn.BatchNorm1d(self.hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(GNNLayer(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        # Post-processing layers
        self.postdense1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.postnorm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.readout = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def use_physics_loss(self):
        return self.physics_loss

    def is_supervised(self):
        return self.supervised

    def is_complex(self):
        return self.complex

    def is_analytical(self):
        return False

    def forward(self, data):
        # Data format:
        #   x features: [p_mw, q_mvar, hops_to_slack]
        #   edge_attr features: [r_pu, x_pu]
        #   y labels: [vm_pu, va_degree]
        #   slack_info (global): [slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]

        x = torch.nan_to_num(data.x, nan=0.0)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
        edge_index = data.edge_index

        # If 'batch' does not exist, make batch of 1
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = data.x.new_zeros(x.size(0), dtype=torch.long)

        # Handle batched data: slack_info contains slack info for each graph in the batch
        # data.batch maps each node to its corresponding graph in the batch
        # data.slack_info has shape [batch_size * 4] - need to reshape and index properly

        batch_size = data.batch.max().item() + 1  # Number of graphs in batch
        slack_info_per_graph = data.slack_info.view(
            batch_size, 4
        )  # Reshape to [batch_size, 4]

        # For each node, get the slack info from its corresponding graph
        # data.batch[i] tells us which graph node i belongs to
        node_slack_info = slack_info_per_graph[data.batch]  # Shape: [num_nodes, 4]

        # Append slack info to each node feature
        # Now x has shape: [p_mw, q_mvar, hops_to_slack, slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu] (7 features)
        x = torch.cat([x, node_slack_info], dim=1)

        orig_x = x

        # Pre-processing
        x = self.leakyReLU(self.prenorm(self.predense1(x)))
        x = self.leakyReLU(self.prenorm2(self.predense2(x)))

        # GNN Layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = self.leakyReLU(x)

        # Post-processing
        x = self.postdense1(x)
        x = self.postnorm(x)
        x = self.leakyReLU(x)

        # Readout
        x = self.readout(x)
        if self.residuals:
            # Adding the slack bus's voltage components makes the NN's task to predict the residuals,
            x = x + orig_x[:, 3:5]  # Add slack_vm_pu and slack_va_degree
        return x


class NormedGNN_Residuals(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=True,
            physics_loss=False,
            supervised=True,
            complex=False,
        )

class NormedGNN_PhysicsLoss_Supervised(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=False,
            physics_loss=True,
            supervised=True,
            complex=False,
        )


class NormedGNN_Complex(nn.Module):
    class ComplexBatchNorm1d(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.bn_real = nn.BatchNorm1d(num_features)
            self.bn_imag = nn.BatchNorm1d(num_features)

        def forward(self, x):
            return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))

    class ComplexGNNLayer(MessagePassing):
        def __init__(self, in_c, out_c):
            super().__init__(aggr="add")
            # Linear applied after aggregation
            self.lin = ComplexLinear(in_c, out_c)
            # Linear for the root/self contribution
            self.lin_root = ComplexLinear(in_c, out_c)

        def forward(self, x, edge_index, edge_weight=None):
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            return self.propagate(edge_index, x=x, edge_weight=edge_weight)

        def message(self, x_j, edge_weight):
            return edge_weight.view(-1, 1).to(x_j.dtype) * self.lin(x_j)

        def update(self, inputs, x):
            return self.lin_root(x) + inputs

    class ComplexGraphConv(MessagePassing):
        def __init__(self, in_c, out_c):
            super().__init__(aggr="add")
            # Linear applied after aggregation (like torch_geometric GraphConv)
            self.lin = ComplexLinear(in_c, out_c)
            # Linear for the root/self contribution
            self.lin_root = ComplexLinear(in_c, out_c)

        def forward(self, x, edge_index, edge_weight=None):
            # Allow x to be a single Tensor or a pair (x_src, x_dst)
            if isinstance(x, torch.Tensor):
                x = (x, x)

            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

            # apply linear after aggregation
            out = self.lin(out)

            x_r = x[1]
            if x_r is not None:
                out = out + self.lin_root(x_r)

            return out

        def message(self, x_j, edge_weight=None):
            if edge_weight is None:
                return x_j
            return edge_weight.to(x_j.dtype).view(-1, 1) * x_j

    def __init__(
        self,
        input_dim=4,
        output_dim=1,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
    ):
        super().__init__()

        self.residuals = residuals
        self.physics_loss = physics_loss
        self.supervised = supervised
        self.complex = True
        self.hidden_dim = 128
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyReLU = complex_relu

        # Node feature processing layers
        self.predense1 = ComplexLinear(self.input_dim, self.hidden_dim * 2)
        self.prenorm = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim * 2)

        self.predense2 = ComplexLinear(self.hidden_dim * 2, self.hidden_dim)
        self.prenorm2 = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(
                NormedGNN_Complex.ComplexGNNLayer(self.hidden_dim, self.hidden_dim)
            )
            self.norms.append(NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim))

        self.postdense1 = ComplexLinear(self.hidden_dim, self.hidden_dim * 2)
        self.postnorm = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim * 2)

        self.readout = ComplexLinear(self.hidden_dim * 2, self.output_dim)

    def use_physics_loss(self):
        return self.physics_loss

    def is_supervised(self):
        return self.supervised

    def is_complex(self):
        return self.complex

    def is_analytical(self):
        return False

    def forward(self, data):
        # Data format:
        #   x features: [complex_power, hops_to_slack]
        #   edge_attr features: [complex_impedance]
        #   y labels: [complex_voltage]
        #   slack_info (global): [slack_complex_voltage, slack_complex_impedance]

        x = torch.nan_to_num(data.x, nan=0.0)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
        edge_index = data.edge_index

        # If 'batch' does not exist, make batch of 1
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = data.x.new_zeros(x.size(0), dtype=torch.long)

        # Handle batched data: slack_info contains slack info for each graph in the batch
        # data.batch maps each node to its corresponding graph in the batch
        # data.slack_info has shape [batch_size * 2] - need to reshape and index properly

        batch_size = data.batch.max().item() + 1  # Number of graphs in batch
        slack_info_per_graph = data.slack_info.view(
            batch_size, 2
        )  # Reshape to [batch_size, 2]

        # For each node, get the slack info from its corresponding graph
        # data.batch[i] tells us which graph node i belongs to
        node_slack_info = slack_info_per_graph[data.batch]  # Shape: [num_nodes, 2]

        # Append slack info to each node feature
        # Now x has shape: [complex_power, hops_to_slack, slack_complex_voltage, slack_complex_impedance] (4 features)
        x = torch.cat([x, node_slack_info], dim=1)

        orig_x = x

        # Pre-processing
        x = self.leakyReLU(self.prenorm(self.predense1(x)))
        x = self.leakyReLU(self.prenorm2(self.predense2(x)))

        # GNN Layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = self.leakyReLU(x)

        # Post-processing
        x = self.postdense1(x)
        x = self.postnorm(x)
        x = self.leakyReLU(x)

        # Readout
        x = self.readout(x)
        if self.residuals:
            # Adding the slack bus's voltage components makes the NN's task to predict the residuals,
            x = x + orig_x[:, 2:3]  # Add slack_vm_pu and slack_va_degree (as one complex voltage)
        return x

class DC_PF(nn.Module):
    """ Implements the DC Power Flow as a neural network module."""
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
        return data.dc_pf

class DC_PF_Slack(DC_PF):
    """ Sets all voltage magnitudes to slack bus voltage magnitude."""
    def __init__(self):
        super().__init__()

    def forward(self, data):
        out = data.dc_pf
        out[:, 0] = torch.ones(len(out))*data.slack_info[0]
        return out

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

class XGBModelWrapper:
    """
    Wrapper for darts XGBModel for sequential voltage prediction along grid paths.
    
    This model predicts voltage at each node along a path from slack bus to leaf nodes.
    It uses:
    - Lagged target values (lags=1): Previous voltage [V_{i}, theta_{i}] from parent node
    - Past covariates: Branch parameters [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
    
    Note: V_i, theta_i are NOT in covariates - they come from target lags.
    This ensures clean separation between training (uses true lags) and testing
    (uses predicted lags via recursive prediction).
    
    The model is designed for:
    - Training: Uses true parent voltages from target series lags
    - Testing: Recursively builds target history using predictions
    """
    
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42, prediction_scheme='basic'):
        """
        Initialize the XGBModel wrapper.
        
        Args:
            lags (int): Number of lagged target values to use. Default 1 (parent voltage).
            lags_future_covariates (list[int]): List of lagged covariate steps to use. Default [0].
            output_chunk_length (int): Number of steps to predict at once. Default 1.
            random_state (int): Random seed for reproducibility.
        """
        self.lags = lags
        self.lags_future_covariates = lags_future_covariates
        # Note: Can perhaps use more past and future lags in the model.
        self.model = XGBModel(
            lags=lags,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            random_state=random_state,
            multi_models=True,  # Separate model for each output component (V, theta)
            n_estimators=200,  # Number of boosting rounds
            max_depth=7,  # Maximum tree depth
            learning_rate=0.1,  # Learning rate
        )
        assert prediction_scheme in ['basic', 'average', 'linear'], "Invalid prediction scheme"
        self.prediction_scheme = prediction_scheme
        self._is_fitted = False
    
    @property
    def __name__(self):
        return "XGBModelWrapper"
    
    def fit(self, target_series_train, covariate_series_train, target_series_val, covariate_series_val, verbose=False):
        """
        Fit the model on a list of path sequences.
        
        Args:
            target_series_list: List of darts TimeSeries with targets [V_j, theta_j]
            covariate_series_list: List of darts TimeSeries with covariates (8 features)
            verbose: Whether to print progress
        """
        # # Filter out sequences that are too short for the lag configuration
        # # Need at least lags + 1 steps to have enough history
        # min_length = max(self.lags, self.lags_future_covariates) + 1
        
        # valid_targets = []
        # valid_covariates = []
        
        # for target, cov in zip(target_series_train, covariate_series_train):
        #     if target.n_timesteps >= min_length:
        #         valid_targets.append(target)
        #         valid_covariates.append(cov)
        
        # if verbose:
        #     print(f"Training on {len(valid_targets)} sequences (filtered from {len(target_series_train)})")
        
        # self.model.fit(
        #     series=valid_targets,
        #     past_covariates=valid_covariates
        # )
        self.model.fit(series=target_series_train,
                       future_covariates=covariate_series_train,
                       val_series=target_series_val,
                       val_future_covariates=covariate_series_val,
                       verbose=verbose)
        self._is_fitted = True
        
    def get_validation_error(self):
        """
        Get the final validation error after training.
        Returns:
            final_error: Final validation RMSE after training
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting validation error.")
        # Access the underlying XGBoost model's evaluation results
        eval_results = self.model.model.evals_result()

        # {'validation_0': {'rmse': [0.5, 0.4, 0.35]}}
        final_error = eval_results['validation_0']['rmse'][-1]
        return final_error
    
    def _predict_sequence(self, n, voltage_history, path_covariates):
        """
        Predict the next voltage given history.
        
        Args:
            voltage_history: TimeSeries of past targets [V, theta] up to parent node
            path_covariates: TimeSeries of covariates up to and including current step
            
        Returns:
            numpy array of shape (2,) with [V_j, theta_j] prediction
        """
        pred = self.model.predict(
            n=n,
            series=voltage_history,
            future_covariates=path_covariates
        )
        return pred.values()

    def predict_basic(self, num_nodes, paths):
        """
        Predict voltages along all paths in the sample using basic scheme.
        """
        slack_voltage_series = paths[0]['target_series'][0:1]
        predictions = np.ones((num_nodes, 2))
        predictions[0] = slack_voltage_series[0].values().flatten() # Store slack voltage

        for path_info in paths:
            path_length = len(path_info['path'])
            if path_length <= 1:
                # Slack node, already stored
                continue
            target_node = path_info['target_node']
            covariate_series_test = path_info['covariate_series'][1:] # Exclude slack step
            # Predict all nodes from slack to target autoregressively
            all_preds = self._predict_sequence(n=path_length-1,
                                               voltage_history=slack_voltage_series,
                                               path_covariates=covariate_series_test)
            predictions[target_node] = all_preds[-1]

        return predictions
    
    def predict_average(self, num_nodes, paths):
        """
        Predict voltages along all paths in the sample using averaging scheme.
        """
        slack_voltage_series = paths[0]['target_series'][0:1]
        predictions = {i: [] for i in range(num_nodes)}
        predictions[0] = [slack_voltage_series[0].values().flatten()] # Store slack voltage

        for path_info in paths:
            path_nodes = path_info['path']
            path_length = len(path_nodes)
            if path_length <= 1:
                # Slack node, already stored
                continue
            covariate_series_test = path_info['covariate_series'][1:] # Exclude slack step
            # Predict all nodes from slack to target autoregressively
            all_preds = self._predict_sequence(n=path_length-1,
                                               voltage_history=slack_voltage_series,
                                               path_covariates=covariate_series_test)
            for i, node in enumerate(path_nodes[1:]):
                predictions[node].append(all_preds[i])

        return np.array([np.mean(predictions[i], axis=0) for i in range(num_nodes)])
    
    def predict_linear(self, num_nodes, paths):
        """
        Predict voltages along all paths in the sample using linear scheme.
        """
        # Sort paths by length (BFS order ensures parents are processed first)
        sorted_paths = sorted(paths, key=lambda p: len(p['path']))
        predictions = np.ones((num_nodes, 2))

        # Initialize slack voltage from the first path's target series
        slack_voltage = paths[0]['target_series'][0].values().flatten()
        predictions[0] = slack_voltage

        for path_info in sorted_paths:
            path = path_info['path']
            target_node = path_info['target_node']
            path_length = len(path)
            
            if path_length <= 1:
                # Slack node - skip
                continue

            parent_node = path[-2]
            
            # Build minimal TimeSeries with aligned time indices
            # Parent voltage is at time index (path_length - 2)
            # Target covariate is at time index (path_length - 1)
            parent_voltage = predictions[parent_node]  # np.array of shape (2,)
            
            # Create parent voltage series with correct time index
            parent_time_idx = path_length - 2
            parent_voltage_series = TimeSeries.from_times_and_values(
                times=pd.RangeIndex(start=parent_time_idx, stop=parent_time_idx + 1),
                values=parent_voltage.reshape(1, 2)
            )
            
            # Only need covariates for parent→target edge (last 2 steps to satisfy [0] lag)
            # Get covariates with original time indices preserved
            target_covs = path_info['covariate_series'][-2:]  # Keeps original indices
            
            # Predict single step
            pred = self._predict_sequence(n=1,
                                          voltage_history=parent_voltage_series,
                                          path_covariates=target_covs)
            predictions[target_node] = pred.flatten()

        return predictions 

    def predict(self, sample):
        """
        Predict voltages along all paths in the sample.
        
        Args:
            sample: Dictionary with keys:
                - 'grid_type': grid_type,
                - 'sample_idx': sample_idx,
                - 'num_nodes': num_nodes,
                - 'paths': Dictionary with keys:
                    - 'target_series': darts TimeSeries of targets [V_j, theta_j]
                    - 'covariate_series': darts TimeSeries of covariates (8 features)
                    - 'path': list of node indices
                    - 'target_node': int
                - 'true_voltages': true_voltages,

        Returns:
            predictions: np.array of shape (num_nodes, 2) with predicted [V_j, theta_j] for all nodes
        """
        num_nodes = sample['num_nodes']
        paths = sample['paths']

        # Note: Possible inference schemes:
        # 1. For every path, passing in the slack voltage as previous series,
        #    then using the cov features of whole path to predict the last node.
        #    N^2 scaling.
        # 2. Averaging the predictions of nodes on the path to the target node,
        #    because they will be predicted multiple times. N^2 scaling.
        # 3. Predicting in order of path lengths, then can use the prediction
        #    of the previous node to predict the next one in one step, rather
        #    using the autoregression of the predict method. N scaling.
        # 4. Using only the minimum paths so inference goes faster and we dont
        #    have to do data alterations during inference. N scaling.

        if self.prediction_scheme == 'basic':
            return self.predict_basic(num_nodes, paths)
        elif self.prediction_scheme == 'average':
            # Implement averaging scheme if needed
            return self.predict_average(num_nodes, paths)
        elif self.prediction_scheme == 'linear':
            # Implement linear scheme if needed
            return self.predict_linear(num_nodes, paths)
        else:
            raise ValueError("Invalid prediction scheme")

    def is_fitted(self):
        return self._is_fitted

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        # This should not be used for sequential prediction
        raise NotImplementedError("Use the predict() method for sequential prediction.")

class XGBModel_Basic(XGBModelWrapper):
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42):
        super().__init__(lags=lags,
                         lags_future_covariates=lags_future_covariates,
                         output_chunk_length=output_chunk_length,
                         random_state=random_state,
                         prediction_scheme='basic')

class XGBModel_Average(XGBModelWrapper):
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42):
        super().__init__(lags=lags,
                         lags_future_covariates=lags_future_covariates,
                         output_chunk_length=output_chunk_length,
                         random_state=random_state,
                         prediction_scheme='average')

class XGBModel_Linear(XGBModelWrapper):
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42):
        super().__init__(lags=lags,
                         lags_future_covariates=lags_future_covariates,
                         output_chunk_length=output_chunk_length,
                         random_state=random_state,
                         prediction_scheme='linear')