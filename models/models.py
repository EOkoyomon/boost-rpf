import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from darts import TimeSeries

# from complexPyTorch.complexFunctions import complex_relu
# from complexPyTorch.complexLayers import ComplexLinear
from darts.models import XGBModel
from darts.dataprocessing.transformers import Diff
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
    
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42, prediction_scheme='basic', normalize=False, residuals=False):
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
            multi_models=True,  # Separate model for each future lag (no effect when output_chunk_length=1)
            n_estimators=200,  # Number of boosting rounds
            max_depth=7,  # Maximum tree depth
            learning_rate=0.5,  # Learning rate
        )
        assert prediction_scheme in ['basic', 'average', 'linear'], "Invalid prediction scheme"
        self.prediction_scheme = prediction_scheme
        self.normalize = normalize
        self.residuals = residuals
        self._is_fitted = False
    
    @property
    def __name__(self):
        return "XGBModelWrapper"
    
    def fit(self, target_series_train, covariate_series_train, target_series_val, covariate_series_val, verbose=False):
        """
        Fit the model on a list of path sequences.
        
        Args:
            target_series_list: List of numpy arrays with targets [V_j, theta_j]
            covariate_series_list: List of numpy arrays with covariates (8 features)
            verbose: Whether to print progress
        """
        if self.residuals:
            residual_train = []
            for target, cov in zip(target_series_train, covariate_series_train):
                # Extract the physics approximation (ensure it's the same shape)
                physics_approx = cov[:, 6:8]  # V_LDF_j, theta_LDF_j
                # Residual = True - Physics
                residual_train.append(target - physics_approx)

            residual_val = []
            for target, cov in zip(target_series_val, covariate_series_val):
                physics_approx = cov[:, 6:8]  # V_LDF_j, theta_LDF_j
                residual_val.append(target - physics_approx)

            target_series_train = residual_train
            target_series_val = residual_val

        if self.normalize:
            # Fit scalers on training data
            target_arrays = np.concatenate(target_series_train)
            covariate_arrays = np.concatenate(covariate_series_train)

            self.target_scaler = StandardScaler().fit(target_arrays)
            self.covariate_scaler = StandardScaler().fit(covariate_arrays)
            
            target_series_train = [TimeSeries.from_values(self.target_scaler.transform(ts)) for ts in target_series_train]
            covariate_series_train = [TimeSeries.from_values(self.covariate_scaler.transform(ts)) for ts in covariate_series_train]
            target_series_val = [TimeSeries.from_values(self.target_scaler.transform(ts)) for ts in target_series_val]
            covariate_series_val = [TimeSeries.from_values(self.covariate_scaler.transform(ts)) for ts in covariate_series_val]
        else:
            target_series_train = [TimeSeries.from_values(ts) for ts in target_series_train]
            covariate_series_train = [TimeSeries.from_values(ts) for ts in covariate_series_train]
            target_series_val = [TimeSeries.from_values(ts) for ts in target_series_val]
            covariate_series_val = [TimeSeries.from_values(ts) for ts in covariate_series_val]

        self.differencer = Diff(lags=1).fit(target_series_train)
        target_series_train = self.differencer.transform(target_series_train)
        target_series_val = self.differencer.transform(target_series_val)

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
        if self.normalize:
            voltage_history = TimeSeries.from_values(self.target_scaler.transform(voltage_history.values()))
            path_covariates = TimeSeries.from_values(self.covariate_scaler.transform(path_covariates.values()))

        voltage_history = self.differencer.transform(voltage_history)
        pred = self.model.predict(
            n=n,
            series=voltage_history,
            future_covariates=path_covariates
        )
        pred = self.differencer.inverse_transform(pred)
        if self.normalize:
            return self.target_scaler.inverse_transform(pred.values())
        else:
            return pred.values()

    def predict_basic(self, num_nodes, paths):
        """
        Predict voltages along all paths in the sample using basic scheme.
        """
        slack_voltage_series = paths[0]['target_series'][0:1]
        predictions = np.ones((num_nodes, 2))
        predictions[0] = slack_voltage_series[0].flatten() # Store slack voltage

        for path_info in paths:
            path_length = len(path_info['path'])
            if path_length <= 1:
                # Slack node, already stored
                continue
            target_node = path_info['target_node']
            covariate_series_test = TimeSeries.from_values(path_info['covariate_series'])[1:] # Exclude slack step
            # Predict all nodes from slack to target autoregressively
            all_preds = self._predict_sequence(n=path_length-1,
                                               voltage_history=TimeSeries.from_values(slack_voltage_series),
                                               path_covariates=covariate_series_test)
            if self.residuals:
                predictions[target_node] = all_preds[-1] + covariate_series_test.values()[-1, 6:8]
            else:
                predictions[target_node] = all_preds[-1]

        return predictions
    
    def predict_average(self, num_nodes, paths):
        """
        Predict voltages along all paths in the sample using averaging scheme.
        """
        slack_voltage_series = paths[0]['target_series'][0:1]
        predictions = {i: [] for i in range(num_nodes)}
        predictions[0] = [slack_voltage_series[0].flatten()] # Store slack voltage

        for path_info in paths:
            path_nodes = path_info['path']
            path_length = len(path_nodes)
            if path_length <= 1:
                # Slack node, already stored
                continue
            covariate_series_test = TimeSeries.from_values(path_info['covariate_series'])[1:] # Exclude slack step
            # Predict all nodes from slack to target autoregressively
            all_preds = self._predict_sequence(n=path_length-1,
                                               voltage_history=TimeSeries.from_values(slack_voltage_series),
                                               path_covariates=covariate_series_test)
            for i, node in enumerate(path_nodes[1:]):
                if self.residuals:
                    predictions[node].append(all_preds[i] + covariate_series_test.values()[-(path_length - i), 6:8])
                else:
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
        slack_voltage = paths[0]['target_series'][0].flatten()
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
            covariate_series = TimeSeries.from_values(path_info['covariate_series'])
            target_covs = covariate_series[-2:]  # Keeps original indices
            
            # Predict single step
            pred = self._predict_sequence(n=1,
                                          voltage_history=parent_voltage_series,
                                          path_covariates=target_covs)
            predictions[target_node] = pred.flatten()
            if self.residuals:
                predictions[target_node] += target_covs.values()[-1, 6:8]

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
        # self.model = XGBModel(
        #     lags=self.lags,
        #     lags_future_covariates=self.lags_future_covariates,
        #     output_chunk_length=1,
        #     random_state=42,
        #     multi_models=True,
        #     subsample=0.8,
        #     max_depth=9,
        #     n_estimators=200,
        #     learning_rate=0.02,
        #     min_child_weight=20, # If your paths are very short (length 2), a weight of 20 might be too high. This parameter controls how much "evidence" (samples) a leaf node needs. If it's too high, the model will refuse to split and will just predict a flat average
        #     colsample_bytree=0.9
        # )

class XGBModel_Normalized(XGBModelWrapper):
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42):
        super().__init__(lags=lags,
                         lags_future_covariates=lags_future_covariates,
                         output_chunk_length=output_chunk_length,
                         random_state=random_state,
                         prediction_scheme='linear',
                         normalize=True)

class NativeXGBModelWrapper:
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear', 
                 normalize=False, use_residuals=False, use_diff=True):
        self.lags = lags
        self.random_state = random_state
        self.prediction_scheme = prediction_scheme
        self.normalize = normalize
        assert not (use_residuals and use_diff), "Cannot use both residuals and differencing."
        self.use_residuals = use_residuals
        self.use_diff = use_diff

        # Native XGBRegressor with multi-output support
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.5,
            random_state=random_state,
            # 'one_output_per_tree' is similar to Darts' multi_models=True
            multi_strategy="one_output_per_tree", 
            objective="reg:squarederror"
        )
        
        self.target_scaler = StandardScaler() if normalize else None
        self.covariate_scaler = StandardScaler() if normalize else None
        self._is_fitted = False

    def _create_tabular_data(self, target_series_list, covariate_series_list):
        """
        Creates tabular dataset. 
        X = [Target_lag_1, ..., Target_lag_n, Covariate_t]
        y = [Target_t] OR [Delta_t]
        """
        X_all, y_all = [], []
        
        for target, cov in zip(target_series_list, covariate_series_list):
            # 1. Differencing (Optional)
            if self.use_diff:
                # Pad with 0s at the start to keep length same as original series
                # This ensures path length 2 (Slack -> Node 1) is preserved.
                target_to_use = np.diff(target, axis=0, prepend=target[0:1])
            elif self.use_residuals:
                target_to_use = target - cov[:, 6:8]
            else:
                target_to_use = target

            # 2. Windowing / Lagging
            # We start from index 1 because index 0 is the Slack Bus (Input/History)
            for t in range(1, len(target)):
                # The 'lag' is the absolute voltage of the parent node (t-1)
                # This is true REGARDLESS of whether we predict absolute, diff, or residuals.
                parent_val = target[t-1] 
                current_cov = cov[t]
                
                X_all.append(np.concatenate([parent_val.flatten(), current_cov.flatten()]))
                y_all.append(target_to_use[t])
                
        return np.array(X_all), np.array(y_all)

    def fit(self, target_train, cov_train, target_val, cov_val, verbose=False):
        # 1. Create tabular data
        X_train, y_train = self._create_tabular_data(target_train, cov_train)
        X_val, y_val = self._create_tabular_data(target_val, cov_val)

        # 2. Normalization
        if self.normalize:
            self.covariate_scaler.fit(X_train)
            X_train = self.covariate_scaler.transform(X_train)
            X_val = self.covariate_scaler.transform(X_val)
            self.target_scaler.fit(y_train)
            y_train = self.target_scaler.transform(y_train)
            y_val = self.target_scaler.transform(y_val)

        # 3. Fit the model
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
        self._is_fitted = True

    def get_validation_error(self):
        """
        Get the final validation error from the internal XGBoost evaluator.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting validation error.")
        
        # Access results from the native model
        # eval_set=[(X_val, y_val)] in .fit() corresponds to 'validation_0'
        eval_results = self.model.evals_result()
        
        # XGBoost returns a list of scores for each iteration (boosting round)
        # We take the last value from the first (and only) validation set
        # The default key is usually 'rmse' for regression, but we use 'rmse' specifically
        try:
            final_error = eval_results['validation_0']['rmse'][-1]
        except KeyError:
            # Fallback if the metric name differs (e.g., if using custom objectives)
            metric_name = list(eval_results['validation_0'].keys())[0]
            final_error = eval_results['validation_0'][metric_name][-1]
            
        return final_error

    def _predict_step(self, parent_val, current_covs):
        """Predicts a single step forward [V_j, theta_j]"""
        # 1. Create the tabular input        
        X = np.concatenate([parent_val.flatten(), current_covs.flatten()]).reshape(1, -1)

        # 2. If model was trained on normalized data, scale the inputs 
        if self.normalize:
            X = self.covariate_scaler.transform(X)

        # 3. Predict one step
        pred = self.model.predict(X)
        
        # 4. Inverse scale if necessary
        if self.normalize:
            pred = self.target_scaler.inverse_transform(pred.reshape(1, -1))
        
        return pred.flatten()

    def predict_linear(self, num_nodes, paths):
        """Recursive 1-step prediction along the grid topology"""
        sorted_paths = sorted(paths, key=lambda p: len(p['path']))
        predictions = np.zeros((num_nodes, 2))
        
        # Slack Bus initialization
        slack_val = paths[0]['target_series'][0]
        predictions[0] = slack_val

        for path_info in sorted_paths:
            path = path_info['path']
            if len(path) <= 1: continue
            
            target_node = path_info['target_node']
            parent_node = path[-2]
            
            # 1. Get Parent Voltage (Target Lag)
            v_parent = predictions[parent_node]
            # 2. Get Branch Covariates
            cov_target = path_info['covariate_series'][-1:] # Features of the current node/edge
            
            out = self._predict_step(v_parent.reshape(1,2), cov_target)
            
            # Final prediction
            if self.use_diff:
                # If model predicts deltas: Child = Parent + Delta
                predictions[target_node] = v_parent + out
            elif self.use_residuals:
                # If model predicts residuals: Child = Physics + Predicted_Residual
                predictions[target_node] = cov_target[0, 6:8] + out
            else:
                # If model predicts absolute: Child = Predicted_Absolute
                predictions[target_node] = out

        return predictions

    def predict(self, sample):
        if self.prediction_scheme == 'linear':
            return self.predict_linear(sample['num_nodes'], sample['paths'])
        # Only using the linear method going forward for NativeXGBModelWrapper.
        raise NotImplementedError(f"Scheme {self.prediction_scheme} not implemented.")
    
class XGB_Absolute(NativeXGBModelWrapper):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear', 
                 normalize=False):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=False)
        
class XGB_Absolute_Normalized(XGB_Absolute):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear'):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)

class XGB_Parent(NativeXGBModelWrapper):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear', 
                 normalize=False):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=True)

class XGB_Parent_Normalized(XGB_Parent):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear'):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)
        
class XGB_LDF(NativeXGBModelWrapper):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear', 
                 normalize=False):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=True,
                         use_diff=False)
        
class XGB_LDF_Normalized(XGB_LDF):
    def __init__(self, lags=1, random_state=42, prediction_scheme='linear'):
        super().__init__(lags=lags,
                         random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)

# class PathTransformerModel(nn.Module):
#     """
#     Transformer model for sequence-to-sequence voltage prediction along grid paths.
    
#     Treats the entire path as a sequence (like a sentence):
#     - Input: Covariates for each node in the path [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
#     - Output: Predicted voltages [V_j, theta_j] for each node in the path
#     - The slack voltage is provided as context (prepended or used as encoder input)
    
#     Shorter paths are padded to the maximum path length in the batch.
#     """
    
#     def __init__(self, input_dim=8, output_dim=2, hidden_dim=64, num_heads=4, 
#                  num_encoder_layers=3, num_decoder_layers=3, dropout=0.1, max_seq_len=100):
#         super().__init__()
        
#         self.input_dim = input_dim  # Covariate features
#         self.output_dim = output_dim  # [V, theta]
#         self.hidden_dim = hidden_dim
#         self.max_seq_len = max_seq_len
        
#         # Input embedding for covariates
#         self.input_embed = nn.Linear(input_dim, hidden_dim)
        
#         # Embedding for target (voltage) - used in decoder
#         self.target_embed = nn.Linear(output_dim, hidden_dim)
        
#         # Positional encoding
#         self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
#         nn.init.normal_(self.pos_encoding, std=0.02)
        
#         # Transformer
#         self.transformer = nn.Transformer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Output projection
#         self.output_proj = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, covariates, slack_voltage, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
#         """
#         Forward pass.
        
#         Args:
#             covariates: [batch, seq_len, input_dim] - path covariates
#             slack_voltage: [batch, 1, output_dim] - slack bus voltage as start token
#             src_mask: Optional source mask
#             tgt_mask: Optional target mask (causal)
#             src_padding_mask: [batch, seq_len] - True for padded positions
#             tgt_padding_mask: [batch, seq_len] - True for padded positions
            
#         Returns:
#             predictions: [batch, seq_len, output_dim] - predicted voltages for each position
#         """
#         batch_size, seq_len, _ = covariates.shape
        
#         # Embed covariates and add positional encoding
#         src = self.input_embed(covariates) + self.pos_encoding[:, :seq_len, :]
        
#         # Create target sequence: start with slack voltage, then zeros for positions to predict
#         # We'll use teacher forcing during training (shift targets)
#         tgt_input = torch.zeros(batch_size, seq_len, self.output_dim, device=covariates.device)
#         tgt_input[:, 0, :] = slack_voltage.squeeze(1)  # First position is slack voltage
        
#         # Embed target and add positional encoding
#         tgt = self.target_embed(tgt_input) + self.pos_encoding[:, :seq_len, :]
        
#         # Generate causal mask for decoder (can't look ahead)
#         if tgt_mask is None:
#             tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=covariates.device)
        
#         # Transformer forward
#         output = self.transformer(
#             src, tgt,
#             src_mask=src_mask,
#             tgt_mask=tgt_mask,
#             src_key_padding_mask=src_padding_mask,
#             tgt_key_padding_mask=tgt_padding_mask
#         )
        
#         # Project to output dimension
#         predictions = self.output_proj(output)
        
#         return predictions


# class PathTransformerWrapper:
#     """
#     Wrapper for PathTransformerModel for sequential voltage prediction along grid paths.
    
#     This model processes entire paths as sequences (like sentences in NLP):
#     - Input: Full path covariates [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
#     - Output: All voltages along the path [V_j, theta_j] predicted in one forward pass
#     - Padding: Shorter paths are padded to batch max length
    
#     The slack voltage is used as the "start token" - the only known voltage.
#     """
    
#     def __init__(self, hidden_dim=64, num_heads=4, num_encoder_layers=3, num_decoder_layers=3,
#                  dropout=0.1, max_seq_len=100, n_epochs=50, batch_size=64, lr=1e-3, random_state=42):
#         """
#         Initialize the PathTransformer wrapper.
#         """
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.dropout = dropout
#         self.max_seq_len = max_seq_len
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.random_state = random_state
        
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)
        
#         self.model = PathTransformerModel(
#             input_dim=8,  # covariates
#             output_dim=2,  # [V, theta]
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dropout=dropout,
#             max_seq_len=max_seq_len
#         )
        
#         self._is_fitted = False
#         self.device = torch.device('cpu')  # Use CPU to avoid MPS issues
#         self.model.to(self.device)
        
#     @property
#     def __name__(self):
#         return "PathTransformerWrapper"
    
#     def _prepare_batch(self, target_series_list, covariate_series_list):
#         """
#         Prepare a padded batch from lists of TimeSeries.
        
#         Returns:
#             covariates: [batch, max_len, 8] padded covariate tensor
#             targets: [batch, max_len, 2] padded target tensor  
#             slack_voltages: [batch, 1, 2] slack voltage for each path
#             padding_mask: [batch, max_len] True for padded positions
#             lengths: [batch] original lengths
#         """
#         batch_size = len(target_series_list)
        
#         # Get lengths and find max
#         lengths = [len(ts) for ts in target_series_list]
#         max_len = max(lengths)
        
#         # Initialize padded tensors
#         covariates = torch.zeros(batch_size, max_len, 8, dtype=torch.float32)
#         targets = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
#         slack_voltages = torch.zeros(batch_size, 1, 2, dtype=torch.float32)
#         padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = masked
        
#         for i, (target_ts, cov_ts) in enumerate(zip(target_series_list, covariate_series_list)):
#             seq_len = lengths[i]
            
#             # Extract values
#             target_vals = target_ts.values().astype(np.float32)
#             cov_vals = cov_ts.values().astype(np.float32)
            
#             # Fill tensors
#             covariates[i, :seq_len, :] = torch.from_numpy(cov_vals)
#             targets[i, :seq_len, :] = torch.from_numpy(target_vals)
#             slack_voltages[i, 0, :] = torch.from_numpy(target_vals[0])  # First position is slack
#             padding_mask[i, :seq_len] = False  # Not masked for real positions
            
#         return covariates, targets, slack_voltages, padding_mask, lengths

#     def get_validation_error(self):
#         """Get the final validation error after training."""
#         if not self._is_fitted:
#             raise RuntimeError("Model must be fitted before getting validation error.")
#         return self._val_error
    
#     def fit(self, target_series_train, covariate_series_train, target_series_val, covariate_series_val, verbose=True):
#         """
#         Fit the model on a list of path sequences.
#         """
#         self.model.train()
#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
#         criterion = nn.MSELoss(reduction='none')  # Per-element loss for masking
        
#         # Filter sequences with length >= 2 (need at least slack + one node)
#         train_targets = []
#         train_covs = []
#         for t, c in zip(target_series_train, covariate_series_train):
#             if len(t) >= 2:
#                 train_targets.append(t)
#                 train_covs.append(c)
        
#         val_targets = []
#         val_covs = []
#         for t, c in zip(target_series_val, covariate_series_val):
#             if len(t) >= 2:
#                 val_targets.append(t)
#                 val_covs.append(c)
        
#         if verbose:
#             print(f"Training on {len(train_targets)} sequences, validating on {len(val_targets)}")
        
#         n_batches = (len(train_targets) + self.batch_size - 1) // self.batch_size
        
#         for epoch in range(self.n_epochs):
#             print(f"Epoch {epoch+1}/{self.n_epochs}")
#             # Shuffle training data
#             indices = np.random.permutation(len(train_targets))
#             train_targets_shuffled = [train_targets[i] for i in indices]
#             train_covs_shuffled = [train_covs[i] for i in indices]
            
#             epoch_loss = 0.0
#             for batch_idx in tqdm(range(n_batches)):
#                 start_idx = batch_idx * self.batch_size
#                 end_idx = min(start_idx + self.batch_size, len(train_targets))
                
#                 batch_targets = train_targets_shuffled[start_idx:end_idx]
#                 batch_covs = train_covs_shuffled[start_idx:end_idx]
                
#                 covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                     batch_targets, batch_covs
#                 )
                
#                 covariates = covariates.to(self.device)
#                 targets = targets.to(self.device)
#                 slack_voltages = slack_voltages.to(self.device)
#                 padding_mask = padding_mask.to(self.device)
                
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 predictions = self.model(
#                     covariates, slack_voltages,
#                     src_padding_mask=padding_mask,
#                     tgt_padding_mask=padding_mask
#                 )
                
#                 # Compute masked loss (ignore padding and slack position)
#                 loss_mask = ~padding_mask  # [batch, seq_len]
#                 loss_mask[:, 0] = False  # Don't compute loss on slack voltage (it's given)
                
#                 loss = criterion(predictions, targets)  # [batch, seq_len, 2]
#                 loss = loss.mean(dim=-1)  # [batch, seq_len]
#                 loss = (loss * loss_mask).sum() / loss_mask.sum()
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
            
#             epoch_loss /= n_batches
            
#             if verbose and (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")
        
#         # Compute validation error
#         self.model.eval()
#         with torch.no_grad():
#             covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                 val_targets, val_covs
#             )
#             covariates = covariates.to(self.device)
#             targets = targets.to(self.device)
#             slack_voltages = slack_voltages.to(self.device)
#             padding_mask = padding_mask.to(self.device)
            
#             predictions = self.model(covariates, slack_voltages, src_padding_mask=padding_mask, tgt_padding_mask=padding_mask)
            
#             loss_mask = ~padding_mask
#             loss_mask[:, 0] = False
            
#             val_loss = criterion(predictions, targets).mean(dim=-1)
#             val_loss = (val_loss * loss_mask).sum() / loss_mask.sum()
#             self._val_error = val_loss.item()
        
#         if verbose:
#             print(f"Validation MSE: {self._val_error:.6f}")
        
#         self._is_fitted = True
    
#     def predict(self, sample):
#         """
#         Predict voltages for all nodes in a sample.
        
#         For each path, predicts the entire sequence in one forward pass,
#         then aggregates predictions for nodes that appear in multiple paths.
#         """
#         self.model.eval()
#         num_nodes = sample['num_nodes']
#         paths = sample['paths']
        
#         predictions = {i: [] for i in range(num_nodes)}
        
#         # Get slack voltage from first path
#         slack_voltage = paths[0]['target_series'][0].values().flatten()
#         predictions[0] = [slack_voltage]
        
#         with torch.no_grad():
#             for path_info in paths:
#                 path = path_info['path']
#                 if len(path) <= 1:
#                     continue
                
#                 # Prepare single sequence
#                 target_ts = path_info['target_series']
#                 cov_ts = path_info['covariate_series']
                
#                 covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                     [target_ts], [cov_ts]
#                 )
                
#                 covariates = covariates.to(self.device)
#                 slack_voltages = slack_voltages.to(self.device)
#                 padding_mask = padding_mask.to(self.device)
                
#                 # Predict
#                 preds = self.model(covariates, slack_voltages, src_padding_mask=padding_mask, tgt_padding_mask=padding_mask)
#                 preds = preds[0].cpu().numpy()  # [seq_len, 2]
                
#                 # Store predictions for each node in path (skip slack at index 0)
#                 for i, node in enumerate(path[1:], start=1):
#                     predictions[node].append(preds[i])
        
#         # Average predictions for nodes appearing in multiple paths
#         result = np.zeros((num_nodes, 2))
#         for node in range(num_nodes):
#             if predictions[node]:
#                 result[node] = np.mean(predictions[node], axis=0)
        
#         return result

#     def is_fitted(self):
#         return self._is_fitted

#     def use_physics_loss(self):
#         return False

#     def is_supervised(self):
#         return True

#     def is_complex(self):
#         return False

#     def is_analytical(self):
#         return False

#     def forward(self, data):
#         raise NotImplementedError("Use the predict() method for sequential prediction.")


# class TransformerModel_Linear(PathTransformerWrapper):
#     """Path Transformer model for voltage prediction."""
#     def __init__(self, random_state=42):
#         super().__init__(
#             hidden_dim=64,
#             num_heads=4,
#             num_encoder_layers=2,
#             num_decoder_layers=2,
#             dropout=0.1,
#             max_seq_len=150,
#             n_epochs=2,
#             batch_size=2048,
#             lr=1e-3,
#             random_state=random_state
#         )