import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Diff
from darts.models import XGBModel
from sklearn.preprocessing import StandardScaler


def get_paths_from_loader(loader):
    target_series_all = []
    covariate_series_all = []

    for sample in loader:
        for path_data in sample['paths']:
            target_series_all.append(path_data['targets'])
            covariate_series_all.append(path_data['features'])
    return target_series_all, covariate_series_all

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
    
    def fit(self, loader_train, loader_val, verbose=False):
        """
        Fit the model on a list of path sequences.
        
        Args:
            loader_train: DataLoader for training data
            loader_val: DataLoader for validation data
            verbose: Whether to print progress
        """
        target_series_train, covariate_series_train = get_paths_from_loader(loader_train)
        target_series_val, covariate_series_val = get_paths_from_loader(loader_val)

        print(f"Collected {len(target_series_train)} training paths", flush=True)
        print(f"Collected {len(target_series_val)} validation paths", flush=True)

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
        slack_voltage_series = paths[0]['targets'][0:1]
        predictions = np.ones((num_nodes, 2))
        predictions[0] = slack_voltage_series[0].flatten() # Store slack voltage

        for path_info in paths:
            path_length = len(path_info['path'])
            if path_length <= 1:
                # Slack node, already stored
                continue
            target_node = path_info['target_node']
            covariate_series_test = TimeSeries.from_values(path_info['features'])[1:] # Exclude slack step
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
        slack_voltage_series = paths[0]['targets'][0:1]
        predictions = {i: [] for i in range(num_nodes)}
        predictions[0] = [slack_voltage_series[0].flatten()] # Store slack voltage

        for path_info in paths:
            path_nodes = path_info['path']
            path_length = len(path_nodes)
            if path_length <= 1:
                # Slack node, already stored
                continue
            covariate_series_test = TimeSeries.from_values(path_info['features'])[1:] # Exclude slack step
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
        slack_voltage = paths[0]['targets'][0].flatten()
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
            covariate_series = TimeSeries.from_values(path_info['features'])
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
                    - 'targets': darts TimeSeries of targets [V_j, theta_j]
                    - 'features': darts TimeSeries of covariates (8 features)
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

class XGBModel_Normalized(XGBModelWrapper):
    def __init__(self, lags=1, lags_future_covariates=[0], output_chunk_length=1, random_state=42):
        super().__init__(lags=lags,
                         lags_future_covariates=lags_future_covariates,
                         output_chunk_length=output_chunk_length,
                         random_state=random_state,
                         prediction_scheme='linear',
                         normalize=True)
