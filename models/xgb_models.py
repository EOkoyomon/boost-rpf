import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def get_paths_from_loader(loader):
    target_series_all = []
    covariate_series_all = []

    for sample in loader:
        for path_data in sample['paths']:
            target_series_all.append(path_data['targets'])
            covariate_series_all.append(path_data['features'])
    return target_series_all, covariate_series_all

class BiasCorrector:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,          # Keep it shallow to avoid overfitting noise
            learning_rate=0.1,
            multi_strategy="multi_output_tree", 
            objective='reg:squarederror'
        )

    def split_data_by_sample(self, data_loader, X, y, predictions):
        X_samples = []
        y_samples = []
        predictions_samples = []
        start = end = 0
        for sample in data_loader:
            start = end
            for path_data in sample['paths']:
                # Every path has one target node
                path_length = len(path_data['targets']) - 1  # Exclude slack step
                end += path_length

            X_samples.append(X[start:end])
            y_samples.append(y[start:end])
            predictions_samples.append(predictions[start:end])  # Exclude slack node prediction

            # x_sample = [None]*sample['num_nodes']
            # target_sample = [None]*sample['num_nodes']
            # for path_data in sample['paths']:
            #     # Every path has one target node
            #     path_length = len(path_data['targets']) - 1  # Exclude slack step
            #     end += path_length
            #     x_sample[path_data['target_node']] = X[end-1]  # Append target node's features
            #     target_sample[path_data['target_node']] = y[end-1] # Append target node's true voltage

            # X_samples.append(np.array(x_sample[1:])) # Exclude slack node
            # y_samples.append(np.array(target_sample[1:])) # Exclude slack node
            # predictions.append(predictor.predict_linear(sample['num_nodes'], sample['paths'], use_corrector_if_available=False)[1:])  # Exclude slack node prediction

        return X_samples, y_samples, predictions_samples

    def _create_tabular_features(self, X, predictions, Y=None):
        """
        Creates tabular dataset. 
        X = [X_stats, prediction_stats]
        y = [mean(y_vm - predictions_vm), mean(y_va - predictions_va)]

        Args:
            X: List of np.arrays of shape (num_nodes, d) - input features
            predictions: List of np.arrays of shape (num_nodes, 2) - model predictions
            Y: (Optional) List of np.arrays of shape (num_nodes, 2) - true voltages
        """
        X_all, Y_all = [], []
        if Y is None:
            Y = [np.zeros_like(pred) for pred in predictions]  # Dummy zero targets for prediction

        for x, y, pred in zip(X, Y, predictions):
            # 1. Meta-data
            length = len(pred) # Number of non-slack nodes to predict

            # 2. Input Stats (d Dimensions)
            # We calculate Mean and Std for all d dimensions to capture the 'state' of the grid
            x_mean = np.mean(x, axis=0)  # Shape (d,)
            x_std = np.std(x, axis=0)    # Shape (d,)
            x_min = np.min(x, axis=0)    # Shape (d,)
            x_max = np.max(x, axis=0)    # Shape (d,)

            # 3. Prediction Stats (2 Dimensions)
            pred_mean = np.mean(pred, axis=0)       # Shape (2,)
            pred_mean_std = np.std(pred, axis=0)    # Shape (2,)
            pred_min = np.min(pred, axis=0)         # Shape (2,)
            pred_max = np.max(pred, axis=0)         # Shape (2,)

            # 4. Concatenate everything into one long feature vector
            row = np.concatenate([
                [length],
                x_mean, x_std, x_min, x_max,
                pred_mean, pred_mean_std, pred_min, pred_max
            ])
            X_all.append(row)
            Y_all.append(np.mean(y - pred, axis=0))  # Mean error (bias) over the path

        return np.array(X_all), np.array(Y_all)

    def fit(self, loader_train, X_train, y_train, pred_train, loader_val, X_val, y_val, pred_val, verbose=True):
        """
        Train the corrector to predict the MEAN ERROR (Bias)
        """
        print(f"Training Bias Corrector...", flush=True, end=' ')
        # For every sample, get the subarray for the X, y, and get the node predictions.
        X_samples_train, y_samples_train, pred_samples_train = self.split_data_by_sample(loader_train, X_train, y_train, pred_train)
        X_samples_val, y_samples_val, pred_samples_val = self.split_data_by_sample(loader_val, X_val, y_val, pred_val) 

        # Using the X, y, and predictions, create the feature matrix and target vector.
        X_bias_train, y_bias_train = self._create_tabular_features(X_samples_train, pred_samples_train, y_samples_train)
        X_bias_val, y_bias_val = self._create_tabular_features(X_samples_val, pred_samples_val, y_samples_val)

        # 3. Train XGBoost
        self.model.fit(X_bias_train, y_bias_train, eval_set=[(X_bias_val, y_bias_val)], verbose=verbose)
        print("complete.", flush=True)
        if verbose:
            print(f"Final Validation RMSE: {self.model.evals_result()['validation_0']['rmse'][-1]}", flush=True)

    def predict(self, X, model1_predictions):
        """
        Returns the scalar offset to add to all voltage predictions.
        """
        X_features, _ = self._create_tabular_features([X], [model1_predictions])  # No y provided
        predicted_bias = self.model.predict(X_features) # Output shape (N_samples, 2)
        return predicted_bias


class NativeXGBModelWrapper:
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False, use_residuals=False, use_diff=True, use_corrector=False):
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
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=1.0,
            # 'one_output_per_tree' is similar to Darts' multi_models=True
            multi_strategy="multi_output_tree",
            # multi_strategy="one_output_per_tree",
            objective="reg:squarederror"
        )
        # self.model = xgb.XGBRFRegressor(
        #     n_estimators=200,
        #     max_depth=7,
        #     # Note: learning_rate is usually 1.0 for RF;
        #     # setting it to 0.5 may result in underfitting.
        #     learning_rate=1.0,
        #     random_state=random_state,
        #     # Random Forests require subsampling and colsample to work effectively
        #     subsample=0.8,
        #     colsample_bynode=0.8,
        #     multi_strategy="one_output_per_tree",
        #     objective="reg:squarederror"
        # )
        
        self.target_scaler = StandardScaler() if normalize else None
        self.covariate_scaler = StandardScaler() if normalize else None
        self.corrector = BiasCorrector() if use_corrector else None
        self._is_fitted = False

    def _create_tabular_data(self, target_series_list, covariate_series_list):
        """
        Creates tabular dataset.
        X = [Target_lag_1, ..., Target_lag_n, Covariate_t]
        y = [Target_t] OR [Delta_t]

        Args:
            target_series_list: List of np.arrays of shape (T, 2) with target voltages
            covariate_series_list: List of np.arrays of shape (T, 8) with covariates
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
                parent_cov = cov[t-1]
                current_cov = cov[t]
                # current_cov[6:8] = 0.0  # Zero out physics approx
                
                X_all.append(np.concatenate([parent_val.flatten(), parent_cov.flatten(), current_cov.flatten()]))
                y_all.append(target_to_use[t])
                
        return np.array(X_all), np.array(y_all)

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

        # 1. Create tabular data
        X_train, y_train = self._create_tabular_data(target_series_train, covariate_series_train)
        X_val, y_val = self._create_tabular_data(target_series_val, covariate_series_val)

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

        # 4. Train the bias corrector
        if self.corrector is not None:
            pred_train = self.model.predict(X_train)
            pred_val = self.model.predict(X_val)
            self.corrector.fit(loader_train, X_train, y_train, pred_train, loader_val,
                               X_val, y_val, pred_val, verbose=verbose)

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

    def _predict_step(self, X):
        """Predicts a single step forward [V_j, theta_j]"""
        # 1. If model was trained on normalized data, scale the inputs
        if self.normalize:
            X = self.covariate_scaler.transform(X)

        # 2. Predict one step
        pred = self.model.predict(X)
        
        # 3. Inverse scale if necessary
        if self.normalize:
            pred = self.target_scaler.inverse_transform(pred.reshape(1, -1))
        
        return pred.flatten()

    def predict_linear(self, num_nodes, paths):
        """Recursive 1-step prediction along the grid topology"""
        sorted_paths = sorted(paths, key=lambda p: len(p['path']))
        predictions = np.zeros((num_nodes, 2))
        
        # Slack Bus initialization
        slack_val = paths[0]['targets'][0]
        predictions[0] = slack_val
        X_all = [None]*num_nodes

        for path_info in sorted_paths:
            path = path_info['path']
            if len(path) <= 1: continue
            
            target_node = path_info['target_node']
            parent_node = path[-2]
            
            # 1. Get Parent Voltage (Target Lag)
            v_parent = predictions[parent_node]
            # 2. Get Branch Covariates
            cov_parent = path_info['features'][-2]  # Covariates of the parent node/edge
            cov_target = path_info['features'][-1] # Features of the current node/edge
            
            X = np.concatenate([v_parent.flatten(), cov_parent.flatten(), cov_target.flatten()])
            out = self._predict_step(X.reshape(1, -1))
            X_all[target_node] = X

            # 5. Apply bias correction
            if self.corrector is not None:
                bias = self.corrector.predict(X.reshape(1, -1), out.reshape(1, -1))  # Exclude slack node
                # print(f"Applying Bias Correction: {bias}", flush=True)
                predictions[1:] += bias
            
            # Final prediction
            if self.use_diff:
                # If model predicts deltas: Child = Parent + Delta
                predictions[target_node] = v_parent + out
            elif self.use_residuals:
                # If model predicts residuals: Child = Physics + Predicted_Residual
                predictions[target_node] = cov_target[6:8] + out
            else:
                # If model predicts absolute: Child = Predicted_Absolute
                predictions[target_node] = out

        # 5. Apply bias correction
        # if self.corrector is not None:
        #     bias = self.corrector.predict(np.array(X_all[1:]), predictions[1:])  # Exclude slack node
        #     print(f"Applying Bias Correction: {bias}", flush=True)
        #     predictions[1:] += bias

        return predictions

    def predict(self, sample):
        if self.prediction_scheme == 'linear':
            return self.predict_linear(sample['num_nodes'], sample['paths'])
        # Only using the linear method going forward for NativeXGBModelWrapper.
        raise NotImplementedError(f"Scheme {self.prediction_scheme} not implemented.")
    
class XGB_Absolute(NativeXGBModelWrapper):
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=False)
        
class XGB_Absolute_Normalized(XGB_Absolute):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)

class XGB_Parent(NativeXGBModelWrapper):
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False, use_corrector=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=True,
                         use_corrector=use_corrector)

class XGB_Parent_Normalized(XGB_Parent):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)
        
class XGB_Parent_Corrected(XGB_Parent):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         use_corrector=True)

class XGB_LDF(NativeXGBModelWrapper):
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False, use_corrector=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=True,
                         use_diff=False,
                         use_corrector=use_corrector)
        
class XGB_LDF_Normalized(XGB_LDF):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)

class XGB_LDF_Corrected(XGB_LDF):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         use_corrector=True)
