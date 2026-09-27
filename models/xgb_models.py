import ctypes

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from utils.data_utils import compute_sample_weights

# ---------------------------------------------------------------------------
# Fast inference path
#
# Recursive prediction down a feeder is dominated by per-call overhead in the
# XGBoost Python layer (~120us/call for the sklearn `predict` wrapper), not by
# tree traversal. Two things remove almost all of it:
#
#   1. Depth batching. V_j = f(V_parent(j), ...) is sequential only in the hop
#      distance from the slack bus, not in the node count: every node at the
#      same depth has its parent already resolved, so a whole depth level goes
#      out in one call. That is D calls instead of N (e.g. 19 instead of 135
#      on 1-MV-urban), and it is exact, not an approximation.
#   2. Calling XGBoosterPredictFromDense directly, which skips DMatrix
#      construction, feature validation and the per-call config round trip.
#
# The C API entry points below are private to xgboost, so the import is guarded
# and falls back to `Booster.inplace_predict` (still depth batched) if the
# installed version moves them.
# ---------------------------------------------------------------------------
try:
    from xgboost.core import _LIB, _check_call, c_bst_ulong, make_jcargs

    try:
        from xgboost._data_utils import array_interface
    except ImportError:  # xgboost < 3.0
        from xgboost.data import _array_interface as array_interface

    # Prediction options are fixed for our use, so build the JSON config once.
    _PREDICT_ARGS = make_jcargs(
        type=0,              # value, not margin
        training=False,
        iteration_begin=0,
        iteration_end=0,
        missing=np.nan,
        strict_shape=False,
        cache_id=0,
    )
    _CAPI_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the installed xgboost
    _CAPI_AVAILABLE = False
    _PREDICT_ARGS = None


def _predict_block_capi(handle, iface, num_rows):
    """Predict one contiguous block via the XGBoost C API.

    `iface` is the cached __array_interface__ JSON of the block. The returned
    array is a view into XGBoost-owned memory that stays valid only until the
    next prediction call on this booster, so callers must consume it
    immediately (all call sites below feed it straight into an assignment).
    """
    preds = ctypes.POINTER(ctypes.c_float)()
    shape = ctypes.POINTER(c_bst_ulong)()
    dims = c_bst_ulong()
    _check_call(
        _LIB.XGBoosterPredictFromDense(
            handle,
            iface,
            _PREDICT_ARGS,
            ctypes.c_void_p(),   # no proxy DMatrix (no base_margin)
            ctypes.byref(shape),
            ctypes.byref(dims),
            ctypes.byref(preds),
        )
    )
    total = 1
    for i in range(dims.value):
        total *= int(shape[i])
    return np.ctypeslib.as_array(preds, shape=(total,)).reshape(num_rows, total // num_rows)


class _PathPlan:
    """Depth-batched layout of one grid's buses.

    Rows are ordered by depth so that each depth level is a contiguous slice of
    a single (num_rows, 18) float32 buffer. The 16 covariate columns do not
    depend on any prediction and are filled once here; only the 2 parent-voltage
    columns are rewritten as the sweep descends.
    """

    __slots__ = ('num_nodes', 'X', 'parent', 'target', 'res_base', 'levels', 'ifaces')

    def __init__(self, sample, normalize_mean=None, normalize_scale=None):
        features = sample['features']
        bus_depth = sample['depth']
        num_nodes = sample['num_nodes']

        # Buses 1..N-1 sorted by depth; ties keep ascending bus order.
        target = np.argsort(bus_depth[1:], kind='stable') + 1
        n = len(target)
        parent = np.asarray(sample['parent'], dtype=np.intp)[target]

        self.num_nodes = num_nodes
        # float64 throughout, so the bytes handed to XGBoost are the same ones
        # the per-node sweep builds. XGBoost casts to float32 internally, and a
        # float64 block benchmarks identically, so this costs nothing.
        self.X = np.zeros((n, 18))
        self.parent = parent
        self.target = target.astype(np.intp)
        self.res_base = features[target][:, 6:8].copy()

        covariates = self.X[:, 2:18]  # filled in place
        covariates[:, 0:8] = features[parent]
        covariates[:, 8:16] = features[target]

        if normalize_mean is not None:
            # StandardScaler.transform is elementwise, so the static columns can
            # be scaled once here instead of on every step.
            covariates -= normalize_mean[2:18]
            covariates /= normalize_scale[2:18]

        if n == 0:
            self.levels = []
        else:
            depths = np.asarray(bus_depth, dtype=np.intp)[target]
            bounds = np.concatenate([[0], np.flatnonzero(np.diff(depths)) + 1, [n]])
            self.levels = [(int(bounds[k]), int(bounds[k + 1])) for k in range(len(bounds) - 1)]

        # Views into X are contiguous and their pointers are stable for the
        # lifetime of this plan, so the array interface is built once per level.
        self.ifaces = (
            [array_interface(self.X[a:b]) for a, b in self.levels] if _CAPI_AVAILABLE else None
        )


def path_row_order(parent, depth):
    """Row indices that reproduce the slack-to-every-bus path data.

    The old format emitted one row per step of every slack-to-bus path.
    Every such row is a copy of a per-bus row (bus j maps to row j-1 of
    the per-bus matrix), so replaying that order reproduces the old training
    matrix exactly (same rows, same multiplicities, same sequence).
    The order matters because `subsample` draws rows in data order, so
    weighting or regrouping the duplicates changes the trees that get built.

    Returns:
        np.array: (sum of bus depths,) indices into the per-bus rows.
    """
    num_nodes = len(parent)
    chains = [None] * num_nodes
    chains[0] = []  # the slack contributes no row
    # Shallowest first, so a bus's parent chain is always built before its own.
    for node in np.argsort(depth, kind='stable'):
        if node != 0:
            chains[node] = chains[parent[node]] + [node - 1]

    order = []
    for target_node in range(1, num_nodes):
        order.extend(chains[target_node])
    return np.asarray(order, dtype=np.int64)

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
                 normalize=False, use_residuals=False, use_diff=True, use_corrector=False,
                 use_fast_predict=True, predict_nthread=1,
                 weight_scheme='subtree', expand_weights=False):
        self.random_state = random_state
        self.prediction_scheme = prediction_scheme
        self.normalize = normalize
        # Depth-batched inference.
        self.use_fast_predict = use_fast_predict
        # Tiny per-level batches do not amortize OpenMP fan-out: single-threaded
        # prediction is ~2x faster than the default here.
        self.predict_nthread = predict_nthread
        self._fast_ctx = None
        assert not (use_residuals and use_diff), "Cannot use both residuals and differencing."
        self.use_residuals = use_residuals
        self.use_diff = use_diff
        # How much each branch transition counts during fitting.
        self.weight_scheme = weight_scheme
        # Legacy mode: duplicate the rows instead of weighting them.
        assert not (use_corrector and expand_weights), \
            "The bias corrector indexes one row per bus, so it cannot be combined with expand_weights."
        self.expand_weights = expand_weights

        # Native XGBRegressor with multi-output support
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.5,
            random_state=random_state,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=1.0,
            multi_strategy="multi_output_tree",
            objective="reg:squarederror"
        )
        
        self.target_scaler = StandardScaler() if normalize else None
        self.covariate_scaler = StandardScaler() if normalize else None
        self.corrector = BiasCorrector() if use_corrector else None
        self._is_fitted = False

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_fast_ctx', None)  # holds a ctypes handle
        return state

    def __setstate__(self, state):
        # Defaults for models pickled before the fast predict path was added.
        state.setdefault('use_fast_predict', True)
        state.setdefault('predict_nthread', 1)
        state.setdefault('weight_scheme', 'subtree')
        state.setdefault('expand_weights', False)
        self.__dict__.update(state)
        self._fast_ctx = None

    def _create_tabular_data(self, samples):
        """
        Creates the tabular dataset: one row per bus (excluding the slack).

        X = [V_parent, theta_parent, Covariates_parent, Covariates_j]
        y = [V_j, theta_j] OR [Delta_j] OR [Residual_j]
        w = weight of bus j under self.weight_scheme

        Args:
            samples: List of sample dicts from get_grid_paths / load_precomputed_paths
        """
        X_all, y_all, w_all = [], [], []

        for sample in samples:
            features = sample['features']
            targets = sample['targets']
            parent = sample['parent']

            # Buses 1..N-1 are the prediction targets; parent[1:] indexes their parents.
            p = parent[1:]

            # The 'lag' is the absolute voltage of the parent bus. This is true REGARDLESS
            # of whether we predict absolute, diff, or residuals.
            X_all.append(np.hstack([targets[p], features[p], features[1:]]))

            if self.use_diff:
                y_all.append(targets[1:] - targets[p])
            elif self.use_residuals:
                y_all.append(targets[1:] - features[1:, 6:8]) # Residual from physics baseline
            else:
                y_all.append(targets[1:])

            if self.expand_weights:
                # Legacy fitting: materialize the duplicated rows in the original path
                # order instead of weighting the unique rows.
                idx = path_row_order(parent, sample['depth'])
                X_all[-1] = X_all[-1][idx]
                y_all[-1] = y_all[-1][idx]
                w_all.append(np.ones(len(idx)))
            else:
                w_all.append(compute_sample_weights(parent, sample['depth'], self.weight_scheme))

        return np.vstack(X_all), np.vstack(y_all), np.concatenate(w_all)

    def fit(self, loader_train, loader_val, verbose=False):
        """
        Fit the model on a list of per-bus network samples.

        Args:
            loader_train: DataLoader for training data
            loader_val: DataLoader for validation data
            verbose: Whether to print progress
        """
        # 1. Create tabular data (one row per bus, weighted by self.weight_scheme)
        X_train, y_train, w_train = self._create_tabular_data(loader_train)
        X_val, y_val, w_val = self._create_tabular_data(loader_val)

        if self.expand_weights:
            # The duplication already carries the weighting, so the rows stay unweighted.
            w_train = w_val = None

        print(f"Collected {len(X_train)} training rows from {len(loader_train)} networks", flush=True)
        print(f"Collected {len(X_val)} validation rows from {len(loader_val)} networks", flush=True)

        # 2. Normalization (weighted, to match the weighting used for fitting)
        if self.normalize:
            self.covariate_scaler.fit(X_train, sample_weight=w_train)
            X_train = self.covariate_scaler.transform(X_train)
            X_val = self.covariate_scaler.transform(X_val)
            self.target_scaler.fit(y_train, sample_weight=w_train)
            y_train = self.target_scaler.transform(y_train)
            y_val = self.target_scaler.transform(y_val)

        # 3. Fit the model. The validation weights keep get_validation_error on the same
        # scale as the training objective (in expanded mode the duplication carries it).
        weight_kwargs = {} if w_train is None else {'sample_weight': w_train,
                                                    'sample_weight_eval_set': [w_val]}
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       verbose=verbose,
                       **weight_kwargs)

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

    def _predict_linear_reference(self, sample):
        """Recursive 1-step prediction along the grid topology, one node per call.

        Reference implementation. `_predict_linear_fast` is equivalent and much
        faster; this is kept so the two can be diffed.
        """
        features = sample['features']
        targets = sample['targets']
        parent = sample['parent']
        num_nodes = sample['num_nodes']

        predictions = np.zeros((num_nodes, 2))

        # Slack Bus initialization
        predictions[0] = targets[0]

        # Walking the buses in order of increasing depth guarantees that a bus's parent has
        # already been predicted by the time we reach it.
        order = np.argsort(sample['depth'], kind='stable')

        for target_node in order:
            if target_node == 0:
                continue

            parent_node = parent[target_node]

            # 1. Get Parent Voltage (Target Lag)
            v_parent = predictions[parent_node]
            # 2. Get Branch Covariates
            cov_parent = features[parent_node]  # Covariates of the parent node/edge
            cov_target = features[target_node]  # Features of the current node/edge

            X = np.concatenate([v_parent.flatten(), cov_parent.flatten(), cov_target.flatten()])
            out = self._predict_step(X.reshape(1, -1))

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

        return predictions

    def _get_fast_ctx(self):
        """Booster handle plus cached scaler arrays, built once per model."""
        if self._fast_ctx is None:
            booster = self.model.get_booster()
            if self.predict_nthread is not None:
                booster.set_param('nthread', self.predict_nthread)
            ctx = {'booster': booster, 'handle': booster.handle}
            if self.normalize:
                ctx['cov_mean'] = self.covariate_scaler.mean_
                ctx['cov_scale'] = self.covariate_scaler.scale_
                ctx['tgt_mean'] = self.target_scaler.mean_
                ctx['tgt_scale'] = self.target_scaler.scale_
            self._fast_ctx = ctx
        return self._fast_ctx

    def _predict_linear_fast(self, sample):
        """Depth-batched equivalent of `_predict_linear_reference`.

        One XGBoost call per depth level instead of one per node. Produces
        bit-identical output.
        """
        ctx = self._get_fast_ctx()
        normalize = self.normalize
        plan = _PathPlan(
            sample,
            normalize_mean=ctx['cov_mean'] if normalize else None,
            normalize_scale=ctx['cov_scale'] if normalize else None,
        )

        num_nodes = sample['num_nodes']
        predictions = np.zeros((num_nodes, 2))
        predictions[0] = sample['targets'][0]  # Slack bus initialization

        X = plan.X
        parent, target = plan.parent, plan.target
        use_capi = _CAPI_AVAILABLE
        handle, booster = ctx['handle'], ctx['booster']

        for level, (a, b) in enumerate(plan.levels):
            block = X[a:b]
            v_parent = predictions[parent[a:b]]

            # Only the parent-voltage columns change as the sweep descends.
            if normalize:
                block[:, 0:2] = (v_parent - ctx['cov_mean'][0:2]) / ctx['cov_scale'][0:2]
            else:
                block[:, 0:2] = v_parent

            if use_capi:
                out = _predict_block_capi(handle, plan.ifaces[level], b - a)
            else:
                out = booster.inplace_predict(block)

            if normalize:
                out = np.asarray(out, dtype=np.float32).copy()
                out *= ctx['tgt_scale']   # float32 in-place: rounds like sklearn
                out += ctx['tgt_mean']

            if self.use_diff:
                # Model predicts deltas: Child = Parent + Delta
                predictions[target[a:b]] = v_parent + out
            elif self.use_residuals:
                # Model predicts residuals: Child = Physics + Predicted_Residual
                predictions[target[a:b]] = plan.res_base[a:b] + out
            else:
                # Model predicts absolute voltages
                predictions[target[a:b]] = out

        return predictions

    def predict_linear(self, sample):
        """Recursive 1-step prediction along the grid topology."""
        # The corrector mutates `predictions` inside the node loop, so it has no
        # depth-batched equivalent; those variants use the reference sweep.
        if self.use_fast_predict and self.corrector is None:
            return self._predict_linear_fast(sample)
        return self._predict_linear_reference(sample)

    def predict(self, sample):
        if self.prediction_scheme == 'linear':
            return self.predict_linear(sample)
        # Only using the linear method going forward for NativeXGBModelWrapper.
        raise NotImplementedError(f"Scheme {self.prediction_scheme} not implemented.")
    
    def save(self, filepath):
        """Saves the entire wrapper state, including scalers and sub-models."""
        if not self._is_fitted:
            print("Warning: Saving a model that hasn't been fitted yet.")

        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads the wrapper from a file."""
        return joblib.load(filepath)

class XGB_Absolute(NativeXGBModelWrapper):
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False, use_fast_predict=False, expand_weights=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=False,
                         use_fast_predict=use_fast_predict,
                         expand_weights=expand_weights)
        
class XGB_Absolute_Fast(XGB_Absolute):
    def __init__(self):
        super().__init__(use_fast_predict=True)

class XGB_Absolute_Normalized(XGB_Absolute):
    def __init__(self, random_state=42, prediction_scheme='linear'):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=True)

class XGB_Parent(NativeXGBModelWrapper):
    def __init__(self, random_state=42, prediction_scheme='linear',
                 normalize=False, use_corrector=False, use_fast_predict=False, expand_weights=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=False,
                         use_diff=True,
                         use_corrector=use_corrector,
                         use_fast_predict=use_fast_predict,
                         expand_weights=expand_weights)

class XGB_Parent_Fast(XGB_Parent):
    def __init__(self):
        super().__init__(use_fast_predict=True)

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
                 normalize=False, use_corrector=False, use_fast_predict=False, expand_weights=False):
        super().__init__(random_state=random_state,
                         prediction_scheme=prediction_scheme,
                         normalize=normalize,
                         use_residuals=True,
                         use_diff=False,
                         use_corrector=use_corrector,
                         use_fast_predict=use_fast_predict,
                         expand_weights=expand_weights)

class XGB_LDF_Fast(XGB_LDF):
    def __init__(self):
        super().__init__(use_fast_predict=True)
        
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
