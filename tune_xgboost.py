"""
Hyperparameter tuning for XGBoost model using simple grid/random search.
Integrates with existing data loading and evaluation infrastructure.
"""
import argparse
import itertools
import json
import os
import random
from datetime import datetime

import pandas as pd
from darts.models import XGBModel

from models.models import XGBModel_Linear
from utils.data_utils import get_dataloaders
from utils.training_utils import get_dist_grid_codes, setup_seeds, test_sequential, train_sequential


def get_param_grid():
    """Define the hyperparameter search space."""
    return {
        # High priority
        'n_estimators': [100],#[50, 100, 200, 500],
        'max_depth': [7], #[5, 7, 9, 12],
        'learning_rate': [0.02], #[0.05, 0.1, 0.2, 0.3],
        # Medium priority
        'min_child_weight': [20], #[1, 5],
        'subsample': [0.9, 1.0], # [0.8, 1.0],
        'colsample_bytree': [0.8], #[0.8, 1.0],
        # Lower priority (optional - uncomment to include)
        # 'reg_alpha': [0, 0.1],
        # 'reg_lambda': [1.0, 10.0],
    }


def get_random_params(param_grid, n_samples=20):
    """Sample random hyperparameter combinations."""
    param_combinations = []
    for _ in range(n_samples):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        param_combinations.append(params)
    return param_combinations


def get_grid_params(param_grid):
    """Generate all hyperparameter combinations (grid search)."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def create_tuned_model_class(params):
    """
    Create a TunedXGBModel class that inherits from XGBModel_Linear
    but uses custom hyperparameters.
    """
    class TunedXGBModel(XGBModel_Linear):
        """XGBModel_Linear with custom hyperparameters for tuning."""
        
        def __init__(self):
            # Call parent init to set up prediction scheme and other attributes
            super().__init__()
            # Replace the model with one using tuned hyperparameters
            self.model = XGBModel(
                lags=self.lags,
                lags_future_covariates=self.lags_future_covariates,
                output_chunk_length=1,
                random_state=42,
                multi_models=True,
                **params  # Inject hyperparameters
            )
        
        @property
        def __name__(self):
            return "TunedXGBModel"
    
    return TunedXGBModel


def evaluate_params(params, loader_train, loader_val, loader_test):
    """Evaluate a single hyperparameter configuration."""
    # Create model with tuned params
    model_class = create_tuned_model_class(params)
    model = model_class()

    # Train model
    train_time, _ = train_sequential(model, loader_train, loader_val)
    
    # Test using the model's predict method
    rmse_vm, rmse_va, _ = test_sequential(model, loader_test)
    
    return {
        'rmse_vm': rmse_vm,
        'rmse_va': rmse_va,
        'train_time': train_time,
        'combined_score': rmse_vm + 0.1 * rmse_va,  # Weighted combination
    }


def run_tuning(args):
    """Run hyperparameter tuning."""
    setup_seeds()
    
    # Load data
    print("Loading data...")
    grids = get_dist_grid_codes(scenario=1)
    loader_train, loader_val, loader_test = get_dataloaders(
        args.data_dir, grids[:-1], testing_grid=grids[-1], batch_size=16, paths=True
    )
    print(f"Data loaded: {len(loader_train)} train, {len(loader_val)} val, {len(loader_test)} test samples")
    
    # Get hyperparameter combinations
    param_grid = get_param_grid()
    
    if args.search_type == 'grid':
        param_combinations = get_grid_params(param_grid)
        print(f"Grid search: {len(param_combinations)} combinations")
    else:
        param_combinations = get_random_params(param_grid, n_samples=args.n_samples)
        print(f"Random search: {args.n_samples} combinations")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('out', f'tuning_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tuning
    results = []
    best_score = float('inf')
    best_params = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
        
        try:
            metrics = evaluate_params(params, loader_train, loader_val, loader_test)
            
            result = {**params, **metrics}
            results.append(result)
            
            print(f"  RMSE V_m: {metrics['rmse_vm']:.6f}, RMSE V_a: {metrics['rmse_va']:.4f}, "
                  f"Time: {metrics['train_time']:.1f}s")
            
            if metrics['combined_score'] < best_score:
                best_score = metrics['combined_score']
                best_params = params.copy()
                print("  *** New best! ***")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('combined_score')
    results_file = os.path.join(output_dir, 'tuning_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save best params
    best_params_file = os.path.join(output_dir, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump({'params': best_params, 'score': best_score}, f, indent=2)
    print(f"Best params saved to: {best_params_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best parameters: {best_params}")
    print(f"Best combined score: {best_score:.6f}")
    print("\nTop 5 configurations:")
    print(results_df.head().to_string())
    
    return best_params


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for XGBoost')
    parser.add_argument('--data_dir', required=True, help='Path to data directory')
    parser.add_argument('--search_type', choices=['grid', 'random'], default='random',
                        help='Search strategy (default: random)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of random samples (default: 20)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    best_params = run_tuning(args)
