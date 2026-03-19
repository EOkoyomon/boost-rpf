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

# Add project root to sys.path for imports
import sys
from pathlib import Path
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from utils.data_utils import get_dataloaders
from utils.training_utils import get_lv_grid_codes, setup_seeds, get_device, train, test
from models.arma_gnn import ARMA_GNN
from models.mlp import GlobalMLP


def get_param_grid_gnn():
    """Define the hyperparameter search space."""
    return {
        "batch_size": [16, 32, 64],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [3000],
        "num_layers": [6, 7, 8, 9, 10],
        "hidden_dim": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "patience": [200],
    }

def get_param_grid_mlp():
    """Define the hyperparameter search space."""
    return {
        "batch_size": [16, 32, 64],
        "learning_rate": [0.0001, 0.00001, 0.000001],
        "epochs": [10000],
        "hidden_dim": [64, 128, 256, 512],
        "dropout": [0.0, 0.1, 0.2],
        "patience": [500],
    }


def get_random_params(param_grid, n_samples=20):
    """Sample random hyperparameter combinations."""
    param_combinations = []
    MAX_DUPLICATES = 10
    duplicates = 0
    i = 0
    while i < n_samples:
        params = {k: random.choice(v) for k, v in param_grid.items()}
        if params not in param_combinations:
            param_combinations.append(params)
            i += 1
        else:
            duplicates += 1
            if duplicates >= MAX_DUPLICATES:
                print("Max duplicates reached, stopping early.")
                break
    return param_combinations


def get_grid_params(param_grid):
    """Generate all hyperparameter combinations (grid search)."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

class TunedARMAGNN(ARMA_GNN):
    """ARMA_GNN with custom hyperparameters for tuning."""
    def __init__(self):
        # Call parent init to set up prediction scheme and other attributes
        super().__init__(
            num_layers=params['num_layers'],
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
        )
    
    @property
    def __name__(self):
        return "TunedARMAGNN"

class TunedGlobalMLP(GlobalMLP):
    """GlobalMLP with custom hyperparameters for tuning."""
    def __init__(self):
        # Call parent init to set up prediction scheme and other attributes
        super().__init__(
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
        )
    
    @property
    def __name__(self):
        return "TunedGlobalMLP"

def create_tuned_model_class(model, params):
    """
    Create a TunedARMAGNN class that inherits from ARMA_GNN
    but uses custom hyperparameters.
    """
    tuned_model_class = None
    if model == 'arma-gnn':
        class TunedARMAGNN(ARMA_GNN):
            """ARMA_GNN with custom hyperparameters for tuning."""
            def __init__(self):
                # Call parent init to set up prediction scheme and other attributes
                super().__init__(
                    num_layers=params['num_layers'],
                    hidden_dim=params['hidden_dim'],
                    dropout=params['dropout'],
                )
            
            @property
            def __name__(self):
                return "TunedARMAGNN"
        tuned_model_class = TunedARMAGNN
    elif model == 'global-mlp':
        class TunedGlobalMLP(GlobalMLP):
            """GlobalMLP with custom hyperparameters for tuning."""
            def __init__(self):
                # Call parent init to set up prediction scheme and other attributes
                super().__init__(
                    hidden_dim=params['hidden_dim'],
                    dropout=params['dropout'],
                )
            
            @property
            def __name__(self):
                return "TunedGlobalMLP"
        tuned_model_class = TunedGlobalMLP
    else:
        raise ValueError(f"Unknown model class: {model}")
    
    return tuned_model_class


def evaluate_params(model, params, loader_train, loader_val, loader_test):
    """Evaluate a single hyperparameter configuration."""
    # Create model with tuned params
    model_class = create_tuned_model_class(model, params)
    model = model_class()
    device = get_device()

    # Train model
    _, _, _, _, _, train_time = train(model,
                                      device,
                                      loader_train,
                                      loader_val,
                                      epochs=params['epochs'],
                                      learning_rate=params['learning_rate'], 
                                      early_stopping=True,
                                      patience=params['patience'],
                                      best_val_weights=True)
    
    # Test using the model's predict method
    rmse_vm, rmse_va, _ = test(model, device, loader_test)
    
    return {
        'rmse_vm': rmse_vm,
        'rmse_va': rmse_va,
        'train_time': train_time,
        'combined_score': rmse_vm + 0.01 * rmse_va,  # Weighted combination
    }


def run_tuning(args):
    """Run hyperparameter tuning."""
    setup_seeds()
    
    # Load data
    print("Loading data...")
    # Get grids to process
    if args.kerber:
        grids = ['Kerber_Dorfnetz']
    else:
        grids = get_lv_grid_codes(scenario=1)
    
    tabular = (args.model == 'global-mlp')
    loader_train, loader_val, loader_test = get_dataloaders(
        args.data_dir, grids, testing_grid=None, batch_size=16, tabular=tabular
    )
    print(f"Data loaded: {len(loader_train)} train, {len(loader_val)} val, {len(loader_test)} test samples")
    
    # Get hyperparameter combinations
    param_grid = {}
    if args.model == 'arma-gnn':
        param_grid = get_param_grid_gnn()
    elif args.model == 'global-mlp':
        param_grid = get_param_grid_mlp()
    
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
            metrics = evaluate_params(args.model, params, loader_train, loader_val, loader_test)
            
            result = {"model": args.model, **params, **metrics}
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
        json.dump({'model': args.model, 'params': best_params, 'score': best_score}, f, indent=2)
    print(f"Best params saved to: {best_params_file}")
    
    TOP_N = 10
    # Print summary
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best parameters: {best_params}")
    print(f"Best combined score: {best_score:.6f}")
    print(f"\nTop {TOP_N} configurations:")
    print(results_df.head(TOP_N).to_string())
    
    return best_params


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for XGBoost')
    parser.add_argument('--data_dir', required=True, help='Path to data directory')
    parser.add_argument("--model", default='arma-gnn', choices=['arma-gnn', 'global-mlp'])
    parser.add_argument("--kerber", action="store_true")
    parser.add_argument('--search_type', choices=['grid', 'random'], default='random',
                        help='Search strategy (default: random)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of random samples (default: 20)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    best_params = run_tuning(args)
