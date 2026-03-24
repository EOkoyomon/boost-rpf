import argparse
import os

import pandas as pd
import torch

from models.arma_gnn import (
    ARMA_GNN,
)
from models.dcpf_models import (
    DC_PF,
    DC_PF_Slack,
)
from models.gnn_models import (
    NormedGNN,
)
from models.lindistflow import (
    DistFlow,
    LinDistFlow,
)
from models.mlp import (
    CustomNormedMLP,
    GlobalMLP
)
from models.pg_models import (
    get_pg_model,
)
from models.powerflownet import (
    PowerFlowNet,
)
from models.xgb_darts_models import (
    XGBModel_Basic,
    XGBModel_Linear,
    XGBModel_Normalized,
)
from models.xgb_models import (
    XGB_LDF,
    XGB_Absolute,
    XGB_Absolute_Normalized,
    XGB_LDF_Normalized,
    XGB_Parent,
    XGB_Parent_Corrected,
    XGB_Parent_Normalized,
)
from utils.data_utils import get_dataloaders
from utils.training_utils import (
    create_log_dir,
    get_device,
    get_lv_grid_codes,
    get_model_save_path,
    plot_loss,
    setup_seeds,
    plot_error_accumulation,
    test,
    test_sequential,
    train,
    train_sequential,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        default=["n-gnn"],
        nargs="+",
        choices=["ALL"] + list(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save_results",
        action="store_true"
    )
    parser.add_argument(
        "--plot",
        action="store_true"
    )
    parser.add_argument(
        "--save_model",
        action="store_true"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true"
    )
    parser.add_argument(
        "--load_model_dir",
        required=False,
    )
    parser.add_argument(
        "--load_model_name",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--experiment",
        type=int,
        help="The experiment group to analyze (1, 2, or 3). If not provided, analyzes all experiments together."
    )
    args = parser.parse_args()
    return args

def evaluate_performance(model_class,
                         loader_train,
                         loader_val,
                         loader_test,
                         epochs=100,
                         learning_rate=0.001,
                         patience=50,
                         log_dir=None,
                         plot=False,
                         save_model=False,
                         eval_only=False,
                         load_model_dir=None,
                         model_load_experiment_id='0',
                         experiment_id='0'):
    # Log information about training run
    early_stopping=True
    best_val_weights=True
    print(f'\n{locals()}', flush=True)
    if log_dir:
        # Log locals to config file
        config_path = os.path.join(log_dir, f'config_{experiment_id}.txt')
        with open(config_path, 'w') as f:
            for key, value in locals().items():
                f.write(f'{key}: {value}\n')


    model_weights_path = ''
    if save_model:
        assert log_dir, 'Need to pass a log_dir path in order to save model or plot loss'
        model_weights_path = get_model_save_path(log_dir, experiment_id)

    # PyTorch setup
    device = get_device()
    print(f"Training using {device}", flush=True)

    # Create model
    model = model_class().to(device)

    if load_model_dir:
        load_model_path = get_model_save_path(load_model_dir,
                                              model_id=model_load_experiment_id)
        model.load_state_dict(torch.load(load_model_path,
                                         weights_only=True,
                                         map_location=device))

    train_loss_vec = val_loss_vec = best_val_loss = corresponding_train_loss = total_epochs = train_time = 0
    if not eval_only:
        # Train the model
        train_loss_vec, val_loss_vec, best_val_loss, corresponding_train_loss, total_epochs, train_time = \
            train(model=model,
                device=device,
                loader_train=loader_train,
                loader_val=loader_val,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping=early_stopping,
                patience=patience,
                best_val_weights=best_val_weights,
                save_model_to=model_weights_path,
                log_epochs=True)

        # Plot the model
        if plot and log_dir is not None:
            plot_loss(log_dir,
                    model_class.__name__,
                    train_loss_vec,
                    val_loss_vec,
                    fig_id=experiment_id)

    # Test the model
    rmse_vm, rmse_va, inference_time_ms = test(model=model,
                                               device=device,
                                               loader_test=loader_test,
                                               plot=plot)

    return rmse_vm, rmse_va, best_val_loss, corresponding_train_loss, total_epochs, train_time, inference_time_ms

def evaluate_performance_sequential(model_class,
                                    loader_train,
                                    loader_val,
                                    loader_test,
                                    epochs=100,
                                    learning_rate=0.001,
                                    patience=50,
                                    log_dir=None,
                                    plot=False,
                                    save_model=False,
                                    eval_only=False,
                                    load_model_dir=None,
                                    model_load_experiment_id='0',
                                    experiment_id='0'):
    """
    Evaluate sequential models that operate on path sequences.

    This function:
        1. Extracts all paths from training samples
        2. Fits the model using path sequences (model uses lags for parent voltage)
        3. Tests using recursive prediction
        4. Computes RMSE metrics

    Args:
        model_class: The sequential model class (e.g., XGBModelWrapper)
        loader_train: List of sample dicts from get_grid_paths for training
        loader_val: List of sample dicts from get_grid_paths for validation (unused for now)
        loader_test: List of sample dicts from get_grid_paths for testing
        epochs: Ignored for XGBoost, kept for interface compatibility
        Other args: For interface compatibility with evaluate_performance

    Returns:
        Tuple of (rmse_vm, rmse_va, mape_vm, mape_va, best_val_loss, train_loss, epochs, train_time)
    """

    if save_model:
        assert log_dir, 'Need to pass a log_dir path in order to save model or plot loss'
        save_model_path = get_model_save_path(log_dir, experiment_id, sequential=True)

    # Create model
    if load_model_dir:

        load_model_path = get_model_save_path(load_model_dir,
                                              model_id=model_load_experiment_id,
                                              sequential=True)
        model = model_class.load(load_model_path)
    else:
        model = model_class()

    # Train the model
    train_time = validation_error = -1
    if not eval_only:
        train_time, validation_error = train_sequential(model=model,
                                    loader_train=loader_train,
                                    loader_val=loader_val)

    if save_model:
        model.save(save_model_path)

    # Test the model
    rmse_vm, rmse_va, inference_time_ms = test_sequential(model=model,
                                                          loader_test=loader_test,
                                                          plot=plot)

    corresponding_train_loss = total_epochs = -1

    return rmse_vm, rmse_va, validation_error, corresponding_train_loss, total_epochs, train_time, inference_time_ms

# Get models to evaluate
MODEL_CLASSES = {
    "ldf": LinDistFlow,
    "distflow": DistFlow,
    "n-gnn": NormedGNN,
    "xgb-absolute": XGB_Absolute,
    "xgb-parent": XGB_Parent,
    "xgb-ldf": XGB_LDF,
    "pfnet": PowerFlowNet,
    "pg-transformer": get_pg_model("Transformer"),
    "custom-mlp": CustomNormedMLP,
    "global-mlp": GlobalMLP,
    "arma-gnn": ARMA_GNN,
}
ANALYTICAL_MODELS = [DC_PF, DC_PF_Slack, LinDistFlow, DistFlow]
SEQUENTIAL_MODELS = [XGBModel_Basic,
                     XGBModel_Linear,
                     XGBModel_Normalized,
                     XGB_Absolute,
                     XGB_Parent,
                     XGB_LDF,
                     XGB_Absolute_Normalized,
                     XGB_Parent_Normalized,
                     XGB_LDF_Normalized,
                     XGB_Parent_Corrected]
TABULAR_MODELS = [CustomNormedMLP, GlobalMLP]
REAL_VALUED_GRAPH_MODELS = set(MODEL_CLASSES.values()) - set(SEQUENTIAL_MODELS) - set(TABULAR_MODELS)

def run_benchmark(args):
    # Argument parsing and validation
    if args.load_model_dir:
        assert len(args.model) == 1 and args.model[0].upper() != 'ALL', "When loading a model, please specify a single model type, not 'ALL'."

    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience
    save_results = args.save_results
    plot = args.plot
    save_model = args.save_model
    eval_only = args.eval_only
    load_model_dir = args.load_model_dir
    load_model_name = args.load_model_name
    
    # Set up training, logging, and experiment cases
    setup_seeds(args.seed)
    
    log_dir = None
    if save_results or save_model:
        # Create a new log directory for each model
        log_dir = create_log_dir()
    
    # Select experiment cases based on argument
    selected_experiments = [args.experiment] if args.experiment else [1, 2, 3]

    # Only compare LV networks because radial
    test_cases = []
    test_cases.append((1, ['Kerber_Dorfnetz'], None)) # Experiment 1: Kerber Dorfnetz
    grids_to_compare = get_lv_grid_codes(scenario=1)
    test_cases.append((2, grids_to_compare, None))  # Experiment 2: Heterogenous Grids (all grids)
    for grid in grids_to_compare:
        test_cases.append((3, [g for g in grids_to_compare if g != grid], grid))  # Experiment 3: OOD (Leave-one-out scenarios)

    # Filter test cases based on selected experiments
    test_cases = [case for case in test_cases if case[0] in selected_experiments]

    # Set up results tracking
    if save_results and log_dir:
        results_file = os.path.join(log_dir, 'results_summary.csv')
        column_names = [
            'experiment',
            'model',
            'testing_grid',
            'rmse_vm_pu',
            'rmse_va_degree',
            'best_val_loss',
            'corresponding_train_loss',
            'total_epochs',
            'train_time',
            'inference_time_ms',
            'seed'
        ]
        # Create a DataFrame for the results
        pd.DataFrame(columns=column_names).to_csv(results_file)
        print(f'\nResults will be saved to: {results_file}\n', flush=True)

    models_to_evaluate = []
    if len(args.model) == 1 and args.model[0].upper() == 'ALL':
        models_to_evaluate = list(MODEL_CLASSES.values())
    else:
        models_to_evaluate = [MODEL_CLASSES[m] for m in args.model]

    # Run evaluations
    for experiment, training_grids, testing_grid in test_cases:
        # Get data loaders

        need_real_valued_data = len(set(models_to_evaluate) & set(REAL_VALUED_GRAPH_MODELS)) > 0
        need_real_valued_path_data = len(set(models_to_evaluate) & set(SEQUENTIAL_MODELS)) > 0
        need_real_valued_tabular_data = len(set(models_to_evaluate) & set(TABULAR_MODELS)) > 0

        # If no real models are being evaluated, skip loading real data
        if need_real_valued_data:
            loader_train_real, loader_val_real, loader_test_real = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size
            )

        # If no path-based models are being evaluated, skip loading path-based data
        if need_real_valued_path_data:
            loader_train_path, loader_val_path, loader_test_path = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size, paths=True
            )

        # If no tabular models are being evaluated, skip loading tabular data
        if need_real_valued_tabular_data:
            loader_train_tabular, loader_val_tabular, loader_test_tabular = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size, tabular=True
            )

        # Keep track of results
        results = []

        for model in models_to_evaluate:
            if model in SEQUENTIAL_MODELS:
                loader_train, loader_val, loader_test = loader_train_path, loader_val_path, loader_test_path
            elif model in TABULAR_MODELS:
                loader_train, loader_val, loader_test = loader_train_tabular, loader_val_tabular, loader_test_tabular
            else:
                loader_train, loader_val, loader_test = loader_train_real, loader_val_real, loader_test_real
            print('\n--------------------------------------------------', flush=True)
            case_name = ''
            if training_grids[0] == 'Kerber_Dorfnetz':
                case_name = 'Kerber_Dorfnetz'
            elif testing_grid:
                case_name = testing_grid
            else:
                case_name = 'all'
            print(f'\nEvaluating model: {model.__name__} | Testing grid: {case_name}', flush=True)
            # Train and test model
            if model in ANALYTICAL_MODELS:
                rmse_vm, rmse_va, inference_time_ms = test(model(), get_device(), loader_test)
                best_val_loss, corresponding_train_loss, total_epochs, train_time = 0, 0, 0, 0
            elif model in SEQUENTIAL_MODELS:
                # For sequential models, use path-based evaluation
                rmse_vm, rmse_va, best_val_loss, corresponding_train_loss, total_epochs, train_time, inference_time_ms = \
                    evaluate_performance_sequential(model_class=model,
                                        loader_train=loader_train,
                                        loader_val=loader_val,
                                        loader_test=loader_test,
                                        epochs=epochs,
                                        learning_rate=learning_rate,
                                        patience=patience,
                                        log_dir=log_dir,
                                        plot=plot,
                                        save_model=save_model,
                                        eval_only=eval_only,
                                        load_model_dir=load_model_dir,
                                        model_load_experiment_id=f"{load_model_name if load_model_name else model.__name__}_{case_name}",
                                        experiment_id=f"{model.__name__}_{case_name}")
            else:
                rmse_vm, rmse_va, best_val_loss, corresponding_train_loss, total_epochs, train_time, inference_time_ms = \
                    evaluate_performance(model_class=model,
                                        loader_train=loader_train,
                                        loader_val=loader_val,
                                        loader_test=loader_test,
                                        epochs=epochs,
                                        learning_rate=learning_rate,
                                        patience=patience,
                                        log_dir=log_dir,
                                        plot=plot,
                                        save_model=save_model,
                                        eval_only=eval_only,
                                        load_model_dir=load_model_dir,
                                        model_load_experiment_id=f"{load_model_name if load_model_name else model.__name__}_{case_name}",
                                        experiment_id=f"{model.__name__}_{case_name}")
            
            results.append(
                (
                    experiment,
                    model.__name__,
                    case_name,
                    rmse_vm,
                    rmse_va,
                    best_val_loss,
                    corresponding_train_loss,
                    total_epochs,
                    train_time,
                    inference_time_ms,
                    args.seed
                )
            )
            print(f'\nCompleted evaluation for model: {model.__name__}', flush=True)
            stats = f'\trmse_vm: {rmse_vm}\n\trmse_va: {rmse_va}\n\tbest_val_loss: {best_val_loss}\n\tcorresponding_train_loss: {corresponding_train_loss}\n\ttotal_epochs: {total_epochs}\n\ttime (s): {train_time}\n\tinference_time_ms: {inference_time_ms}'
            print(stats, flush=True)

        if save_results and log_dir:
            # Create a DataFrame for the results
            results_df = pd.DataFrame(results, columns=column_names)
            
            # Append to existing results file after each test case
            assert(results_file is not None), "results_file should not be None if save_results is True"
            results_df.to_csv(results_file, mode='a', index=True, header=False)
            print(f'\nAppended results to: {results_file}', flush=True)

        print('\n==================================================', flush=True)

    print('\nAll evaluations completed.', flush=True)

if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args)
