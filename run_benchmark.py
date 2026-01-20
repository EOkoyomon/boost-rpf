import argparse
import os

import pandas as pd
import torch

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
    get_dist_grid_codes,
    get_model_save_path,
    plot_loss,
    setup_seeds,
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
        default="xgb-basic",
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
        default="NormedGNN",
    )
    args = parser.parse_args()
    return args

def evaluate_performance(model_class,
                         loader_train,
                         loader_val,
                         loader_test,
                         epochs=100,
                         log_dir=None,
                         plot=False,
                         save_model=False,
                         eval_only=False,
                         load_model_dir=None,
                         model_load_experiment_id='0',
                         experiment_id='0'):
    # Log information about training run
    learning_rate=1e-3
    early_stopping=True
    patience=750
    best_val_weights=True
    print(f'\n{locals()}', flush=True)

    model_weights_path = ''
    if save_model:
        assert log_dir, 'Need to pass a log_dir path in order to save model or plot loss'
        model_weights_path = get_model_save_path(log_dir, experiment_id)

    # PyTorch setup
    device = get_device()
    print(f"Training using {device}", flush=True)

    # Create model
    model = model_class().to(device)
    if not model.is_supervised():
        learning_rate = 1e-2

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
                log_epochs=(log_dir is not None))

        # Plot the model
        if plot:
            plot_loss(log_dir,
                    model_class.__name__,
                    train_loss_vec,
                    val_loss_vec,
                    fig_id=experiment_id)

    # Test the model
    rmse_vm, rmse_va, mape_vm, mape_va = test(model=model,
                                            device=device,
                                            loader_test=loader_test)

    return rmse_vm, rmse_va, mape_vm, mape_va, best_val_loss, corresponding_train_loss, total_epochs, train_time

def evaluate_performance_sequential(model_class,
                                    loader_train,
                                    loader_val,
                                    loader_test,
                                    epochs=100,
                                    plot=False,
                                    save_model=False,
                                    eval_only=False,
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
    # Create model
    model = model_class()

    # Train the model
    train_time, validation_error = train_sequential(model=model,
                                  loader_train=loader_train,
                                  loader_val=loader_val)

    # Test the model
    rmse_vm, rmse_va, inference_time_ms = test_sequential(model=model,
                                                          loader_test=loader_test,
                                                          plot=plot)

    corresponding_train_loss = total_epochs = -1

    return rmse_vm, rmse_va, validation_error, corresponding_train_loss, total_epochs, train_time, inference_time_ms

# Get models to evaluate
MODEL_CLASSES = {
    # "dist-flow": DistFlow,
    # "lin-dist-flow": LinDistFlow,
    # "n-gnn": NormedGNN,
    # "dc-pf": DC_PF,
    # "dc-pf-slack": DC_PF_Slack,
    # "xgb-basic": XGBModel_Basic,
    # "xgb-linear": XGBModel_Linear,
    # "xgb-normalized": XGBModel_Normalized,
    "xgb-absolute": XGB_Absolute,
    "xgb-parent": XGB_Parent,
    "xgb-ldf": XGB_LDF,
    "xgb-absolute-n": XGB_Absolute_Normalized,
    "xgb-parent-n": XGB_Parent_Normalized,
    "xgb-ldf-n": XGB_LDF_Normalized,
    "xgb-parent-corrected": XGB_Parent_Corrected,
}
COMPLEX_MODELS = []
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
REAL_VALUED_GRAPH_MODELS = set(MODEL_CLASSES.values()) - set(COMPLEX_MODELS) - set(SEQUENTIAL_MODELS)

def run_benchmark(args):
    # Argument parsing and validation
    if args.load_model_dir:
        assert args.model.upper() != 'ALL', "When loading a model, please specify a single model type, not 'ALL'."

    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    save_results = args.save_results
    plot = args.plot
    save_model = args.save_model
    eval_only = args.eval_only
    load_model_dir = args.load_model_dir
    load_model_name = args.load_model_name
    
    # Set up training, logging, and experiment cases
    setup_seeds()
    
    log_dir = None
    if save_results or save_model:
        # Create a new log directory for each model
        log_dir = create_log_dir()
    
    grids_to_compare = get_dist_grid_codes(scenario=1)
    test_cases = [(grids_to_compare, None)]  # All grids scenario
    for grid in grids_to_compare:
        test_cases.append(([g for g in grids_to_compare if g != grid], grid))  # Leave-one-out scenarios
    # test_cases = test_cases[:1] + test_cases[-4:-3] + test_cases[-1:]

    # Set up results tracking
    if save_results and log_dir:
        results_file = os.path.join(log_dir, 'results_summary.csv')
        column_names = [
            'model',
            'testing_grid',
            'rmse_vm_pu',
            'rmse_va_degree',
            'best_val_loss',
            'corresponding_train_loss',
            'total_epochs',
            'train_time',
            'inference_time_ms'
        ]
        # Create a DataFrame for the results
        pd.DataFrame(columns=column_names).to_csv(results_file)
        print(f'\nResults will be saved to: {results_file}\n', flush=True)

    models_to_evaluate = [MODEL_CLASSES[args.model]] if args.model.upper() != 'ALL' else list(MODEL_CLASSES.values())

    # Run evaluations
    for training_grids, testing_grid in test_cases:
        # Get data loaders

        need_real_valued_data = len(set(models_to_evaluate) & set(REAL_VALUED_GRAPH_MODELS)) > 0
        need_complex_valued_data = len(set(models_to_evaluate) & set(COMPLEX_MODELS)) > 0
        need_real_valued_path_data = len(set(models_to_evaluate) & set(SEQUENTIAL_MODELS)) > 0

        # If no real models are being evaluated, skip loading real data
        if need_real_valued_data:
            loader_train_real, loader_val_real, loader_test_real = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size
            )

        # If no complex models are being evaluated, skip loading complex data
        if need_complex_valued_data:
            loader_train_complex, loader_val_complex, loader_test_complex = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size, complex=True
            )

        # If no path-based models are being evaluated, skip loading path-based data
        if need_real_valued_path_data:
            loader_train_path, loader_val_path, loader_test_path = get_dataloaders(
                data_dir, training_grids, testing_grid, batch_size=batch_size, paths=True
            )

        # Keep track of results
        results = []

        for model in models_to_evaluate:
            # Use complex data loaders for complex models
            if model in COMPLEX_MODELS:
                loader_train, loader_val, loader_test = loader_train_complex, loader_val_complex, loader_test_complex
            elif model in SEQUENTIAL_MODELS:
                loader_train, loader_val, loader_test = loader_train_path, loader_val_path, loader_test_path
            else:
                loader_train, loader_val, loader_test = loader_train_real, loader_val_real, loader_test_real
            print('\n--------------------------------------------------', flush=True)
            print(f'\nEvaluating model: {model.__name__} | Testing grid: {testing_grid if testing_grid else "All Grids"}', flush=True)
            # Train and test model
            if model in ANALYTICAL_MODELS:
                rmse_vm, rmse_va, mape_vm, mape_va = test(model(), get_device(), loader_test)
                best_val_loss, corresponding_train_loss, total_epochs, train_time = 0, 0, 0, 0
                inference_time_ms = -1
            elif model in SEQUENTIAL_MODELS:
                # For sequential models, use path-based evaluation
                rmse_vm, rmse_va, best_val_loss, corresponding_train_loss, total_epochs, train_time, inference_time_ms = \
                    evaluate_performance_sequential(model_class=model,
                                        loader_train=loader_train,
                                        loader_val=loader_val,
                                        loader_test=loader_test,
                                        epochs=epochs,
                                        plot=plot,
                                        save_model=save_model,
                                        eval_only=eval_only,
                                        experiment_id=f"{model.__name__}_{testing_grid if testing_grid else 'all'}")
            else:
                rmse_vm, rmse_va, mape_vm, mape_va, best_val_loss, corresponding_train_loss, total_epochs, train_time = \
                    evaluate_performance(model_class=model,
                                        loader_train=loader_train,
                                        loader_val=loader_val,
                                        loader_test=loader_test,
                                        epochs=epochs,
                                        log_dir=log_dir,
                                        plot=plot,
                                        save_model=save_model,
                                        eval_only=eval_only,
                                        load_model_dir=load_model_dir,
                                        model_load_experiment_id=f"{load_model_name}_{testing_grid if testing_grid else 'all'}",
                                        experiment_id=f"{model.__name__}_{testing_grid if testing_grid else 'all'}")
                inference_time_ms = -1
            
            results.append(
                (
                    model.__name__,
                    testing_grid if testing_grid else 'all',
                    rmse_vm,
                    rmse_va,
                    best_val_loss,
                    corresponding_train_loss,
                    total_epochs,
                    train_time,
                    inference_time_ms
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
