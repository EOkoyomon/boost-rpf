#!/usr/bin/env python3
"""
Script to analyze model performance from CSV results and generate LaTeX table.

This script:
1. Reads CSV data with model performance metrics
2. Calculates statistics (min, max, mean, std) for each model type
3. Generates a comprehensive LaTeX table with bolded best values
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_and_analyze_data(csv_file_path):
    """
    Load CSV data and calculate statistics for each model type.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with model statistics
    """
    # Load the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Filter out entries where testing_grid == "all"
    # df_filtered = df[df['testing_grid'] != 'all']
    df_filtered = df.copy()
    
    # Define the metrics columns we want to analyze
    metrics = ['rmse_vm_pu', 'rmse_va_degree'] # 'mape_vm_pu', 'mape_va_degree']
    
    # Get unique model types
    model_types = df_filtered['model'].unique()
    
    # Calculate statistics for each model
    results = {}

    for model in sorted(model_types.tolist()):
        model_data = df_filtered[df_filtered['model'] == model]
        results[model] = {}
        
        for metric in metrics:
            values = model_data[metric].values
            results[model][metric] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)  # Sample standard deviation
            }
    
    return results, metrics, model_types


def find_best_values(results, metrics):
    """
    Find the best (minimum) value for each metric and statistic combination.
    
    Args:
        results (dict): Dictionary with model statistics
        metrics (list): List of metric names
        
    Returns:
        dict: Dictionary indicating which model has the best value for each metric/statistic
        dict: Dictionary indicating which model has the second best value for each metric/statistic
    """
    best_values = {}
    second_best_values = {}
    stats = ['min', 'max', 'mean', 'std']
    
    for metric in metrics:
        best_values[metric] = {}
        second_best_values[metric] = {}
        for stat in stats:
            # For all metrics, lower is better
            min_value = float('inf')
            second_min_value = float('inf')
            best_model = None
            second_best_model = None
            
            for model in results.keys():
                value = results[model][metric][stat]
                if value < min_value:
                    # Move the previous best to second best
                    second_min_value = min_value
                    second_best_model = best_model
                    # Record the new best
                    min_value = value
                    best_model = model
                elif value < second_min_value:
                    second_min_value = value
                    second_best_model = model

            best_values[metric][stat] = best_model
            second_best_values[metric][stat] = second_best_model
    
    return best_values, second_best_values


def format_number(value, precision=4):
    """Format number with appropriate precision."""
    return f"{value:.{precision}f}"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def format_model_name(model_name):
    """Format model name for display in table."""
    # Create shorter, cleaner display names
    name_mappings = {
        'LinDistFlow': 'LinDistFlow',
    }
    return name_mappings.get(model_name, model_name.replace('_', '-'))


def generate_latex_table(results, metrics, model_types, best_values, second_best_values, mean_only=False):
    """
    Generate LaTeX table code with the results.
    Models are on rows, metrics are on columns, split into RMSE and MAPE sections.
    
    Args:
        results (dict): Dictionary with model statistics
        metrics (list): List of metric names
        model_types (list): List of model type names
        best_values (dict): Dictionary indicating best values
        second_best_values (dict): Dictionary indicating second best values
        
    Returns:
        str: LaTeX table code
    """
    # Define metric display names and group them
    metric_groups = {
        'RMSE': {
            'rmse_vm_pu': 'VM (p.u.)',
            'rmse_va_degree': 'VA (deg)'
        },
        # 'MAPE': {
        #     'mape_vm_pu': 'VM (\\%)',
        #     'mape_va_degree': 'VA (\\%)'
        # }
    }
    
    # Define statistic display names
    if mean_only:
        stat_names = {
            'mean': 'Mean'
        }
    else:
        stat_names = {
            'min': 'Min',
            'max': 'Max',
            'mean': 'Mean',
            'std': 'Std'
        }
    
    # Start building the LaTeX table
    latex_code = []
    
    # Table header
    latex_code.append("\\begin{table}[htbp]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Model Performance Comparison: Statistical Summary}")
    latex_code.append("\\label{tab:model_performance}")
    
    # Define column specification: Model name + 4 stats × 2 metrics for each group
    # Format: Model | Min | Max | Mean | Std | Min | Max | Mean | Std |
    num_stat_cols = len(stat_names)  # min, max, mean, std
    num_metrics_per_group = 2  # VM and VA for each group
    total_metric_cols = num_stat_cols * num_metrics_per_group
    col_spec = "l" + "c" * total_metric_cols  # Use paragraph column for model names to handle long names
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\toprule")
    
    # Process each metric group (RMSE and XXXX)
    for group_idx, (group_name, group_metrics) in enumerate(metric_groups.items()):
        # Create multi-level header
        if group_idx > 0:
            latex_code.append("\\midrule")
        
        # Group header row
        header_line1 = "\\textbf{Model}"
        for metric_key, metric_display in group_metrics.items():
            header_line1 += f" & \\multicolumn{{{num_stat_cols}}}{{c}}{{\\textbf{{{group_name} {metric_display}}}}}"
        header_line1 += " \\\\"
        latex_code.append(header_line1)
        
        # Sub-header row with statistics
        header_line2 = ""
        for metric_key, metric_display in group_metrics.items():
            header_line2 += " & " + " & ".join([f"\\textbf{{{stat}}}" for stat in stat_names.values()])
        header_line2 += " \\\\"
        latex_code.append(header_line2)
        latex_code.append("\\midrule")
        
        # Data rows for each model
        for model in model_types:
            formatted_model_name = format_model_name(model)
            row_data = [f"\\textbf{{{formatted_model_name}}}"]
            
            # Add data for each metric in this group
            for metric_key in group_metrics.keys():
                for stat in stat_names.keys():
                    value = results[model][metric_key][stat]
                    formatted_value = format_number(value)
                    
                    # Bold the value if it's the best (minimum) for this metric/stat combination
                    if best_values[metric_key][stat] == model:
                        formatted_value = f"\\textbf{{{formatted_value}}}"
                    # if second_best_values[metric_key][stat] == model:
                    #     formatted_value = f"\\underline{{{formatted_value}}}"
                    
                    row_data.append(formatted_value)
            
            row = " & ".join(row_data) + " \\\\"
            latex_code.append(row)
    
    # Table footer
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\end{table}")
    
    return "\n".join(latex_code)

def format_model_capacity(num):
    """
    Formats a number into a human-readable string (e.g., 1.2M, 4.5B).
    """
    if num < 1000:
        return str(num)

    for unit in ['K', 'M', 'B', 'T']:
        num /= 1000.0
        if abs(num) < 1000:
            return f"{num:.1f}{unit}"

    return f"{num:.1f}P" # Handles Quadrillions (Peta) just in case

def generate_latex_table_raw(results_csv_file_path, model_stats_csv_file_path=None):
    """
    Generate LaTeX table directly from CSV data, grouped by testing_grid.
    Bolds the smallest value for each metric within each testing_grid group.
    
    Args:
        results_csv_file_path (str): Path to the CSV file with results
        model_stats_csv_file_path (str): Path to the CSV file with model stats
        
    Returns:
        str: LaTeX table code
    """
    # Load the CSV data
    df = pd.read_csv(results_csv_file_path)

    # Handling potential unnamed index columns that might have been saved in the CSVs
    # (This step is often necessary when CSVs are saved with index=True)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Define the columns we want to include in order
    columns = ['testing_grid', 'model', 'rmse_vm_pu', 'rmse_va_degree', 'train_time', 'inference_time_ms']
    # columns = ['testing_grid', 'model', 'rmse_vm_pu', 'rmse_va_degree', 'mape_vm_pu', 'mape_va_degree', 'train_time']
    metrics_for_bolding = columns[2:]  # Metrics start from index 2
    # metrics_for_bolding = ['rmse_vm_pu', 'rmse_va_degree', 'mape_vm_pu', 'mape_va_degree', 'train_time']

    if model_stats_csv_file_path:
        stats_df = pd.read_csv(model_stats_csv_file_path)
        if 'Unnamed: 0' in stats_df.columns:
            stats_df = stats_df.drop(columns=['Unnamed: 0'])
        df = pd.merge(df, stats_df, on=['testing_grid', 'model'], how='inner')
        model_stats_columns = ['inference_time_ms']#, 'num_params']
        columns.extend(model_stats_columns)
        metrics_for_bolding.extend(model_stats_columns)
    
    # Select and sort the data
    df_selected = df[columns].copy()
    df_selected = df_selected.sort_values(['testing_grid', 'model'])
    
    # Find best (minimum) values for each metric within each testing_grid group
    best_values = {}
    for grid in df_selected['testing_grid'].unique():
        grid_data = df_selected[df_selected['testing_grid'] == grid]
        best_values[grid] = {}

        for metric in metrics_for_bolding:
            # 1. Group by model
            mean_model_performance = grid_data.groupby('model')[metrics_for_bolding].agg(['mean']).reset_index()
            # 2. Find the min value
            min_value = mean_model_performance[metric].min()
            # 3. Filter using np.isclose (checks for equality within a small tolerance)
            mask = np.isclose(mean_model_performance[metric], min_value)
            # 4. Get the best models list
            best_models_df = mean_model_performance[mask]
            best_models = best_models_df['model'].tolist() 

            # Append to best_values dictionary
            best_values[grid][metric] = {
                'value': min_value,
                'models': best_models
            }
    
    # Start building the LaTeX table
    latex_code = []
    
    # Table header
    latex_code.append("\\begin{table}[htbp]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Raw Model Performance Results by Testing Grid}")
    latex_code.append("\\label{tab:raw_model_performance}")
    
    # Define column specification
    col_spec = "llccccr"  # l for text, c for numbers, r for time
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\toprule")
    
    # Create header row
    headers = [
        "\\textbf{Testing Grid}",
        "\\textbf{Model}",
        "\\textbf{RMSE VM}",
        "\\textbf{RMSE VA}",
        # "\\textbf{MAPE VM}",
        # "\\textbf{MAPE VA}",
        "\\textbf{Train (s)}",
        "\\textbf{Inference (ms)}",
    ]
    if model_stats_csv_file_path:
        headers.extend([
            "\\textbf{Inference (ms)}",
            "\\textbf{Capacity}",
        ])
    header_row = " & ".join(headers) + " \\\\"
    latex_code.append(header_row)
    latex_code.append("\\midrule")

    for i, grid in enumerate(df_selected['testing_grid'].unique()):
        grid_df = df_selected[df_selected['testing_grid'] == grid]
        if i > 0:
            latex_code.append("\\midrule")
        grid_name_latex = grid
        if grid_name_latex == 'all':
            grid_name_latex = 'All (Known)'
        elif grid_name_latex.startswith('1-'):
            grid_name_latex = grid_name_latex.split('--')[0][2:]
        else:
            grid_name_latex = grid_name_latex.replace('_', ' ')

        for model_name in grid_df['model'].unique():
            grid_model_df = grid_df[grid_df['model'] == model_name] # Should have the different seed runs
            # Format metric values, bolding if they are the best in this grid
            formatted_values = []
            # Grid and model names (not bolded based on metrics)
            formatted_values.extend([
                grid_name_latex,
                format_model_name(model_name)
            ])
            for metric in metrics_for_bolding:
                mean_value = grid_model_df[metric].mean() # Mean of the seeds
                std_value = grid_model_df[metric].std() # Std of the seeds
                # value = row[metric]
                if metric == 'train_time':
                    formatted_mean_value = f"{mean_value:.1f}"
                    formatted_std_value = f"{std_value:.1f}"
                    # formatted_value = f"{value:.1f}"
                elif metric == 'num_params':
                    # Should only be one.
                    formatted_mean_value = format_model_capacity(mean_value)
                    formatted_std_value = format_model_capacity(std_value)
                    # formatted_value = format_model_capacity(value)
                else:
                    formatted_mean_value = format_number(mean_value, precision=5)
                    formatted_std_value = format_number(std_value, precision=5)
                    # formatted_value = format_number(value, precision=5)

                formatted_value = f"{formatted_mean_value} {{\\scriptsize $\\pm$ {formatted_std_value}}}"

                # Bold if this model has the best value for this metric in this grid
                if metric != 'num_params' and model_name in best_values[grid][metric]['models']:
                    formatted_value = f"\\textbf{{{formatted_value}}}"
                    formatted_value = f"\\textbf{{{formatted_value}}}"
                    formatted_value = f"\\textbf{{{formatted_value}}}"
                
                formatted_values.append(formatted_value)
            
            latex_row = " & ".join(formatted_values) + " \\\\"
            latex_code.append(latex_row)
    
    # Table footer
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\end{table}")
    
    return "\n".join(latex_code)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze model performance and generate LaTeX table"
    )
    parser.add_argument(
        "csv_file", 
        help="Path to the CSV file containing model results"
    )
    parser.add_argument(
        "--model_stats_file",
        required=False,
        help="Path to the CSV file containing model stats. See 'example_model_stats.csv' for expected format. If not provided, model stats columns will be skipped."
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for LaTeX table (optional, prints to stdout if not provided)"
    )
    parser.add_argument(
        "--raw", "-r",
        action="store_true",
        help="Generate raw data table instead of statistical summary"
    )
    parser.add_argument(
        "--mean", "-m",
        action="store_true",
        help="Generate table with mean values only instead of full statistical summary"
    )
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' not found!")
        return 1
    
    try:
        print(f"Loading data from {csv_path}...")
        
        if args.raw:
            model_stats_path = None
            if args.model_stats_file:
                model_stats_path = Path(args.model_stats_file)
                if not model_stats_path.exists():
                    model_stats_path = None
            # Generate raw data table
            latex_table = generate_latex_table_raw(csv_path, model_stats_path)
            # Initialize variables for consistency (not used in raw mode)
            results, metrics, best_values = None, None, None
        else:
            # Load and analyze data for statistical summary
            results, metrics, model_types = load_and_analyze_data(csv_path)
            
            print(f"Found {len(model_types)} model types: {', '.join(model_types)}")
            print(f"Analyzing {len(metrics)} metrics: {', '.join(metrics)}")
            
            # Find best values
            best_values, second_best_values = find_best_values(results, metrics)
            
            # Generate statistical summary table
            latex_table = generate_latex_table(results, metrics, model_types, best_values, second_best_values=second_best_values, mean_only=args.mean)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(latex_table)
            print(f"LaTeX table saved to {output_path}")
        else:
            print("\nGenerated LaTeX Table:")
            print("=" * 50)
            print(latex_table)
        
        # Print summary statistics only for statistical summary table
        if not args.raw and metrics is not None and best_values is not None and results is not None:
            print("\nSummary:")
            print("-" * 40)
            for metric in metrics:
                print(f"\n{metric.upper()}:")
                stats = ['min', 'max', 'mean', 'std']
                if args.mean:
                    stats = ['mean']
                for stat in stats:
                    best_model = best_values[metric][stat]
                    best_value = results[best_model][metric][stat]
                    print(f"  Best {stat}: {best_model} ({format_number(best_value)})")
        
        return 0
        
    except Exception as e:
        raise e
        return 1


if __name__ == "__main__":
    exit(main())