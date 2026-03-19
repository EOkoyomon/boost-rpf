#%%
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_inference_time_scaling(df_all, grid_to_num_noodes, save=False):
    # For every model in the dataframe, plot inference time vs number of nodes as a line plot
    df_all = df_all.copy()

    plt.figure(figsize=(10, 6))
    # testing_grids = df_all[df_all['testing_grid'] != 'all']['testing_grid'].unique()

    sorted_grids = sorted(grid_to_num_noodes.items(), key=lambda x: x[1])
    for model in df_all['model'].unique():
        df_model = df_all[df_all['model'] == model]
        x = []
        y = []
        for grid, num_nodes in sorted_grids:
            if grid in df_model['testing_grid'].values:
                if num_nodes is not None:
                    inference_time = df_model[df_model['testing_grid'] == grid]['inference_time_ms'].values.mean()
                    x.append(num_nodes)
                    y.append(inference_time)
        plt.plot(x, y, label=model, marker='o', linestyle='-')
    plt.xlabel('Number of Nodes', fontsize=22)
    plt.ylabel('Inference Time (ms)', fontsize=22)
    # Add a little padding to the title
    plt.title('Inference Time Scaling', fontsize=24, pad=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Save high resolution vector figure (pdf)
    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        path = FIGURES_DIR / "inference_time_scaling.pdf"
        plt.savefig(path, format="pdf", dpi=300)
    else:
        plt.show()

def plot_vm_boxplot(df_all, metric, save=False):

    # 1. Filter and prepare data
    df_ood = df_all[(df_all['testing_grid'] != 'all') & (df_all['testing_grid'] != 'Kerber_Dorfnetz')].copy()
    df_ood['model_label'] = df_ood['model'].str.replace('_', '-')

    # 2. Balanced Figure Size - 8 wide, 6 tall is a standard "sweet spot" for papers
    plt.figure(figsize=(8, 6)) 
    sns.set_theme(style="white")

    # 3. Add the Boxplot
    # Width 0.6 is a good middle ground (Seaborn default is 0.8, your previous was 0.4)
    ax = sns.boxplot(
        data=df_ood, 
        x='model_label', 
        y=metric, 
        palette='viridis', 
        showfliers=False, 
        width=0.6, 
        boxprops=dict(alpha=0.6)
    )

    # 4. Add individual data points
    # Jitter 0.2 keeps points centered but well-distributed
    sns.stripplot(
        data=df_ood, 
        x='model_label', 
        y=metric, 
        color='black', 
        alpha=0.3, 
        jitter=0.2, 
        size=4
    )

    # 5. Add small grid lines
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 6. Formatting & Log Scale
    plt.yscale('log')
    plt.title(f'Voltage {"Magnitude" if "vm" in metric else "Angle"} OOD RMSE', fontsize=24, pad=20)
    plt.xlabel('Model', fontsize=20)
    plt.ylabel(f'RMSE ({metric.split("_")[-1]})', fontsize=20)

    # Standard rotation for readability
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)

    # Ensure everything fits within the PDF boundaries
    plt.tight_layout()

    # Save high resolution vector figure
    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        path = FIGURES_DIR / f"voltage_{metric}_boxplot.pdf"
        plt.savefig(path, format="pdf", dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures for the paper.")
    parser.add_argument("--exp1_results", type=str, default="experiment_1_results_with_seed.csv", help="Path to experiment 1 results CSV.")
    parser.add_argument("--exp2_results", type=str, default="experiment_2_results_with_seed.csv", help="Path to experiment 2 results CSV.")
    parser.add_argument("--exp3_results", type=str, default="experiment_3_results_with_seed.csv", help="Path to experiment 3 results CSV.")
    parser.add_argument("--save", action="store_true", help="Save figures as high resolution PDFs.")
    args = parser.parse_args()

    FIGURES_DIR = Path("figures")

    df_exp1 = pd.read_csv(args.exp1_results)
    df_exp2 = pd.read_csv(args.exp2_results)
    df_exp3 = pd.read_csv(args.exp3_results)

    # Combine dataframes
    df_all = pd.concat([df_exp1, df_exp2, df_exp3], ignore_index=True)

    # Get mean and std of 'train_time' grouped by model
    grouped = df_all.groupby('model')['train_time'].agg(['mean', 'std']).reset_index()

    # Print out the results using mean +- std format, with valued rounded to 2 decimal places
    for index, row in grouped.iterrows():
        mean_time = round(row['mean'], 2)
        std_time = round(row['std'], 2)
        print(f"{row['model']}: {mean_time} +- {std_time} seconds")

    grid_to_num_noodes = {
        'Kerber_Dorfnetz': 116,
        '1-LV-rural1--1-no_sw': 14,
        '1-LV-rural2--1-no_sw': 96,
        '1-LV-rural3--1-no_sw': 128,
        '1-LV-semiurb4--1-no_sw': 43,
        '1-LV-semiurb5--1-no_sw': 110,
        '1-LV-urban6--1-no_sw': 58,
    }

    for testing_grid in grid_to_num_noodes.keys():
        ldf_inference_time = df_all.loc[(df_all['model'] == 'LinDistFlow') & (df_all['testing_grid'] == testing_grid), 'inference_time_ms'].values[0]
        # Add ldf inference time to XGB_Absolute, XGB_Parent, XGB_LDF models for the same testing grid
        for model in ['XGB_Absolute', 'XGB_Parent', 'XGB_LDF']:
            df_all.loc[(df_all['model'] == model) & (df_all['testing_grid'] == testing_grid), 'inference_time_ms'] += ldf_inference_time

    plot_inference_time_scaling(df_all, grid_to_num_noodes, save=args.save)
    plot_vm_boxplot(df_all, metric='rmse_vm_pu', save=args.save)
    plot_vm_boxplot(df_all, metric='rmse_va_degree', save=args.save)

    if args.save:
        print(f"\nFigures saved to directory: {FIGURES_DIR}/")