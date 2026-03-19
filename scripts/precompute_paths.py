"""
Pre-compute path data for sequential models and save to disk.

This script processes all grids and saves the computed path features/targets
as numpy arrays, which can be quickly loaded and converted to TimeSeries later.

Usage:
    python precompute_paths.py --data_dir data/ENGAGE_dataset/
"""

import argparse
import os
import pickle

from tqdm import tqdm

# Add project root to sys.path for imports
import sys
from pathlib import Path
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from utils.data_utils import get_grid_paths
from utils.training_utils import get_lv_grid_codes



def precompute_and_save_paths(data_dir, grid_type, slack_vm_pu=1.025, slack_va_degree=-150.0):
    """
    Pre-compute path data and save as numpy arrays (without TimeSeries objects).
    
    Saves to: {data_dir}/{grid_type}/train/dataset_sequential.pkl
    
    The saved format stores raw numpy arrays which are fast to load.
    TimeSeries objects are created on-demand when loading.
    """ 
    # Load and prepare all samples
    all_samples = get_grid_paths(data_dir, grid_type, slack_vm_pu, slack_va_degree)
    
    # Feature and target column names (for reference when loading)
    feature_names = [
        'r_ij', 'x_ij', 
        'P_j', 'Q_j', 'P_agg_j', 'Q_agg_j',
        'V_LDF_j', 'theta_LDF_j'
    ]
    target_names = ['V_j', 'theta_j']

    # Save to pickle file
    output_path = os.path.join(data_dir, grid_type, 'train', 'dataset_sequential.pkl')
    
    # Store metadata alongside the data
    save_data = {
        'feature_names': feature_names,
        'target_names': target_names,
        'slack_vm_pu': slack_vm_pu,
        'slack_va_degree': slack_va_degree,
        'samples': all_samples,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved {len(all_samples)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute path data for sequential models")
    parser.add_argument("--data_dir", type=str, default="data/ENGAGE_dataset/",
                        help="Base directory containing grid datasets")
    parser.add_argument("--grids", type=str, nargs="+", default=None,
                        help="Specific grids to process (default: all scenario 1 grids)")
    args = parser.parse_args()
    
    # Get grids to process
    if args.grids:
        grids = args.grids
    else:
        grids = get_lv_grid_codes(scenario=1)
    
    print(f"Pre-computing paths for {len(grids)} grids...")
    
    for grid in tqdm(grids, desc="Grids"):
        precompute_and_save_paths(args.data_dir, grid)
    
    print("\nDone! Pre-computed path data saved.")


if __name__ == '__main__':
    main()
