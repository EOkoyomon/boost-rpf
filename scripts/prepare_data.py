import os
import argparse
import pandas as pd
import pandapower as pp
import torch
from tqdm import tqdm

# Add project root to sys.path for imports
import sys
from pathlib import Path
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from utils.training_utils import get_lv_grid_codes

def get_pp_sources(output_base_dir, grid_type):
    """Get the list of solved source networks for a specific grid type.
    Args:
        output_base_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of sb grid (e.g., '1-LV-rural1--0-no_sw', '1-MV-urban--1-no_sw', etc.)
    """
    dataset_source_path = os.path.join(output_base_dir, grid_type, 'train', 'dataset_src.csv')
    sources_absolute = pd.read_csv(dataset_source_path, index_col=0)['src'].to_list()
    sources = [os.path.join(output_base_dir, grid_type, 'train', src.split('/')[-1]) for src in sources_absolute]
    return sources

def prepare_pyg_graphs(data_dir, grid_type):
    """
    Load PyTorch Geometric graphs and augment them with extra pandapower network information.
    Args:
        data_dir (str): Base directory where datasets are stored.
        grid_type (str): The type of sb grid (e.g., '1-LV-rural1--0-no_sw', '1-MV-urban--1-no_sw', etc.)

    Returns:
        list of torch_geometric.data.Data: List of PyTorch Geometric Data objects.
    """
    dataset_path = os.path.join(data_dir, grid_type, 'train', 'dataset.pt')
    pyg_dataset = torch.load(dataset_path, weights_only=False)
    sources = get_pp_sources(data_dir, grid_type)
    for data, src in tqdm(zip(pyg_dataset, sources)):
        net = pp.from_json(src)
        pp.runpp(net, verbose=False)
        ppci = net["_ppc"]["internal"]
        data.ppci = ppci
    return pyg_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare PyTorch Geometric graph datasets with pandapower information.")
    parser.add_argument("--data_dir", type=str, default="data/ENGAGE_dataset/", help="Base directory where datasets are stored.")
    args = parser.parse_args()
    data_dir = args.data_dir
    training_grids = get_lv_grid_codes(scenario=1)

    for grid in training_grids:
        print(grid)
        dataset = prepare_pyg_graphs(data_dir, grid)
        filename = os.path.join(data_dir, grid, 'train', 'dataset_with_ppci.pt')
        torch.save(dataset, filename)