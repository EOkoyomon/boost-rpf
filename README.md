# BOOST-RPF

Repository accompanying the paper **"BOOST-RPF: Boosted Sequential Trees for Radial Power Flow"** (Paper identifier: **[TODO: arXiv/DOI]**).

BOOST-RPF reformulates radial power-flow voltage prediction from global graph regression to sequential root-to-leaf path learning. The framework evaluates three variants (Absolute Voltage, Parent Residual, and Physics-Informed Residual), with a focus on strong out-of-distribution generalization and linear scaling.

## Getting Started

### Option A: pip

```bash
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate boost-rpf
```

## Data

- **ENGAGE dataset**: Use the ENGAGE benchmark resources here: https://zenodo.org/records/15464235
  - This repository expects that **the full dataset** is downloaded and renamed to `data/ENGAGE_dataset/` (relative to this project root).
  - You should then run  
    &emsp; `python scripts/prepare_data.py`  
    to add relevant pandapower information to the data objects (needed for some models).
  - This data is used to run Experiment 2 (Heterogeneous Grids) and Experiment 3 (OOD).
- **Kerber data generation**: workflow via `scripts/graph_gen.py`.
  - The script supports synthetic grid generation and includes helpers for Kerber-style data creation.
  - This script was modified from the [ENGAGE project repo](https://gitlab.lrz.de/energy-management-technologies-public/engage/) and requires powerdata-gen library as a submodule/dependency (see https://gitlab.lrz.de/energy-management-technologies-public/engage#dependencies for details on how to set this up smoothly).  
    Note: You can alternatively just use copy the [powerdata-gen](https://github.com/bdonon/powerdata-gen) folder into this repo and do any necessary file updates to fix versioning issues.
  - To generate, run:  
    &emsp; `python scripts/graph_gen.py --size 1800 --grid kerber`  
    Then move the resulting `Kerber_Dorfnetz/` folder under `data/ENGAGE_dataset/`.
  - This data is used to run Experiment 1 (Known Grids).

Expected directory structure:
```
data/
└── ENGAGE_dataset/
    ├── Kerber_Dorfnetz/
    ├── 1-LV-rural1--1-no_sw/
    ├── 1-LV-rural2--1-no_sw/
    ├── 1-MV-comm--1-no_sw/
    └── ... (other grid configurations)
```

## Benchmark Script

The main entry point is `run_benchmark.py`. At a high level, it:

- Parses experiment configuration (dataset path, model choices, training/evaluation flags)
- Builds train/validation/test splits for selected grids
- Trains or loads analytical / neural / sequential baselines
- Runs evaluation and reports RMSE metrics and inference time
- Optionally writes result summaries and model checkpoints

### Usage

```bash
# See usage
python run_benchmark.py -h

# Example benchmark run
python run_benchmark.py --data_dir data/ENGAGE_dataset --model xgb-parent --experiment 1
```

### Reproducing paper results

Run the following commands (separated by model group) using the three independent seeds `[12, 67, 43]`.

```bash
# XGB models
python run_benchmark.py --data_dir data/ENGAGE_dataset/ --model xgb-absolute xgb-parent xgb-ldf --save_model --save_results --seed <SEED>

# ARMA-GNN
python run_benchmark.py --data_dir data/ENGAGE_dataset/ --model arma-gnn --epochs 10000 --batch_size 64 --patience 500 --lr 0.001 --save_model --save_results --seed <SEED>

# GlobalMLP
python run_benchmark.py --data_dir data/ENGAGE_dataset/ --model global-mlp --epochs 10000 --batch_size 64 --patience 500 --lr 0.000001 --save_model --save_results --seed <SEED>

# DistFlow models
python run_benchmark.py --data_dir data/ENGAGE_dataset/ --model distflow ldf --save_results --seed <SEED>
```

## `scripts/` overview

- `scripts/prepare_data.py`: utility to augment the PyG graph datasets with more grid information from pandapower.
- `scripts/graph_gen.py`: data-generation utilities for pandapower-based grids, including Kerber.
- `scripts/precompute_paths.py`: precomputes sequential path features/targets and saves them for faster loading.
- `scripts/tune_xgboost.py`: random/grid search for XGBoost-based sequential models (`xgb-absolute`, `xgb-parent`, `xgb-ldf`).
- `scripts/tune_nns.py`: random/grid search for neural baselines (`arma-gnn`, `global-mlp`).
- `scripts/make_figures.py`: generates evaluation figures (e.g., OOD boxplots, inference scaling).
- `scripts/make_tables.py`: aggregates CSV metrics and generates LaTeX tables.
- `scripts/__init__.py`: package marker for script imports.

## License

This project is released under the **Apache License 2.0**.

<!-- If you use **BOOST-RPF** in your research, please cite our paper: -->
The full paper is currently under review. If you find this code or the BOOST-RPF algorithm useful, please cite our preprint:
```
@article{okoyomon2026boostrpf,
  title={BOOST-RPF: Boosted Sequential Trees for Radial Power Flow},
  author={Okoyomon, Ehimare and Goebel, Christoph},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026},
  url={https://arxiv.org/abs/2603.xxxxx},
  eprint={2603.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
