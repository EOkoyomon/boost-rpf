import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json
import numpy as np
import argparse

# Add project root to sys.path for imports
import sys
from pathlib import Path
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from run_benchmark import MODEL_CLASSES

def count_xgb_params(wrapper, count_feature_index=True, include_scalers=False):
    booster = wrapper.model.get_booster()
    cfg = json.loads(booster.save_raw(raw_format="json"))

    learner = cfg["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    n_targets = int(learner["learner_model_param"].get("num_target", 1))

    n_trees = len(trees)
    n_splits = n_leaves = n_leaf_values = 0
    for t in trees:
        is_leaf = np.asarray(t["left_children"]) == -1
        # Leaf vector size: 1 for single-target trees (0 in old versions), n_targets for multi-output
        k = max(int(t["tree_param"].get("size_leaf_vector", 1)), 1)
        n_leaves += int(is_leaf.sum())
        n_splits += int((~is_leaf).sum())
        n_leaf_values += int(is_leaf.sum()) * k

    per_split = 2 if count_feature_index else 1   # threshold (+ feature index)
    total = n_splits * per_split + n_leaf_values + n_targets  # + base_score per target

    if include_scalers and wrapper.normalize:
        for s in (wrapper.covariate_scaler, wrapper.target_scaler):
            total += s.mean_.size + s.scale_.size

    return {
        "trees": n_trees,
        "split_nodes": n_splits,
        "leaves": n_leaves,
        "leaf_values": n_leaf_values,
        "total_params": total,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to the XGB model file (joblib format).")
parser.add_argument("--class_name", type=str, default="xgb-absolute", choices=MODEL_CLASSES.keys(), help="Name of the model class to use for loading the model.")
args = parser.parse_args()

model = MODEL_CLASSES[args.class_name].load(args.model)
print(count_xgb_params(model)["total_params"])