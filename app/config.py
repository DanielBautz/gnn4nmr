"""Central paths, defaults and device resolution for the web app."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
SDF_PRIMARY_DIR = os.path.join(DATA_DIR, "orca_xyz_formate")
SDF_FALLBACK_DIR = os.path.join(DATA_DIR, "converted_sdf_files")

DEFAULT_DATASET_FILE = "all_graphs_with_length_filtered.pkl"

EDGE_IN_DIM = 10

OPERATOR_TYPES = [
    "SAGEConv",
    "GraphConv",
    "GCNConv",
    "GATConv",
    "GATv2Conv",
    "TransformerConv",
    "GINEConv",
    "NNConv",
]

TRAINING_DEFAULTS = {
    "model_name": "",
    "dataset_file": "",  # empty -> active dataset
    "operator_type": "SAGEConv",
    "hidden_dim": 128,
    "out_dim": 128,
    "num_gnn_layers": 3,
    "encoder_dropout": 0.1,
    "gnnlayer_dropout": 0.1,
    "lr": 2e-4,
    "weight_decay": 5e-5,
    "batch_size": 4,
    "num_epochs": 80,
    "seed": 0,
    "split_ratio": [0.8, 0.1, 0.1],
    "loss_weight_H": 10.0,
    "loss_weight_C": 1.0,
    "scheduler_factor": 0.7,
    "scheduler_patience": 15,
    "early_stopping_patience": 10,
}


def resolve_device():
    import torch

    env_device = os.environ.get("GNN4NMR_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
