import argparse
import wandb
import torch
import random
import numpy as np

from torch_geometric.loader import DataLoader
from dataloader import MoleculeDataset
from model import GNNModel
from train import run_training

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pickle", type=str, default="data/all_graphs.pkl")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    # Die folgenden Parameter werden ggf. vom Sweep überschrieben
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--project_name", type=str, default="gnn_shift_prediction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # set global seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(project=args.project_name)

    config = wandb.config
    lr = config.learning_rate if "learning_rate" in config else args.learning_rate
    hidden_channels = config.hidden_channels if "hidden_channels" in config else args.hidden_channels
    num_layers = config.num_layers if "num_layers" in config else args.num_layers
    batch_size = config.batch_size if "batch_size" in config else args.batch_size
    epochs = args.epochs

    # Einmalige Konvertierung, falls data.pkl noch nicht existiert
    MoleculeDataset.from_pickle(args.data_pickle, args.root)
    dataset = MoleculeDataset(args.root)

    num_data = len(dataset)
    train_split = int(0.8 * num_data)
    val_split = int(0.1 * num_data)
    test_split = num_data - train_split - val_split

    train_dataset = dataset[:train_split]
    val_dataset = dataset[train_split:train_split+val_split]
    test_dataset = dataset[train_split+val_split:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_channels = dataset[0].x.size(1)
    out_channels = 2  # Shift für C und H

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.config.update({
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": device.type,
        "seed": args.seed
    }, allow_val_change=True)

    model = GNNModel(in_channels, hidden_channels, out_channels, num_layers).to(device)

    num_params = count_parameters(model)
    wandb.summary["num_trainable_params"] = num_params
    wandb.watch(model, log="all")

    # Starte Training (run_training kümmert sich ums Speichern des besten Modells)
    run_training(model, train_loader, val_loader, test_loader, epochs, device, lr)