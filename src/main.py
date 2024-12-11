import argparse
import wandb
import torch
from torch_geometric.loader import DataLoader
from dataloader import MoleculeDataset
from model import GNNModel
from train import run_training

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pickle", type=str, default="data/all_graphs.pkl", help="Path to the pickle file containing the graphs.")
    parser.add_argument("--root", type=str, default="data", help="Path to store processed data.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--project_name", type=str, default="gnn_shift_prediction")
    parser.add_argument("--run_name", type=str, default="run_1")
    args = parser.parse_args()

    # Init wandb
    wandb.init(project=args.project_name, name=args.run_name)

    # Einmalige Konvertierung der originalen NetworkX-Pickle-Datei, falls noch nicht geschehen
    # MoleculeDataset.from_pickle(args.data_pickle, args.root)
    
    dataset = MoleculeDataset(args.root)
    
    # Datensatz splitten
    num_data = len(dataset)
    train_split = int(0.8 * num_data)
    val_split = int(0.1 * num_data)
    test_split = num_data - train_split - val_split
    train_dataset = dataset[:train_split]
    val_dataset = dataset[train_split:train_split+val_split]
    test_dataset = dataset[train_split+val_split:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_channels = dataset[0].x.size(1)
    out_channels = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device.type
    })

    model = GNNModel(in_channels, args.hidden_channels, out_channels, args.num_layers).to(device)

    # Anzahl der Parameter loggen
    num_params = count_parameters(model)
    wandb.summary["num_trainable_params"] = num_params

    # Modell "watchen"
    wandb.watch(model, log="all")

    run_training(model, train_loader, val_loader, test_loader, args.epochs, device)
