import argparse
import wandb
import torch
from torch_geometric.loader import DataLoader
from dataloader import MoleculeDataset
from model import GNNModel
from train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pickle", type=str, default="all_graphs.pkl", help="Path to the pickle file containing the graphs.")
    parser.add_argument("--root", type=str, default="./data", help="Path to store processed data.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--project_name", type=str, default="gnn_molecule_regression")
    args = parser.parse_args()

    # Init wandb
    wandb.init(project=args.project_name)

    # Datensatz laden oder konvertieren
    # Wenn noch nicht konvertiert, einmalig:
    # MoleculeDataset.from_pickle(args.data_pickle, args.root)
   # Einmalige Konvertierung der originalen NX-Graph-Pickle-Datei zu PyG-Format
    MoleculeDataset.from_pickle("data/all_graphs.pkl", "data")

    # Anschließend kann das Dataset wie gewohnt geladen werden
    dataset = MoleculeDataset("data")
        
    # Datensatz splitten
    # Hier ein einfacher Split. In der Realität besser einen reproduzierbaren Split verwenden.
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

    # Feature-Dimension bestimmen: in_channels = dataset[0].x.shape[1]
    in_channels = dataset[0].x.size(1)
    # Wir wollen shift_high-low, also out_channels=1
    out_channels = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(in_channels, args.hidden_channels, out_channels, args.num_layers).to(device)

    run_training(model, train_loader, val_loader, test_loader, args.epochs, device)
