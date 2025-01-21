# main.py
import torch
import wandb
from dataloader import get_dataloaders
from model import HeteroGNNModel
from train import train_model

def main():
    wandb.init(project="hetero_gnn_molecules", config={
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 16,
        "hidden_dim": 64,
        "num_layers": 2
    })
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_dataloaders(root='data', batch_size=config.batch_size)

    model = HeteroGNNModel(
        c_in=38,
        h_in=32,
        o_in=6,
        e_in=4,
        hidden_dim=config.hidden_dim,
        out_dim=1,
        num_layers=config.num_layers
    ).to(device)

    model, test_loss = train_model(model, train_loader, val_loader, test_loader, config.epochs, config.lr, device)
    print("Final Test Loss:", test_loss)

    wandb.finish()

if __name__ == "__main__":
    main()
