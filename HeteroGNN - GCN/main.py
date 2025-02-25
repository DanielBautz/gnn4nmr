# main.py
import random
import numpy as np
import torch
import wandb
from dataloader import create_dataloaders
from model import HeteroGNNModel
from train import train_model

def main():
    # 1) Weights & Biases init
    wandb.init(project="gnn_shift_prediction")
    config = wandb.config
    
    # 2) Globalen Seed definieren
    config.seed = 0 #random.randint(1, 100)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 3) Hyperparameter
    config.batch_size = 4
    config.hidden_dim = 32 # for encoder
    config.out_dim = 64 # for encoder output and GNN
    config.num_epochs = 100
    config.lr = 5e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4) Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size
    )
    
    # 5) Modell erstellen
    example_data = next(iter(train_loader))
    in_dim_dict = {}
    for ntype in example_data.node_types:
        if example_data[ntype].x is not None:
            in_dim_dict[ntype] = example_data[ntype].x.size(-1)
        else:
            in_dim_dict[ntype] = 0
    
    model = HeteroGNNModel(
        in_dim_dict, 
        hidden_dim=config.hidden_dim, 
        out_dim=config.out_dim
    ).to(device)
    
    # 6) Trainieren
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        device, 
        config
    )
    
    # 7) Beenden
    wandb.finish()

if __name__ == "__main__":
    main()
