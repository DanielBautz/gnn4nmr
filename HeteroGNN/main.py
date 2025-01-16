# main.py
import torch
import wandb
from dataloader import create_dataloaders
from model import HeteroGNNModel
from train import train_model

def main():
    wandb.init(project="gnn_shift_prediction")
    config = wandb.config
    
    # Hyper-Parameter
    config.batch_size = 4
    config.hidden_dim = 32
    config.out_dim = 16
    config.num_epochs = 10
    config.lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1) Dataloaders: train / val / test
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size
    )
    
    # 2) Modell erstellen
    example_data = next(iter(train_loader))
    in_dim_dict = {}
    for ntype in example_data.node_types:
        if example_data[ntype].x is not None:
            in_dim_dict[ntype] = example_data[ntype].x.size(-1)
        else:
            in_dim_dict[ntype] = 0
    
    model = HeteroGNNModel(in_dim_dict, hidden_dim=config.hidden_dim, out_dim=config.out_dim).to(device)
    
    # 3) Trainieren (mit train_model aus train.py)
    trained_model = train_model(model, train_loader, val_loader, test_loader, device, config)
    
    # 4) Fertig
    wandb.finish()

if __name__ == "__main__":
    main()
