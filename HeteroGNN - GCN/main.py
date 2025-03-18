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
    config.seed = 0  # random.randint(1, 100)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 3) Hyperparameter
    config.batch_size = 4
    config.hidden_dim = 32      # für den Encoder
    config.out_dim = 64         # für Encoder-Output und GNN
    config.num_epochs = 100
    config.lr = 5e-4

    # Neue Konfigurationsvariablen
    config.split_ratio = (0.8, 0.1, 0.1)
    config.encoder_dropout = 0.1
    config.gnnlayer_dropout = 0.1
    config.num_gnn_layers = 2
    config.optimizer = "Adam"   # Optionen: "Adam", "SGD", etc.
    config.weight_decay = 0.0
    config.scheduler_factor = 0.5
    config.scheduler_patience = 10
    config.loss_weight_H = 10
    config.loss_weight_C = 1
    config.normalize_edge_features = False
    config.normalize_node_features = True

    # Neuer Parameter: Operator-Typ für das GNN 
    if not hasattr(config, "operator_type"):
        config.operator_type = "GINEConv"  # Alternativen: "GCNConv", "GATConv", "SAGEConv", "GATv2Conv", "GraphConv", "NNConv", "GINEConv" 

    # Optionale zusätzliche Parameter für den Operator
    if not hasattr(config, "operator_kwargs"):
        config.operator_kwargs = {}
    if config.operator_type == "GATConv" or config.operator_type == "GATv2Conv":
        config.operator_kwargs['add_self_loops'] = False    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4) Dataloaders (mit split_ratio aus config)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size, 
        split_ratio=config.split_ratio,
        normalize_node_features=config.normalize_node_features,
        normalize_edge_features=config.normalize_edge_features
        
    )
    
    # 5) Modell erstellen
    example_data = next(iter(train_loader))
    in_dim_dict = {}
    for ntype in example_data.node_types:
        if example_data[ntype].x is not None:
            in_dim_dict[ntype] = example_data[ntype].x.size(-1)
            print(f"Node type {ntype} has input dimension {in_dim_dict[ntype]}")
        else:
            in_dim_dict[ntype] = 0
    
    model = HeteroGNNModel(
        in_dim_dict, 
        hidden_dim=config.hidden_dim, 
        out_dim=config.out_dim,
        encoder_dropout=config.encoder_dropout,
        gnnlayer_dropout=config.gnnlayer_dropout,
        num_gnn_layers=config.num_gnn_layers,
        operator_type=config.operator_type,
        operator_kwargs=config.operator_kwargs,
        edge_in_dim=10
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
