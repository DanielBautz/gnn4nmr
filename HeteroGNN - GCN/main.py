import random
import numpy as np
import torch
import wandb
from dataloader import create_dataloaders
from model import HeteroGNNModel
from train import train_model
from explainer import explain_nodes
import os


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
    
    # 3) Hyperparameter - Zurück zu ursprünglichen Werten für bessere Genauigkeit
    config.batch_size = 4  # Original batch size for better accuracy
    config.hidden_dim = 32  # für den Encoder
    config.out_dim = 64     # für Encoder-Output und GNN
    config.num_epochs = 100
    config.lr = 5e-4

    # Neue Konfigurationsvariablen
    config.split_ratio = (0.8, 0.1, 0.1)
    config.encoder_dropout = 0.1
    config.gnnlayer_dropout = 0.1
    config.num_gnn_layers = 2
    config.optimizer = "Adam"  # Optionen: "Adam", "SGD", etc.
    config.weight_decay = 0.0
    config.scheduler_factor = 0.5
    config.scheduler_patience = 10
    config.loss_weight_H = 10
    config.loss_weight_C = 1
    config.normalize_edge_features = False  # wenn deaktiviert, Mittelwert = 0, Std = 1
    config.normalize_node_features = True

    # Performance-Optimierungen mit angepassten Werten für bessere Genauigkeit
    config.use_mixed_precision = False  # Deaktiviere Mixed Precision für bessere Genauigkeit
    config.gradient_accumulation_steps = 1  # Deaktiviere Gradient Accumulation
    
    # DataLoader-Optimierungen
    num_cpu_cores = os.cpu_count() or 1
    # Reduziere Anzahl der Worker für stabileres Training
    config.num_workers = min(2, num_cpu_cores)
    config.pin_memory = True
    # Persistent workers nur aktivieren wenn worker > 0
    config.persistent_workers = config.num_workers > 0
    
    # Neuer Parameter: Operator-Typ für das GNN 
    if not hasattr(config, "operator_type"):
        config.operator_type = "GINEConv"  # Alternativen: "GCNConv", "GATConv", "SAGEConv", "GATv2Conv", "GraphConv", "NNConv", "GINEConv", "TransformerConv" 

    # Optionale zusätzliche Parameter für den Operator
    if not hasattr(config, "operator_kwargs"):
        config.operator_kwargs = {}
    if config.operator_type == "GATConv" or config.operator_type == "GATv2Conv":
        config.operator_kwargs['add_self_loops'] = False
    
    # Aktiviere cuDNN Benchmarking für optimale Performance, wenn nicht im Debug-Modus
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4) Dataloaders (mit optimierten Parametern)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size, 
        split_ratio=config.split_ratio,
        normalize_node_features=config.normalize_node_features,
        normalize_edge_features=config.normalize_edge_features,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    # Speichernutzung optimieren
    torch.cuda.empty_cache()
    
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
        edge_in_dim=10,
        use_activation_checkpointing=False  # Deaktiviere für bessere Genauigkeit
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
    
    # Clean up
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()