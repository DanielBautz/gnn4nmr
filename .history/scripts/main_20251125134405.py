import random
import numpy as np
import torch
import wandb
from dataloader import create_dataloaders
from model import HeteroGNNModel
from train import train_model


def main():
    # 1) Weights & Biases init
    wandb.init(project="gnn_shift_prediction_100")
    config = wandb.config
    
    # 2) Globalen Seed definieren
    config.seed = 0  # random.randint(1, 100)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 3) Hyperparameter
    config.batch_size = 1
    config.hidden_dim = 128      # für den Encoder
    config.out_dim = 128        # für Encoder-Output und GNN
    config.num_epochs = 80
    config.lr = 2e-4

    # Neue Konfigurationsvariablen
    config.split_ratio = (0.825, 0, 0.175)
    config.encoder_dropout = 0.1
    config.gnnlayer_dropout = 0.1
    config.num_gnn_layers = 3
    config.optimizer = "Adam"   # Optionen: "Adam", "SGD", etc.
    config.weight_decay = 5e-5
    config.scheduler_factor = 0.7
    config.scheduler_patience = 15
    config.loss_weight_H = 1
    config.loss_weight_C = 0
    config.normalize_edge_features = True # wenn deaktiviert, Mittelwert = 0, Std = 1
    config.normalize_node_features = True

    # Neuer Parameter: Operator-Typ für das GNN 
    if not hasattr(config, "operator_type"):
        config.operator_type = "SAGEConv"  # Alternativen: "GCNConv", "GATConv", "SAGEConv", "GATv2Conv", "GraphConv", "NNConv", "GINEConv", "TransformerConv" 

    # Optionale zusätzliche Parameter für den Operator
    if not hasattr(config, "operator_kwargs"):
        config.operator_kwargs = {}
    if config.operator_type == "GATConv" or config.operator_type == "GATv2Conv":
        config.operator_kwargs['add_self_loops'] = False    
    
    # Neue Parameter für detaillierte Vorhersagen
    config.output_detailed_predictions = True
    config.output_dir = "results"  # Verzeichnis für die Ausgabe
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4) Dataloaders (mit split_ratio aus config)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size, 
        split_ratio=config.split_ratio,
        normalize_node_features=config.normalize_node_features,
        normalize_edge_features=config.normalize_edge_features
        
    )
    
    # 5) Modell erstellen

    in_dim_dict = {
        "H": 34,
        "C": 39,
        "Others": 16
    }
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate GNN model for shift prediction")
    parser.add_argument("--output-predictions", action="store_true", 
                        help="Output detailed predictions for test set")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to store prediction results")
                        
    args = parser.parse_args()
    
    # Initialisiere Wandb und lade die Argumente
    wandb.init(project="gnn_shift_prediction_100")
    config = wandb.config
    
    # Übertrage die Kommandozeilenargumente in die Konfiguration
    if args.output_predictions:
        config.output_detailed_predictions = True
    if args.output_dir:
        config.output_dir = args.output_dir
    
    main()