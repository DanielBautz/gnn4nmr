import random
import numpy as np
import torch
import wandb
from dataloader import create_dataloaders
from model import HeteroGNNModel
from train import train_model
from explainer import explain_nodes
import copy
import pandas as pd
import os


def main():
    # 1) Weights & Biases init
    wandb.init(project="gnn_shift_prediction_lokal")
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
    config.loss_weight_H = 0
    config.loss_weight_C = 1
    config.normalize_edge_features = False # wenn deaktiviert, Mittelwert = 0, Std = 1
    config.normalize_node_features = True

    # Neuer Parameter: Operator-Typ für das GNN 
    if not hasattr(config, "operator_type"):
        config.operator_type = "GINEConv"  # Alternativen: "GCNConv", "GATConv", "SAGEConv", "GATv2Conv", "GraphConv", "NNConv", "GINEConv", "TransformerConv" 

    # Optionale zusätzliche Parameter für den Operator
    if not hasattr(config, "operator_kwargs"):
        config.operator_kwargs = {}
    if config.operator_type == "GATConv" or config.operator_type == "GATv2Conv":
        config.operator_kwargs['add_self_loops'] = False
        
    # Neuer Parameter: k-fold Cross-Validation
    if not hasattr(config, "k_folds"):
        config.k_folds = 10  # Standard ist 1 (keine Cross-Validation)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Überprüfen, ob die erforderlichen Ordnerstrukturen existieren
    if not os.path.exists('models'):
        os.makedirs('models')
    
    if config.k_folds == 1:
        # Standard-Training ohne Cross-Validation
        # 4) Dataloaders (mit split_ratio aus config)
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=config.batch_size, 
            split_ratio=config.split_ratio,
            normalize_node_features=config.normalize_node_features,
            normalize_edge_features=config.normalize_edge_features,
            random_seed=config.seed
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
        trained_model, _, _ = train_model(
            model, 
            train_loader, 
            val_loader, 
            test_loader, 
            device, 
            config
        )
    else:
        # K-Fold Cross-Validation
        fold_results = {'val_scores': [], 'test_scores': []}
        
        # Lade einen Batch, um die Feature-Dimensionen zu erhalten
        # (wir müssen dies nur einmal tun)
        temp_train_loader, _, _ = create_dataloaders(
            batch_size=config.batch_size,
            k_folds=config.k_folds,
            fold_idx=0,
            random_seed=config.seed
        )
        example_data = next(iter(temp_train_loader))
        in_dim_dict = {}
        for ntype in example_data.node_types:
            if example_data[ntype].x is not None:
                in_dim_dict[ntype] = example_data[ntype].x.size(-1)
                print(f"Node type {ntype} has input dimension {in_dim_dict[ntype]}")
            else:
                in_dim_dict[ntype] = 0
        
        for fold_idx in range(config.k_folds):
            print(f"\n{'='*20} Fold {fold_idx+1}/{config.k_folds} {'='*20}")
            
            # Erstelle Dataloaders für diesen Fold
            train_loader, val_loader, test_loader = create_dataloaders(
                batch_size=config.batch_size,
                normalize_node_features=config.normalize_node_features,
                normalize_edge_features=config.normalize_edge_features,
                k_folds=config.k_folds, 
                fold_idx=fold_idx,
                random_seed=config.seed
            )
            
            # Erstelle ein frisches Modell für jeden Fold
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
            
            # Trainiere das Modell für diesen Fold
            trained_model, val_score, test_score = train_model(
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                config,
                fold_idx=fold_idx
            )
            
            # Speichere das beste Modell für diesen Fold
            torch.save(trained_model.state_dict(), f"models/best_model_fold_{fold_idx}.pt")
            
            # Sammle Ergebnisse
            fold_results['val_scores'].append(val_score)
            fold_results['test_scores'].append(test_score)
        
        # Berechne und logge die durchschnittlichen Scores über alle Folds
        avg_val_score = sum(fold_results['val_scores']) / len(fold_results['val_scores'])
        avg_test_score = sum(fold_results['test_scores']) / len(fold_results['test_scores'])
        
        std_val_score = np.std(fold_results['val_scores'])
        std_test_score = np.std(fold_results['test_scores'])
        
        wandb.log({
            "avg_val_score": avg_val_score,
            "avg_test_score": avg_test_score,
            "std_val_score": std_val_score,
            "std_test_score": std_test_score
        })
        
        print("\n" + "="*50)
        print(f"k-fold Cross-Validation Results (k={config.k_folds}):")
        print(f"Average Validation Score: {avg_val_score:.4f} ± {std_val_score:.4f}")
        print(f"Average Test Score: {avg_test_score:.4f} ± {std_test_score:.4f}")
        print("="*50)
        
        # Optional: Wähle das beste Modell aus allen Folds
        best_fold_idx = np.argmin(fold_results['val_scores'])
        print(f"Best model was from fold {best_fold_idx+1} with validation score {fold_results['val_scores'][best_fold_idx]:.4f}")
        
        # Lade das beste Modell
        best_model = HeteroGNNModel(
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
        
        best_model.load_state_dict(torch.load(f"models/best_model_fold_{best_fold_idx}.pt"))
        # Speichere das beste Modell als "best_model.pt"
        torch.save(best_model.state_dict(), "best_model.pt")
        
        trained_model = best_model
    
    # 7) Beenden
    wandb.finish()

    # Wenn explainer.py implementiert ist, kann Folgendes verwendet werden:
    # Angenommen, test_loader liefert HeteroData-Batches:
    #test_batch = next(iter(test_loader))
    # Wähle beispielsweise Knoten des Typs "H" aus dem Testbatch:
    #node_indices_to_explain = [0, 1, 2, 3]  # Passe die Indizes an deine Bedürfnisse an
    # Erkläre die ausgewählten Knoten (z. B. mit 100 Epochen für den Explainer)
    #explanations = explain_nodes(trained_model, test_batch, "H", node_indices_to_explain, epochs=100)

if __name__ == "__main__":
    main()