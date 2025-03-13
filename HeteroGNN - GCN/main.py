import random
import numpy as np
import torch
import wandb
from dataloader import create_kfold_dataloaders, create_dataloaders
from model import HeteroGNNModel
from train import train_model

def main():
    wandb.init(project="gnn_shift_prediction")
    config = wandb.config
    
    # Globaler Seed
    config.seed = 0
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Hyperparameter
    config.batch_size = 4
    config.hidden_dim = 32
    config.out_dim = 64
    config.num_epochs = 100
    config.lr = 5e-4

    # Neue Konfigurationsvariablen
    config.split_ratio = (0.8, 0.1, 0.1)  # wird nur bei Standard-Split verwendet
    config.encoder_dropout = 0.1
    config.gnnlayer_dropout = 0.1
    config.num_gnn_layers = 2
    config.optimizer = "Adam"
    config.weight_decay = 0.0
    config.scheduler_factor = 0.5
    config.scheduler_patience = 10
    config.loss_weight_H = 10
    config.loss_weight_C = 1

    # GNN Operator
    if not hasattr(config, "operator_type"):
        config.operator_type = "GraphConv"
    if not hasattr(config, "operator_kwargs"):
        config.operator_kwargs = {}
    if config.operator_type in ["GATConv", "GATv2Conv"]:
        config.operator_kwargs['add_self_loops'] = False    
    
    # Neuer Parameter: Anzahl der Folds (k)
    config.k_folds = 1  # # if k_folds > 1, führe k-fold Cross-Validation durch. Sonst Standard-Split
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Da du die Eingabedimensionen bereits kennst, definieren wir sie direkt:
    in_dim_dict = {
        "H": 41,
        "C": 48,
        "Others": 17
    }
    
    if config.k_folds > 1:
        folds = create_kfold_dataloaders(
            batch_size=config.batch_size,
            split_ratio=config.split_ratio,
            k_folds=config.k_folds
        )
        
        test_errors = []
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds):
            print(f"\n*** Fold {fold_idx+1} von {config.k_folds} ***")
            
            # Modell direkt mit bekannten Eingabedimensionen erzeugen:
            model = HeteroGNNModel(
                in_dim_dict,
                hidden_dim=config.hidden_dim,
                out_dim=config.out_dim,
                encoder_dropout=config.encoder_dropout,
                gnnlayer_dropout=config.gnnlayer_dropout,
                num_gnn_layers=config.num_gnn_layers,
                operator_type=config.operator_type,
                operator_kwargs=config.operator_kwargs
            ).to(device)
            
            # Trainiere und erhalte Testmetriken
            model, metrics = train_model(model, train_loader, val_loader, test_loader, device, config, fold=fold_idx)
            test_errors.append(metrics)
        
        # Berechne finale Durchschnittswerte über alle Folds
        final_test_mse_H = np.mean([m["test_mse_H"] for m in test_errors])
        final_test_mae_H = np.mean([m["test_mae_H"] for m in test_errors])
        final_test_mse_C = np.mean([m["test_mse_C"] for m in test_errors])
        final_test_mae_C = np.mean([m["test_mae_C"] for m in test_errors])
        
        print("\n=== Durchschnittliche Testmetriken über alle Folds ===")
        print(f"Test MSE H: {final_test_mse_H:.4f}, Test MAE H: {final_test_mae_H:.4f}")
        print(f"Test MSE C: {final_test_mse_C:.4f}, Test MAE C: {final_test_mae_C:.4f}")
        
        # Logge die finalen Durchschnittswerte unter den gewünschten Namen
        wandb.log({
            "test_mse_H": final_test_mse_H,
            "test_mae_H": final_test_mae_H,
            "test_mse_C": final_test_mse_C,
            "test_mae_C": final_test_mae_C
        })
        
        # Erstelle zusätzlich eine Tabelle mit den Fold-spezifischen Fehlern
        fold_table = wandb.Table(columns=["Fold", "test_mse_H", "test_mae_H", "test_mse_C", "test_mae_C"])
        for idx, metrics in enumerate(test_errors):
            fold_table.add_data(idx+1, metrics["test_mse_H"], metrics["test_mae_H"], metrics["test_mse_C"], metrics["test_mae_C"])
        wandb.log({"fold_metrics_table": fold_table})
        
    else:
        # Falls kein k-fold-Modus genutzt wird, verwende den Standard-Split
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=config.batch_size,
            split_ratio=config.split_ratio
        )
        
        model = HeteroGNNModel(
            in_dim_dict,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            encoder_dropout=config.encoder_dropout,
            gnnlayer_dropout=config.gnnlayer_dropout,
            num_gnn_layers=config.num_gnn_layers,
            operator_type=config.operator_type,
            operator_kwargs=config.operator_kwargs
        ).to(device)
        
        train_model(model, train_loader, val_loader, test_loader, device, config)
    
    wandb.finish()

if __name__ == "__main__":
    main()
