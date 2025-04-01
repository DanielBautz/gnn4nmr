import random
import numpy as np
import torch
import wandb
import os
import json
import argparse
from dataloader import create_kfold_dataloaders
from model import HeteroGNNModel
# Importiere die neue train_model Funktion mit Early Stopping
from train_with_early_stopping import train_model

def load_config(config_path):
    """Lädt die beste Konfiguration aus einer JSON-Datei"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, path):
    """Speichert die Konfiguration in einer JSON-Datei"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Trainiert das beste Modell mit der optimierten Konfiguration')
    parser.add_argument('--config', type=str, default='best_config.json', help='Pfad zur besten Konfiguration')
    parser.add_argument('--sweep_id', type=str, help='W&B Sweep ID zum Laden der besten Konfiguration')
    parser.add_argument('--project', type=str, default='gnn_shift_prediction_sweep', help='W&B Projektname')
    args = parser.parse_args()

    # Stelle sicher, dass model-Ordner existiert
    os.makedirs("model", exist_ok=True)
    
    # Lade die beste Konfiguration
    if args.sweep_id:
        # Hole beste Konfiguration aus W&B Sweep
        api = wandb.Api()
        sweep = api.sweep(f"{args.project}/{args.sweep_id}")
        best_run = sweep.best_run()
        config_dict = best_run.config
        
        # Speichere die beste Konfiguration
        save_config(config_dict, 'best_config.json')
        print(f"Beste Konfiguration aus Sweep {args.sweep_id} geladen und gespeichert in best_config.json")
    else:
        # Lade Konfiguration aus Datei
        config_dict = load_config(args.config)
        print(f"Konfiguration aus {args.config} geladen")
    
    # Initialisiere W&B für das finale Training
    run = wandb.init(project=args.project, name="best_model_final", config=config_dict)
    config = wandb.config
    
    # Setze Seed für Reproduzierbarkeit
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setze konstante Werte
    config.seed = seed
    config.num_epochs = 100
    config.split_ratio = (0.8, 0.1, 0.1)
    config.k_folds = 5
    
    # Geräteerkennung
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CUDA-Optimierungen
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = config.benchmark
        torch.backends.cudnn.deterministic = True
        
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # K-Fold Cross Validation
    test_metrics_across_folds = {
        'test_mse_H': [], 'test_mae_H': [], 
        'test_mse_C': [], 'test_mae_C': [],
        'val_score': []
    }
    
    for fold_idx in range(config.k_folds):
        print(f"\n====== FOLD {fold_idx+1}/{config.k_folds} ======")
        
        # Erstelle Dataloaders
        train_loader, val_loader, test_loader = create_kfold_dataloaders(
            batch_size=int(config.batch_size),
            n_folds=config.k_folds,
            fold_idx=fold_idx,
            split_ratio=config.split_ratio,
            normalize_node_features=config.normalize_node_features,
            normalize_edge_features=config.normalize_edge_features,
            num_workers=int(config.num_workers),
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers
        )
        
        # Hole Feature-Dimensionen
        example_data = next(iter(train_loader))
        in_dim_dict = {}
        for ntype in example_data.node_types:
            if example_data[ntype].x is not None:
                in_dim_dict[ntype] = example_data[ntype].x.size(-1)
                print(f"Node type {ntype} has input dimension {in_dim_dict[ntype]}")
            else:
                in_dim_dict[ntype] = 0
        
        # Operator-spezifische Konfiguration
        operator_kwargs = {}
        if config.operator_type == "GATConv" or config.operator_type == "GATv2Conv":
            operator_kwargs['add_self_loops'] = False
            
        # Initialisiere Modell
        model = HeteroGNNModel(
            in_dim_dict, 
            hidden_dim=int(config.hidden_dim),
            out_dim=int(config.out_dim),
            encoder_dropout=float(config.encoder_dropout),
            gnnlayer_dropout=float(config.gnnlayer_dropout),
            num_gnn_layers=int(config.num_gnn_layers),
            operator_type=config.operator_type,
            operator_kwargs=operator_kwargs,
            edge_in_dim=10
        ).to(device)
        
        # Modellpfad
        model_path = f"model/best_model_fold_{fold_idx}_final.pt"
        
        # Trainiere das Modell
        trained_result = train_model(
            model, 
            train_loader, 
            val_loader, 
            test_loader, 
            device, 
            config,
            model_path=model_path,
            fold_idx=fold_idx
        )
        
        # Evaluiere und sammle Metriken
        test_mse_H, test_mae_H, test_mse_C, test_mae_C, val_score = trained_result['test_metrics']
        test_metrics_across_folds['test_mse_H'].append(test_mse_H)
        test_metrics_across_folds['test_mae_H'].append(test_mae_H)
        test_metrics_across_folds['test_mse_C'].append(test_mse_C)
        test_metrics_across_folds['test_mae_C'].append(test_mae_C)
        test_metrics_across_folds['val_score'].append(val_score)
        
        # Logge Metriken
        wandb.log({
            f"fold_{fold_idx}_test_mse_H": test_mse_H,
            f"fold_{fold_idx}_test_mae_H": test_mae_H,
            f"fold_{fold_idx}_test_mse_C": test_mse_C,
            f"fold_{fold_idx}_test_mae_C": test_mae_C,
            f"fold_{fold_idx}_val_score": val_score
        })
        
        # Cache leeren
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Berechne Durchschnitt und Standardabweichung
    avg_metrics = {}
    std_metrics = {}
    for metric_name, values in test_metrics_across_folds.items():
        avg_metrics[f"avg_{metric_name}"] = np.mean(values)
        std_metrics[f"std_{metric_name}"] = np.std(values)
    
    # Sammle Early Stopping Statistiken
    early_stopping_stats = {
        'early_stopped': [result.get('early_stopped', False) for result in test_metrics_across_folds],
        'epochs_trained': [result.get('epochs_trained', config.num_epochs) for result in test_metrics_across_folds]
    }
    
    # Berechne durchschnittliche Trainingszeiten
    avg_epochs = np.mean(early_stopping_stats['epochs_trained'])
    
    # Logge zusammenfassende Metriken
    wandb.log({
        **avg_metrics, 
        **std_metrics,
        'avg_epochs_trained': avg_epochs,
        'early_stopped_folds': sum(early_stopping_stats['early_stopped'])
    })
    
    # Gib Zusammenfassung aus
    print("\n====== FINAL MODEL PERFORMANCE SUMMARY ======")
    print(f"Average Test MSE H: {avg_metrics['avg_test_mse_H']:.4f} ± {std_metrics['std_test_mse_H']:.4f}")
    print(f"Average Test MAE H: {avg_metrics['avg_test_mae_H']:.4f} ± {std_metrics['std_test_mae_H']:.4f}")
    print(f"Average Test MSE C: {avg_metrics['avg_test_mse_C']:.4f} ± {std_metrics['std_test_mse_C']:.4f}")
    print(f"Average Test MAE C: {avg_metrics['avg_test_mae_C']:.4f} ± {std_metrics['std_test_mae_C']:.4f}")
    print(f"Average Val Score: {avg_metrics['avg_val_score']:.4f} ± {std_metrics['std_val_score']:.4f}")
    print(f"Average Epochs Trained: {avg_epochs:.1f}")
    print(f"Early Stopping aktiviert in {sum(early_stopping_stats['early_stopped'])}/{config.k_folds} Folds")
    
    # Speichere finale Ergebnisse
    final_results = {
        "config": config_dict,
        "metrics": avg_metrics,
        "std_metrics": std_metrics
    }
    with open("final_model_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\nFinale Ergebnisse gespeichert in final_model_results.json")

if __name__ == "__main__":
    main()