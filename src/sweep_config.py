import wandb

"""
Fortgeschrittene Sweep-Konfiguration mit:
- Bedingte Parameter (abhängig von anderen Parametern)
- Parameter-Gruppierung für hierarchische Suche
- Early Termination für ineffiziente Runs
"""

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_score',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10,
        's': 2,
    },
    'parameters': {
        # Modellarchitektur
        'model_size': {
            'distribution': 'categorical',
            'values': ['small', 'medium', 'large']
        },

        # Lernraten-Strategie
        'lr_strategy': {
            'distribution': 'categorical',
            'values': ['static', 'scheduler', 'cyclic']
        },
        
        # GNN-Typ
        'operator_type': {
            'values': ['GCNConv', 'GATConv', 'GINEConv', 'GraphConv', 'TransformerConv']
        },
        
        # Datenverarbeitung
        #'normalize_edge_features': {'values': [True, False]},
        #'normalize_node_features': {'values': [True, False]},
        
        # Batch-Größe - wichtig für Speichernutzung
        'batch_size': {'values': [16]},
        
        # Regularisierung
        'encoder_dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        'gnnlayer_dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
        
        # Verlustgewichtung
        'loss_weight_H': {'values': [1, 5, 10, 15]},
        'loss_weight_C': {'values': [1, 2, 5]},
        
        # Früher Abbruch
        'early_stop_patience': {'values': [10, 20, 30]},
    },
    
    # Parameters mit Bedingungen/abhängigkeiten
    'conditions': [
        # Modellgröße beeinflusst versteckte Dimensionen
        {
            'condition': {'parameter': 'model_size', 'value': 'small'},
            'result': {
                'parameters': {
                    'hidden_dim': {'value': 16}, 
                    'out_dim': {'value': 32},
                    'num_gnn_layers': {'values': [1, 2]}
                }
            }
        },
        {
            'condition': {'parameter': 'model_size', 'value': 'medium'},
            'result': {
                'parameters': {
                    'hidden_dim': {'value': 32}, 
                    'out_dim': {'value': 64},
                    'num_gnn_layers': {'values': [2, 3]}
                }
            }
        },
        {
            'condition': {'parameter': 'model_size', 'value': 'large'},
            'result': {
                'parameters': {
                    'hidden_dim': {'value': 64}, 
                    'out_dim': {'value': 128},
                    'num_gnn_layers': {'values': [3, 4]}
                }
            }
        },
        
        # Lernraten-Strategie beeinflusst Lernrate und Scheduler
        {
            'condition': {'parameter': 'lr_strategy', 'value': 'static'},
            'result': {
                'parameters': {
                    'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
                    'scheduler_type': {'value': None}
                }
            }
        },
        {
            'condition': {'parameter': 'lr_strategy', 'value': 'scheduler'},
            'result': {
                'parameters': {
                    'lr': {'distribution': 'log_uniform_values', 'min': 5e-4, 'max': 5e-3},
                    'scheduler_type': {'value': 'ReduceLROnPlateau'},
                    'scheduler_factor': {'values': [0.3, 0.5, 0.7]},
                    'scheduler_patience': {'values': [5, 10, 15]}
                }
            }
        },
        {
            'condition': {'parameter': 'lr_strategy', 'value': 'cyclic'},
            'result': {
                'parameters': {
                    'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
                    'scheduler_type': {'value': 'CosineAnnealingLR'}
                }
            }
        },
        
        # Spezifische Parameter für bestimmte Operatoren
        {
            'condition': {'parameter': 'operator_type', 'value': 'GATConv'},
            'result': {
                'parameters': {
                    'gat_heads': {'values': [1, 2, 4, 8]},
                    'gat_dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.6}
                }
            }
        },
        {
            'condition': {'parameter': 'operator_type', 'value': 'TransformerConv'},
            'result': {
                'parameters': {
                    'transformer_heads': {'values': [1, 2, 4]},
                    'transformer_dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.6}
                }
            }
        }
    ]
}

def create_sweep():
    # Erstelle den Sweep
    sweep_id = wandb.sweep(sweep_config, project="gnn_shift_prediction_sweep")
    print(f"Sweep erstellt mit ID: {sweep_id}")
    return sweep_id

if __name__ == "__main__":
    sweep_id = create_sweep()
    print(f"Starte Sweep mit dem Befehl: wandb agent {sweep_id}")