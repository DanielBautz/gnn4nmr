import wandb
#from main import sweep_config  # Falls Sie sweep_config in main.py definieren, alternativ hier kopieren

sweep_config = {
    'method': 'bayes',  # z.B. 'grid', 'random', oder 'bayes'
    'metric': {
      'name': 'val_mse',   # Metrik, nach der optimiert wird (Val Loss o.ä.)
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]   # verschiedene LR
        },
        'hidden_channels': {
            'values': [32, 64, 128]            # verschiedene Hidden Größen
        },
        'num_layers': {
            'values': [2, 3, 4]                # verschiedene Anzahl Layer
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        # Weitere Hyperparameter können ergänzt werden
    }
}

# Sweep erstellen
sweep_id = wandb.sweep(sweep_config, project="gnn_shift_prediction")

# Agent starten:
# Dies kann auch in der Kommandozeile geschehen:
# wandb agent <username>/<projectname>/<sweep_id>

# Oder in Python:
import subprocess
subprocess.run(["wandb", "agent", f"{wandb.env.get_username()}/gnn_shift_prediction/{sweep_id}"])
