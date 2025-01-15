import wandb
#from main import sweep_config

sweep_config = {
    'method': 'bayes',  # zB 'grid', 'random', oder 'bayes'
    'metric': {
      'name': 'val_mse',   # 
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]  
        },
        'hidden_channels': {
            'values': [32, 64, 128]
        },
        'num_layers': {
            'values': [2, 3, 4]               
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
    }
}

# Sweep erstellen
sweep_id = wandb.sweep(sweep_config, project="gnn_shift_prediction")

# Agent starten:
import subprocess
subprocess.run(["wandb", "agent", f"{wandb.env.get_username()}/gnn_shift_prediction/{sweep_id}"])
