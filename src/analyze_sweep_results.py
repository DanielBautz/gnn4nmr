import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def fetch_sweep_runs(sweep_id, project="gnn_shift_prediction"):
    """Lädt alle Runs eines Sweeps in ein Pandas DataFrame."""
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    runs = sweep.runs
    
    # Sammle Daten für DataFrame
    runs_data = []
    for run in runs:
        # Basisinfos
        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url
        }
        
        # Füge alle config Parameter hinzu
        for key, value in run.config.items():
            if key not in run_data:
                run_data[f"config/{key}"] = value
        
        # Füge summary Metriken hinzu
        for key, value in run.summary.items():
            if key not in run_data and not key.startswith("_"):
                run_data[f"metrics/{key}"] = value
        
        runs_data.append(run_data)
    
    # Erstelle DataFrame
    df = pd.DataFrame(runs_data)
    return df

def plot_parameter_importance(df, target_metric="metrics/val_score"):
    """Visualisiert die Wichtigkeit der Hyperparameter für die Zielmetrik."""
    # Extrahiere nur Konfigurationsparameter
    config_cols = [col for col in df.columns if col.startswith("config/")]
    
    # Berechne Korrelation mit der Zielmetrik
    corr_data = []
    for col in config_cols:
        if df[col].dtype in [np.int64, np.float64]:
            corr = df[col].corr(df[target_metric])
            if not np.isnan(corr):
                corr_data.append((col.replace("config/", ""), abs(corr)))
    
    # Sortiere nach Korrelationsstärke
    corr_data.sort(key=lambda x: x[1], reverse=True)
    
    # Plotte die Top-10
    plt.figure(figsize=(10, 6))
    params = [x[0] for x in corr_data[:10]]
    importance = [x[1] for x in corr_data[:10]]
    
    sns.barplot(x=importance, y=params)
    plt.title(f"Top-10 Parameter nach Einfluss auf {target_metric}")
    plt.xlabel("Absolute Korrelation")
    plt.tight_layout()
    plt.savefig("parameter_importance.png")
    print(f"Parameter importance plot saved to parameter_importance.png")

def plot_parallel_coordinates(df, target_metric="metrics/val_score", top_n=20):
    """Erstellt einen Parallel Coordinates Plot für die besten Runs."""
    # Wähle die besten n Runs
    df_sorted = df.sort_values(by=target_metric)
    df_top = df_sorted.head(top_n)
    
    # Wähle numerische Parameter und die Zielmetrik
    numeric_cols = []
    for col in df_top.columns:
        if col.startswith("config/") and df_top[col].dtype in [np.int64, np.float64]:
            numeric_cols.append(col)
    
    plot_cols = numeric_cols + [target_metric]
    df_plot = df_top[plot_cols].copy()
    
    # Normalisiere Daten für bessere Darstellung
    for col in plot_cols:
        df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())
    
    # Plotte
    plt.figure(figsize=(14, 8))
    pd.plotting.parallel_coordinates(
        df_plot, 
        target_metric, 
        colormap=plt.cm.coolwarm
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Parallel Coordinates der Top {top_n} Runs")
    plt.tight_layout()
    plt.savefig("parallel_coordinates.png")
    print(f"Parallel coordinates plot saved to parallel_coordinates.png")

def plot_learning_curves(df, target_metric="metrics/val_score", top_n=5):
    """Zeigt die Lernkurven der besten Runs."""
    api = wandb.Api()
    
    # Sortiere nach der Zielmetrik und nimm die besten n
    df_sorted = df.sort_values(by=target_metric)
    top_runs = df_sorted.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    for _, row in top_runs.iterrows():
        run = api.run(f"{row['id']}")
        history = run.scan_history(keys=["val_mae_H", "val_mae_C", "epoch"])
        
        # Konvertiere zu DataFrame für einfachere Handhabung
        history_df = pd.DataFrame(history)
        
        if not history_df.empty:
            plt.plot(history_df["epoch"], history_df["val_mae_H"], 
                     label=f"H-MAE: {row['name']}")
            plt.plot(history_df["epoch"], history_df["val_mae_C"], 
                     label=f"C-MAE: {row['name']}", linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title(f"Lernkurven der Top {top_n} Modelle")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    print(f"Learning curves plot saved to learning_curves.png")

def print_best_configs(df, target_metric="metrics/val_score", top_n=3):
    """Gibt die Konfigurationen der besten Runs aus."""
    df_sorted = df.sort_values(by=target_metric)
    top_runs = df_sorted.head(top_n)
    
    print(f"\n=== Top {top_n} Konfigurationen basierend auf {target_metric} ===\n")
    
    for i, (_, row) in enumerate(top_runs.iterrows()):
        print(f"Rank {i+1}: Run '{row['name']}' ({row['id']})")
        print(f"  {target_metric}: {row[target_metric]:.4f}")
        
        if 'metrics/test_mae_H' in row:
            print(f"  Test MAE H: {row['metrics/test_mae_H']:.4f}")
        if 'metrics/test_mae_C' in row:
            print(f"  Test MAE C: {row['metrics/test_mae_C']:.4f}")
        
        print("  Konfiguration:")
        for col in sorted([c for c in row.index if c.startswith("config/")]):
            param_name = col.replace("config/", "")
            print(f"    {param_name}: {row[col]}")
        print()
    
    # Speichere die beste Konfiguration als Python-Datei
    if len(top_runs) > 0:
        best_run = top_runs.iloc[0]
        with open("best_config.py", "w") as f:
            f.write("# Beste Hyperparameter-Konfiguration aus Sweep\n\n")
            f.write("best_config = {\n")
            for col in sorted([c for c in best_run.index if c.startswith("config/")]):
                param_name = col.replace("config/", "")
                value = best_run[col]
                
                # Format basierend auf Datentyp
                if isinstance(value, str):
                    f.write(f"    '{param_name}': '{value}',\n")
                else:
                    f.write(f"    '{param_name}': {value},\n")
            f.write("}\n")
        print("Beste Konfiguration gespeichert in best_config.py")

def main():
    parser = argparse.ArgumentParser(description='Analyse der W&B Sweep Ergebnisse')
    parser.add_argument('--sweep-id', type=str, required=True, help='Sweep ID from wandb')
    parser.add_argument('--project', type=str, default="gnn_shift_prediction", help='W&B Projektname')
    parser.add_argument('--metric', type=str, default="metrics/val_score", help='Zielmetrik für die Optimierung')
    parser.add_argument('--top-n', type=int, default=5, help='Anzahl der Top-Runs für die Analyse')
    args = parser.parse_args()
    
    print(f"Analysiere Sweep {args.sweep_id} im Projekt {args.project}...")
    
    # API Token einrichten
    try:
        api = wandb.Api()
    except:
        print("Bitte zunächst mit 'wandb login' einloggen")
        return
    
    # Lade Sweep-Daten
    print("Lade Runs aus dem Sweep...")
    df = fetch_sweep_runs(args.sweep_id, args.project)
    
    if df.empty:
        print("Keine Runs gefunden!")
        return
    
    print(f"Gefunden: {len(df)} Runs")
    
    # Führe Analysen durch
    print("\nGeneriere Plots und Analysen...")
    plot_parameter_importance(df, args.metric)
    plot_parallel_coordinates(df, args.metric, args.top_n)
    plot_learning_curves(df, args.metric, args.top_n)
    print_best_configs(df, args.metric, args.top_n)
    
    print("\nAnalyse abgeschlossen!")

if __name__ == "__main__":
    main()