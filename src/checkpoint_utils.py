import os
import json
import time
import wandb
from typing import Dict, Any, Optional

# Standardpfad für Checkpoint-Verzeichnis
CHECKPOINT_DIR = "sweep_checkpoints"

def ensure_checkpoint_dir(sweep_id: str) -> str:
    """Stellt sicher, dass das Checkpoint-Verzeichnis für den angegebenen Sweep existiert."""
    sweep_dir = os.path.join(CHECKPOINT_DIR, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    return sweep_dir

def save_sweep_checkpoint(sweep_id: str, run_id: str, config: Dict[str, Any], 
                          metrics: Dict[str, Any], global_step: int) -> str:
    """
    Speichert einen Checkpoint des aktuellen Sweep-Runs.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        run_id: ID des aktuellen W&B-Runs
        config: Die Konfiguration des Runs
        metrics: Die aktuellen Metriken
        global_step: Der aktuelle Schritt oder die Iteration
        
    Returns:
        Path zum gespeicherten Checkpoint
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    
    # Erstelle Checkpoint-Daten
    checkpoint_data = {
        "timestamp": time.time(),
        "sweep_id": sweep_id,
        "run_id": run_id,
        "config": config,
        "metrics": metrics,
        "global_step": global_step
    }
    
    # Generiere Checkpoint-Dateiname
    checkpoint_path = os.path.join(sweep_dir, f"checkpoint_{run_id}_{global_step}.json")
    
    # Speichere Checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Aktualisiere den latest_checkpoint-Verweis
    latest_path = os.path.join(sweep_dir, "latest_checkpoint.json")
    with open(latest_path, 'w') as f:
        json.dump({
            "latest_checkpoint": checkpoint_path,
            "timestamp": time.time(),
            "run_id": run_id,
            "global_step": global_step
        }, f, indent=2)
    
    print(f"Checkpoint gespeichert: {checkpoint_path}")
    return checkpoint_path

def load_sweep_checkpoint(sweep_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Lädt einen Checkpoint für den angegebenen Sweep.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        run_id: Optionale ID eines spezifischen Runs. Wenn None, wird der neueste Checkpoint geladen.
        
    Returns:
        Die geladenen Checkpoint-Daten oder ein leeres Dict, wenn kein Checkpoint gefunden wurde.
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    
    # Wenn run_id angegeben ist, versuche diesen spezifischen Checkpoint zu laden
    if run_id:
        # Finde den neuesten Checkpoint für diesen Run
        checkpoints = [f for f in os.listdir(sweep_dir) if f.startswith(f"checkpoint_{run_id}")]
        if checkpoints:
            # Sortiere nach globalem Schritt (Extrahiere die Nummer am Ende des Dateinamens)
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            checkpoint_path = os.path.join(sweep_dir, checkpoints[0])
            
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
    
    # Ansonsten versuche, den neuesten Checkpoint zu laden
    latest_path = os.path.join(sweep_dir, "latest_checkpoint.json")
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            latest_info = json.load(f)
        
        if os.path.exists(latest_info["latest_checkpoint"]):
            with open(latest_info["latest_checkpoint"], 'r') as f:
                return json.load(f)
    
    # Wenn kein Checkpoint gefunden wurde, gib ein leeres Dict zurück
    return {}

def list_sweep_checkpoints(sweep_id: str) -> Dict[str, Any]:
    """
    Listet alle verfügbaren Checkpoints für einen bestimmten Sweep auf.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        
    Returns:
        Dictionary mit Informationen zu allen Checkpoints
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    
    # Wenn das Verzeichnis nicht existiert oder leer ist
    if not os.path.exists(sweep_dir) or not os.listdir(sweep_dir):
        return {"checkpoints": [], "count": 0}
    
    checkpoints = []
    for filename in os.listdir(sweep_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".json"):
            checkpoint_path = os.path.join(sweep_dir, filename)
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                checkpoints.append({
                    "filename": filename,
                    "path": checkpoint_path,
                    "run_id": checkpoint_data.get("run_id", "unknown"),
                    "timestamp": checkpoint_data.get("timestamp", 0),
                    "global_step": checkpoint_data.get("global_step", 0),
                    "metrics": checkpoint_data.get("metrics", {})
                })
    
    # Sortiere Checkpoints nach Zeitstempel (neueste zuerst)
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "checkpoints": checkpoints,
        "count": len(checkpoints),
        "latest": checkpoints[0] if checkpoints else None
    }

def save_best_config(sweep_id: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """
    Speichert die beste Konfiguration aus einem Sweep.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        config: Die beste Konfiguration
        metrics: Die zugehörigen Metriken
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    
    # Erstelle best_config Daten
    best_config_data = {
        "timestamp": time.time(),
        "sweep_id": sweep_id,
        "config": config,
        "metrics": metrics
    }
    
    # Speichere best_config
    best_config_path = os.path.join(sweep_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config_data, f, indent=2)
    
    print(f"Beste Konfiguration gespeichert: {best_config_path}")

def update_sweep_progress(sweep_id: str, total_runs: int, completed_runs: int, 
                         best_run_id: Optional[str] = None, 
                         best_metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Aktualisiert den Fortschritt eines Sweeps.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        total_runs: Gesamtzahl der Runs in diesem Sweep
        completed_runs: Anzahl der abgeschlossenen Runs
        best_run_id: ID des besten Runs (optional)
        best_metrics: Metriken des besten Runs (optional)
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    
    # Erstelle Fortschrittsdaten
    progress_data = {
        "timestamp": time.time(),
        "sweep_id": sweep_id,
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "progress_percentage": (completed_runs / total_runs) * 100 if total_runs > 0 else 0,
        "best_run_id": best_run_id,
        "best_metrics": best_metrics
    }
    
    # Speichere Fortschritt
    progress_path = os.path.join(sweep_dir, "sweep_progress.json")
    with open(progress_path, 'w') as f:
        json.dump(progress_data, f, indent=2)

def get_sweep_progress(sweep_id: str) -> Dict[str, Any]:
    """
    Holt den aktuellen Fortschritt eines Sweeps.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        
    Returns:
        Dictionary mit Fortschrittsinformationen
    """
    sweep_dir = ensure_checkpoint_dir(sweep_id)
    progress_path = os.path.join(sweep_dir, "sweep_progress.json")
    
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    
    return {
        "sweep_id": sweep_id,
        "total_runs": 0,
        "completed_runs": 0,
        "progress_percentage": 0,
        "best_run_id": None,
        "best_metrics": None
    }

def get_best_config_from_api(sweep_id: str, project: str = "gnn_shift_prediction_sweep") -> Dict[str, Any]:
    """
    Holt die beste Konfiguration aus der W&B API.
    
    Args:
        sweep_id: ID des W&B-Sweeps
        project: Name des W&B-Projekts
        
    Returns:
        Dictionary mit der besten Konfiguration und den zugehörigen Metriken
    """
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{project}/{sweep_id}")
        best_run = sweep.best_run()
        
        return {
            "config": best_run.config,
            "metrics": {
                "summary": best_run.summary._json_dict,
                "state": best_run.state
            },
            "run_id": best_run.id,
            "name": best_run.name,
            "url": best_run.url
        }
    except Exception as e:
        print(f"Fehler beim Abrufen der besten Konfiguration aus der API: {e}")
        return {}