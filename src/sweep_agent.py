import os
import sys
import wandb
import subprocess
import argparse
import json
import signal

# Importiere die notwendigen Module
try:
    from sweep_config import create_sweep
except ImportError:
    print("WARNUNG: sweep_config konnte nicht importiert werden. Eine neue Sweep-ID wird erstellt.")
    
    def create_sweep():
        """Fallback-Funktion, wenn sweep_config nicht importiert werden kann"""
        api = wandb.Api()
        sweep = api.sweep(wandb.sweep(
            {
                'method': 'bayes',
                'metric': {'name': 'avg_val_score', 'goal': 'minimize'},
                'parameters': {
                    'hidden_dim': {'values': [32, 64, 128]},
                    'out_dim': {'values': [64, 128, 256]},
                    'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
                }
            },
            project="gnn_shift_prediction_sweep"
        ))
        return sweep.id

# Globale Variable für Sweep-Fortschrittsverfolgung
GLOBAL_RUN_COUNT = 0
TOTAL_RUNS = 0
SWEEP_ID = None
SHOULD_EXIT = False

def signal_handler(sig, frame):
    """Signal-Handler für sauberes Beenden"""
    global SHOULD_EXIT
    print(f"\nEmpfangenes Signal {sig}. Bereite sauberes Beenden vor...")
    SHOULD_EXIT = True

# Registriere Signal-Handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def modify_script_for_sweep():
    """
    Erstellt eine temporäre Kopie von main.py mit angepassten Imports für Sweeps
    """
    main_script_path = os.path.join(os.path.dirname(__file__), "main.py")
    sweep_script_path = os.path.join(os.path.dirname(__file__), "main_for_sweep.py")
    
    # Lese die originale main.py
    with open(main_script_path, 'r') as f:
        content = f.read()
    
    # Ersetze 'from src.' durch 'from '
    modified_content = content.replace('from src.', 'from ')
    
    # Schreibe die modifizierte Version in main_for_sweep.py
    with open(sweep_script_path, 'w') as f:
        f.write(modified_content)
    
    return sweep_script_path

def run_sweep():
    """Startet einen neuen Prozess für jeden Sweep-Run"""
    global GLOBAL_RUN_COUNT, SWEEP_ID, SHOULD_EXIT
    
    # Inkrementiere den Zähler
    GLOBAL_RUN_COUNT += 1
    
    # Erstelle Umgebungsvariablen für den Run
    env = os.environ.copy()
    env["SWEEP_ID"] = SWEEP_ID
    env["RUN_INDEX"] = str(GLOBAL_RUN_COUNT)
    
    print(f"\n=== Starte Sweep-Run {GLOBAL_RUN_COUNT}/{TOTAL_RUNS} ===")
    
    # Erstelle eine modifizierte Version von main.py
    sweep_script_path = modify_script_for_sweep()
    
    # Benutze den aktuellen Python-Interpreter
    cmd = [sys.executable, sweep_script_path]
    
    # Führe den Prozess für diesen Run aus
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Lese die Ausgabe in Echtzeit
    while True:
        if SHOULD_EXIT:
            print("Beende aktuellen Run wegen Unterbrechungssignal...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
            
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Hole die restlichen Ausgaben
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout)
    if stderr:
        print(f"FEHLER: {stderr}")
    
    # Wenn Exit-Signal empfangen wurde, beende das Skript
    if SHOULD_EXIT:
        print(f"Sweep wurde bei Run {GLOBAL_RUN_COUNT}/{TOTAL_RUNS} unterbrochen.")
        print(f"Führe zum Fortsetzen aus: python -m src.sweep_agent {TOTAL_RUNS} {SWEEP_ID} --resume")
        sys.exit(0)
    
    return process.returncode

def parse_args():
    """Kommandozeilenargumente parsen"""
    parser = argparse.ArgumentParser(description='Sweep-Agent für die Hyperparameter-Optimierung')
    parser.add_argument('num_runs', type=int, nargs='?', default=20,
                        help='Anzahl der Runs pro Agent (Standard: 20)')
    parser.add_argument('sweep_id', type=str, nargs='?', default=None,
                        help='Sweep-ID (optional, sonst wird ein neuer Sweep erstellt)')
    parser.add_argument('--project', type=str, default="gnn_shift_prediction_sweep",
                        help='W&B Projektname (Standard: gnn_shift_prediction_sweep)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TOTAL_RUNS = args.num_runs
    SWEEP_ID = args.sweep_id
    
    # Wenn kein Sweep-ID angegeben ist, erstelle einen neuen Sweep
    if not SWEEP_ID:
        SWEEP_ID = create_sweep()
        print(f"Neuer Sweep erstellt mit ID: {SWEEP_ID}")
    
    # Führe die Sweep-Runs aus
    while GLOBAL_RUN_COUNT < TOTAL_RUNS:
        return_code = run_sweep()
        if return_code != 0:
            print(f"Run beendet mit Fehlercode {return_code}.")
        
        # Prüfe, ob wir beenden sollen
        if SHOULD_EXIT:
            break
    
    # Zusammenfassung am Ende
    if GLOBAL_RUN_COUNT >= TOTAL_RUNS:
        print(f"\nSweep {SWEEP_ID} abgeschlossen!")
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{args.project}/{SWEEP_ID}")
            
            # Sortiere die Runs nach der Zielmetrik
            sweep_runs = sorted(sweep.runs, key=lambda run: run.summary.get("avg_val_score", float("inf")))
            if sweep_runs:
                best_run = sweep_runs[0]
                print(f"Bester Run: {best_run.id} - {best_run.name}")
                print(f"URL: {best_run.url}")
                print("Beste Metriken:")
                print(json.dumps(best_run.summary._json_dict, indent=2))
        except Exception as e:
            print(f"Fehler beim Abrufen der besten Konfiguration: {e}")