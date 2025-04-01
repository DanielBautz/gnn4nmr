import numpy as np
import torch
import os

class EarlyStopping:
    """
    Early Stopping zur Vermeidung von Overfitting.
    Beendet das Training, wenn sich eine Metrik für 'patience' Epochen nicht verbessert.
    
    Args:
        patience (int): Anzahl an Epochen, die gewartet wird, bis das Training beendet wird
        verbose (bool): Falls True, werden Meldungen ausgegeben
        delta (float): Minimale Änderung, die als Verbesserung zählt
        path (str): Pfad zum Speichern des Checkpoints
        trace_func (Callable): Funktion zum Tracen (z.B. print)
        mode (str): 'min' für Minimierung, 'max' für Maximierung der Metrik
    """
    def __init__(self, patience=20, verbose=True, delta=0, path='checkpoint.pt', trace_func=print, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        
    def __call__(self, val_score, model):
        """
        Ruft die Early Stopping Logik auf
        
        Args:
            val_score (float): Validierungs-Metrik
            model (nn.Module): Modell, das gespeichert werden soll
        """
        score = -val_score if self.mode == 'min' else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
            
    def save_checkpoint(self, val_score, model):
        """Speichert das Modell, wenn die Validierungs-Metrik sich verbessert hat."""
        if self.verbose:
            score_info = f'decreased to {val_score:.6f}' if self.mode == 'min' else f'increased to {val_score:.6f}'
            self.trace_func(f'Validation score {score_info}. Speichere Modell...')
        
        # Stelle sicher, dass der Verzeichnispfad existiert, falls es einen gibt
        dirname = os.path.dirname(self.path)
        if dirname:  # Nur versuchen, das Verzeichnis zu erstellen, wenn der Pfad eine Verzeichniskomponente hat
            os.makedirs(dirname, exist_ok=True)
        
        torch.save(model.state_dict(), self.path)
        self.val_score_min = val_score