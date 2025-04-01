import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import autocast, GradScaler
from early_stopping import EarlyStopping
import os

def compute_metrics(pred, target):
    """Berechnet MSE und MAE."""
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse, mae

def train_one_epoch(model, dataloader, device, optimizer, config, scaler=None):
    """
    Trainiert über eine Epoche.
    Berechnet separate Metriken für H und C und kombiniert den Loss.
    Unterstützt Mixed Precision Training mit scaler.
    """
    model.train()
    
    total_mse_H, total_mae_H, count_H = 0.0, 0.0, 0
    total_mse_C, total_mae_C, count_C = 0.0, 0.0, 0
    
    for batch_data in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)  # Asynchroner Transfer
        optimizer.zero_grad(set_to_none=True)  # Effizienter als .zero_grad()
        
        x_dict, y_dict = {}, {}
        for ntype in batch_data.node_types:
            x_dict[ntype] = batch_data[ntype].x
            y_dict[ntype] = getattr(batch_data[ntype], 'y', None)
        
        edge_index_dict = {}
        edge_attr_dict = {}
        for store in batch_data.edge_stores:
            src, rel, dst = store._key
            edge_index_dict[(src, rel, dst)] = store.edge_index
            edge_attr_dict[(src, rel, dst)] = store.edge_attr
        
        # Mixed Precision Training wenn scaler vorhanden
        if scaler is not None:
            with autocast():
                out_dict = model(x_dict, edge_index_dict, edge_attr_dict)
                loss_terms = []
                
                # H
                if out_dict['H'] is not None and y_dict['H'] is not None:
                    valid_mask = ~torch.isnan(y_dict['H'])
                    if valid_mask.sum() > 0:
                        pred_H = out_dict['H'][valid_mask]
                        target_H = y_dict['H'][valid_mask]
                        mse_H, mae_H = compute_metrics(pred_H, target_H)
                        loss_terms.append(mae_H * config.loss_weight_H)
                        total_mse_H += mse_H.item()
                        total_mae_H += mae_H.item()
                        count_H += 1
                
                # C
                if out_dict['C'] is not None and y_dict['C'] is not None:
                    valid_mask = ~torch.isnan(y_dict['C'])
                    if valid_mask.sum() > 0:
                        pred_C = out_dict['C'][valid_mask]
                        target_C = y_dict['C'][valid_mask]
                        mse_C, mae_C = compute_metrics(pred_C, target_C)
                        loss_terms.append(mae_C * config.loss_weight_C)
                        total_mse_C += mse_C.item()
                        total_mae_C += mae_C.item()
                        count_C += 1
                
                if len(loss_terms) > 0:
                    loss = torch.stack(loss_terms).mean()
                    
            # Mixed Precision Backprop
            if len(loss_terms) > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            # Original FP32 Durchlauf wenn kein scaler
            out_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            loss_terms = []
            
            # H
            if out_dict['H'] is not None and y_dict['H'] is not None:
                valid_mask = ~torch.isnan(y_dict['H'])
                if valid_mask.sum() > 0:
                    pred_H = out_dict['H'][valid_mask]
                    target_H = y_dict['H'][valid_mask]
                    mse_H, mae_H = compute_metrics(pred_H, target_H)
                    loss_terms.append(mae_H * config.loss_weight_H)
                    total_mse_H += mse_H.item()
                    total_mae_H += mae_H.item()
                    count_H += 1
            
            # C
            if out_dict['C'] is not None and y_dict['C'] is not None:
                valid_mask = ~torch.isnan(y_dict['C'])
                if valid_mask.sum() > 0:
                    pred_C = out_dict['C'][valid_mask]
                    target_C = y_dict['C'][valid_mask]
                    mse_C, mae_C = compute_metrics(pred_C, target_C)
                    loss_terms.append(mae_C * config.loss_weight_C)
                    total_mse_C += mse_C.item()
                    total_mae_C += mae_C.item()
                    count_C += 1
            
            if len(loss_terms) > 0:
                loss = torch.stack(loss_terms).mean()
                loss.backward()
                optimizer.step()
    
    train_mse_H = total_mse_H / count_H if count_H > 0 else 0.0
    train_mae_H = total_mae_H / count_H if count_H > 0 else 0.0
    train_mse_C = total_mse_C / count_C if count_C > 0 else 0.0
    train_mae_C = total_mae_C / count_C if count_C > 0 else 0.0
    
    return train_mse_H, train_mae_H, train_mse_C, train_mae_C

@torch.no_grad()
def evaluate_with_config(model, dataloader, device, config):
    """
    Berechnet MSE/MAE für H und C und liefert auch einen 'val_score' zurück,
    wobei der Score als gewichteter Durchschnitt der MAEs berechnet wird.
    """
    model.eval()
    
    total_mse_H, total_mae_H, count_H = 0.0, 0.0, 0
    total_mse_C, total_mae_C, count_C = 0.0, 0.0, 0
    
    for batch_data in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        
        x_dict, y_dict = {}, {}
        for ntype in batch_data.node_types:
            x_dict[ntype] = batch_data[ntype].x
            y_dict[ntype] = getattr(batch_data[ntype], 'y', None)
        
        edge_index_dict = {}
        edge_attr_dict = {}
        for store in batch_data.edge_stores:
            src, rel, dst = store._key
            edge_index_dict[(src, rel, dst)] = store.edge_index
            edge_attr_dict[(src, rel, dst)] = store.edge_attr
        
        out_dict = model(x_dict, edge_index_dict, edge_attr_dict)
        
        # H
        if out_dict['H'] is not None and y_dict['H'] is not None:
            valid_mask = ~torch.isnan(y_dict['H'])
            if valid_mask.sum() > 0:
                pred_H = out_dict['H'][valid_mask]
                target_H = y_dict['H'][valid_mask]
                mse_H, mae_H = compute_metrics(pred_H, target_H)
                total_mse_H += mse_H.item()
                total_mae_H += mae_H.item()
                count_H += 1
        
        # C
        if out_dict['C'] is not None and y_dict['C'] is not None:
            valid_mask = ~torch.isnan(y_dict['C'])
            if valid_mask.sum() > 0:
                pred_C = out_dict['C'][valid_mask]
                target_C = y_dict['C'][valid_mask]
                mse_C, mae_C = compute_metrics(pred_C, target_C)
                total_mse_C += mse_C.item()
                total_mae_C += mae_C.item()
                count_C += 1
    
    val_mse_H = total_mse_H / count_H if count_H > 0 else 0.0
    val_mae_H = total_mae_H / count_H if count_H > 0 else 0.0
    val_mse_C = total_mse_C / count_C if count_C > 0 else 0.0
    val_mae_C = total_mae_C / count_C if count_C > 0 else 0.0
    
    val_score = (val_mae_H * config.loss_weight_H + val_mae_C * config.loss_weight_C) / 2.0
    return val_mse_H, val_mae_H, val_mse_C, val_mae_C, val_score

def train_model(model, train_loader, val_loader, test_loader, device, config, model_path="best_model.pt", fold_idx=None):
    """
    Haupttrainingsschleife mit Mixed Precision Support und Early Stopping:
      - Single-Model, multi-task (H und C).
      - Loggt train/val/test MSE und MAE für beide Targets.
      - Verwendet Mixed Precision wenn use_amp=True.
      - Early Stopping nach patience Epochen ohne Verbesserung.
      
    Args:
        model: Das zu trainierende Modell
        train_loader, val_loader, test_loader: Die entsprechenden DataLoader
        device: Das Device (cuda/cpu) auf dem trainiert wird
        config: Die Konfiguration mit Hyperparametern
        model_path: Speicherpfad für das beste Modell (Standardwert: "best_model.pt")
        fold_idx: Der aktuelle Fold-Index (nur bei k-fold Cross-Validation, sonst None)
        
    Returns:
        Ein Dictionary mit dem trainierten Modell und den Test-Metriken
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Update model path to include the models directory if needed
    if not os.path.dirname(model_path):
        model_path = os.path.join("models", model_path)

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=True
    )
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if hasattr(config, 'use_amp') and config.use_amp and device.type == 'cuda' else None
    
    # Early Stopping initialisieren
    early_stopping_patience = config.early_stop_patience if hasattr(config, 'early_stop_patience') else 20
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, verbose=True, 
        path=model_path, mode='min'
    )
    
    # Benchmark-Modus für bessere Geschwindigkeit (gut für wiederholte Netzwerkgrößen)
    if device.type == 'cuda' and hasattr(config, 'benchmark') and config.benchmark:
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(config.num_epochs):
        train_mse_H, train_mae_H, train_mse_C, train_mae_C = train_one_epoch(
            model, train_loader, device, optimizer, config, scaler=scaler
        )
        val_mse_H, val_mae_H, val_mse_C, val_mae_C, val_score = evaluate_with_config(
            model, val_loader, device, config
        )
        
        scheduler.step(val_score)
        
        # Early Stopping überprüfen
        early_stopping(val_score, model)
        
        # Prefix für Logging-Keys bei k-fold
        prefix = f"fold_{fold_idx}_" if fold_idx is not None else ""
        
        wandb.log({
            f"{prefix}epoch": epoch,
            f"{prefix}train_mse_C": train_mse_C,
            f"{prefix}train_mae_C": train_mae_C,
            f"{prefix}train_mse_H": train_mse_H,
            f"{prefix}train_mae_H": train_mae_H,
            f"{prefix}val_mse_C": val_mse_C,
            f"{prefix}val_mae_C": val_mae_C,
            f"{prefix}val_mse_H": val_mse_H,
            f"{prefix}val_mae_H": val_mae_H,
            f"{prefix}val_score": val_score,
            f"{prefix}lr": optimizer.param_groups[0]["lr"],
            f"{prefix}early_stopping_counter": early_stopping.counter
        })
        
        fold_info = f"Fold {fold_idx+1} - " if fold_idx is not None else ""
        print(f"{fold_info}Epoch [{epoch+1}/{config.num_epochs}]")
        print(f"  Train   => H: MSE={train_mse_H:.4f}, MAE={train_mae_H:.4f} |"
              f"  C: MSE={train_mse_C:.4f}, MAE={train_mae_C:.4f}")
        print(f"  Val     => H: MSE={val_mse_H:.4f}, MAE={val_mae_H:.4f} |"
              f"  C: MSE={val_mse_C:.4f}, MAE={val_mae_C:.4f} | val_score={val_score:.4f}")
        
        # Early Stopping prüfen
        if early_stopping.early_stop:
            print(f"Early Stopping aktiviert in Epoche {epoch+1}.")
            break
    
    # Lade das beste Modell (wurde bereits von Early Stopping gespeichert)
    model.load_state_dict(torch.load(model_path))
    test_mse_H, test_mae_H, test_mse_C, test_mae_C, test_score = evaluate_with_config(model, test_loader, device, config)
    
    # Prefix für Logging-Keys bei k-fold
    prefix = f"fold_{fold_idx}_" if fold_idx is not None else ""
    
    wandb.log({
        f"{prefix}test_mse_C": test_mse_C,
        f"{prefix}test_mae_C": test_mae_C,
        f"{prefix}test_mse_H": test_mse_H,
        f"{prefix}test_mae_H": test_mae_H,
        f"{prefix}epochs_trained": epoch + 1,
        f"{prefix}early_stopped": early_stopping.early_stop
    })
    
    fold_info = f"Fold {fold_idx+1} - " if fold_idx is not None else ""
    print(f"{fold_info}** Test ** => H: MSE={test_mse_H:.4f}, MAE={test_mae_H:.4f} |"
          f"  C: MSE={test_mse_C:.4f}, MAE={test_mae_C:.4f}")
    
    # Gib ein Ergebnisdictionary zurück, das sowohl das Modell als auch die Test-Metriken enthält
    return {
        "model": model,
        "test_metrics": (test_mse_H, test_mae_H, test_mse_C, test_mae_C, test_score),
        "epochs_trained": epoch + 1,
        "early_stopped": early_stopping.early_stop
    }