import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import autocast, GradScaler

def compute_metrics(pred, target):
    """Berechnet MSE und MAE."""
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse, mae

def train_one_epoch(model, dataloader, device, optimizer, config, scaler=None):
    """
    Trainiert über eine Epoche mit optionaler Mixed Precision.
    Berechnet separate Metriken für H und C und kombiniert den Loss.
    """
    model.train()
    
    total_mse_H, total_mae_H, count_H = 0.0, 0.0, 0
    total_mse_C, total_mae_C, count_C = 0.0, 0.0, 0
    
    # Gradient accumulation setup
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    optimizer.zero_grad()
    
    for i, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device, non_blocking=config.get('pin_memory', True))
        
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
        
        # Use autocast for mixed precision training if enabled
        with autocast(enabled=config.get('use_mixed_precision', False)):
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
                # Scale the loss according to accumulation steps
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
        
        # Backward pass with mixed precision scaling if enabled
        if len(loss_terms) > 0:
            if scaler is not None and config.get('use_mixed_precision', False):
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                    # Unscale before optimizer step to properly handle gradient clipping
                    scaler.unscale_(optimizer)
                    
                    # Optional gradient clipping
                    if config.get('gradient_clip_val', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.gradient_clip_val
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                    # Optional gradient clipping
                    if config.get('gradient_clip_val', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.gradient_clip_val
                        )
                    
                    optimizer.step()
                    optimizer.zero_grad()
    
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
    
    # Use autocast for evaluation only if mixed precision is enabled
    with autocast(enabled=config.get('use_mixed_precision', False)):
        for batch_data in dataloader:
            batch_data = batch_data.to(device, non_blocking=config.get('pin_memory', True))
            
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
    
    val_score = (val_mae_H * config.loss_weight_H + val_mae_C * config.loss_weight_C) / (config.loss_weight_H + config.loss_weight_C)
    return val_mse_H, val_mae_H, val_mse_C, val_mae_C, val_score

def train_model(model, train_loader, val_loader, test_loader, device, config):
    """
    Haupttrainingsschleife mit optionaler Mixed Precision und Gradient Accumulation.
    """
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    # Set up mixed precision training if enabled
    scaler = None
    if config.get('use_mixed_precision', False):
        scaler = GradScaler()
    
    # Let's use a more robust scheduler setup
    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
    else:
        # Default to ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.scheduler_factor, 
            patience=config.scheduler_patience, verbose=True
        )
    
    best_val_score = float('inf')
    best_epoch = 0
    no_improve_count = 0
    early_stop_patience = config.get('early_stop_patience', 20)
    
    for epoch in range(config.num_epochs):
        train_mse_H, train_mae_H, train_mse_C, train_mae_C = train_one_epoch(
            model, train_loader, device, optimizer, config, scaler
        )
        val_mse_H, val_mae_H, val_mse_C, val_mae_C, val_score = evaluate_with_config(
            model, val_loader, device, config
        )
        
        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_score)
        else:
            scheduler.step()
        
        # Logging
        wandb.log({
            "epoch": epoch,
            "train_mse_C": train_mse_C,
            "train_mae_C": train_mae_C,
            "train_mse_H": train_mse_H,
            "train_mae_H": train_mae_H,
            "val_mse_C": val_mse_C,
            "val_mae_C": val_mae_C,
            "val_mse_H": val_mse_H,
            "val_mae_H": val_mae_H,
            "val_score": val_score,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}]")
        print(f"  Train   => H: MSE={train_mse_H:.4f}, MAE={train_mae_H:.4f} |"
              f"  C: MSE={train_mse_C:.4f}, MAE={train_mae_C:.4f}")
        print(f"  Val     => H: MSE={val_mse_H:.4f}, MAE={val_mae_H:.4f} |"
              f"  C: MSE={val_mse_C:.4f}, MAE={val_mae_C:.4f} | val_score={val_score:.4f}")
        
        # Check if this is the best model so far
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ** New best model saved (val_score={val_score:.4f})")
        else:
            no_improve_count += 1
            print(f"  -- No improvement for {no_improve_count} epochs. Best: {best_val_score:.4f} at epoch {best_epoch+1}")
            
            # Early stopping
            if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
                print(f"Early stopping triggered after {no_improve_count} epochs without improvement")
                break
    
    print(f"Training completed. Loading best model from epoch {best_epoch+1}")
    model.load_state_dict(torch.load("best_model.pt"))
    
    # Final evaluation on test set
    test_mse_H, test_mae_H, test_mse_C, test_mae_C, _ = evaluate_with_config(
        model, test_loader, device, config
    )
    
    wandb.log({
        "test_mse_C": test_mse_C,
        "test_mae_C": test_mae_C,
        "test_mse_H": test_mse_H,
        "test_mae_H": test_mae_H,
        "best_epoch": best_epoch
    })
    
    print(f"** Test ** => H: MSE={test_mse_H:.4f}, MAE={test_mae_H:.4f} |"
          f"  C: MSE={test_mse_C:.4f}, MAE={test_mae_C:.4f}")
    
    return model