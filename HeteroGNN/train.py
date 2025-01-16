# train.py
import torch
import torch.nn.functional as F
import wandb

def compute_metrics(pred, target):
    """Hilfsfunktion, um MSE und MAE zu berechnen."""
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse, mae

def train_one_epoch(model, dataloader, device, optimizer, epoch):
    """Trainingsschleife für eine Epoche. Loggt train_mse & train_mae."""
    model.train()
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        
        # Node-Features und -Labels in Dictionaries
        x_dict = {}
        y_dict = {}
        for ntype in batch_data.node_types:
            x_dict[ntype] = batch_data[ntype].x
            y_dict[ntype] = getattr(batch_data[ntype], 'y', None)
        
        edge_index_dict = {}
        for store in batch_data.edge_stores:
            src, rel, dst = store._key
            edge_index_dict[(src, rel, dst)] = store.edge_index
        
        out_dict = model(x_dict, edge_index_dict)
        
        # Aufsummieren der Fehler für H und C
        mse_sum = 0.0
        mae_sum = 0.0
        valid_ntype_count = 0
        
        for ntype in ['H', 'C']:
            if out_dict.get(ntype) is not None and y_dict.get(ntype) is not None:
                valid_mask = ~torch.isnan(y_dict[ntype])
                if valid_mask.sum() > 0:
                    pred = out_dict[ntype][valid_mask.view(-1)]
                    target = y_dict[ntype][valid_mask.view(-1)]
                    
                    mse_ntype, mae_ntype = compute_metrics(pred, target)
                    mse_sum += mse_ntype.item()
                    mae_sum += mae_ntype.item()
                    valid_ntype_count += 1
        
        if valid_ntype_count > 0:
            # Mittelwert über H und C bilden
            mse_avg = mse_sum / valid_ntype_count
            mae_avg = mae_sum / valid_ntype_count
            
            # Loss für Backprop (hier MSE)
            # In PyTorch-Geometric-Szenarien rechnet man oft direkt auf pred, target. 
            # Hier machen wir's vereinfacht: 
            loss = torch.tensor(mse_avg, requires_grad=True)
            loss.backward()
            
            total_mse += mse_avg
            total_mae += mae_avg
            count += 1
        else:
            # Falls kein H oder C im Batch, skip
            continue
        
        optimizer.step()
    
    if count > 0:
        epoch_mse = total_mse / count
        epoch_mae = total_mae / count
    else:
        epoch_mse = 0.0
        epoch_mae = 0.0
    
    wandb.log({
        "train_mse": epoch_mse,
        "train_mae": epoch_mae,
        "epoch": epoch
    })
    
    return epoch_mse, epoch_mae

@torch.no_grad()
def evaluate(model, dataloader, device, epoch, prefix="val"):
    """
    Evaluationsroutine. Kann prefix="val" oder "test" sein.
    Loggt val_mse, val_mae oder test_mse, test_mae entsprechend.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        
        x_dict = {}
        y_dict = {}
        for ntype in batch_data.node_types:
            x_dict[ntype] = batch_data[ntype].x
            y_dict[ntype] = getattr(batch_data[ntype], 'y', None)
        
        edge_index_dict = {}
        for store in batch_data.edge_stores:
            src, rel, dst = store._key
            edge_index_dict[(src, rel, dst)] = store.edge_index
        
        out_dict = model(x_dict, edge_index_dict)
        
        mse_sum = 0.0
        mae_sum = 0.0
        valid_ntype_count = 0
        
        for ntype in ['H', 'C']:
            if out_dict.get(ntype) is not None and y_dict.get(ntype) is not None:
                valid_mask = ~torch.isnan(y_dict[ntype])
                if valid_mask.sum() > 0:
                    pred = out_dict[ntype][valid_mask.view(-1)]
                    target = y_dict[ntype][valid_mask.view(-1)]
                    
                    mse_ntype, mae_ntype = compute_metrics(pred, target)
                    mse_sum += mse_ntype.item()
                    mae_sum += mae_ntype.item()
                    valid_ntype_count += 1
        
        if valid_ntype_count > 0:
            total_mse += (mse_sum / valid_ntype_count)
            total_mae += (mae_sum / valid_ntype_count)
            count += 1

    if count > 0:
        epoch_mse = total_mse / count
        epoch_mae = total_mae / count
    else:
        epoch_mse = 0.0
        epoch_mae = 0.0
    
    wandb.log({
        f"{prefix}_mse": epoch_mse,
        f"{prefix}_mae": epoch_mae,
        "epoch": epoch
    })
    
    return epoch_mse, epoch_mae

def train_model(model, train_loader, val_loader, test_loader, device, config):
    """
    Haupt-Trainingsfunktion:
      - Enthält Epocenschleife für Training und Validation.
      - Lädt am Ende das beste Modell und führt (falls gewünscht) Test-Evaluierung durch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    best_val_mse = float('inf')
    
    for epoch in range(config.num_epochs):
        train_mse, train_mae = train_one_epoch(model, train_loader, device, optimizer, epoch)
        val_mse, val_mae = evaluate(model, val_loader, device, epoch, prefix="val")
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}]")
        print(f" - train_mse: {train_mse:.4f} | train_mae: {train_mae:.4f}")
        print(f" - val_mse:   {val_mse:.4f}   | val_mae:   {val_mae:.4f}")
        
        # optional: Best model speichern
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), "best_model.pt")
    
    # 4) Test-Evaluierung mit Best-Model (optional)
    model.load_state_dict(torch.load("best_model.pt"))
    test_mse, test_mae = evaluate(model, test_loader, device, epoch=config.num_epochs, prefix="test")
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
    return model
