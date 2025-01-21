# train.py
import torch
import torch.nn as nn
import wandb

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Extrahieren von edge_attr_dict
        edge_attr_dict = {}
        for store in data.edge_stores:
            src, rel, dst = store._key
            key = (src, dst)
            edge_attr_dict[key] = store.edge_attr if hasattr(store, 'edge_attr') else None

        out_c, out_h = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        
        # Targets
        y_c = data['C'].y
        y_h = data['H'].y
        
        loss_c = criterion(out_c, y_c) if out_c.numel() > 0 else 0.0
        loss_h = criterion(out_h, y_h) if out_h.numel() > 0 else 0.0
        
        loss = loss_c + loss_h
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        
        edge_attr_dict = {}
        for store in data.edge_stores:
            src, rel, dst = store._key
            key = (src, dst)
            edge_attr_dict[key] = store.edge_attr if hasattr(store, 'edge_attr') else None

        out_c, out_h = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        
        y_c = data['C'].y
        y_h = data['H'].y
        
        loss_c = criterion(out_c, y_c) if out_c.numel() > 0 else 0.0
        loss_h = criterion(out_h, y_h) if out_h.numel() > 0 else 0.0
        loss = loss_c + loss_h
        total_loss += loss.item()
    return total_loss / len(loader)

def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Test mit bestem Modell
    model.load_state_dict(best_model_state)
    test_loss = evaluate(model, test_loader, device)
    wandb.log({'test_loss': test_loss})
    return model, test_loss
