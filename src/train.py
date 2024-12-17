import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import wandb

def mse_and_mae(pred, target):
    # pred und target: [N] Tensor
    # Gibt MSE und MAE zurück
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse.item(), mae.item()

def train(model, loader, optimizer, device):
    model.train()
    total_nodes = 0
    running_mse = 0.0
    running_mae = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()

        mask = ~torch.isnan(data.y)
        # Predictions und Targets filtern
        pred = out[mask]
        tgt = data.y[mask]

        if pred.numel() == 0:
            # Falls in diesem Batch keine C/H Atome sind, weiter
            continue

        loss = F.mse_loss(pred, tgt)
        loss.backward()
        optimizer.step()

        # Logging von Metriken
        batch_mse, batch_mae = mse_and_mae(pred, tgt)
        n = mask.sum().item()
        running_mse += batch_mse * n
        running_mae += batch_mae * n
        total_nodes += n

    if total_nodes == 0:
        return np.nan, np.nan
    avg_mse = running_mse / total_nodes
    avg_mae = running_mae / total_nodes
    return avg_mse, avg_mae

def test(model, loader, device):
    model.eval()
    total_nodes = 0
    running_mse = 0.0
    running_mae = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).squeeze()

            mask = ~torch.isnan(data.y)
            pred = out[mask]
            tgt = data.y[mask]

            if pred.numel() == 0:
                continue

            batch_mse, batch_mae = mse_and_mae(pred, tgt)
            n = mask.sum().item()
            running_mse += batch_mse * n
            running_mae += batch_mae * n
            total_nodes += n

    if total_nodes == 0:
        return np.nan, np.nan
    avg_mse = running_mse / total_nodes
    avg_mae = running_mae / total_nodes
    return avg_mse, avg_mae

def run_training(model, train_loader, val_loader, test_loader, epochs, device, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        train_mse, train_mae = train(model, train_loader, optimizer, device)
        val_mse, val_mae = test(model, val_loader, device)
        wandb.log({
            "epoch": epoch,
            "train_mse": train_mse, 
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae
        })

    test_mse, test_mae = test(model, test_loader, device)
    wandb.log({
        "test_mse": test_mse,
        "test_mae": test_mae
    })
