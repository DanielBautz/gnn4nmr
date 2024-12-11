import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import wandb

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # out: [num_nodes, out_channels] -> out_channels=1 für Regression
        # y: [num_nodes]

        # Maskiere Knoten, die kein shift_high-low haben (nan)
        mask = ~torch.isnan(data.y)
        loss = F.mse_loss(out[mask].squeeze(), data.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
    return total_loss / sum([d.num_nodes for d in loader.dataset])

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            mask = ~torch.isnan(data.y)
            loss = F.mse_loss(out[mask].squeeze(), data.y[mask])
            total_loss += loss.item() * data.num_nodes
    return total_loss / sum([d.num_nodes for d in loader.dataset])

def run_training(model, train_loader, val_loader, test_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = test(model, val_loader, device)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    test_loss = test(model, test_loader, device)
    wandb.log({"test_loss": test_loss})
    print(f"Test Loss: {test_loss:.4f}")
