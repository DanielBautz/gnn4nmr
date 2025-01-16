import copy  # NEU: für deepcopy
import torch
import torch.nn.functional as F
import numpy as np
import wandb

def mse_and_mae(pred, target):
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse.item(), mae.item()

def train(model, loader, optimizer, device):
    model.train()

    total_C, total_H = 0, 0
    running_mse_C, running_mae_C = 0.0, 0.0
    running_mse_H, running_mae_H = 0.0, 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)  # shape: [N, 2]

        # Targets
        tgt_C = data.y[:, 0]
        tgt_H = data.y[:, 1]

        # Masken
        mask_C = ~torch.isnan(tgt_C)
        mask_H = ~torch.isnan(tgt_H)

        pred_C = out[mask_C, 0]
        pred_H = out[mask_H, 1]
        real_C = tgt_C[mask_C]
        real_H = tgt_H[mask_H]

        if pred_C.numel() == 0 and pred_H.numel() == 0:
            continue

        loss_C = F.mse_loss(pred_C, real_C) if pred_C.numel() > 0 else 0.0
        loss_H = F.mse_loss(pred_H, real_H) if pred_H.numel() > 0 else 0.0
        loss = loss_C + loss_H
        loss.backward()
        optimizer.step()

        # Logging
        if pred_C.numel() > 0:
            mse_c, mae_c = mse_and_mae(pred_C, real_C)
            running_mse_C += mse_c * pred_C.numel()
            running_mae_C += mae_c * pred_C.numel()
            total_C += pred_C.numel()

        if pred_H.numel() > 0:
            mse_h, mae_h = mse_and_mae(pred_H, real_H)
            running_mse_H += mse_h * pred_H.numel()
            running_mae_H += mae_h * pred_H.numel()
            total_H += pred_H.numel()

    avg_mse_C = running_mse_C / total_C if total_C > 0 else np.nan
    avg_mae_C = running_mae_C / total_C if total_C > 0 else np.nan
    avg_mse_H = running_mse_H / total_H if total_H > 0 else np.nan
    avg_mae_H = running_mae_H / total_H if total_H > 0 else np.nan

    return avg_mse_C, avg_mae_C, avg_mse_H, avg_mae_H

def test(model, loader, device):
    model.eval()

    total_C, total_H = 0, 0
    running_mse_C, running_mae_C = 0.0, 0.0
    running_mse_H, running_mae_H = 0.0, 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)

            tgt_C = data.y[:, 0]
            tgt_H = data.y[:, 1]

            mask_C = ~torch.isnan(tgt_C)
            mask_H = ~torch.isnan(tgt_H)

            pred_C = out[mask_C, 0]
            pred_H = out[mask_H, 1]
            real_C = tgt_C[mask_C]
            real_H = tgt_H[mask_H]

            if pred_C.numel() > 0:
                mse_c, mae_c = mse_and_mae(pred_C, real_C)
                running_mse_C += mse_c * pred_C.numel()
                running_mae_C += mae_c * pred_C.numel()
                total_C += pred_C.numel()

            if pred_H.numel() > 0:
                mse_h, mae_h = mse_and_mae(pred_H, real_H)
                running_mse_H += mse_h * pred_H.numel()
                running_mae_H += mae_h * pred_H.numel()
                total_H += pred_H.numel()

    avg_mse_C = running_mse_C / total_C if total_C > 0 else np.nan
    avg_mae_C = running_mae_C / total_C if total_C > 0 else np.nan
    avg_mse_H = running_mse_H / total_H if total_H > 0 else np.nan
    avg_mae_H = running_mae_H / total_H if total_H > 0 else np.nan

    return avg_mse_C, avg_mae_C, avg_mse_H, avg_mae_H

def run_training(model, train_loader, val_loader, test_loader, epochs, device, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # NEU: Hier speichern wir das beste Modell
    best_val_score = float('inf')
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_mse_C, train_mae_C, train_mse_H, train_mae_H = train(model, train_loader, optimizer, device)
        val_mse_C, val_mae_C, val_mse_H, val_mae_H = test(model, val_loader, device)

        # Beispiel: Wir nehmen die Summe der MSEs als "Val-Score"
        # Du kannst stattdessen auch (val_mse_C + val_mse_H)/2 nehmen, etc.
        val_score = (val_mse_C if not np.isnan(val_mse_C) else 0) \
                    + (val_mse_H if not np.isnan(val_mse_H) else 0)

        # Check, ob neues bestes Modell
        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(model.state_dict())

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
            "val_score": val_score
        })

    # Nach dem Training: Lade das beste Modell wieder
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Jetzt Testevaluation mit dem besten Modell
    test_mse_C, test_mae_C, test_mse_H, test_mae_H = test(model, test_loader, device)
    wandb.log({
        "test_mse_C": test_mse_C,
        "test_mae_C": test_mae_C,
        "test_mse_H": test_mse_H,
        "test_mae_H": test_mae_H
    })

    # Modell lokal speichern (z.B. als "best_model.pt")
    torch.save(model.state_dict(), "best_model.pt")
