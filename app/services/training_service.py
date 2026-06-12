"""Background training: compound-based split, train.py-equivalent loop, no wandb.

The loop mirrors scripts/train.py semantics:
- masked MAE per node type (NaN targets skipped)
- loss = mean(mae_H * loss_weight_H, mae_C * loss_weight_C)
- Adam + ReduceLROnPlateau on val_score
- val_score = (val_mae_H * w_H + val_mae_C * w_C) / 2
- early stopping on val_score, best state_dict kept in memory

Progress is written into state.train_job under state.train_lock and polled by
the frontend. The thread trains its own model instance and only writes
checkpoint files; activating a checkpoint stays an explicit user action.
"""
import random
import threading
import time
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from . import pathsetup  # noqa: F401
from .. import config
from . import dataset_service, model_service


def start_training(state, train_config):
    with state.train_lock:
        if state.train_job is not None and state.train_job["status"] in ("running", "stopping"):
            return None
        state.train_stop = threading.Event()
        state.train_job = {
            "status": "running",
            "model_name": train_config["model_name"] or train_config["operator_type"],
            "config": train_config,
            "epoch": 0,
            "num_epochs": train_config["num_epochs"],
            "history": [],
            "best_epoch": None,
            "best_val_score": None,
            "val_fallback": None,
            "stopped_early": False,
            "checkpoint_id": None,
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "_started_monotonic": time.monotonic(),
        }
        job = state.train_job
    thread = threading.Thread(target=_train_worker, args=(state, train_config), daemon=True)
    state.train_thread = thread
    thread.start()
    return job


def training_status(state):
    with state.train_lock:
        job = state.train_job
        if job is None:
            return {"status": "idle"}
        snapshot = {k: v for k, v in job.items() if not k.startswith("_")}
        end = job.get("_finished_monotonic") or time.monotonic()
        snapshot["elapsed_s"] = round(end - job["_started_monotonic"], 1)
        snapshot["history"] = list(job["history"])
        return snapshot


def request_stop(state):
    with state.train_lock:
        job = state.train_job
        if job is None or job["status"] != "running":
            return False
        job["status"] = "stopping"
    state.train_stop.set()
    return True


def _update_job(state, **kwargs):
    with state.train_lock:
        if state.train_job is not None:
            state.train_job.update(kwargs)


def _append_history(state, record):
    with state.train_lock:
        if state.train_job is not None:
            state.train_job["history"].append(record)
            state.train_job["epoch"] = record["epoch"]


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _train_worker(state, cfg):
    try:
        _run_training(state, cfg)
    except Exception as exc:  # surface anything to the UI
        _update_job(state, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        _update_job(state, _finished_monotonic=time.monotonic())


def _run_training(state, cfg):
    device = state.device
    stop_event = state.train_stop

    file_name = cfg.get("dataset_file") or state.active_dataset_file
    if not file_name:
        raise RuntimeError("No dataset loaded.")
    # Loaded into the registry without making it the Explore view's active
    # dataset; first load of a new file takes a few seconds (stat scan).
    entry = dataset_service.load_dataset(state, file_name, make_active=False)
    dataset = entry["dataset"]

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    in_dim_dict = dataset_service.infer_in_dim_dict(dataset)

    # Compound-based split (parity with scripts/dataloader.create_dataloaders).
    compound_to_indices = {}
    for idx, nx_g in enumerate(dataset.nx_graphs):
        compound = nx_g.graph.get("compound", "unknown")
        compound_to_indices.setdefault(compound, []).append(idx)
    compounds = sorted(compound_to_indices.keys(), key=str)
    random.Random(seed).shuffle(compounds)

    r_train, r_val, _ = cfg["split_ratio"]
    n = len(compounds)
    train_end = int(r_train * n)
    val_end = train_end + int(r_val * n)
    split_compounds = {
        "train": compounds[:train_end],
        "val": compounds[train_end:val_end],
        "test": compounds[val_end:],
    }
    indices = {
        part: [i for comp in comps for i in compound_to_indices[comp]]
        for part, comps in split_compounds.items()
    }

    val_fallback = None
    if not indices["val"]:
        if indices["test"]:
            indices["val"] = indices["test"]
            val_fallback = "test"
        else:
            indices["val"] = indices["train"]
            val_fallback = "train"
    if not indices["train"]:
        raise RuntimeError("Train split is empty; adjust split_ratio.")
    _update_job(state, val_fallback=val_fallback)

    batch_size = int(cfg["batch_size"])
    train_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, indices["train"]), batch_size=batch_size, shuffle=True
    )
    val_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, indices["val"]), batch_size=batch_size, shuffle=False
    )
    test_loader = (
        PyGDataLoader(
            torch.utils.data.Subset(dataset, indices["test"]), batch_size=batch_size, shuffle=False
        )
        if indices["test"]
        else None
    )

    model_config = {
        "in_dim_dict": in_dim_dict,
        "hidden_dim": int(cfg["hidden_dim"]),
        "out_dim": int(cfg["out_dim"]),
        "encoder_dropout": float(cfg["encoder_dropout"]),
        "gnnlayer_dropout": float(cfg["gnnlayer_dropout"]),
        "num_gnn_layers": int(cfg["num_gnn_layers"]),
        "operator_type": cfg["operator_type"],
        "operator_kwargs": cfg.get("operator_kwargs") or {},
        "edge_in_dim": config.EDGE_IN_DIM,
    }
    model = model_service.build_model(model_config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(cfg["scheduler_factor"]),
        patience=int(cfg["scheduler_patience"]),
    )

    w_h = float(cfg["loss_weight_H"])
    w_c = float(cfg["loss_weight_C"])
    patience = int(cfg["early_stopping_patience"])

    best_state = None
    best_val_score = float("inf")
    best_epoch = None
    early_counter = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, int(cfg["num_epochs"]) + 1):
        if stop_event.is_set():
            break
        train_metrics = _run_epoch(model, train_loader, device, optimizer, w_h, w_c, stop_event)
        if train_metrics is None:  # stopped mid-epoch
            break
        val_metrics = _run_epoch(model, val_loader, device, None, w_h, w_c, stop_event)
        if val_metrics is None:
            break
        epochs_run = epoch

        val_score = (val_metrics["mae_H"] * w_h + val_metrics["mae_C"] * w_c) / 2
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        _append_history(
            state,
            {
                "epoch": epoch,
                "train_mae_H": train_metrics["mae_H"],
                "train_mse_H": train_metrics["mse_H"],
                "train_mae_C": train_metrics["mae_C"],
                "train_mse_C": train_metrics["mse_C"],
                "val_mae_H": val_metrics["mae_H"],
                "val_mse_H": val_metrics["mse_H"],
                "val_mae_C": val_metrics["mae_C"],
                "val_mse_C": val_metrics["mse_C"],
                "val_score": val_score,
                "lr": lr_now,
            },
        )

        if val_score <= best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early_counter = 0
            _update_job(state, best_epoch=best_epoch, best_val_score=best_val_score)
        else:
            early_counter += 1
            if early_counter >= patience:
                stopped_early = True
                break

    if stop_event.is_set():
        stopped_early = True

    if best_state is None:
        # Stopped before the first epoch finished: keep current weights.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs_run

    # Test evaluation with the best weights.
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = None
    if test_loader is not None:
        test_metrics = _run_epoch(model, test_loader, device, None, w_h, w_c, None)

    history = training_status(state).get("history", [])
    best_row = next((h for h in history if h["epoch"] == best_epoch), None)

    model_id = model_service.make_model_id(cfg["model_name"], cfg["operator_type"])
    sidecar = {
        "model_id": model_id,
        "display_name": cfg["model_name"] or f"{cfg['operator_type']} {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": entry["file_name"],
        "model_config": model_config,
        "training_config": {k: v for k, v in cfg.items()},
        "split": {**{f"{k}_compounds": [str(c) for c in v] for k, v in split_compounds.items()},
                  "val_fallback": val_fallback},
        "metrics": {
            "best_epoch": best_epoch,
            "best_val_score": best_val_score if best_val_score != float("inf") else None,
            "best_val_mae_H": best_row["val_mae_H"] if best_row else None,
            "best_val_mae_C": best_row["val_mae_C"] if best_row else None,
            "test_mae_H": test_metrics["mae_H"] if test_metrics else None,
            "test_mse_H": test_metrics["mse_H"] if test_metrics else None,
            "test_mae_C": test_metrics["mae_C"] if test_metrics else None,
            "test_mse_C": test_metrics["mse_C"] if test_metrics else None,
        },
        "epochs_trained": epochs_run,
        "stopped_early": stopped_early,
        "history": history,
    }
    norm_stats = dataset.norm_stats if dataset.normalize_node_features else None
    edge_stats = {
        "edge_length_mean": float(dataset.edge_length_mean),
        "edge_length_std": float(dataset.edge_length_std),
        "edge_order_mean": float(dataset.edge_order_mean),
        "edge_order_std": float(dataset.edge_order_std),
    }
    model_service.save_checkpoint(model_id, best_state, sidecar, norm_stats, edge_stats)

    _update_job(state, status="done", checkpoint_id=model_id, stopped_early=stopped_early)


def _run_epoch(model, loader, device, optimizer, w_h, w_c, stop_event):
    """One pass over a loader. optimizer=None -> eval mode. Returns averaged
    metrics, or None if stop_event fired mid-epoch."""
    training = optimizer is not None
    model.train() if training else model.eval()

    sums = {"mse_H": 0.0, "mae_H": 0.0, "mse_C": 0.0, "mae_C": 0.0}
    counts = {"H": 0, "C": 0}

    for batch in loader:
        if stop_event is not None and stop_event.is_set():
            return None
        batch = batch.to(device)
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        y_dict = {nt: getattr(batch[nt], "y", None) for nt in batch.node_types}
        edge_index_dict = {}
        edge_attr_dict = {}
        for store in batch.edge_stores:
            edge_index_dict[store._key] = store.edge_index
            if getattr(store, "edge_attr", None) is not None:
                edge_attr_dict[store._key] = store.edge_attr

        if training:
            optimizer.zero_grad()
            out = model(x_dict, edge_index_dict, edge_attr_dict or None)
        else:
            with torch.no_grad():
                out = model(x_dict, edge_index_dict, edge_attr_dict or None)

        loss_terms = []
        for ntype, weight in (("H", w_h), ("C", w_c)):
            pred_all = out.get(ntype)
            target_all = y_dict.get(ntype)
            if pred_all is None or target_all is None:
                continue
            mask = ~torch.isnan(target_all)
            if mask.sum() == 0:
                continue
            pred = pred_all[mask]
            target = target_all[mask]
            mse = torch.mean((pred - target) ** 2)
            mae = torch.mean(torch.abs(pred - target))
            sums[f"mse_{ntype}"] += float(mse)
            sums[f"mae_{ntype}"] += float(mae)
            counts[ntype] += 1
            if training:
                loss_terms.append(mae * weight)

        if training and loss_terms:
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            optimizer.step()

    return {
        "mse_H": sums["mse_H"] / counts["H"] if counts["H"] else float("nan"),
        "mae_H": sums["mae_H"] / counts["H"] if counts["H"] else float("nan"),
        "mse_C": sums["mse_C"] / counts["C"] if counts["C"] else float("nan"),
        "mae_C": sums["mae_C"] / counts["C"] if counts["C"] else float("nan"),
    }
