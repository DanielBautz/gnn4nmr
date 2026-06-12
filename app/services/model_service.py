"""Model construction, checkpoint persistence and per-graph prediction.

Checkpoint layout in models/:
    <model_id>.pt          best state_dict
    <model_id>.stats.pkl   {"norm_stats": ..., "edge_stats": {...}} used at train time
    <model_id>.json        sidecar: model/training config, split, metrics, history
"""
import json
import math
import os
import pickle
import re
from datetime import datetime

import torch

from . import pathsetup  # noqa: F401
from .. import config
from . import dataset_service
from .jsonutil import to_jsonable

from model import HeteroGNNModel  # noqa: E402


class SafeModelAdapter(torch.nn.Module):
    """HeteroGNNModel.forward rebinds x_dict entries and (for GINEConv)
    re-projects edge_attr tensors in place. Hand it copies so callers can
    reuse their dicts across repeated forwards (predict loops, IG steps)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = dict(x_dict)
        ea = {k: v.clone() for k, v in edge_attr_dict.items()} if edge_attr_dict else None
        return self.base(x, edge_index_dict, ea)


def build_model(model_config, device):
    operator_type = model_config["operator_type"]
    operator_kwargs = dict(model_config.get("operator_kwargs") or {})
    if operator_type in ("GATConv", "GATv2Conv"):
        operator_kwargs.setdefault("add_self_loops", False)
    if operator_type == "NNConv":
        operator_kwargs.setdefault("edge_dim", model_config.get("edge_in_dim", config.EDGE_IN_DIM))
    model = HeteroGNNModel(
        in_dim_dict=model_config["in_dim_dict"],
        hidden_dim=model_config["hidden_dim"],
        out_dim=model_config["out_dim"],
        encoder_dropout=model_config["encoder_dropout"],
        gnnlayer_dropout=model_config["gnnlayer_dropout"],
        num_gnn_layers=model_config["num_gnn_layers"],
        operator_type=operator_type,
        operator_kwargs=operator_kwargs,
        edge_in_dim=model_config.get("edge_in_dim", config.EDGE_IN_DIM),
    ).to(device)
    return model


def make_model_id(model_name, operator_type):
    base = model_name.strip() or operator_type
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-") or operator_type
    return f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_checkpoint(model_id, state_dict, sidecar, norm_stats, edge_stats):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(state_dict, os.path.join(config.MODELS_DIR, f"{model_id}.pt"))
    with open(os.path.join(config.MODELS_DIR, f"{model_id}.stats.pkl"), "wb") as handle:
        pickle.dump({"norm_stats": norm_stats, "edge_stats": edge_stats}, handle)
    with open(os.path.join(config.MODELS_DIR, f"{model_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(sidecar), handle, indent=2)


def list_models(state):
    models = []
    if os.path.isdir(config.MODELS_DIR):
        for name in sorted(os.listdir(config.MODELS_DIR)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(config.MODELS_DIR, name), encoding="utf-8") as handle:
                    sidecar = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if "model_id" not in sidecar:
                continue
            models.append(_model_summary(sidecar))
    active_id = None
    with state.model_lock:
        if state.active_model is not None:
            active_id = state.active_model["model_id"]
    for m in models:
        m["active"] = m["model_id"] == active_id
    models.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return models, active_id


def _model_summary(sidecar):
    metrics = sidecar.get("metrics", {})
    return {
        "model_id": sidecar["model_id"],
        "display_name": sidecar.get("display_name", sidecar["model_id"]),
        "operator_type": sidecar.get("model_config", {}).get("operator_type"),
        "created_at": sidecar.get("created_at"),
        "dataset_file": sidecar.get("dataset_file"),
        "epochs_trained": sidecar.get("epochs_trained"),
        "stopped_early": sidecar.get("stopped_early", False),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_score": metrics.get("best_val_score"),
        "best_val_mae_H": metrics.get("best_val_mae_H"),
        "best_val_mae_C": metrics.get("best_val_mae_C"),
        "test_mae_H": metrics.get("test_mae_H"),
        "test_mae_C": metrics.get("test_mae_C"),
    }


def load_model(state, model_id):
    """Activate a checkpoint: rebuild model + normalized dataset view."""
    sidecar_path = os.path.join(config.MODELS_DIR, f"{model_id}.json")
    weights_path = os.path.join(config.MODELS_DIR, f"{model_id}.pt")
    stats_path = os.path.join(config.MODELS_DIR, f"{model_id}.stats.pkl")
    for path in (sidecar_path, weights_path, stats_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file missing: {os.path.basename(path)}")

    with open(sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    with open(stats_path, "rb") as handle:
        stats = pickle.load(handle)

    model = build_model(sidecar["model_config"], state.device)
    state_dict = torch.load(weights_path, map_location=state.device)
    model.load_state_dict(state_dict)
    model.eval()
    adapter = SafeModelAdapter(model)

    entry = state.active_dataset()
    view = None
    if entry is not None:
        view = dataset_service.make_normalized_view(
            entry["nx_graphs"], stats.get("norm_stats"), stats.get("edge_stats")
        )

    with state.model_lock:
        state.active_model = {
            "model_id": model_id,
            "sidecar": sidecar,
            "stats": stats,
            "adapter": adapter,
            "view": view,
        }
    state.cache_clear()
    return _model_summary(sidecar)


def rebuild_active_view(state):
    """Rebuild the active model's normalized view after a dataset switch."""
    with state.model_lock:
        active = state.active_model
    if active is None:
        return
    entry = state.active_dataset()
    if entry is None:
        return
    view = dataset_service.make_normalized_view(
        entry["nx_graphs"],
        active["stats"].get("norm_stats"),
        active["stats"].get("edge_stats"),
    )
    with state.model_lock:
        if state.active_model is active:
            active["view"] = view


def active_model_summary(state):
    with state.model_lock:
        active = state.active_model
    if active is None:
        return None
    return _model_summary(active["sidecar"])


def predict_graph(state, graph_idx):
    """Run the active model on one graph; map predictions back to atoms."""
    with state.model_lock:
        active = state.active_model
    if active is None or active["view"] is None:
        raise LookupError("No active model loaded.")

    entry = state.active_dataset()
    node_map = dataset_service.get_node_map(entry, graph_idx)
    data = active["view"][graph_idx]

    x_dict = {nt: data[nt].x.to(state.device) for nt in data.node_types}
    edge_index_dict = {}
    edge_attr_dict = {}
    for store in data.edge_stores:
        edge_index_dict[store._key] = store.edge_index.to(state.device)
        if getattr(store, "edge_attr", None) is not None:
            edge_attr_dict[store._key] = store.edge_attr.to(state.device)

    with state.inference_lock:
        with torch.no_grad():
            out = active["adapter"](x_dict, edge_index_dict, edge_attr_dict or None)

    predictions = {}
    summary = {}
    for ntype in ("H", "C"):
        rows = []
        abs_errors = []
        if ntype in data.node_types and out.get(ntype) is not None:
            preds = out[ntype].squeeze(-1).cpu()
            targets = data[ntype].y.squeeze(-1).cpu() if hasattr(data[ntype], "y") else None
            for local_idx in range(preds.shape[0]):
                pred = float(preds[local_idx])
                gt = float(targets[local_idx]) if targets is not None else float("nan")
                has_gt = math.isfinite(gt)
                if has_gt:
                    abs_errors.append(abs(pred - gt))
                rows.append(
                    {
                        "type_local_idx": local_idx,
                        "atom_index": node_map["by_type"][ntype][local_idx],
                        "pred": pred,
                        "ground_truth": gt if has_gt else None,
                        "abs_error": abs(pred - gt) if has_gt else None,
                    }
                )
        predictions[ntype] = rows
        summary[f"mae_{ntype}"] = sum(abs_errors) / len(abs_errors) if abs_errors else None
        summary[f"n_{ntype}_labeled"] = len(abs_errors)

    return {
        "graph_idx": graph_idx,
        "model_id": active["model_id"],
        "predictions": predictions,
        "summary": summary,
    }
