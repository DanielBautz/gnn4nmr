"""JSON API blueprint. Thin request handling; logic lives in app/services."""
from flask import Blueprint, jsonify, request

from . import config
from .state import STATE
from .services import (
    dataset_service,
    explain_service,
    model_service,
    training_service,
)
from .services.jsonutil import to_jsonable

api = Blueprint("api", __name__, url_prefix="/api")


def _ok(payload, status=200):
    return jsonify(to_jsonable(payload)), status


def _err(code, message, status):
    return jsonify({"error": code, "message": message}), status


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@api.get("/health")
def health():
    return _ok({"ok": True, "device": str(STATE.device)})


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@api.get("/datasets")
def datasets():
    files = dataset_service.list_dataset_files()
    with STATE.dataset_lock:
        loaded = dict(STATE.datasets)
        active = STATE.active_dataset_file
    items = []
    for name in files:
        entry = loaded.get(name)
        items.append(
            {
                "file_name": name,
                "loaded": entry is not None,
                "active": name == active,
                "num_graphs": len(entry["nx_graphs"]) if entry else None,
                "num_compounds": entry["num_compounds"] if entry else None,
            }
        )
    return _ok({"datasets": items, "active": active})


@api.post("/datasets/select")
def select_dataset():
    body = request.get_json(silent=True) or {}
    file_name = body.get("file_name")
    if not file_name:
        return _err("bad_request", "file_name is required.", 400)
    if STATE.training_active():
        return _err("training_running", "Cannot switch datasets while training.", 409)
    try:
        entry = dataset_service.load_dataset(STATE, file_name)
    except FileNotFoundError as exc:
        return _err("not_found", str(exc), 404)
    except Exception as exc:
        return _err("load_failed", f"Could not load dataset: {exc}", 400)
    model_service.rebuild_active_view(STATE)
    STATE.cache_clear()
    return _ok(
        {
            "ok": True,
            "file_name": file_name,
            "num_graphs": len(entry["nx_graphs"]),
            "num_compounds": entry["num_compounds"],
        }
    )


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
def _active_entry_or_error():
    entry = STATE.active_dataset()
    if entry is None:
        return None, _err("no_dataset", "No dataset loaded.", 409)
    return entry, None


def _check_graph_idx(entry, graph_idx):
    if graph_idx < 0 or graph_idx >= len(entry["nx_graphs"]):
        return _err("not_found", f"graph_idx {graph_idx} out of range.", 404)
    return None


@api.get("/graphs")
def graphs():
    entry, error = _active_entry_or_error()
    if error:
        return error
    return _ok({"graphs": entry["summaries"], "dataset_file": entry["file_name"]})


@api.get("/graphs/<int:graph_idx>")
def graph_detail(graph_idx):
    entry, error = _active_entry_or_error()
    if error:
        return error
    error = _check_graph_idx(entry, graph_idx)
    if error:
        return error
    return _ok(dataset_service.molecule_detail(entry, graph_idx))


@api.get("/graphs/<int:graph_idx>/predictions")
def graph_predictions(graph_idx):
    entry, error = _active_entry_or_error()
    if error:
        return error
    error = _check_graph_idx(entry, graph_idx)
    if error:
        return error
    try:
        return _ok(model_service.predict_graph(STATE, graph_idx))
    except LookupError:
        return _err("no_active_model", "No model loaded. Train or load one first.", 409)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@api.post("/train")
def train_start():
    entry, error = _active_entry_or_error()
    if error:
        return error
    body = request.get_json(silent=True) or {}

    cfg = dict(config.TRAINING_DEFAULTS)
    unknown = set(body) - set(cfg)
    if unknown:
        return _err("bad_request", f"Unknown training parameters: {sorted(unknown)}", 400)
    cfg.update(body)

    if cfg["operator_type"] not in config.OPERATOR_TYPES:
        return _err(
            "bad_request",
            f"operator_type must be one of {config.OPERATOR_TYPES}",
            400,
        )
    cfg["dataset_file"] = str(cfg.get("dataset_file") or "")
    if not cfg["dataset_file"]:
        cfg["dataset_file"] = entry["file_name"]
    elif cfg["dataset_file"] not in dataset_service.list_dataset_files():
        return _err("bad_request", f"Unknown dataset file: {cfg['dataset_file']}", 400)
    try:
        cfg["model_name"] = str(cfg.get("model_name") or "")
        for key in ("hidden_dim", "out_dim", "num_gnn_layers", "batch_size", "num_epochs",
                    "seed", "scheduler_patience", "early_stopping_patience"):
            cfg[key] = int(cfg[key])
        for key in ("encoder_dropout", "gnnlayer_dropout", "lr", "weight_decay",
                    "loss_weight_H", "loss_weight_C", "scheduler_factor"):
            cfg[key] = float(cfg[key])
        ratio = [float(r) for r in cfg["split_ratio"]]
        if len(ratio) != 3 or any(r < 0 for r in ratio) or sum(ratio) > 1.0001 or ratio[0] <= 0:
            raise ValueError("split_ratio must be three fractions with train > 0.")
        cfg["split_ratio"] = ratio
        if cfg["num_epochs"] < 1 or cfg["batch_size"] < 1 or cfg["num_gnn_layers"] < 1:
            raise ValueError("num_epochs, batch_size and num_gnn_layers must be >= 1.")
    except (TypeError, ValueError) as exc:
        return _err("bad_request", f"Invalid training parameters: {exc}", 400)

    job = training_service.start_training(STATE, cfg)
    if job is None:
        return _err("training_already_running", "A training job is already running.", 409)
    return _ok({"ok": True, "job": training_service.training_status(STATE)}, status=202)


@api.get("/train/status")
def train_status():
    return _ok(training_service.training_status(STATE))


@api.post("/train/stop")
def train_stop():
    if not training_service.request_stop(STATE):
        return _err("not_running", "No training job is running.", 409)
    return _ok({"ok": True})


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@api.get("/models")
def models():
    items, active_id = model_service.list_models(STATE)
    return _ok({"models": items, "active_model_id": active_id})


@api.post("/models/<model_id>/load")
def model_load(model_id):
    try:
        summary = model_service.load_model(STATE, model_id)
    except FileNotFoundError as exc:
        return _err("not_found", str(exc), 404)
    except Exception as exc:
        return _err("load_failed", f"Could not load model: {exc}", 400)
    return _ok({"ok": True, "model": summary})


@api.get("/models/active")
def model_active():
    return _ok({"model": model_service.active_model_summary(STATE), "device": str(STATE.device)})


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
@api.post("/explain")
def explain():
    entry, error = _active_entry_or_error()
    if error:
        return error
    with STATE.model_lock:
        active = STATE.active_model
    if active is None:
        return _err("no_active_model", "No model loaded. Train or load one first.", 409)

    body = request.get_json(silent=True) or {}
    try:
        graph_idx = int(body["graph_idx"])
        node_type = str(body["node_type"])
        local_idx = int(body["type_local_idx"])
        method = str(body["method"])
    except (KeyError, TypeError, ValueError):
        return _err(
            "bad_request",
            "graph_idx, node_type, type_local_idx and method are required.",
            400,
        )

    error = _check_graph_idx(entry, graph_idx)
    if error:
        return error
    if node_type not in ("H", "C"):
        return _err("bad_request", "node_type must be 'H' or 'C' (predicted types).", 400)

    node_map = dataset_service.get_node_map(entry, graph_idx)
    if local_idx < 0 or local_idx >= len(node_map["by_type"][node_type]):
        return _err("bad_request", f"type_local_idx {local_idx} out of range.", 400)

    try:
        params = explain_service.normalize_params(method, body.get("params"))
    except ValueError as exc:
        return _err("bad_request", str(exc), 400)

    if params["explanation_type"] == "phenomenon":
        atom_index = node_map["by_type"][node_type][local_idx]
        gt = entry["nx_graphs"][graph_idx].nodes[atom_index].get(dataset_service.TARGET_ATTR)
        try:
            valid_gt = gt is not None and gt == gt  # not NaN
        except Exception:
            valid_gt = False
        if not valid_gt:
            return _err(
                "bad_request",
                "Phenomenon explanations need a ground-truth shift; this atom has none.",
                400,
            )

    explain_request = {
        "model_id": active["model_id"],
        "graph_idx": graph_idx,
        "node_type": node_type,
        "type_local_idx": local_idx,
        "method": method,
        "params": params,
    }
    response = explain_service.submit(STATE, explain_request)
    return _ok(response, status=200 if response["status"] == "done" else 202)


@api.get("/explain/jobs/<job_id>")
def explain_job(job_id):
    status = explain_service.job_status(STATE, job_id)
    if status is None:
        return _err("not_found", "Unknown or expired job id.", 404)
    return _ok(status)
