"""Node-level explanations: GNNExplainer (PyG) and Integrated Gradients (Captum).

Both methods produce a uniform result shape consumed by the frontend:
    {
        method, model_id, graph_idx, node_type, type_local_idx, atom_index,
        prediction, ground_truth, params, elapsed_s,
        own_features: [{name, raw_value, importance (0..1), signed (-1..1)}],
        neighbors:    [{atom_index, element, node_type, type_local_idx, hops, importance}],
        bonds:        [{a1, a2, importance}] | None   (None for IG),
        convergence_delta: float | None
    }

Explanations run in worker threads (job polling) because IG with neighbor
attributions can take tens of seconds on CPU. Results are cached per
(model, dataset, node, method, params).
"""
import hashlib
import json
import math
import threading
import time
import uuid

import networkx as nx
import torch

from . import pathsetup  # noqa: F401
from . import dataset_service, features

from scripts.explainer.explainer_utils import (  # noqa: E402
    NodeTypeRegressionWrapper,
    heterodata_to_dicts,
)
from scripts.explainer.ig_explainer import compute_ig_explanation  # noqa: E402

GNN_EXPLAINER_DEFAULTS = {
    "epochs": 100,
    "lr": 0.01,
    "explanation_type": "model",
    "k_hops": 2,
    # Regularization coefficients (PyG GNNExplainer `coeffs`). Higher size
    # coefficients push toward sparser (more compact) masks; higher entropy
    # coefficients push mask values toward a crisp 0/1 selection.
    "edge_size": 0.005,
    "node_feat_size": 1.0,
    "edge_ent": 1.0,
    "node_feat_ent": 0.1,
}
# Regularization coefficient keys exposed to the UI, clamped to non-negative.
GNN_EXPLAINER_COEFFS = ("edge_size", "node_feat_size", "edge_ent", "node_feat_ent")
IG_DEFAULTS = {
    "n_steps": 50,
    "baseline_type": "zero",
    "include_neighbors": True,
    "k_hops": 1,
    "explanation_type": "model",
}
IG_BASELINES = ("zero", "mean", "min", "max", "random")
MAX_JOBS = 20


def normalize_params(method, params):
    params = params or {}
    if method == "gnn_explainer":
        merged = {**GNN_EXPLAINER_DEFAULTS}
        merged["epochs"] = int(_clamp(params.get("epochs", merged["epochs"]), 10, 500))
        merged["lr"] = float(params.get("lr", merged["lr"]))
        merged["explanation_type"] = _expl_type(params)
        merged["k_hops"] = int(_clamp(params.get("k_hops", merged["k_hops"]), 1, 4))
        for coeff in GNN_EXPLAINER_COEFFS:
            merged[coeff] = _coeff(params.get(coeff), merged[coeff])
        return merged
    if method == "integrated_gradients":
        merged = {**IG_DEFAULTS}
        merged["n_steps"] = int(_clamp(params.get("n_steps", merged["n_steps"]), 8, 128))
        baseline = str(params.get("baseline_type", merged["baseline_type"]))
        if baseline not in IG_BASELINES:
            raise ValueError(f"baseline_type must be one of {IG_BASELINES}")
        merged["baseline_type"] = baseline
        merged["include_neighbors"] = bool(params.get("include_neighbors", True))
        merged["k_hops"] = int(_clamp(params.get("k_hops", merged["k_hops"]), 1, 4))
        merged["explanation_type"] = _expl_type(params)
        return merged
    raise ValueError(f"Unknown method '{method}'")


def _expl_type(params):
    value = str(params.get("explanation_type", "model"))
    if value not in ("model", "phenomenon"):
        raise ValueError("explanation_type must be 'model' or 'phenomenon'")
    return value


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _coeff(value, default):
    """A non-negative regularization coefficient, falling back to `default`
    for missing/blank/non-finite input (a cleared number field sends NaN)."""
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return _clamp(value, 0.0, 100.0)


def cache_key(state, request):
    params_hash = hashlib.sha1(
        json.dumps(request["params"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return (
        request["model_id"],
        state.active_dataset_file,
        request["graph_idx"],
        request["node_type"],
        request["type_local_idx"],
        request["method"],
        params_hash,
    )


def submit(state, request):
    """Return a cached result immediately or spawn a job. The request must be
    pre-validated (active model, indices, phenomenon/ground-truth)."""
    key = cache_key(state, request)
    cached = state.cache_get(key)
    if cached is not None:
        return {"status": "done", "cached": True, "result": cached}

    job_id = uuid.uuid4().hex[:12]
    with state.jobs_lock:
        state.explain_jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "_started_monotonic": time.monotonic(),
        }
        while len(state.explain_jobs) > MAX_JOBS:
            state.explain_jobs.popitem(last=False)

    thread = threading.Thread(
        target=_job_worker, args=(state, job_id, request, key), daemon=True
    )
    thread.start()
    return {"status": "pending", "job_id": job_id}


def job_status(state, job_id):
    with state.jobs_lock:
        job = state.explain_jobs.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
            "elapsed_s": round(time.monotonic() - job["_started_monotonic"], 1),
        }


def _job_worker(state, job_id, request, key):
    try:
        with state.inference_lock:
            result = _run_explanation(state, request)
        state.cache_put(key, result)
        with state.jobs_lock:
            if job_id in state.explain_jobs:
                state.explain_jobs[job_id].update(status="done", result=result)
    except Exception as exc:
        with state.jobs_lock:
            if job_id in state.explain_jobs:
                state.explain_jobs[job_id].update(
                    status="error", error=f"{type(exc).__name__}: {exc}"
                )


# --------------------------------------------------------------------------- #
# Core computation (called with inference_lock held)
# --------------------------------------------------------------------------- #
def _run_explanation(state, request):
    started = time.monotonic()
    device = state.device
    graph_idx = request["graph_idx"]
    node_type = request["node_type"]
    local_idx = request["type_local_idx"]
    method = request["method"]
    params = request["params"]

    with state.model_lock:
        active = state.active_model
    if active is None or active["view"] is None:
        raise LookupError("No active model loaded.")

    entry = state.active_dataset()
    nx_g = entry["nx_graphs"][graph_idx]
    node_map = dataset_service.get_node_map(entry, graph_idx)
    atom_index = node_map["by_type"][node_type][local_idx]

    data = active["view"][graph_idx]
    x_dict, edge_index_dict, edge_attr_dict, y_dict = heterodata_to_dicts(data)
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    if edge_attr_dict is not None:
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

    hops_map = nx.single_source_shortest_path_length(nx_g, atom_index, cutoff=params["k_hops"])
    hops_map.pop(atom_index, None)

    adapter = active["adapter"]
    with torch.no_grad():
        out = adapter(x_dict, edge_index_dict, edge_attr_dict)
    prediction = float(out[node_type].squeeze(-1)[local_idx])

    gt_raw = nx_g.nodes[atom_index].get(dataset_service.TARGET_ATTR, float("nan"))
    try:
        gt_raw = float(gt_raw)
    except (TypeError, ValueError):
        gt_raw = float("nan")
    ground_truth = gt_raw if math.isfinite(gt_raw) else None

    if method == "gnn_explainer":
        own, neighbors, bonds, delta = _run_gnn_explainer(
            adapter, node_type, local_idx, x_dict, edge_index_dict, edge_attr_dict,
            y_dict, params, node_map, nx_g, hops_map,
        )
    else:
        own, neighbors, bonds, delta = _run_integrated_gradients(
            adapter, data, node_type, local_idx, x_dict, edge_index_dict, edge_attr_dict,
            y_dict, params, node_map, nx_g, hops_map, device,
        )

    return {
        "method": method,
        "model_id": active["model_id"],
        "graph_idx": graph_idx,
        "node_type": node_type,
        "type_local_idx": local_idx,
        "atom_index": atom_index,
        "element": nx_g.nodes[atom_index]["element"],
        "prediction": prediction,
        "ground_truth": ground_truth,
        "params": params,
        "elapsed_s": round(time.monotonic() - started, 1),
        "own_features": own,
        "neighbors": neighbors,
        "bonds": bonds,
        "convergence_delta": delta,
    }


def _run_gnn_explainer(adapter, node_type, local_idx, x_dict, edge_index_dict,
                       edge_attr_dict, y_dict, params, node_map, nx_g, hops_map):
    from torch_geometric.explain import Explainer
    from torch_geometric.explain.algorithm import GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    wrapped = NodeTypeRegressionWrapper(adapter, node_type)
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(
            epochs=params["epochs"],
            lr=params["lr"],
            edge_size=params["edge_size"],
            node_feat_size=params["node_feat_size"],
            edge_ent=params["edge_ent"],
            node_feat_ent=params["node_feat_ent"],
        ),
        explanation_type=params["explanation_type"],
        model_config=ModelConfig(mode="regression", task_level="node", return_type="raw"),
        node_mask_type="attributes",
        edge_mask_type="object",
    )
    target = None
    if params["explanation_type"] == "phenomenon":
        target = y_dict[node_type].view(-1)

    try:
        explanation = explainer(
            x_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            target=target,
            index=local_idx,
        )
    finally:
        _clear_masks_safely(wrapped)

    node_mask_dict = {k: v.detach().cpu() for k, v in explanation.node_mask_dict.items()}
    edge_mask_dict = {k: v.detach().cpu() for k, v in explanation.edge_mask_dict.items()}

    atom_index = node_map["by_type"][node_type][local_idx]
    own = _own_feature_list(
        node_mask_dict[node_type][local_idx], node_type, nx_g.nodes[atom_index], signed=False
    )
    neighbors = _neighbor_list_from_masks(node_mask_dict, node_map, nx_g, hops_map)
    bonds = _bond_list(edge_mask_dict, edge_index_dict, node_map,
                       allowed_atoms=set(hops_map) | {atom_index})
    return own, neighbors, bonds, None


def _run_integrated_gradients(adapter, data, node_type, local_idx, x_dict,
                              edge_index_dict, edge_attr_dict, y_dict, params,
                              node_map, nx_g, hops_map, device):
    target = y_dict[node_type] if params["explanation_type"] == "phenomenon" else None
    result = compute_ig_explanation(
        base_model=adapter,
        data=data,
        node_type=node_type,
        node_idx=local_idx,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        device=device,
        target=target,
        n_steps=params["n_steps"],
        baseline_type=params["baseline_type"],
        include_neighbors=params["include_neighbors"],
        k_hops=params["k_hops"],
        explanation_type=params["explanation_type"],
    )

    own_mask = result["node_mask_dict"][node_type]
    own_row = own_mask[0] if own_mask.dim() == 2 else own_mask
    atom_index = node_map["by_type"][node_type][local_idx]
    own = _own_feature_list(
        own_row.detach().cpu(), node_type, nx_g.nodes[atom_index], signed=True
    )

    neighbors = []
    full = result.get("node_mask_full_dict") or {}
    valid = result.get("node_mask_full_valid") or {}
    if params["include_neighbors"] and full:
        scored = []
        for ntype, mask in full.items():
            mask = mask.detach().cpu()
            valid_flags = valid.get(ntype)
            for li in range(mask.shape[0]):
                if ntype == node_type and li == local_idx:
                    continue
                if valid_flags is not None and not bool(valid_flags[li]):
                    continue
                atom = node_map["by_type"][ntype][li]
                scored.append((float(mask[li].abs().mean()), atom, ntype, li))
        neighbors = _format_neighbors(scored, nx_g, hops_map)

    delta = result.get("convergence_delta")
    if delta is not None:
        try:
            delta = float(delta if not hasattr(delta, "item") else delta.item())
            if not math.isfinite(delta):
                delta = None
        except (TypeError, ValueError):
            delta = None
    return own, neighbors, None, delta


def _clear_masks_safely(model):
    try:
        from torch_geometric.explain.algorithm.utils import clear_masks

        clear_masks(model)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Tensor -> JSON helpers
# --------------------------------------------------------------------------- #
def _own_feature_list(values, node_type, atom_attrs, signed):
    values = values.view(-1)
    names = features.FEATURE_NAMES[node_type]
    raw_values = features.raw_values(atom_attrs, node_type)
    max_abs = float(values.abs().max()) if values.numel() else 0.0
    entries = []
    for i in range(values.numel()):
        score = float(values[i])
        if max_abs > 0:
            importance = abs(score) / max_abs
            signed_val = score / max_abs if signed else importance
        else:
            importance = 0.0
            signed_val = 0.0
        entries.append(
            {
                "feature_idx": i,
                "name": names[i] if i < len(names) else f"feature_{i}",
                "raw_value": raw_values[i] if i < len(raw_values) else None,
                "importance": importance,
                "signed": signed_val,
            }
        )
    entries.sort(key=lambda e: e["importance"], reverse=True)
    return entries


def _neighbor_list_from_masks(node_mask_dict, node_map, nx_g, hops_map):
    scored = []
    for atom, hops in hops_map.items():
        ntype, li = node_map["by_atom"][atom]
        mask = node_mask_dict.get(ntype)
        if mask is None or li >= mask.shape[0]:
            continue
        scored.append((float(mask[li].abs().mean()), atom, ntype, li))
    return _format_neighbors(scored, nx_g, hops_map)


def _format_neighbors(scored, nx_g, hops_map):
    if not scored:
        return []
    max_score = max(s for s, *_ in scored)
    neighbors = []
    for score, atom, ntype, li in scored:
        neighbors.append(
            {
                "atom_index": atom,
                "element": nx_g.nodes[atom]["element"],
                "node_type": ntype,
                "type_local_idx": li,
                "hops": hops_map.get(atom),
                "importance": (score / max_score) if max_score > 0 else 0.0,
                "raw_score": score,
            }
        )
    neighbors.sort(key=lambda n: n["importance"], reverse=True)
    return neighbors


def _bond_list(edge_mask_dict, edge_index_dict, node_map, allowed_atoms):
    pair_importance = {}
    for etype, mask in edge_mask_dict.items():
        edge_index = edge_index_dict.get(etype)
        if edge_index is None:
            continue
        src_type, _, dst_type = etype
        mask = mask.view(-1)
        edge_index = edge_index.cpu()
        limit = min(int(edge_index.shape[1]), int(mask.shape[0]))
        for pos in range(limit):
            u = node_map["by_type"][src_type][int(edge_index[0, pos])]
            v = node_map["by_type"][dst_type][int(edge_index[1, pos])]
            key = (min(u, v), max(u, v))
            value = abs(float(mask[pos]))
            if value > pair_importance.get(key, -1.0):
                pair_importance[key] = value

    bonds = [
        {"a1": a1, "a2": a2, "importance": imp}
        for (a1, a2), imp in pair_importance.items()
        if a1 in allowed_atoms and a2 in allowed_atoms
    ]
    if not bonds:
        return []
    max_imp = max(b["importance"] for b in bonds)
    if max_imp > 0:
        for b in bonds:
            b["importance"] = b["importance"] / max_imp
    bonds.sort(key=lambda b: b["importance"], reverse=True)
    return bonds
