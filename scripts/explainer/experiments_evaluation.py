from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from tqdm.auto import tqdm as _tqdm_auto
except Exception:
    _tqdm_auto = None
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

from scripts.explainer.explainer_utils import (
    NodeTypeRegressionWrapper,
    build_dataset,
    get_device,
    heterodata_to_dicts,
    load_config,
    load_stats,
    load_trained_model,
)
from scripts.explainer.ig_explainer import compute_ig_explanation


EXP_METHOD_GNN = "gnnexplainer"
EXP_METHOD_IG = "integrated_gradients"
EXP_ALLOWED_NODE_TYPES = ("H", "C", "Others")
EXP_PLOT_METHOD_COLORS = {
    EXP_METHOD_GNN: "#004e9f",
    EXP_METHOD_IG: "#fcba00",
}
EXP_PLOT_NEUTRAL_COLOR = "#909085"


def exp_pretty_method_name(method: str) -> str:
    method_norm = str(method).strip().lower()
    if method_norm == EXP_METHOD_GNN:
        return "GNNExplainer"
    if method_norm == EXP_METHOD_IG:
        return "Integrated Gradients"
    return method_norm.replace("_", " ").title()


def exp_pretty_metric_name(metric: str) -> str:
    metric_norm = str(metric).strip().lower()
    if metric_norm == "fid_plus_model":
        return "Fidelity+"
    if metric_norm == "fid_minus_model":
        return "Fidelity-"
    return metric_norm.replace("_", " ").title()


def exp_pretty_mask_name(mask_variant: str) -> str:
    mask_norm = str(mask_variant).strip().lower()
    if mask_norm == "zero":
        return "Zero Mask"
    if mask_norm == "scientific":
        return "Scientific Mask"
    return mask_norm.replace("_", " ").title()


@dataclass
class ExperimentContext:
    model_path: Path
    data_path: Path
    config_path: Path
    norm_stats_path: Path
    edge_stats_path: Path
    split_path: Path
    output_dir: Path


def safe_tqdm(
    iterable,
    total: Optional[int],
    desc: str,
    enabled: bool = True,
    use_tqdm: bool = True,
):
    if not enabled:
        return iterable
    if use_tqdm and _tqdm_auto is not None:
        return _tqdm_auto(iterable, total=total, desc=desc, leave=False)
    return iterable


def resolve_path(path_like: str, root: Optional[Path] = None) -> Path:
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.exists():
        return path_obj
    if root is None:
        return path_obj
    candidate = root / path_obj
    if candidate.exists():
        return candidate
    return candidate


def load_split_metadata(split_path: Path) -> Optional[Dict[str, Any]]:
    if not split_path.exists():
        return None
    try:
        with split_path.open("rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def exp_load_eval_graph_indices(split_path: str, total_graphs: Optional[int] = None) -> List[int]:
    split_meta = load_split_metadata(Path(split_path))
    if not split_meta:
        return []

    candidates = (
        split_meta.get("test_graph_indices"),
        split_meta.get("test_indices"),
        split_meta.get("test_graphs"),
    )
    for candidate in candidates:
        if not isinstance(candidate, (list, tuple)):
            continue
        result: List[int] = []
        seen = set()
        for value in candidate:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if total_graphs is not None and (idx < 0 or idx >= total_graphs):
                continue
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        if result:
            result.sort()
            return result
    return []


def select_scope_graph_indices(
    dataset: Any,
    split_path: Path,
    graph_scope: str,
) -> List[int]:
    scope = (graph_scope or "test_split").strip().lower()
    if scope not in {"test_split", "full_dataset"}:
        raise ValueError("graph_scope must be one of: test_split | full_dataset")

    if scope == "full_dataset":
        return list(range(len(dataset)))

    # test_split
    test_indices = exp_load_eval_graph_indices(str(split_path), total_graphs=len(dataset))
    if test_indices:
        return test_indices

    # controlled fallback
    return list(range(len(dataset)))


def first_graph_indices_per_component(
    dataset: Any,
    graph_indices: Sequence[int],
    component_key: str = "compound",
) -> List[int]:
    selected: Dict[str, int] = {}
    for graph_idx in sorted(int(i) for i in graph_indices):
        nx_g = dataset.nx_graphs[graph_idx]
        comp_val = nx_g.graph.get(component_key, "unknown")
        comp_key = str(comp_val)
        if comp_key not in selected:
            selected[comp_key] = graph_idx
    return sorted(selected.values())


def get_train_graph_indices(split_path: Path, total_graphs: int) -> List[int]:
    split_meta = load_split_metadata(split_path)
    if split_meta:
        candidates = (
            split_meta.get("train_graph_indices"),
            split_meta.get("train_indices"),
            split_meta.get("train_graphs"),
        )
        for candidate in candidates:
            if not isinstance(candidate, (list, tuple)):
                continue
            seen = set()
            result: List[int] = []
            for value in candidate:
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < total_graphs and idx not in seen:
                    seen.add(idx)
                    result.append(idx)
            if result:
                result.sort()
                return result

    fallback_size = max(1, int(0.8 * total_graphs))
    return list(range(fallback_size))


def exp_topk_indices_from_importance(importance: np.ndarray, sparsity: float) -> np.ndarray:
    values = np.asarray(importance, dtype=float).reshape(-1)
    d = int(values.size)
    if d <= 0:
        return np.array([], dtype=np.int64)

    keep_ratio = min(max(1.0 - float(sparsity), 0.0), 1.0)
    k_keep = max(1, int(math.ceil(keep_ratio * d)))
    k_keep = min(k_keep, d)

    abs_values = np.abs(values)
    abs_values = np.where(np.isfinite(abs_values), abs_values, -np.inf)
    order = np.argsort(-abs_values)
    return order[:k_keep].astype(np.int64)


def actual_sparsity(total_dim: int, k_keep: int) -> float:
    if total_dim <= 0:
        return float("nan")
    return 1.0 - (float(k_keep) / float(total_dim))


def get_target_value(y_dict: Dict[str, Optional[torch.Tensor]], node_type: str, node_idx: int) -> float:
    y_tensor = y_dict.get(node_type)
    if y_tensor is None:
        return float("nan")
    flat = y_tensor.reshape(-1)
    if node_idx < 0 or node_idx >= flat.size(0):
        return float("nan")
    value = flat[node_idx]
    if torch.isnan(value):
        return float("nan")
    return float(value.item())


def exp_predict_single_node(
    model: torch.nn.Module,
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
    node_type: str,
    node_idx: int,
) -> float:
    x_input = {nt: feat.clone() for nt, feat in x_dict.items()}
    if edge_attr_dict is None:
        edge_attr_input = None
    else:
        edge_attr_input = {
            etype: (attrs.clone() if attrs is not None else None)
            for etype, attrs in edge_attr_dict.items()
        }

    with torch.no_grad():
        out_dict = model(x_input, edge_index_dict, edge_attr_input)
        if node_type not in out_dict or out_dict[node_type] is None:
            raise ValueError(f"Model output does not contain node type '{node_type}'.")
        values = out_dict[node_type].reshape(-1)
        if node_idx < 0 or node_idx >= values.size(0):
            raise IndexError(
                f"Node index {node_idx} out of bounds for node type "
                f"'{node_type}' with size {values.size(0)}."
            )
        return float(values[node_idx].item())


def exp_build_mask_baseline(
    mode: str,
    ig_baseline_mode: str,
    target_features: torch.Tensor,
    node_features: torch.Tensor,
    node_type: str,
    dataset: Optional[Any] = None,
    train_graph_indices: Optional[List[int]] = None,
    median_cache: Optional[Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = None,
    elem_distribution_cache: Optional[Dict[str, torch.Tensor]] = None,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    mode = (mode or "match_ig_baseline").lower()
    ig_baseline_mode = (ig_baseline_mode or "scientific").lower()

    if mode not in {"zero", "mean", "match_ig_baseline"}:
        raise ValueError("mask baseline mode must be one of: zero | mean | match_ig_baseline")

    if mode == "zero":
        return torch.zeros_like(target_features)
    if mode == "mean":
        return torch.mean(node_features, dim=0)

    # match_ig_baseline
    if ig_baseline_mode == "zero":
        return torch.zeros_like(target_features)
    if ig_baseline_mode == "mean":
        return torch.mean(node_features, dim=0)
    if ig_baseline_mode == "random":
        mean_val = torch.mean(node_features, dim=0)
        std_val = torch.std(node_features, dim=0)
        return mean_val + std_val * torch.randn_like(target_features)
    if ig_baseline_mode == "min":
        return torch.min(node_features, dim=0)[0]
    if ig_baseline_mode == "max":
        return torch.max(node_features, dim=0)[0]
    if ig_baseline_mode != "scientific":
        raise ValueError(f"Unsupported ig_baseline_mode: {ig_baseline_mode}")

    if dataset is None:
        raise ValueError("Scientific baseline requires dataset.")
    if train_graph_indices is None:
        train_graph_indices = list(range(max(1, int(0.8 * len(dataset)))))
    if median_cache is None:
        median_cache = {}
    if elem_distribution_cache is None:
        elem_distribution_cache = {}
    if cache_dir is None:
        cache_dir = Path("baselines")

    from scripts.explainer.baselines import (
        build_scientific_baseline,
        load_or_compute_element_distribution,
        load_or_compute_medians,
    )

    if node_type not in median_cache:
        median_cache[node_type] = load_or_compute_medians(
            dataset=dataset,
            train_graph_indices=train_graph_indices,
            node_type=node_type,
            cache_dir=str(cache_dir),
        )
    if "value" not in elem_distribution_cache:
        elem_distribution_cache["value"] = load_or_compute_element_distribution(
            dataset=dataset,
            train_graph_indices=train_graph_indices,
            cache_dir=str(cache_dir),
        )

    median_global, median_by_element = median_cache[node_type]
    elem_distribution = elem_distribution_cache.get("value")

    try:
        baseline_cpu = build_scientific_baseline(
            target_features=target_features.detach().cpu(),
            node_type=node_type,
            median_global=median_global.detach().cpu(),
            median_by_element=median_by_element,
            elem_distribution=elem_distribution.detach().cpu() if elem_distribution is not None else None,
        )
        return baseline_cpu.to(target_features.device)
    except Exception:
        # Robust fallback to avoid dropping all rows if scientific assertions fail.
        return torch.mean(node_features, dim=0)


def exp_extract_gnn_feature_importance(explanation: Any, node_type: str, node_idx: int) -> np.ndarray:
    if not hasattr(explanation, "node_mask_dict"):
        raise ValueError("Explanation has no node_mask_dict.")
    mask = explanation.node_mask_dict.get(node_type)
    if mask is None:
        raise ValueError(f"No node mask for node type '{node_type}'.")

    if mask.dim() == 2:
        if node_idx < 0 or node_idx >= mask.size(0):
            raise IndexError(
                f"Node index {node_idx} out of bounds for node mask with {mask.size(0)} rows."
            )
        values = mask[node_idx]
    else:
        values = mask
    return values.detach().cpu().numpy().astype(float)


def exp_extract_gnn_global_feature_importance(explanation: Any) -> Dict[str, np.ndarray]:
    if not hasattr(explanation, "node_mask_dict"):
        return {}
    node_mask_dict = getattr(explanation, "node_mask_dict", {})
    if not isinstance(node_mask_dict, dict):
        return {}

    out: Dict[str, np.ndarray] = {}
    for curr_type, mask in node_mask_dict.items():
        if mask is None:
            continue
        arr = (
            mask.detach().cpu().numpy().astype(float)
            if isinstance(mask, torch.Tensor)
            else np.asarray(mask, dtype=float)
        )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            continue
        out[str(curr_type)] = arr
    return out


def exp_extract_ig_feature_importance(
    explanation_result: Dict[str, Any],
    node_type: str,
    node_idx: int,
) -> np.ndarray:
    selected_node = explanation_result.get("selected_node", {})
    attrs = selected_node.get("attributions")
    if attrs is not None:
        return np.asarray(attrs, dtype=float).reshape(-1)

    node_mask_dict = explanation_result.get("node_mask_dict", {})
    mask = node_mask_dict.get(node_type)
    if mask is None:
        raise ValueError("IG result does not contain node attributions.")

    if mask.dim() == 2:
        row_idx = int(node_idx)
        if row_idx < 0 or row_idx >= mask.size(0):
            row_idx = 0
        values = mask[row_idx]
    else:
        values = mask
    return values.detach().cpu().numpy().astype(float)


def exp_extract_ig_global_feature_importance(
    explanation_result: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    full_mask_dict = explanation_result.get("node_mask_full_dict", {})
    full_valid_dict = explanation_result.get("node_mask_full_valid", {})
    out: Dict[str, np.ndarray] = {}

    if isinstance(full_mask_dict, dict):
        for node_type, mask in full_mask_dict.items():
            if mask is None:
                continue
            arr = (
                mask.detach().cpu().numpy().astype(float)
                if isinstance(mask, torch.Tensor)
                else np.asarray(mask, dtype=float)
            )
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                continue

            valid = None
            if isinstance(full_valid_dict, dict):
                valid_mask = full_valid_dict.get(node_type)
                if valid_mask is not None:
                    valid = (
                        valid_mask.detach().cpu().numpy().astype(bool).reshape(-1)
                        if isinstance(valid_mask, torch.Tensor)
                        else np.asarray(valid_mask, dtype=bool).reshape(-1)
                    )
            if valid is not None and valid.size == arr.shape[0]:
                if not np.any(valid):
                    continue
                arr = arr.copy()
                arr[~valid] = np.nan
            out[node_type] = arr

    if out:
        return out

    # Compatibility fallback: use node_mask_dict when it already contains full matrices.
    node_mask_dict = explanation_result.get("node_mask_dict", {})
    if isinstance(node_mask_dict, dict):
        for node_type, mask in node_mask_dict.items():
            if mask is None:
                continue
            arr = (
                mask.detach().cpu().numpy().astype(float)
                if isinstance(mask, torch.Tensor)
                else np.asarray(mask, dtype=float)
            )
            if arr.ndim == 2 and arr.shape[0] > 1:
                out[node_type] = arr
    return out


def exp_feature_fidelity_for_node(
    base_model: torch.nn.Module,
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
    node_type: str,
    node_idx: int,
    importance: np.ndarray,
    sparsity: float,
    mask_baseline_mode: str,
    ig_baseline_mode: str,
    dataset: Optional[Any] = None,
    train_graph_indices: Optional[List[int]] = None,
    median_cache: Optional[Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = None,
    elem_distribution_cache: Optional[Dict[str, torch.Tensor]] = None,
    cache_dir: Optional[Path] = None,
    importance_by_node_type: Optional[Dict[str, np.ndarray]] = None,
    pred_orig: Optional[float] = None,
    target_value: float = float("nan"),
) -> Dict[str, float]:
    if node_type not in x_dict:
        raise ValueError(f"Node type '{node_type}' not present in x_dict.")

    node_matrix = x_dict[node_type]
    if node_idx < 0 or node_idx >= node_matrix.size(0):
        raise IndexError(f"Node index {node_idx} out of bounds for node type '{node_type}'.")

    feature_count = int(node_matrix.size(1))
    x_drop = {nt: feat.clone() for nt, feat in x_dict.items()}
    x_keep = {nt: feat.clone() for nt, feat in x_dict.items()}

    mask_scope: Dict[str, Dict[int, np.ndarray]] = {}
    if importance_by_node_type:
        for curr_type, curr_importance in importance_by_node_type.items():
            if curr_type not in x_dict or curr_importance is None:
                continue
            arr = np.asarray(curr_importance, dtype=float)
            if arr.ndim == 1:
                # 1D vectors are ambiguous for non-target types; keep only the safe case.
                if curr_type == node_type:
                    mask_scope.setdefault(curr_type, {})[int(node_idx)] = arr.reshape(-1)
                elif int(x_dict[curr_type].size(0)) == 1:
                    mask_scope.setdefault(curr_type, {})[0] = arr.reshape(-1)
                continue
            if arr.ndim != 2:
                continue

            node_limit = min(int(x_dict[curr_type].size(0)), int(arr.shape[0]))
            for curr_idx in range(node_limit):
                row = np.asarray(arr[curr_idx], dtype=float).reshape(-1)
                if row.size == 0 or not np.isfinite(row).any():
                    continue
                mask_scope.setdefault(curr_type, {})[curr_idx] = row

    # Always enforce the selected-node vector to match the current target explanation.
    target_importance = np.asarray(importance, dtype=float).reshape(-1)
    mask_scope.setdefault(node_type, {})[int(node_idx)] = target_importance

    total_feature_count = 0
    total_k_keep = 0
    target_k_keep = int(exp_topk_indices_from_importance(target_importance, sparsity=sparsity).size)

    for curr_type, row_map in mask_scope.items():
        curr_matrix = x_dict[curr_type]
        curr_feature_count = int(curr_matrix.size(1))
        for curr_idx in sorted(row_map.keys()):
            if curr_idx < 0 or curr_idx >= int(curr_matrix.size(0)):
                continue
            row_importance = np.asarray(row_map[curr_idx], dtype=float).reshape(-1)
            if row_importance.size != curr_feature_count:
                if curr_type == node_type and curr_idx == int(node_idx):
                    raise ValueError(
                        f"Importance size mismatch for {curr_type}[{curr_idx}]: "
                        f"got {row_importance.size}, expected {curr_feature_count}."
                    )
                continue

            keep_indices = exp_topk_indices_from_importance(row_importance, sparsity=sparsity)
            k_keep = int(keep_indices.size)

            baseline_vec = exp_build_mask_baseline(
                mode=mask_baseline_mode,
                ig_baseline_mode=ig_baseline_mode,
                target_features=curr_matrix[curr_idx],
                node_features=curr_matrix,
                node_type=curr_type,
                dataset=dataset,
                train_graph_indices=train_graph_indices,
                median_cache=median_cache,
                elem_distribution_cache=elem_distribution_cache,
                cache_dir=cache_dir,
            )

            selected = curr_matrix[curr_idx].clone()
            drop_vec = selected.clone()
            if k_keep > 0:
                drop_vec[keep_indices] = baseline_vec[keep_indices]

            keep_vec = baseline_vec.clone()
            if k_keep > 0:
                keep_vec[keep_indices] = selected[keep_indices]

            x_drop[curr_type][curr_idx] = drop_vec
            x_keep[curr_type][curr_idx] = keep_vec

            total_feature_count += curr_feature_count
            total_k_keep += k_keep
            if curr_type == node_type and curr_idx == int(node_idx):
                target_k_keep = k_keep

    if total_feature_count <= 0:
        total_feature_count = feature_count
        total_k_keep = target_k_keep

    if pred_orig is None:
        pred_orig = exp_predict_single_node(
            base_model,
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            node_type,
            node_idx,
        )

    pred_drop = exp_predict_single_node(
        base_model,
        x_drop,
        edge_index_dict,
        edge_attr_dict,
        node_type,
        node_idx,
    )
    pred_keep = exp_predict_single_node(
        base_model,
        x_keep,
        edge_index_dict,
        edge_attr_dict,
        node_type,
        node_idx,
    )

    fid_plus_model = abs(pred_orig - pred_drop)
    fid_minus_model = abs(pred_orig - pred_keep)

    if np.isfinite(target_value):
        err_orig = abs(pred_orig - target_value)
        err_drop = abs(pred_drop - target_value)
        err_keep = abs(pred_keep - target_value)
        fid_plus_error_delta = err_drop - err_orig
        fid_minus_error_delta = err_keep - err_orig
    else:
        err_orig = float("nan")
        err_drop = float("nan")
        err_keep = float("nan")
        fid_plus_error_delta = float("nan")
        fid_minus_error_delta = float("nan")

    return {
        "pred_orig": float(pred_orig),
        "pred_drop": float(pred_drop),
        "pred_keep": float(pred_keep),
        "target": float(target_value) if np.isfinite(target_value) else float("nan"),
        "error_orig": float(err_orig),
        "error_drop": float(err_drop),
        "error_keep": float(err_keep),
        "fid_plus_model": float(fid_plus_model),
        "fid_minus_model": float(fid_minus_model),
        "fid_plus_error_delta": float(fid_plus_error_delta),
        "fid_minus_error_delta": float(fid_minus_error_delta),
        "n_features": int(total_feature_count),
        "k_features": int(total_k_keep),
        "actual_sparsity_feat": float(actual_sparsity(total_feature_count, total_k_keep)),
        "n_features_target": int(feature_count),
        "k_features_target": int(target_k_keep),
    }


def filter_edges_by_mask(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
    selected_positions: Dict[Tuple[str, str, str], Set[int]],
    mode: str,
) -> Tuple[Dict[Tuple[str, str, str], torch.Tensor], Optional[Dict[Tuple[str, str, str], torch.Tensor]]]:
    new_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    new_edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = {} if edge_attr_dict is not None else None

    for edge_type, edge_index in edge_index_dict.items():
        num_edges = int(edge_index.size(1))

        selected = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
        selected_set = selected_positions.get(edge_type, set())
        if selected_set:
            idx_tensor = torch.tensor(sorted(selected_set), dtype=torch.long, device=edge_index.device)
            idx_tensor = idx_tensor[(idx_tensor >= 0) & (idx_tensor < num_edges)]
            if idx_tensor.numel() > 0:
                selected[idx_tensor] = True

        if mode == "drop_selected":
            keep_mask = ~selected
        elif mode == "keep_selected":
            keep_mask = selected
        else:
            raise ValueError(f"Unknown edge mask mode: {mode}")

        new_edge_index_dict[edge_type] = edge_index[:, keep_mask]

        if new_edge_attr_dict is not None and edge_type in edge_attr_dict and edge_attr_dict[edge_type] is not None:
            new_edge_attr_dict[edge_type] = edge_attr_dict[edge_type][keep_mask]

    return new_edge_index_dict, new_edge_attr_dict


def exp_gnn_edge_fidelity_for_node(
    base_model: torch.nn.Module,
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
    edge_mask_dict: Dict[Tuple[str, str, str], torch.Tensor],
    node_type: str,
    node_idx: int,
    sparsity: float,
    pred_orig: Optional[float] = None,
    target_value: float = float("nan"),
) -> Optional[Dict[str, float]]:
    all_edge_entries: List[Tuple[float, Tuple[str, str, str], int]] = []

    for edge_type, edge_index in edge_index_dict.items():
        num_edges = int(edge_index.size(1))
        if num_edges <= 0:
            continue

        edge_mask = edge_mask_dict.get(edge_type) if isinstance(edge_mask_dict, dict) else None
        mask_vals = None
        if edge_mask is not None:
            mask_vals = edge_mask.view(-1).detach().cpu()

        for pos in range(num_edges):
            if mask_vals is not None and pos < int(mask_vals.shape[0]):
                score_abs = float(abs(mask_vals[pos].item()))
            else:
                # Missing mask entries are treated as zero-importance edges.
                score_abs = 0.0
            all_edge_entries.append((score_abs, edge_type, int(pos)))

    total_edges_considered = len(all_edge_entries)
    if total_edges_considered == 0:
        return None

    keep_ratio = min(max(1.0 - float(sparsity), 0.0), 1.0)
    k_keep = max(1, int(math.ceil(keep_ratio * total_edges_considered)))
    k_keep = min(k_keep, total_edges_considered)

    all_edge_entries.sort(key=lambda item: item[0], reverse=True)
    selected_entries = all_edge_entries[:k_keep]
    selected_positions: Dict[Tuple[str, str, str], Set[int]] = {}
    for _, edge_type, edge_pos in selected_entries:
        selected_positions.setdefault(edge_type, set()).add(edge_pos)

    edge_idx_drop, edge_attr_drop = filter_edges_by_mask(
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        selected_positions=selected_positions,
        mode="drop_selected",
    )
    edge_idx_keep, edge_attr_keep = filter_edges_by_mask(
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        selected_positions=selected_positions,
        mode="keep_selected",
    )

    if pred_orig is None:
        pred_orig = exp_predict_single_node(
            base_model,
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            node_type,
            node_idx,
        )

    pred_drop = exp_predict_single_node(
        base_model,
        x_dict,
        edge_idx_drop,
        edge_attr_drop,
        node_type,
        node_idx,
    )
    pred_keep = exp_predict_single_node(
        base_model,
        x_dict,
        edge_idx_keep,
        edge_attr_keep,
        node_type,
        node_idx,
    )

    fid_plus_model = abs(pred_orig - pred_drop)
    fid_minus_model = abs(pred_orig - pred_keep)

    if np.isfinite(target_value):
        err_orig = abs(pred_orig - target_value)
        err_drop = abs(pred_drop - target_value)
        err_keep = abs(pred_keep - target_value)
        fid_plus_error_delta = err_drop - err_orig
        fid_minus_error_delta = err_keep - err_orig
    else:
        err_orig = float("nan")
        err_drop = float("nan")
        err_keep = float("nan")
        fid_plus_error_delta = float("nan")
        fid_minus_error_delta = float("nan")

    return {
        "pred_orig": float(pred_orig),
        "pred_drop": float(pred_drop),
        "pred_keep": float(pred_keep),
        "target": float(target_value) if np.isfinite(target_value) else float("nan"),
        "error_orig": float(err_orig),
        "error_drop": float(err_drop),
        "error_keep": float(err_keep),
        "fid_plus_model": float(fid_plus_model),
        "fid_minus_model": float(fid_minus_model),
        "fid_plus_error_delta": float(fid_plus_error_delta),
        "fid_minus_error_delta": float(fid_minus_error_delta),
        "n_edges_considered": int(total_edges_considered),
        "k_edges": int(k_keep),
        "actual_sparsity_edge": float(actual_sparsity(total_edges_considered, k_keep)),
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    metric_cols = [
        "fid_plus_model",
        "fid_minus_model",
        "fid_plus_error_delta",
        "fid_minus_error_delta",
        "actual_sparsity_feat",
    ]

    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(["method", "node_type"], dropna=False)
    for (method, node_type), group_df in grouped:
        row: Dict[str, Any] = {
            "method": method,
            "node_type": node_type,
            "n_nodes": int(len(group_df)),
            "n_graphs": int(group_df["graph_idx"].nunique()),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce")
            valid = values.dropna()
            row[f"{metric}_mean"] = float(valid.mean()) if not valid.empty else float("nan")
            row[f"{metric}_std"] = float(valid.std(ddof=0)) if not valid.empty else float("nan")
            row[f"{metric}_count"] = int(valid.shape[0])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["node_type", "method"]).reset_index(drop=True)


def build_fidelity_grid_summary(node_df: pd.DataFrame) -> pd.DataFrame:
    if node_df is None or node_df.empty:
        return pd.DataFrame()

    required = {"mask_variant", "sparsity_target", "method", "node_type"}
    missing = required - set(node_df.columns)
    if missing:
        raise ValueError(
            f"Grid summary requires columns {sorted(required)}, missing: {sorted(missing)}"
        )

    metric_cols = [
        "fid_plus_model",
        "fid_minus_model",
        "fid_plus_error_delta",
        "fid_minus_error_delta",
        "actual_sparsity_feat",
    ]
    grouped = node_df.groupby(
        ["mask_variant", "sparsity_target", "method", "node_type"],
        dropna=False,
    )

    rows: List[Dict[str, Any]] = []
    for keys, group_df in grouped:
        mask_variant, sparsity_target, method, node_type = keys
        row: Dict[str, Any] = {
            "mask_variant": str(mask_variant),
            "sparsity_target": float(sparsity_target),
            "method": str(method),
            "node_type": str(node_type),
            "n_nodes": int(len(group_df)),
            "n_graphs": int(group_df["graph_idx"].nunique()) if "graph_idx" in group_df.columns else int(0),
        }
        for metric in metric_cols:
            if metric not in group_df.columns:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_count"] = int(0)
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce")
            valid = values.dropna()
            row[f"{metric}_mean"] = float(valid.mean()) if not valid.empty else float("nan")
            row[f"{metric}_std"] = float(valid.std(ddof=0)) if not valid.empty else float("nan")
            row[f"{metric}_count"] = int(valid.shape[0])
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["mask_variant", "sparsity_target", "node_type", "method"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def plot_fidelity_vs_sparsity(
    summary_df: pd.DataFrame,
    mask_variant: str,
    metric: str,
    output_path: Path,
    title: str,
    node_type: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    neutral_color: str = EXP_PLOT_NEUTRAL_COLOR,
) -> Path:
    if metric not in {"fid_plus_model", "fid_minus_model"}:
        raise ValueError("metric must be one of: fid_plus_model | fid_minus_model")
    mean_col = f"{metric}_mean"
    metric_label = exp_pretty_metric_name(metric)
    import matplotlib.pyplot as plt

    def _save_placeholder(message: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel(f"{metric_label} (Mean)")
        ax.grid(True, alpha=0.25, color=neutral_color)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    if summary_df is None or summary_df.empty:
        _save_placeholder("No data available.")
        return output_path
    if mean_col not in summary_df.columns:
        _save_placeholder(f"Missing column: {mean_col}")
        return output_path

    plot_df = summary_df[summary_df["mask_variant"] == str(mask_variant)].copy()
    if node_type is not None and "node_type" in plot_df.columns:
        plot_df = plot_df[plot_df["node_type"] == str(node_type)].copy()
    if plot_df.empty:
        if node_type is None:
            _save_placeholder(f"No rows for mask variant '{mask_variant}'.")
        else:
            _save_placeholder(
                f"No rows for mask variant '{mask_variant}' and node type '{node_type}'."
            )
        return output_path

    method_colors = color_map if color_map is not None else EXP_PLOT_METHOD_COLORS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    plotted = False
    if node_type is None and "node_type" in plot_df.columns:
        grouped = plot_df.groupby(["method", "node_type"], dropna=False)
    else:
        grouped = plot_df.groupby(["method"], dropna=False)
    for group_keys, group_df in grouped:
        if isinstance(group_keys, tuple):
            method = str(group_keys[0])
            group_node_type = str(group_keys[1]) if len(group_keys) > 1 else ""
        else:
            method = str(group_keys)
            group_node_type = str(node_type) if node_type is not None else ""
        x = pd.to_numeric(group_df["sparsity_target"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group_df[mean_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        order = np.argsort(x[finite])
        x_plot = x[finite][order]
        y_plot = y[finite][order]
        method_label = exp_pretty_method_name(method)
        if node_type is None and group_node_type:
            series_label = f"{method_label} ({group_node_type})"
        else:
            series_label = method_label
        ax.plot(
            x_plot,
            y_plot,
            marker="o",
            linewidth=1.8,
            color=method_colors.get(method, neutral_color),
            markerfacecolor=method_colors.get(method, neutral_color),
            markeredgecolor=neutral_color,
            label=series_label,
        )
        plotted = True

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No finite values to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Sparsity")
    ax.set_ylabel(f"{metric_label} (Mean)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, color=neutral_color)
    if plotted:
        ax.legend(loc="best", title="Explainer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def shared_node_keys(
    node_df: pd.DataFrame,
    methods: Sequence[str],
    node_types: Sequence[str],
) -> Set[Tuple[int, str, int]]:
    if node_df.empty:
        return set()

    key_cols = ["graph_idx", "node_type", "node_idx"]
    work_df = node_df[node_df["node_type"].isin(node_types)]

    method_sets: List[Set[Tuple[int, str, int]]] = []
    for method in methods:
        mdf = work_df[work_df["method"] == method]
        key_set = set(tuple(row) for row in mdf[key_cols].itertuples(index=False, name=None))
        method_sets.append(key_set)

    if not method_sets:
        return set()

    shared = method_sets[0]
    for keys in method_sets[1:]:
        shared = shared & keys
    return shared


def run_experiments_evaluation(
    context: ExperimentContext,
    node_types: Sequence[str] = ("H", "C"),
    sparsity: float = 0.90,
    mask_baseline_mode: str = "match_ig_baseline",
    ig_baseline_mode: str = "scientific",
    gnn_epochs: int = 200,
    gnn_lr: float = 0.01,
    gnn_explanation_type: str = "phenomenon",
    ig_explanation_type: Optional[str] = None,
    gnn_use_custom_coeffs: bool = False,
    gnn_coeffs: Optional[Dict[str, float]] = None,
    ig_n_steps: int = 64,
    ig_scientific_strict: bool = False,
    graph_scope: str = "test_split",
    first_graph_per_component: bool = True,
    component_key: str = "compound",
    max_graphs: Optional[int] = None,
    max_nodes_per_graph: int = 0,
    include_gnn_edge_report: bool = True,
    seed: int = 42,
    verbose: bool = True,
    progress_enabled: bool = True,
    progress_use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_node_types = [nt for nt in node_types if nt in EXP_ALLOWED_NODE_TYPES]
    if not clean_node_types:
        raise ValueError("No valid node types selected.")

    gnn_explanation_type_norm = str(gnn_explanation_type).strip().lower()
    if gnn_explanation_type_norm not in {"model", "phenomenon"}:
        raise ValueError("gnn_explanation_type must be one of: model | phenomenon")
    if ig_explanation_type is None:
        ig_explanation_type_norm = gnn_explanation_type_norm
    else:
        ig_explanation_type_norm = str(ig_explanation_type).strip().lower()
    if ig_explanation_type_norm not in {"model", "phenomenon"}:
        raise ValueError("ig_explanation_type must be one of: model | phenomenon")

    device = get_device(None)
    rng = random.Random(int(seed))

    config = load_config(str(context.config_path))
    norm_stats, edge_stats = load_stats(str(context.norm_stats_path), str(context.edge_stats_path))
    dataset = build_dataset(str(context.data_path), config, norm_stats=norm_stats, edge_stats=edge_stats)
    base_model = load_trained_model(str(context.model_path), config, device)

    requested_scope = (graph_scope or "test_split").strip().lower()
    scope_graph_indices = select_scope_graph_indices(
        dataset=dataset,
        split_path=context.split_path,
        graph_scope=requested_scope,
    )

    if requested_scope == "test_split":
        split_test_indices = exp_load_eval_graph_indices(str(context.split_path), total_graphs=len(dataset))
        if split_test_indices:
            graph_source = "test_split"
        else:
            graph_source = "full_dataset_fallback_missing_or_invalid_split"
    else:
        graph_source = "full_dataset"

    graph_count_before_component_filter = len(scope_graph_indices)

    if first_graph_per_component:
        eval_graph_indices = first_graph_indices_per_component(
            dataset=dataset,
            graph_indices=scope_graph_indices,
            component_key=component_key,
        )
    else:
        eval_graph_indices = list(scope_graph_indices)

    graph_count_after_component_filter = len(eval_graph_indices)

    if max_graphs is not None and max_graphs > 0 and max_graphs < len(eval_graph_indices):
        eval_graph_indices = eval_graph_indices[: max_graphs]

    train_graph_indices = get_train_graph_indices(context.split_path, total_graphs=len(dataset))

    if verbose:
        print(f"Device: {device}")
        print(f"Requested graph scope: {requested_scope}")
        print(f"Graph source: {graph_source}")
        print(f"First graph per component: {bool(first_graph_per_component)} (key='{component_key}')")
        print(f"Graphs before component filter: {graph_count_before_component_filter}")
        print(f"Graphs after component filter: {graph_count_after_component_filter}")
        print(f"Graphs evaluated: {len(eval_graph_indices)}")
        print(f"Node types: {clean_node_types}")
        print(f"Sparsity target: {sparsity}")
        print(
            f"Explanation types: GNN={gnn_explanation_type_norm}, "
            f"IG={ig_explanation_type_norm}"
        )
        print(
            f"Mask baseline mode: {mask_baseline_mode} "
            f"(IG baseline mode: {ig_baseline_mode})"
        )
        if str(ig_baseline_mode).lower() == "scientific":
            print(f"IG scientific strict mode: {bool(ig_scientific_strict)}")

    median_cache: Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
    elem_distribution_cache: Dict[str, torch.Tensor] = {}
    if ig_baseline_mode == "scientific":
        from scripts.explainer.baselines import (
            load_or_compute_element_distribution,
            load_or_compute_medians,
        )

        for node_type in clean_node_types:
            try:
                median_cache[node_type] = load_or_compute_medians(
                    dataset=dataset,
                    train_graph_indices=train_graph_indices,
                    node_type=node_type,
                    cache_dir=str(context.output_dir.parent / "baselines"),
                )
            except Exception:
                continue
        try:
            elem_distribution_cache["value"] = load_or_compute_element_distribution(
                dataset=dataset,
                train_graph_indices=train_graph_indices,
                cache_dir=str(context.output_dir.parent / "baselines"),
            )
        except Exception:
            pass

    feature_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    graph_total = int(len(eval_graph_indices))
    graph_iter = safe_tqdm(
        eval_graph_indices,
        total=graph_total,
        desc="Graph evaluation",
        enabled=bool(progress_enabled),
        use_tqdm=bool(progress_use_tqdm),
    )
    for graph_pos, graph_idx in enumerate(graph_iter, start=1):
        if verbose and (not progress_enabled or not progress_use_tqdm):
            print(f"[graphs] {graph_pos}/{graph_total}: graph_idx={graph_idx}")
        data = dataset[graph_idx].to(device)
        x_dict_raw, edge_index_dict, edge_attr_dict_raw, y_dict = heterodata_to_dicts(data)

        x_dict_base = {nt: feat.clone() for nt, feat in x_dict_raw.items()}
        if edge_attr_dict_raw is None:
            edge_attr_base = None
        else:
            edge_attr_base = {
                etype: (attrs.clone() if attrs is not None else None)
                for etype, attrs in edge_attr_dict_raw.items()
            }

        for node_type in clean_node_types:
            if node_type not in x_dict_base:
                continue

            num_nodes = int(x_dict_base[node_type].size(0))
            if num_nodes <= 0:
                continue

            node_indices = list(range(num_nodes))
            if max_nodes_per_graph and max_nodes_per_graph > 0 and max_nodes_per_graph < len(node_indices):
                node_indices = sorted(rng.sample(node_indices, max_nodes_per_graph))

            target_tensor = y_dict.get(node_type)
            if target_tensor is not None:
                target_tensor = target_tensor.reshape(-1)

            wrapped_model = NodeTypeRegressionWrapper(base_model, node_type)
            model_config = ModelConfig(
                mode=ModelMode.regression,
                task_level=ModelTaskLevel.node,
                return_type=ModelReturnType.raw,
            )

            if gnn_use_custom_coeffs and gnn_coeffs:
                custom_coeffs = {key: float(value) for key, value in dict(gnn_coeffs).items()}

                def _coeffs_applied(algo: Any) -> bool:
                    algo_coeffs = getattr(algo, "coeffs", None)
                    if not isinstance(algo_coeffs, dict):
                        return False
                    for key, expected in custom_coeffs.items():
                        if key not in algo_coeffs:
                            return False
                        actual = algo_coeffs[key]
                        if not np.isclose(float(actual), float(expected), rtol=0.0, atol=0.0):
                            return False
                    return True

                algorithm = None

                # Preferred on modern PyG: coefficients as explicit kwargs.
                try:
                    candidate = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr, **custom_coeffs)
                    if _coeffs_applied(candidate):
                        algorithm = candidate
                except TypeError:
                    pass

                if algorithm is None:
                    # Robust fallback: instantiate default and patch coeff dict in-place.
                    candidate = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr)
                    algo_coeffs = getattr(candidate, "coeffs", None)
                    if isinstance(algo_coeffs, dict):
                        algo_coeffs.update(custom_coeffs)
                    if not _coeffs_applied(candidate):
                        raise TypeError(
                            "Could not apply custom GNNExplainer coeffs. "
                            "Check torch_geometric version and coeff names."
                        )
                    algorithm = candidate
            else:
                algorithm = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr)

            gnn_explainer = Explainer(
                model=wrapped_model,
                algorithm=algorithm,
                explanation_type=gnn_explanation_type_norm,
                model_config=model_config,
                node_mask_type="attributes",
                edge_mask_type="object",
            )

            gnn_target = target_tensor if (gnn_explanation_type_norm == "phenomenon" and target_tensor is not None) else None
            ig_target = target_tensor if (ig_explanation_type_norm == "phenomenon" and target_tensor is not None) else None

            for node_idx in node_indices:
                target_value = get_target_value(y_dict, node_type, node_idx)
                try:
                    pred_orig = exp_predict_single_node(
                        base_model,
                        x_dict_base,
                        edge_index_dict,
                        edge_attr_base,
                        node_type,
                        node_idx,
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "graph_idx": graph_idx,
                            "node_type": node_type,
                            "node_idx": node_idx,
                            "method": "prediction",
                            "error": str(exc),
                        }
                    )
                    continue

                # GNNExplainer
                try:
                    if gnn_target is not None and torch.isnan(gnn_target[node_idx]):
                        raise ValueError("Target is NaN for phenomenon explanation.")
                    gnn_explanation = gnn_explainer(
                        {nt: feat.clone() for nt, feat in x_dict_base.items()},
                        edge_index_dict,
                        edge_attr_dict=(
                            {etype: (attrs.clone() if attrs is not None else None) for etype, attrs in edge_attr_base.items()}
                            if edge_attr_base is not None
                            else None
                        ),
                        target=gnn_target,
                        index=node_idx,
                    )

                    gnn_importance = exp_extract_gnn_feature_importance(
                        gnn_explanation, node_type=node_type, node_idx=node_idx
                    )
                    gnn_importance_scope = exp_extract_gnn_global_feature_importance(gnn_explanation)
                    gnn_metrics = exp_feature_fidelity_for_node(
                        base_model=base_model,
                        x_dict=x_dict_base,
                        edge_index_dict=edge_index_dict,
                        edge_attr_dict=edge_attr_base,
                        node_type=node_type,
                        node_idx=node_idx,
                        importance=gnn_importance,
                        sparsity=sparsity,
                        mask_baseline_mode=mask_baseline_mode,
                        ig_baseline_mode=ig_baseline_mode,
                        dataset=dataset,
                        train_graph_indices=train_graph_indices,
                        median_cache=median_cache,
                        elem_distribution_cache=elem_distribution_cache,
                        cache_dir=context.output_dir.parent / "baselines",
                        importance_by_node_type=gnn_importance_scope,
                        pred_orig=pred_orig,
                        target_value=target_value,
                    )
                    feature_rows.append(
                        {
                            "graph_idx": int(graph_idx),
                            "node_type": node_type,
                            "node_idx": int(node_idx),
                            "method": EXP_METHOD_GNN,
                            "sparsity_target": float(sparsity),
                            "mask_baseline_mode": mask_baseline_mode,
                            "ig_baseline_mode": ig_baseline_mode,
                            **gnn_metrics,
                        }
                    )

                    if include_gnn_edge_report:
                        edge_metrics = exp_gnn_edge_fidelity_for_node(
                            base_model=base_model,
                            x_dict=x_dict_base,
                            edge_index_dict=edge_index_dict,
                            edge_attr_dict=edge_attr_base,
                            edge_mask_dict=gnn_explanation.edge_mask_dict,
                            node_type=node_type,
                            node_idx=node_idx,
                            sparsity=sparsity,
                            pred_orig=pred_orig,
                            target_value=target_value,
                        )
                        if edge_metrics is not None:
                            edge_rows.append(
                                {
                                    "graph_idx": int(graph_idx),
                                    "node_type": node_type,
                                    "node_idx": int(node_idx),
                                    "method": EXP_METHOD_GNN,
                                    "sparsity_target": float(sparsity),
                                    **edge_metrics,
                                }
                            )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "graph_idx": graph_idx,
                            "node_type": node_type,
                            "node_idx": node_idx,
                            "method": EXP_METHOD_GNN,
                            "error": str(exc),
                        }
                    )

                # Integrated Gradients
                try:
                    if ig_target is not None and torch.isnan(ig_target[node_idx]):
                        raise ValueError("Target is NaN for IG phenomenon explanation.")
                    if ig_explanation_type_norm == "phenomenon" and ig_target is None:
                        raise ValueError("No target tensor for IG phenomenon explanation.")
                    try:
                        ig_result = compute_ig_explanation(
                            base_model=base_model,
                            data=data,
                            node_type=node_type,
                            node_idx=node_idx,
                            x_dict={nt: feat.clone() for nt, feat in x_dict_base.items()},
                            edge_index_dict=edge_index_dict,
                            edge_attr_dict=(
                                {etype: (attrs.clone() if attrs is not None else None) for etype, attrs in edge_attr_base.items()}
                                if edge_attr_base is not None
                                else None
                            ),
                            device=device,
                            target=ig_target,
                            n_steps=ig_n_steps,
                            baseline_type=ig_baseline_mode,
                            include_neighbors=False,
                            include_all_nodes=True,
                            k_hops=1,
                            median_global=(median_cache[node_type][0] if node_type in median_cache else None),
                            median_by_element=(median_cache[node_type][1] if node_type in median_cache else None),
                            elem_distribution=elem_distribution_cache.get("value"),
                            dataset=dataset,
                            train_graph_indices=train_graph_indices,
                            train_split_path=str(context.split_path) if context.split_path.exists() else None,
                            explanation_type=ig_explanation_type_norm,
                        )
                    except Exception as exc_first:
                        scientific_mode = str(ig_baseline_mode).lower() == "scientific"
                        if scientific_mode and (not ig_scientific_strict):
                            ig_result = compute_ig_explanation(
                                base_model=base_model,
                                data=data,
                                node_type=node_type,
                                node_idx=node_idx,
                                x_dict={nt: feat.clone() for nt, feat in x_dict_base.items()},
                                edge_index_dict=edge_index_dict,
                                edge_attr_dict=(
                                    {etype: (attrs.clone() if attrs is not None else None) for etype, attrs in edge_attr_base.items()}
                                    if edge_attr_base is not None
                                    else None
                                ),
                                device=device,
                                target=ig_target,
                                n_steps=ig_n_steps,
                                baseline_type="mean",
                                include_neighbors=False,
                                include_all_nodes=True,
                                k_hops=1,
                                median_global=None,
                                median_by_element=None,
                                elem_distribution=None,
                                dataset=dataset,
                                train_graph_indices=train_graph_indices,
                                train_split_path=str(context.split_path) if context.split_path.exists() else None,
                                explanation_type=ig_explanation_type_norm,
                            )
                        else:
                            raise exc_first

                    ig_importance = exp_extract_ig_feature_importance(
                        ig_result,
                        node_type=node_type,
                        node_idx=node_idx,
                    )
                    ig_importance_scope = exp_extract_ig_global_feature_importance(ig_result)
                    ig_metrics = exp_feature_fidelity_for_node(
                        base_model=base_model,
                        x_dict=x_dict_base,
                        edge_index_dict=edge_index_dict,
                        edge_attr_dict=edge_attr_base,
                        node_type=node_type,
                        node_idx=node_idx,
                        importance=ig_importance,
                        sparsity=sparsity,
                        mask_baseline_mode=mask_baseline_mode,
                        ig_baseline_mode=ig_baseline_mode,
                        dataset=dataset,
                        train_graph_indices=train_graph_indices,
                        median_cache=median_cache,
                        elem_distribution_cache=elem_distribution_cache,
                        cache_dir=context.output_dir.parent / "baselines",
                        importance_by_node_type=ig_importance_scope,
                        pred_orig=pred_orig,
                        target_value=target_value,
                    )
                    feature_rows.append(
                        {
                            "graph_idx": int(graph_idx),
                            "node_type": node_type,
                            "node_idx": int(node_idx),
                            "method": EXP_METHOD_IG,
                            "sparsity_target": float(sparsity),
                            "mask_baseline_mode": mask_baseline_mode,
                            "ig_baseline_mode": ig_baseline_mode,
                            **ig_metrics,
                        }
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "graph_idx": graph_idx,
                            "node_type": node_type,
                            "node_idx": node_idx,
                            "method": EXP_METHOD_IG,
                            "error": str(exc),
                        }
                    )

    node_df = pd.DataFrame(feature_rows)
    gnn_edge_df = pd.DataFrame(edge_rows)
    summary_method_type_df = build_summary_table(node_df)

    shared_keys = shared_node_keys(
        node_df=node_df,
        methods=[EXP_METHOD_GNN, EXP_METHOD_IG],
        node_types=clean_node_types,
    )
    if shared_keys:
        node_key_tuples = list(node_df[["graph_idx", "node_type", "node_idx"]].itertuples(index=False, name=None))
        fair_mask = [key in shared_keys for key in node_key_tuples]
        fair_df = node_df.loc[fair_mask].copy()
    else:
        fair_df = node_df.iloc[0:0].copy()
    summary_fair_df = build_summary_table(fair_df)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    node_csv = context.output_dir / f"node_level_metrics_{timestamp}.csv"
    summary_csv = context.output_dir / f"summary_by_method_and_type_{timestamp}.csv"
    fair_csv = context.output_dir / f"fair_comparison_H_C_{timestamp}.csv"
    edge_csv = context.output_dir / f"gnn_edge_summary_{timestamp}.csv"
    failure_csv = context.output_dir / f"failed_nodes_{timestamp}.csv"
    config_json = context.output_dir / f"run_config_{timestamp}.json"

    node_df.to_csv(node_csv, index=False)
    summary_method_type_df.to_csv(summary_csv, index=False)
    summary_fair_df.to_csv(fair_csv, index=False)
    if include_gnn_edge_report:
        gnn_edge_df.to_csv(edge_csv, index=False)
    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(failure_csv, index=False)

    run_config = {
        "timestamp": timestamp,
        "model_path": str(context.model_path),
        "data_path": str(context.data_path),
        "split_path": str(context.split_path),
        "graph_scope": requested_scope,
        "graph_source": graph_source,
        "first_graph_per_component": bool(first_graph_per_component),
        "component_key": str(component_key),
        "graph_count_before_component_filter": int(graph_count_before_component_filter),
        "graph_count_after_component_filter": int(graph_count_after_component_filter),
        "graphs_evaluated": len(eval_graph_indices),
        "node_types": list(clean_node_types),
        "sparsity_target": float(sparsity),
        "mask_baseline_mode": mask_baseline_mode,
        "ig_baseline_mode": ig_baseline_mode,
        "gnn_epochs": int(gnn_epochs),
        "gnn_lr": float(gnn_lr),
        "gnn_explanation_type": gnn_explanation_type_norm,
        "ig_explanation_type": ig_explanation_type_norm,
        "gnn_use_custom_coeffs": bool(gnn_use_custom_coeffs),
        "gnn_coeffs": (
            {str(k): float(v) for k, v in dict(gnn_coeffs).items()}
            if gnn_use_custom_coeffs and gnn_coeffs
            else None
        ),
        "ig_n_steps": int(ig_n_steps),
        "ig_scientific_strict": bool(ig_scientific_strict),
        "gnn_edge_selection_scope": "global_all_edges",
        "gnn_edge_keep_mode": "topk_only",
        "gnn_edge_ranking": "abs(edge_mask)",
        "max_graphs": int(max_graphs) if max_graphs is not None else None,
        "max_nodes_per_graph": int(max_nodes_per_graph),
        "rows_node_metrics": int(len(node_df)),
        "rows_gnn_edge": int(len(gnn_edge_df)),
        "rows_failures": int(len(failure_df)),
        "shared_nodes_count": int(len(shared_keys)),
        "artifacts": {
            "node_level_metrics": str(node_csv),
            "summary_by_method_and_type": str(summary_csv),
            "fair_comparison_H_C": str(fair_csv),
            "gnn_edge_summary": str(edge_csv) if include_gnn_edge_report else None,
            "failed_nodes": str(failure_csv),
        },
    }
    with config_json.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    if verbose:
        print("Saved artifacts:")
        print(f"  node metrics: {node_csv}")
        print(f"  summary by method/type: {summary_csv}")
        print(f"  fair H/C summary: {fair_csv}")
        if include_gnn_edge_report:
            print(f"  gnn edge summary: {edge_csv}")
        print(f"  failures: {failure_csv}")
        print(f"  config: {config_json}")

    return node_df, summary_method_type_df, summary_fair_df, gnn_edge_df


def build_default_context(
    project_root: Path,
    model_file: str,
    data_file: str,
    split_file: str = "models/graph_split.pkl",
    output_dir: str = "results/experiments",
) -> ExperimentContext:
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    return ExperimentContext(
        model_path=resolve_path(model_file, models_dir),
        data_path=resolve_path(data_file, data_dir),
        config_path=models_dir / "config.pkl",
        norm_stats_path=models_dir / "norm_stats.pkl",
        edge_stats_path=models_dir / "edge_stats.pkl",
        split_path=resolve_path(split_file, project_root),
        output_dir=resolve_path(output_dir, project_root),
    )


def run_fidelity_grid_evaluation(
    context: ExperimentContext,
    node_types: Sequence[str] = ("H", "C"),
    sparsities: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    mask_variants: Sequence[str] = ("zero", "scientific"),
    gnn_epochs: int = 200,
    gnn_lr: float = 0.01,
    gnn_explanation_type: str = "phenomenon",
    ig_n_steps: int = 64,
    max_graphs: Optional[int] = None,
    max_nodes_per_graph: int = 0,
    include_gnn_edge_report: bool = True,
    seed: int = 42,
    verbose: bool = True,
    progress_enabled: bool = True,
    progress_use_tqdm: bool = True,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not sparsities:
        raise ValueError("sparsities must not be empty.")
    if not mask_variants:
        raise ValueError("mask_variants must not be empty.")

    mask_to_baseline = {
        "zero": "zero",
        "scientific": "match_ig_baseline",
    }
    normalized_masks = [str(m).strip().lower() for m in mask_variants]
    invalid_masks = sorted(set(m for m in normalized_masks if m not in mask_to_baseline))
    if invalid_masks:
        raise ValueError(
            f"Unsupported mask variants: {invalid_masks}. Allowed: {sorted(mask_to_baseline.keys())}"
        )

    sparsity_values = [float(s) for s in sparsities]
    for s in sparsity_values:
        if not np.isfinite(s) or s < 0.0 or s > 1.0:
            raise ValueError(f"Invalid sparsity value: {s}. Expected finite value in [0, 1].")

    fixed_gnn_coeffs = {
        "edge_size": 1e-3,
        "edge_ent": 1e-3,
        "node_feat_size": 1e-12,
        "node_feat_ent": 1e-12,
    }

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_name = str(run_name).strip() if run_name is not None else ""
    safe_name = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw_name)
    run_slug = safe_name if safe_name else f"fidelity_grid_{run_stamp}"

    run_dir = Path(context.output_dir) / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    progress_node_csv = run_dir / "grid_progress_node_metrics.csv"
    progress_summary_csv = run_dir / "grid_progress_summary.csv"
    progress_edge_csv = run_dir / "grid_progress_edge_metrics.csv"
    progress_edge_summary_csv = run_dir / "grid_progress_edge_summary.csv"
    progress_status_csv = run_dir / "grid_progress_run_status.csv"
    progress_manifest_json = run_dir / "grid_progress_manifest.json"

    final_node_csv = run_dir / "grid_node_metrics_all.csv"
    final_summary_csv = run_dir / "grid_summary_all.csv"
    final_plot_input_csv = run_dir / "grid_plot_input.csv"
    final_edge_csv = run_dir / "grid_edge_metrics_all.csv"
    final_edge_summary_csv = run_dir / "grid_edge_summary_all.csv"
    final_edge_plot_input_csv = run_dir / "grid_edge_plot_input.csv"
    final_status_csv = run_dir / "grid_run_status.csv"
    final_manifest_json = run_dir / "grid_manifest_final.json"

    run_plan: List[Tuple[float, str]] = []
    for sparsity in sparsity_values:
        for mask_variant in normalized_masks:
            run_plan.append((float(sparsity), str(mask_variant)))

    def _last_file(path: Path, pattern: str) -> Optional[str]:
        matches = sorted(path.glob(pattern))
        if not matches:
            return None
        return str(matches[-1])

    combined_node_parts: List[pd.DataFrame] = []
    combined_edge_parts: List[pd.DataFrame] = []
    run_records: List[Dict[str, Any]] = []

    if verbose:
        print("Starting fidelity grid evaluation")
        print(f"  output dir: {run_dir}")
        print(f"  runs: {len(run_plan)}")
        print(f"  sparsities: {sparsity_values}")
        print(f"  mask variants: {normalized_masks}")
        print("  fixed scope: graph_scope='test_split', first_graph_per_component=True, component_key='compound'")
        print(f"  fixed GNN coeffs: {fixed_gnn_coeffs}")

    total_runs = int(len(run_plan))
    for run_idx, (sparsity, mask_variant) in enumerate(run_plan, start=1):
        run_id = f"run_{run_idx:02d}_s{int(round(sparsity * 100.0)):02d}_{mask_variant}"
        subrun_dir = run_dir / run_id
        subrun_dir.mkdir(parents=True, exist_ok=True)
        sub_context = ExperimentContext(
            model_path=context.model_path,
            data_path=context.data_path,
            config_path=context.config_path,
            norm_stats_path=context.norm_stats_path,
            edge_stats_path=context.edge_stats_path,
            split_path=context.split_path,
            output_dir=subrun_dir,
        )

        status = "success"
        error_message = ""
        node_df_run = pd.DataFrame()
        summary_df_run = pd.DataFrame()
        fair_df_run = pd.DataFrame()
        edge_df_run = pd.DataFrame()

        if verbose:
            print(
                f"[{run_idx}/{total_runs}] sparsity={sparsity:.2f} "
                f"mask_variant={mask_variant} (mask_baseline={mask_to_baseline[mask_variant]})"
            )

        try:
            node_df_run, summary_df_run, fair_df_run, edge_df_run = run_experiments_evaluation(
                context=sub_context,
                node_types=node_types,
                sparsity=float(sparsity),
                mask_baseline_mode=str(mask_to_baseline[mask_variant]),
                ig_baseline_mode="scientific",
                gnn_epochs=int(gnn_epochs),
                gnn_lr=float(gnn_lr),
                gnn_explanation_type=str(gnn_explanation_type),
                ig_explanation_type=str(gnn_explanation_type),
                gnn_use_custom_coeffs=True,
                gnn_coeffs=fixed_gnn_coeffs,
                ig_n_steps=int(ig_n_steps),
                ig_scientific_strict=True,
                graph_scope="test_split",
                first_graph_per_component=True,
                component_key="compound",
                max_graphs=max_graphs,
                max_nodes_per_graph=int(max_nodes_per_graph),
                include_gnn_edge_report=bool(include_gnn_edge_report),
                seed=int(seed),
                verbose=bool(verbose),
                progress_enabled=bool(progress_enabled),
                progress_use_tqdm=bool(progress_use_tqdm),
            )
        except Exception as exc:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  run failed: {error_message}")

        if status == "success" and node_df_run is not None and not node_df_run.empty:
            node_aug = node_df_run.copy()
            node_aug["grid_run_id"] = run_id
            node_aug["mask_variant"] = str(mask_variant)
            node_aug["sparsity_target_requested"] = float(sparsity)
            combined_node_parts.append(node_aug)
        if status == "success" and edge_df_run is not None and not edge_df_run.empty:
            edge_aug = edge_df_run.copy()
            edge_aug["grid_run_id"] = run_id
            edge_aug["mask_variant"] = str(mask_variant)
            edge_aug["sparsity_target_requested"] = float(sparsity)
            combined_edge_parts.append(edge_aug)

        artifacts = {
            "run_output_dir": str(subrun_dir),
            "node_level_metrics": _last_file(subrun_dir, "node_level_metrics_*.csv"),
            "summary_by_method_and_type": _last_file(subrun_dir, "summary_by_method_and_type_*.csv"),
            "fair_comparison": _last_file(subrun_dir, "fair_comparison_H_C_*.csv"),
            "gnn_edge_summary": _last_file(subrun_dir, "gnn_edge_summary_*.csv"),
            "failed_nodes": _last_file(subrun_dir, "failed_nodes_*.csv"),
            "run_config": _last_file(subrun_dir, "run_config_*.json"),
        }
        run_records.append(
            {
                "run_index": int(run_idx),
                "run_id": str(run_id),
                "status": str(status),
                "error": str(error_message),
                "sparsity_target": float(sparsity),
                "mask_variant": str(mask_variant),
                "mask_baseline_mode": str(mask_to_baseline[mask_variant]),
                "ig_baseline_mode": "scientific",
                "node_rows": int(len(node_df_run)) if isinstance(node_df_run, pd.DataFrame) else 0,
                "summary_rows": int(len(summary_df_run)) if isinstance(summary_df_run, pd.DataFrame) else 0,
                "fair_rows": int(len(fair_df_run)) if isinstance(fair_df_run, pd.DataFrame) else 0,
                "edge_rows": int(len(edge_df_run)) if isinstance(edge_df_run, pd.DataFrame) else 0,
                "artifacts": artifacts,
            }
        )

        combined_node_df = (
            pd.concat(combined_node_parts, ignore_index=True)
            if combined_node_parts
            else pd.DataFrame()
        )
        combined_summary_df = (
            build_fidelity_grid_summary(combined_node_df)
            if not combined_node_df.empty
            else pd.DataFrame()
        )
        combined_edge_df = (
            pd.concat(combined_edge_parts, ignore_index=True)
            if combined_edge_parts
            else pd.DataFrame()
        )
        combined_edge_summary_df = (
            build_fidelity_grid_summary(combined_edge_df)
            if not combined_edge_df.empty
            else pd.DataFrame()
        )
        run_status_df = pd.DataFrame(run_records)

        combined_node_df.to_csv(progress_node_csv, index=False)
        combined_summary_df.to_csv(progress_summary_csv, index=False)
        combined_edge_df.to_csv(progress_edge_csv, index=False)
        combined_edge_summary_df.to_csv(progress_edge_summary_csv, index=False)
        run_status_df.to_csv(progress_status_csv, index=False)

        progress_manifest = {
            "updated_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "total_runs": int(total_runs),
            "completed_runs": int(len(run_records)),
            "successful_runs": int(sum(1 for r in run_records if r.get("status") == "success")),
            "failed_runs": int(sum(1 for r in run_records if r.get("status") != "success")),
            "progress_artifacts": {
                "node_metrics": str(progress_node_csv),
                "summary": str(progress_summary_csv),
                "edge_metrics": str(progress_edge_csv),
                "edge_summary": str(progress_edge_summary_csv),
                "run_status": str(progress_status_csv),
            },
            "runs": run_records,
        }
        with progress_manifest_json.open("w", encoding="utf-8") as handle:
            json.dump(progress_manifest, handle, indent=2)

    combined_node_df = (
        pd.concat(combined_node_parts, ignore_index=True)
        if combined_node_parts
        else pd.DataFrame()
    )
    combined_summary_df = (
        build_fidelity_grid_summary(combined_node_df)
        if not combined_node_df.empty
        else pd.DataFrame()
    )
    run_status_df = pd.DataFrame(run_records)
    combined_edge_df = (
        pd.concat(combined_edge_parts, ignore_index=True)
        if combined_edge_parts
        else pd.DataFrame()
    )
    combined_edge_summary_df = (
        build_fidelity_grid_summary(combined_edge_df)
        if not combined_edge_df.empty
        else pd.DataFrame()
    )

    plot_input_cols = [
        "mask_variant",
        "sparsity_target",
        "method",
        "node_type",
        "fid_plus_model_mean",
        "fid_plus_model_std",
        "fid_plus_model_count",
        "fid_minus_model_mean",
        "fid_minus_model_std",
        "fid_minus_model_count",
        "n_nodes",
        "n_graphs",
    ]
    if not combined_summary_df.empty:
        available_cols = [col for col in plot_input_cols if col in combined_summary_df.columns]
        plot_input_df = combined_summary_df[available_cols].copy()
    else:
        plot_input_df = pd.DataFrame(columns=plot_input_cols)
    if not combined_edge_summary_df.empty:
        edge_available_cols = [col for col in plot_input_cols if col in combined_edge_summary_df.columns]
        edge_plot_input_df = combined_edge_summary_df[edge_available_cols].copy()
    else:
        edge_plot_input_df = pd.DataFrame(columns=plot_input_cols)

    combined_node_df.to_csv(final_node_csv, index=False)
    combined_summary_df.to_csv(final_summary_csv, index=False)
    plot_input_df.to_csv(final_plot_input_csv, index=False)
    combined_edge_df.to_csv(final_edge_csv, index=False)
    combined_edge_summary_df.to_csv(final_edge_summary_csv, index=False)
    edge_plot_input_df.to_csv(final_edge_plot_input_csv, index=False)
    run_status_df.to_csv(final_status_csv, index=False)

    def _metric_key(metric_name: str) -> str:
        return "fid_plus" if metric_name == "fid_plus_model" else "fid_minus"

    plot_node_types: List[str] = []
    seen_node_types: Set[str] = set()
    for raw_nt in node_types:
        nt = str(raw_nt).strip()
        if not nt or nt in seen_node_types:
            continue
        seen_node_types.add(nt)
        plot_node_types.append(nt)
    if not plot_node_types:
        plot_node_types = ["H", "C"]

    plot_paths: Dict[str, str] = {}
    plot_masks = ("zero", "scientific")
    plot_metrics = ("fid_plus_model", "fid_minus_model")

    for mask_variant_name in plot_masks:
        mask_label = exp_pretty_mask_name(mask_variant_name)
        for metric_name in plot_metrics:
            metric_key = _metric_key(metric_name)
            metric_label = exp_pretty_metric_name(metric_name)
            for node_type_name in plot_node_types:
                key = f"{metric_key}_{mask_variant_name}_{node_type_name}"
                output_path = plots_dir / f"{key}.png"
                plot_paths[key] = str(output_path)
                title = (
                    f"{metric_label} vs Sparsity ({mask_label}) - Node Type {node_type_name}"
                )
                plot_fidelity_vs_sparsity(
                    summary_df=combined_summary_df,
                    mask_variant=mask_variant_name,
                    metric=metric_name,
                    output_path=output_path,
                    title=title,
                    node_type=node_type_name,
                    color_map=EXP_PLOT_METHOD_COLORS,
                    neutral_color=EXP_PLOT_NEUTRAL_COLOR,
                )

    edge_plot_df = combined_edge_summary_df.copy()
    if "method" in edge_plot_df.columns:
        edge_plot_df = edge_plot_df[edge_plot_df["method"] == EXP_METHOD_GNN].copy()
    for mask_variant_name in plot_masks:
        mask_label = exp_pretty_mask_name(mask_variant_name)
        for metric_name in plot_metrics:
            metric_key = _metric_key(metric_name)
            metric_label = exp_pretty_metric_name(metric_name)
            for node_type_name in plot_node_types:
                key = f"gnn_edge_{metric_key}_{mask_variant_name}_{node_type_name}"
                output_path = plots_dir / f"{key}.png"
                plot_paths[key] = str(output_path)
                title = (
                    f"GNNExplainer Edge {metric_label} vs Sparsity "
                    f"({mask_label}) - Node Type {node_type_name}"
                )
                plot_fidelity_vs_sparsity(
                    summary_df=edge_plot_df,
                    mask_variant=mask_variant_name,
                    metric=metric_name,
                    output_path=output_path,
                    title=title,
                    node_type=node_type_name,
                    color_map=EXP_PLOT_METHOD_COLORS,
                    neutral_color=EXP_PLOT_NEUTRAL_COLOR,
                )

    final_manifest = {
        "timestamp": run_stamp,
        "run_dir": str(run_dir),
        "total_runs": int(total_runs),
        "successful_runs": int(sum(1 for r in run_records if r.get("status") == "success")),
        "failed_runs": int(sum(1 for r in run_records if r.get("status") != "success")),
        "fixed_scope": {
            "graph_scope": "test_split",
            "first_graph_per_component": True,
            "component_key": "compound",
        },
        "grid": {
            "sparsities": [float(s) for s in sparsity_values],
            "mask_variants": list(normalized_masks),
            "mask_variant_to_baseline": mask_to_baseline,
        },
        "ig": {
            "baseline_mode": "scientific",
            "scientific_strict": True,
            "scope": "all_nodes",
            "ig_n_steps": int(ig_n_steps),
            "explanation_type": str(gnn_explanation_type),
        },
        "gnnexplainer": {
            "epochs": int(gnn_epochs),
            "lr": float(gnn_lr),
            "explanation_type": str(gnn_explanation_type),
            "coeffs": fixed_gnn_coeffs,
        },
        "limits": {
            "max_graphs": int(max_graphs) if max_graphs is not None else None,
            "max_nodes_per_graph": int(max_nodes_per_graph),
        },
        "artifacts": {
            "progress_node_metrics": str(progress_node_csv),
            "progress_summary": str(progress_summary_csv),
            "progress_edge_metrics": str(progress_edge_csv),
            "progress_edge_summary": str(progress_edge_summary_csv),
            "progress_status": str(progress_status_csv),
            "progress_manifest": str(progress_manifest_json),
            "node_metrics_all": str(final_node_csv),
            "summary_all": str(final_summary_csv),
            "plot_input": str(final_plot_input_csv),
            "edge_metrics_all": str(final_edge_csv),
            "edge_summary_all": str(final_edge_summary_csv),
            "edge_plot_input": str(final_edge_plot_input_csv),
            "run_status": str(final_status_csv),
            "plots": plot_paths,
        },
        "runs": run_records,
    }
    with final_manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(final_manifest, handle, indent=2)

    if verbose:
        print("Saved fidelity-grid artifacts:")
        print(f"  run dir: {run_dir}")
        print(f"  node metrics (all): {final_node_csv}")
        print(f"  summary (all): {final_summary_csv}")
        print(f"  plot input: {final_plot_input_csv}")
        print(f"  edge metrics (all): {final_edge_csv}")
        print(f"  edge summary (all): {final_edge_summary_csv}")
        print(f"  edge plot input: {final_edge_plot_input_csv}")
        print(f"  run status: {final_status_csv}")
        print(f"  final manifest: {final_manifest_json}")
        for key, path in plot_paths.items():
            print(f"  plot {key}: {path}")

    return {
        "run_dir": str(run_dir),
        "node_df": combined_node_df,
        "summary_df": combined_summary_df,
        "plot_input_df": plot_input_df,
        "edge_df": combined_edge_df,
        "edge_summary_df": combined_edge_summary_df,
        "edge_plot_input_df": edge_plot_input_df,
        "run_status_df": run_status_df,
        "plot_paths": plot_paths,
        "manifest": final_manifest,
        "manifest_path": str(final_manifest_json),
    }


def run_fidelity_grid_smoke_test(
    context: ExperimentContext,
    node_types: Sequence[str] = ("H", "C"),
    gnn_epochs: int = 200,
    gnn_lr: float = 0.01,
    gnn_explanation_type: str = "phenomenon",
    ig_n_steps: int = 64,
    seed: int = 42,
    run_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Minimal smoke test for systematic fidelity grid evaluation.

    Scope is intentionally tiny:
      - sparsities: (0.5,)
      - mask variants: ("zero", "scientific")
      - max_graphs: 1
      - max_nodes_per_graph: 2
    """

    smoke_name = (
        run_name.strip()
        if isinstance(run_name, str) and run_name.strip()
        else f"fidelity_grid_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result = run_fidelity_grid_evaluation(
        context=context,
        node_types=node_types,
        sparsities=(0.5,),
        mask_variants=("zero", "scientific"),
        gnn_epochs=int(gnn_epochs),
        gnn_lr=float(gnn_lr),
        gnn_explanation_type=str(gnn_explanation_type),
        ig_n_steps=int(ig_n_steps),
        max_graphs=1,
        max_nodes_per_graph=2,
        include_gnn_edge_report=True,
        seed=int(seed),
        verbose=bool(verbose),
        progress_enabled=False,
        progress_use_tqdm=False,
        run_name=smoke_name,
    )

    run_dir = Path(result["run_dir"])
    required_files = [
        run_dir / "grid_node_metrics_all.csv",
        run_dir / "grid_summary_all.csv",
        run_dir / "grid_plot_input.csv",
        run_dir / "grid_edge_metrics_all.csv",
        run_dir / "grid_edge_summary_all.csv",
        run_dir / "grid_edge_plot_input.csv",
        run_dir / "grid_run_status.csv",
        run_dir / "grid_manifest_final.json",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise RuntimeError(f"Smoke test failed. Missing artifact files: {missing_files}")

    plot_node_types: List[str] = []
    seen_node_types: Set[str] = set()
    for raw_nt in node_types:
        nt = str(raw_nt).strip()
        if not nt or nt in seen_node_types:
            continue
        seen_node_types.add(nt)
        plot_node_types.append(nt)
    if not plot_node_types:
        plot_node_types = ["H", "C"]

    required_plots: List[str] = []
    for node_type_name in plot_node_types:
        for mask_variant_name in ("zero", "scientific"):
            required_plots.append(f"fid_plus_{mask_variant_name}_{node_type_name}")
            required_plots.append(f"fid_minus_{mask_variant_name}_{node_type_name}")
            required_plots.append(f"gnn_edge_fid_plus_{mask_variant_name}_{node_type_name}")
            required_plots.append(f"gnn_edge_fid_minus_{mask_variant_name}_{node_type_name}")
    missing_plots: List[str] = []
    for key in required_plots:
        plot_path = Path(str(result.get("plot_paths", {}).get(key, "")))
        if not plot_path.exists() or plot_path.stat().st_size <= 0:
            missing_plots.append(key)
    if missing_plots:
        raise RuntimeError(f"Smoke test failed. Missing/empty plot files: {missing_plots}")

    status_df = result.get("run_status_df")
    if isinstance(status_df, pd.DataFrame):
        if len(status_df) != 2:
            raise RuntimeError(f"Smoke test failed. Expected 2 run-status rows, got {len(status_df)}.")
        if "status" in status_df.columns:
            failed = status_df[status_df["status"] != "success"]
            if not failed.empty:
                failed_ids = failed["run_id"].astype(str).tolist() if "run_id" in failed.columns else []
                raise RuntimeError(f"Smoke test failed. Failed runs: {failed_ids}")

    return result
