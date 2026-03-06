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


@dataclass
class ExperimentContext:
    model_path: Path
    data_path: Path
    config_path: Path
    norm_stats_path: Path
    edge_stats_path: Path
    split_path: Path
    output_dir: Path


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
    pred_orig: Optional[float] = None,
    target_value: float = float("nan"),
) -> Dict[str, float]:
    if node_type not in x_dict:
        raise ValueError(f"Node type '{node_type}' not present in x_dict.")

    node_matrix = x_dict[node_type]
    if node_idx < 0 or node_idx >= node_matrix.size(0):
        raise IndexError(f"Node index {node_idx} out of bounds for node type '{node_type}'.")

    feature_count = int(node_matrix.size(1))
    keep_indices = exp_topk_indices_from_importance(importance, sparsity=sparsity)
    k_keep = int(keep_indices.size)

    baseline_vec = exp_build_mask_baseline(
        mode=mask_baseline_mode,
        ig_baseline_mode=ig_baseline_mode,
        target_features=node_matrix[node_idx],
        node_features=node_matrix,
        node_type=node_type,
        dataset=dataset,
        train_graph_indices=train_graph_indices,
        median_cache=median_cache,
        elem_distribution_cache=elem_distribution_cache,
        cache_dir=cache_dir,
    )

    selected = node_matrix[node_idx].clone()
    drop_vec = selected.clone()
    if k_keep > 0:
        drop_vec[keep_indices] = baseline_vec[keep_indices]

    keep_vec = baseline_vec.clone()
    if k_keep > 0:
        keep_vec[keep_indices] = selected[keep_indices]

    x_drop = {nt: feat.clone() for nt, feat in x_dict.items()}
    x_keep = {nt: feat.clone() for nt, feat in x_dict.items()}
    x_drop[node_type][node_idx] = drop_vec
    x_keep[node_type][node_idx] = keep_vec

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
        "n_features": feature_count,
        "k_features": k_keep,
        "actual_sparsity_feat": float(actual_sparsity(feature_count, k_keep)),
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

    for edge_type, edge_mask in edge_mask_dict.items():
        if edge_mask is None or edge_type not in edge_index_dict:
            continue

        edge_index = edge_index_dict[edge_type]

        mask_vals = edge_mask.view(-1).detach().cpu()

        limit = min(int(mask_vals.shape[0]), int(edge_index.size(1)))
        for pos in range(limit):
            score_abs = float(abs(mask_vals[pos].item()))
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
    gnn_use_custom_coeffs: bool = False,
    gnn_coeffs: Optional[Dict[str, float]] = None,
    ig_n_steps: int = 64,
    graph_scope: str = "test_split",
    first_graph_per_component: bool = True,
    component_key: str = "compound",
    max_graphs: Optional[int] = None,
    max_nodes_per_graph: int = 0,
    include_gnn_edge_report: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_node_types = [nt for nt in node_types if nt in EXP_ALLOWED_NODE_TYPES]
    if not clean_node_types:
        raise ValueError("No valid node types selected.")

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
            f"Mask baseline mode: {mask_baseline_mode} "
            f"(IG baseline mode: {ig_baseline_mode})"
        )

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

    for graph_idx in eval_graph_indices:
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
                try:
                    algorithm = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr, coeffs=gnn_coeffs)
                except TypeError:
                    algorithm = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr, **gnn_coeffs)
            else:
                algorithm = GNNExplainer(epochs=gnn_epochs, lr=gnn_lr)

            gnn_explainer = Explainer(
                model=wrapped_model,
                algorithm=algorithm,
                explanation_type=gnn_explanation_type,
                model_config=model_config,
                node_mask_type="attributes",
                edge_mask_type="object",
            )

            gnn_target = target_tensor if (gnn_explanation_type == "phenomenon" and target_tensor is not None) else None

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
                        n_steps=ig_n_steps,
                        baseline_type=ig_baseline_mode,
                        include_neighbors=False,
                        k_hops=1,
                        median_global=(median_cache[node_type][0] if node_type in median_cache else None),
                        median_by_element=(median_cache[node_type][1] if node_type in median_cache else None),
                        elem_distribution=elem_distribution_cache.get("value"),
                        dataset=dataset,
                        train_graph_indices=train_graph_indices,
                        train_split_path=str(context.split_path) if context.split_path.exists() else None,
                    )
                except Exception as exc_first:
                    if ig_baseline_mode == "scientific":
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
                            n_steps=ig_n_steps,
                            baseline_type="mean",
                            include_neighbors=False,
                            k_hops=1,
                            median_global=None,
                            median_by_element=None,
                            elem_distribution=None,
                            dataset=dataset,
                            train_graph_indices=train_graph_indices,
                            train_split_path=str(context.split_path) if context.split_path.exists() else None,
                        )
                    else:
                        raise exc_first

                try:
                    ig_importance = exp_extract_ig_feature_importance(
                        ig_result,
                        node_type=node_type,
                        node_idx=node_idx,
                    )
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
        "gnn_explanation_type": gnn_explanation_type,
        "gnn_use_custom_coeffs": bool(gnn_use_custom_coeffs),
        "ig_n_steps": int(ig_n_steps),
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
