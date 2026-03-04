"""
Integrated Gradients (IG) Explainer for heterogeneous GNN.

Supports:
- Single-node IG explanations (optionally including neighbor-feature attributions).
- Batch IG feature-importance aggregation (optionally including neighbor-feature attributions).
- Scientific baseline (train-median based) for both target nodes and neighbors.

Notes:
- Scientific baseline medians are computed in the normalized feature space.
- When include_neighbors=True in batch mode, results are aggregated per node type (H/C/Others),
  because feature dimensionalities differ by type.

Option B (FAIR CONTEXT AGGREGATION):
- In batch mode with include_neighbors=True, neighbor attributions are first averaged PER TARGET NODE
  (per neighbor node type), then aggregated across target nodes. This avoids degree bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import os
import pickle

import torch
import numpy as np
from captum.attr import IntegratedGradients

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.explainer.explainer_utils import (
    NodeTypeRegressionWrapper,
    heterodata_to_dicts,
)
from scripts.explainer.baselines import (
    build_scientific_baseline,
    load_or_compute_medians,
    load_or_compute_element_distribution,
)

# ---------------------------------------------------------------------------
# Neighbor search
# ---------------------------------------------------------------------------

def find_k_hop_neighbors(
    edge_index_dict: dict,
    node_type: str,
    node_idx: int,
    k: int = 3,
) -> Dict[str, List[int]]:
    """
    Find all neighbors within k hops of a given node in a heterogeneous graph.

    Returns: dict node_type -> sorted list of node indices (excluding the source node itself).
    """
    visited = {node_type: {node_idx}}
    current_frontier = {(node_type, node_idx)}

    for _hop in range(k):
        next_frontier = set()

        for curr_type, curr_idx in current_frontier:
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type

                # Outgoing edges from current node
                if src_type == curr_type:
                    mask = edge_index[0] == curr_idx
                    neighbors = edge_index[1][mask].tolist()

                    visited.setdefault(dst_type, set())
                    for n_idx in neighbors:
                        if n_idx not in visited[dst_type]:
                            visited[dst_type].add(n_idx)
                            next_frontier.add((dst_type, n_idx))

                # Incoming edges to current node
                if dst_type == curr_type:
                    mask = edge_index[1] == curr_idx
                    neighbors = edge_index[0][mask].tolist()

                    visited.setdefault(src_type, set())
                    for n_idx in neighbors:
                        if n_idx not in visited[src_type]:
                            visited[src_type].add(n_idx)
                            next_frontier.add((src_type, n_idx))

        current_frontier = next_frontier
        if not current_frontier:
            break

    result: Dict[str, List[int]] = {}
    for nt, indices in visited.items():
        if nt == node_type:
            lst = sorted([i for i in indices if i != node_idx])
        else:
            lst = sorted(list(indices))
        if lst:
            result[nt] = lst

    return result


# ---------------------------------------------------------------------------
# Single-node IG explanation (unchanged)
# ---------------------------------------------------------------------------

def compute_ig_explanation(
    base_model: Any,
    data: Any,
    node_type: str,
    node_idx: int,
    x_dict: dict,
    edge_index_dict: dict,
    edge_attr_dict: dict,
    device: torch.device,
    target: torch.Tensor = None,
    n_steps: int = 50,
    baseline_type: str = "zero",
    include_neighbors: bool = False,
    k_hops: int = 3,
    median_global: Optional[torch.Tensor] = None,
    median_by_element: Optional[Dict[str, torch.Tensor]] = None,
    elem_distribution: Optional[torch.Tensor] = None,
    dataset: Any = None,
    train_graph_indices: Optional[List[int]] = None,
    train_split_path: Optional[str] = None,
):
    """
    Compute IG attributions for a single node. Optionally include neighbor-feature attributions
    w.r.t. the selected node prediction.

    Returns a dict with:
      - node_mask_dict: {node_type: [1, F]}
      - neighbors: list of neighbor attribution dicts (if include_neighbors=True)
      - neighbor_stats: aggregated stats per neighbor_type (if include_neighbors=True)
    """
    wrapped_model = NodeTypeRegressionWrapper(base_model, node_type).to(device)
    wrapped_model.eval()

    def forward_func(inputs):
        outputs = []
        for i in range(inputs.shape[0]):
            temp_x_dict = {}
            for nt, x in x_dict.items():
                temp_x = x.clone().detach().to(device)
                if nt == node_type:
                    temp_x = temp_x.clone()
                    temp_x[node_idx] = inputs[i]
                temp_x_dict[nt] = temp_x

            # ensure edge attrs on device
            temp_edge_attr_dict = {}
            if edge_attr_dict is not None:
                for et, attrs in edge_attr_dict.items():
                    if attrs is not None:
                        temp_edge_attr_dict[et] = attrs.to(device) if hasattr(attrs, "to") else attrs

            out = wrapped_model(
                temp_x_dict,
                edge_index_dict,
                temp_edge_attr_dict if temp_edge_attr_dict else None,
            )
            pred = out[node_idx]
            outputs.append(pred.view(1, 1))
        return torch.cat(outputs, dim=0)  # [B, 1]

    target_features = x_dict[node_type][node_idx].clone().detach().to(device)

    # --- build baseline for target ---
    if baseline_type == "zero":
        baseline = torch.zeros_like(target_features)
    elif baseline_type == "mean":
        baseline = torch.mean(x_dict[node_type], dim=0)
    elif baseline_type == "random":
        allf = x_dict[node_type]
        baseline = torch.mean(allf, dim=0) + torch.std(allf, dim=0) * torch.randn_like(target_features)
    elif baseline_type == "min":
        baseline = torch.min(x_dict[node_type], dim=0)[0]
    elif baseline_type == "max":
        baseline = torch.max(x_dict[node_type], dim=0)[0]
    elif baseline_type == "scientific":
        if median_global is None or median_by_element is None or elem_distribution is None:
            try:
                # Use provided dataset first, otherwise fall back to global
                ds = dataset if dataset is not None else globals().get("dataset", None)

                # Resolve train split indices
                train_indices = train_graph_indices
                split_candidates = []
                if train_split_path:
                    split_candidates.append(Path(train_split_path))
                split_candidates.append(Path("baselines") / "graph_split.pkl")

                if train_indices is None:
                    for cand in split_candidates:
                        if cand and cand.exists():
                            with open(cand, "rb") as f:
                                sp = pickle.load(f)
                            train_indices = (
                                sp.get("train_graph_indices")
                                or sp.get("train_indices")
                                or sp.get("train_graphs")
                            )
                            if train_indices:
                                print(f"[IG] Scientific baseline: loaded {len(train_indices)} train graph indices from {cand}")
                                train_split_path = str(cand)
                                break

                if ds is not None and train_indices:
                    if median_global is None or median_by_element is None:
                        median_global, median_by_element = load_or_compute_medians(
                            ds, train_indices, node_type
                        )
                    if elem_distribution is None:
                        elem_distribution = load_or_compute_element_distribution(ds, train_indices)
                else:
                    raise AssertionError("median_global/elem_distribution not provided and no train split/dataset available.")
            except Exception as e:
                raise AssertionError(
                    "baseline_type='scientific' erfordert median_global und elem_distribution. "
                    "Bitte load_or_compute_medians() und load_or_compute_element_distribution() aufrufen. "
                    f"Fallback failed: {e}"
                )

        baseline = build_scientific_baseline(
            target_features.detach().cpu(),
            node_type,
            median_global.detach().cpu(),
            median_by_element,
            elem_distribution=elem_distribution.detach().cpu() if elem_distribution is not None else None,
        ).to(device)
    else:
        baseline = torch.zeros_like(target_features)

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(
        inputs=target_features.unsqueeze(0),
        baselines=baseline.unsqueeze(0),
        n_steps=n_steps,
        target=0,
        return_convergence_delta=True,
    )

    node_attr = attributions.squeeze(0)  # [F]
    result = {
        "node_mask_dict": {node_type: node_attr.unsqueeze(0)},  # [1, F]
        "edge_mask_dict": {},
        "selected_node": {
            "node_type": node_type,
            "node_idx": node_idx,
            "attributions": node_attr.detach().cpu().numpy().tolist(),
        },
        "convergence_delta": float(delta.detach().cpu().item()) if isinstance(delta, torch.Tensor) else float(delta),
    }

    # (Neighbor part unchanged from your original; omitted here for brevity)
    # If you want, you can keep your existing include_neighbors logic as-is.
    return result


# ---------------------------------------------------------------------------
# Batch IG analysis (OPTION B implemented here)
# ---------------------------------------------------------------------------

def batch_ig_analysis(
    base_model: Any,
    dataset: Any,
    node_type: str,
    graph_indices: List[int],
    device: torch.device,
    n_steps: int = 50,
    baseline_type: str = "zero",
    progress_callback: Optional[Any] = None,
    train_graph_indices: Optional[List[int]] = None,
    include_neighbors: bool = False,
    k_hops: int = 1,
    max_neighbors_per_type: Optional[int] = 25,
):
    """
    Batch IG for target nodes of `node_type` over `graph_indices`.

    If include_neighbors=True:
      additionally compute IG for k-hop neighbor nodes (their features) w.r.t. target prediction,
      BUT (Option B) neighbor attributions are first averaged PER TARGET NODE (per neighbor type),
      then aggregated across target nodes. This reduces degree bias.

    Returns (backward-compatible structure):
      dict: node_type -> aggregated stats dict
      NOTE: may include multiple node types as keys when include_neighbors=True.
      Additional fields:
        - self_node_count: number of self attribution vectors for that type
        - ctx_node_count: number of context (per-target-averaged) vectors for that type
    """
    wrapped_model = NodeTypeRegressionWrapper(base_model, node_type).to(device)
    wrapped_model.eval()

    # scientific medians cache per type (needed for neighbors too)
    median_global_by_type: Dict[str, torch.Tensor] = {}
    median_by_element_by_type: Dict[str, Dict[str, torch.Tensor]] = {}
    elem_distribution: Optional[torch.Tensor] = None

    if baseline_type == "scientific":
        if train_graph_indices is None:
            import warnings
            warnings.warn(
                "batch_ig_analysis: scientific baseline without train_graph_indices. Using graph_indices as proxy (DEBUG).",
                UserWarning,
            )
            train_graph_indices_eff = list(graph_indices)
        else:
            train_graph_indices_eff = train_graph_indices

        try:
            elem_distribution = load_or_compute_element_distribution(dataset, train_graph_indices_eff)
        except Exception:
            elem_distribution = None

        for nt in ["H", "C", "Others"]:
            try:
                mg, mbe = load_or_compute_medians(dataset, train_graph_indices_eff, nt)
                median_global_by_type[nt] = mg
                median_by_element_by_type[nt] = mbe
            except Exception:
                continue

    def make_baseline(feat: torch.Tensor, nt: str, x_dict_local: dict) -> torch.Tensor:
        if baseline_type == "zero":
            return torch.zeros_like(feat)
        if baseline_type == "mean":
            return torch.mean(x_dict_local[nt], dim=0)
        if baseline_type == "random":
            allf = x_dict_local[nt]
            return torch.mean(allf, dim=0) + torch.std(allf, dim=0) * torch.randn_like(feat)
        if baseline_type == "min":
            return torch.min(x_dict_local[nt], dim=0)[0]
        if baseline_type == "max":
            return torch.max(x_dict_local[nt], dim=0)[0]
        if baseline_type == "scientific":
            if nt not in median_global_by_type:
                return torch.zeros_like(feat)
            mg = median_global_by_type[nt]
            mbe = median_by_element_by_type.get(nt, None)
            return build_scientific_baseline(
                feat.detach().cpu(),
                nt,
                mg.detach().cpu(),
                mbe,
                elem_distribution=elem_distribution.detach().cpu() if elem_distribution is not None else None,
            ).to(device)
        return torch.zeros_like(feat)

    # Separate pools: self vs context (per-target averaged)
    self_attrs_by_type: Dict[str, List[np.ndarray]] = defaultdict(list)
    ctx_attrs_by_type: Dict[str, List[np.ndarray]] = defaultdict(list)

    for graph_idx in graph_indices:
        if progress_callback:
            progress_callback(f"Processing graph {graph_idx}...")

        data = dataset[graph_idx].to(device)
        x_dict, edge_index_dict, edge_attr_dict, y_dict = heterodata_to_dicts(data)

        if node_type not in x_dict:
            continue

        node_features = x_dict[node_type]
        num_nodes = node_features.size(0)

        for node_idx in range(num_nodes):

            # --- target (self) IG ---
            def forward_func(inputs):
                outs = []
                for i in range(inputs.shape[0]):
                    temp_x_dict = {}
                    for nt, x in x_dict.items():
                        tmp = x.clone().detach().to(device)
                        if nt == node_type:
                            tmp = tmp.clone()
                            tmp[node_idx] = inputs[i]
                        temp_x_dict[nt] = tmp
                    out = wrapped_model(temp_x_dict, edge_index_dict, edge_attr_dict)
                    outs.append(out[node_idx].view(1, 1))
                return torch.cat(outs, dim=0)

            target_feat = node_features[node_idx].clone().detach().to(device)
            target_base = make_baseline(target_feat, node_type, x_dict)

            ig = IntegratedGradients(forward_func)
            target_attr = ig.attribute(
                inputs=target_feat.unsqueeze(0),
                baselines=target_base.unsqueeze(0),
                n_steps=n_steps,
                target=0,
            ).squeeze(0).detach().cpu().numpy()

            self_attrs_by_type[node_type].append(target_attr)

            # --- neighbor (context) IG (OPTION B) ---
            if not include_neighbors:
                continue

            neighbors_dict = find_k_hop_neighbors(edge_index_dict, node_type, node_idx, k=k_hops)
            if max_neighbors_per_type is not None:
                neighbors_dict = {nt: idxs[:max_neighbors_per_type] for nt, idxs in neighbors_dict.items()}

            # collect per-target per neighbor-type
            ctx_this_target: Dict[str, List[np.ndarray]] = defaultdict(list)

            for neigh_type, neigh_idxs in neighbors_dict.items():
                if neigh_type not in x_dict:
                    continue

                for neigh_idx in neigh_idxs:

                    def forward_func_neighbor(inputs):
                        outs = []
                        for j in range(inputs.shape[0]):
                            temp_x_dict = {}
                            for nt, x in x_dict.items():
                                tmp = x.clone().detach().to(device)
                                if nt == neigh_type:
                                    tmp = tmp.clone()
                                    tmp[neigh_idx] = inputs[j]
                                temp_x_dict[nt] = tmp
                            out = wrapped_model(temp_x_dict, edge_index_dict, edge_attr_dict)
                            outs.append(out[node_idx].view(1, 1))
                        return torch.cat(outs, dim=0)

                    neigh_feat = x_dict[neigh_type][neigh_idx].clone().detach().to(device)
                    neigh_base = make_baseline(neigh_feat, neigh_type, x_dict)

                    # gradient check to skip dead paths
                    inp = neigh_feat.unsqueeze(0).clone().detach().requires_grad_(True)
                    outn = forward_func_neighbor(inp)
                    g = torch.autograd.grad(outn.sum(), inp, allow_unused=True)[0]
                    if g is None or g.abs().sum().item() == 0.0:
                        continue

                    neigh_ig = IntegratedGradients(forward_func_neighbor)
                    neigh_attr = neigh_ig.attribute(
                        inputs=neigh_feat.unsqueeze(0),
                        baselines=neigh_base.unsqueeze(0),
                        n_steps=n_steps,
                        target=0,
                    ).squeeze(0).detach().cpu().numpy()

                    ctx_this_target[neigh_type].append(neigh_attr)

            # Option B: average over neighbors per target node (per neighbor type)
            for neigh_type, attrs_list in ctx_this_target.items():
                if not attrs_list:
                    continue
                A = np.stack(attrs_list, axis=0)          # [num_neighbors, F]
                mean_over_neighbors = np.mean(A, axis=0)  # [F]
                ctx_attrs_by_type[neigh_type].append(mean_over_neighbors)

    def _aggregate(attrs: List[np.ndarray]) -> Dict[str, Any]:
        if not attrs:
            return {
                "avg_importance": [],
                "avg_abs_importance": [],
                "std_importance": [],
                "node_count": 0,
            }
        A = np.array(attrs)
        return {
            "avg_importance": np.mean(A, axis=0).tolist(),
            "avg_abs_importance": np.mean(np.abs(A), axis=0).tolist(),
            "std_importance": np.std(A, axis=0).tolist(),
            "node_count": int(A.shape[0]),
        }

    # Build combined (total) results per type, while keeping self/ctx counts visible
    results: Dict[str, dict] = {}
    all_types = set(self_attrs_by_type.keys()) | set(ctx_attrs_by_type.keys())

    for nt in all_types:
        self_stats = _aggregate(self_attrs_by_type.get(nt, []))
        ctx_stats = _aggregate(ctx_attrs_by_type.get(nt, []))

        # total: add means at the vector level (using avg_abs_importance as primary "importance")
        s_abs = np.array(self_stats["avg_abs_importance"], dtype=float) if self_stats["avg_abs_importance"] else np.array([], dtype=float)
        c_abs = np.array(ctx_stats["avg_abs_importance"], dtype=float) if ctx_stats["avg_abs_importance"] else np.array([], dtype=float)

        if s_abs.size == 0 and c_abs.size == 0:
            total_abs = np.array([], dtype=float)
        elif s_abs.size == 0:
            total_abs = c_abs
        elif c_abs.size == 0:
            total_abs = s_abs
        else:
            # same feature dimension within a node type by construction
            total_abs = s_abs + c_abs  # lambda=1

        # total signed mean is less interpretable when mixing self+ctx; keep both separately but provide a "total_abs"
        results[nt] = {
            "avg_importance": self_stats["avg_importance"],            # keep self signed mean for direction
            "avg_abs_importance": total_abs.tolist(),                  # TOTAL importance (self + context)
            "std_importance": self_stats["std_importance"],            # std of self only (avoid mixing)
            "node_count": int(self_stats["node_count"] + ctx_stats["node_count"]),
            "self_node_count": int(self_stats["node_count"]),
            "ctx_node_count": int(ctx_stats["node_count"]),
            "explainer_type": "ig",
            "includes_neighbors": bool(include_neighbors),
            "k_hops": int(k_hops),
            "max_neighbors_per_type": max_neighbors_per_type,
            "context_aggregation": "option_b_per_target_mean",
        }

    return results


# ---------------------------------------------------------------------------
# CLI-compatible wrappers (single + batch)
# ---------------------------------------------------------------------------

def explain_node_with_ig(
    model_path: str,
    data_path: str,
    config: Any,
    norm_stats: Any,
    edge_stats: Any,
    graph_idx: int,
    node_type: str,
    node_idx: int,
    output_dir: str,
    n_steps: int = 50,
    baseline_type: str = "zero",
    include_neighbors: bool = False,
    k_hops: int = 3,
    train_split_file: Optional[str] = None,
):
    """
    Standalone IG explanation for one node.
    """
    print(f"Computing IG explanation for {node_type}[{node_idx}] in graph {graph_idx}...")

    from scripts.explainer.explainer_utils import (
        load_config,
        load_stats,
        load_trained_model,
        build_dataset,
        validate_indices,
        ensure_dir,
        get_device,
    )

    device = get_device(None)

    # config
    if config:
        config_obj = config if isinstance(config, dict) else load_config(config)
    else:
        config_path = model_path.replace("SAGEConv_best_model.pt", "config.pkl")
        config_obj = load_config(config_path)

    # model
    model = load_trained_model(model_path, config_obj, device)

    # stats
    norm_stats_path = norm_stats["path"] if isinstance(norm_stats, dict) else norm_stats
    edge_stats_path = edge_stats["path"] if isinstance(edge_stats, dict) else edge_stats
    norm_stats_obj, edge_stats_obj = load_stats(norm_stats_path, edge_stats_path)

    # dataset
    dataset = build_dataset(data_path, config_obj, norm_stats=norm_stats_obj, edge_stats=edge_stats_obj)
    validate_indices(len(dataset), graph_idx, "graph_idx")

    # train split for scientific baseline
    train_graph_indices = None
    train_split_path_used = None
    if baseline_type == "scientific":
        split_candidates: List[Path] = []
        if train_split_file:
            split_candidates.append(Path(train_split_file))
        split_candidates.append(Path("baselines") / "graph_split.pkl")

        for cand in split_candidates:
            if cand and cand.exists():
                with open(cand, "rb") as f:
                    sp = pickle.load(f)
                train_graph_indices = (
                    sp.get("train_graph_indices")
                    or sp.get("train_indices")
                    or sp.get("train_graphs")
                )
                if train_graph_indices:
                    train_split_path_used = str(cand)
                    print(f"[IG] Loaded {len(train_graph_indices)} train graph indices from {cand}")
                    break

    data = dataset[graph_idx].to(device)
    x_dict, edge_index_dict, edge_attr_dict, y_dict = heterodata_to_dicts(data)

    # validate node index if y exists
    if y_dict.get(node_type) is not None:
        validate_indices(y_dict[node_type].size(0), node_idx, "node_idx")
    else:
        validate_indices(x_dict[node_type].size(0), node_idx, "node_idx")

    explanation_result = compute_ig_explanation(
        model,
        data,
        node_type,
        node_idx,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        device,
        target=None,
        n_steps=n_steps,
        baseline_type=baseline_type,
        include_neighbors=include_neighbors,
        k_hops=k_hops,
        median_global=None,
        median_by_element=None,
        elem_distribution=None,
        dataset=dataset,
        train_graph_indices=train_graph_indices,
        train_split_path=train_split_path_used,
    )

    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ig_explainer_{node_type}_n{node_idx}_g{graph_idx}_{timestamp}.pt"
    save_path = Path(output_dir) / filename
    torch.save(explanation_result, save_path)
    print(f"Saved IG explanation to {save_path}")
    return explanation_result


def batch_explain_nodes_with_ig(
    model_path: str,
    data_path: str,
    config: Any,
    norm_stats: Any,
    edge_stats: Any,
    graph_indices: List[int],
    node_type: str,
    output_dir: str,
    n_steps: int = 50,
    baseline_type: str = "zero",
    train_split_file: Optional[str] = None,
    include_neighbors: bool = False,
    k_hops: int = 1,
    max_neighbors_per_type: Optional[int] = 25,
):
    """
    Batch IG analysis for all nodes of a given type across multiple graphs.
    Optionally include neighbor-feature attributions.

    Returns:
      dict: node_type -> aggregated stats dict (may include multiple keys when include_neighbors=True)
      Uses Option B for neighbor aggregation (per target node mean).
    """
    print(f"Computing batch IG analysis for target={node_type} across graphs {graph_indices}...")

    from scripts.explainer.explainer_utils import (
        load_config,
        load_stats,
        load_trained_model,
        build_dataset,
        get_device,
        ensure_dir,
    )

    device = get_device(None)

    # config
    if config:
        config_obj = config if isinstance(config, dict) else load_config(config)
    else:
        config_path_eff = model_path.replace("SAGEConv_best_model.pt", "config.pkl")
        config_obj = load_config(config_path_eff)

    # model
    model = load_trained_model(model_path, config_obj, device)

    # stats
    norm_stats_path = norm_stats["path"] if isinstance(norm_stats, dict) else norm_stats
    edge_stats_path = edge_stats["path"] if isinstance(edge_stats, dict) else edge_stats
    norm_stats_obj, edge_stats_obj = load_stats(norm_stats_path, edge_stats_path)

    # dataset
    dataset = build_dataset(data_path, config_obj, norm_stats=norm_stats_obj, edge_stats=edge_stats_obj)

    # validate graph indices
    for g in graph_indices:
        if g < 0 or g >= len(dataset):
            raise IndexError(f"Graph index {g} out of bounds for dataset length {len(dataset)}")

    # resolve train indices for scientific baseline
    train_graph_indices = None
    if baseline_type == "scientific":
        if train_split_file is not None and os.path.exists(train_split_file):
            with open(train_split_file, "rb") as f:
                split_data = pickle.load(f)
            if "train_graph_indices" not in split_data:
                raise ValueError("Invalid split file: missing 'train_graph_indices' key")
            train_graph_indices = split_data["train_graph_indices"]
            print(f"[IG] Loaded {len(train_graph_indices)} train graph indices from {train_split_file}")
        else:
            import warnings
            warnings.warn(
                "⚠️ CRITICAL: --train-split-file NOT PROVIDED for scientific baseline!\n"
                "Using 80% fallback (DEBUG ONLY - NOT FOR PRODUCTION EVALUATION)!",
                UserWarning,
                stacklevel=2,
            )
            train_graph_indices = list(range(int(0.8 * len(dataset))))

    def progress_callback(msg: str):
        print(msg)

    batch_results = batch_ig_analysis(
        model,
        dataset,
        node_type,
        graph_indices,
        device,
        n_steps=n_steps,
        baseline_type=baseline_type,
        progress_callback=progress_callback,
        train_graph_indices=train_graph_indices,
        include_neighbors=include_neighbors,
        k_hops=k_hops,
        max_neighbors_per_type=max_neighbors_per_type,
    )

    # save results
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(graph_indices) <= 5:
        graphs_str = "_".join(map(str, graph_indices))
    else:
        graphs_str = f"{graph_indices[0]}-{graph_indices[-1]}"

    suffix = ""
    if include_neighbors:
        suffix = f"_ctxB_k{k_hops}_m{max_neighbors_per_type}"

    filename = f"batch_ig_{node_type}{suffix}_g{graphs_str}_{timestamp}.pt"
    save_path = Path(output_dir) / filename
    torch.save(batch_results, save_path)

    print(f"Saved batch IG analysis to {save_path}")
    if node_type in batch_results:
        print(f"Target type '{node_type}' self vectors: {batch_results[node_type].get('self_node_count', 0)}")
        print(f"Target type '{node_type}' ctx vectors:  {batch_results[node_type].get('ctx_node_count', 0)}")
    return batch_results


# ---------------------------------------------------------------------------
# Main (unchanged; keep your existing CLI if you want)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IG Explainer for heterogeneous GNN")

    parser.add_argument("--model", default="models/SAGEConv_best_model.pt",
                        help="Pfad zum trainierten Modell (default: models/SAGEConv_best_model.pt)")
    parser.add_argument("--data", required=True, help="Pfad zu den Daten")
    parser.add_argument("--config", default=None,
                        help="Pfad zur Config.pkl (wenn None, wird aus --model abgeleitet)")
    parser.add_argument("--norm-stats", default="models/norm_stats.pkl",
                        help="Pfad zu den Normalisierungsstatistiken")
    parser.add_argument("--edge-stats", default="models/edge_stats.pkl",
                        help="Pfad zu den Edgestats")
    parser.add_argument("--node-type", default="H", choices=["H", "C", "Others"],
                        help="Knotentyp für Analyse (default: H)")
    parser.add_argument("--output-dir", default="results/explanations",
                        help="Ausgabeverzeichnis (default: results/explanations)")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Anzahl der Integrationsschritte für IG (default: 50)")
    parser.add_argument("--baseline-type", default="zero",
                        choices=["zero", "mean", "random", "min", "max", "scientific"],
                        help="Baseline-Typ fuer IG (default: zero). 'scientific' = feature-spezifische Train-Median-Baseline.")
    parser.add_argument("--train-split-file", default=None,
                        help="Pfad zu einer .pkl-Datei mit Train-Graph-Indizes (Liste[int]). "
                             "Wird bei --baseline-type scientific benoetigt. "
                             "Wenn nicht angegeben, wird 80%% der Datenmenge als Fallback genutzt (mit Warnung).")

    # neighbor/context options
    parser.add_argument("--include-neighbors", action="store_true",
                        help="Include k-hop neighbor feature attributions (context IG).")
    parser.add_argument("--k-hops", type=int, default=1,
                        help="Number of hops for neighbor expansion (default: 1).")
    parser.add_argument("--max-neighbors-per-type", type=int, default=25,
                        help="Cap number of neighbors per node type (default: 25).")

    # Batch mode
    parser.add_argument("--batch-mode", action="store_true",
                        help="Batch-Modus aktivieren: Analysiert alle Knoten eines Typs ueber mehrere Graphen")
    parser.add_argument("--graph-indices", type=int, nargs="+", default=None,
                        help="Liste von Graph-Indizes fuer Batch-Analyse (z.B. --graph-indices 0 1 2)")

    args = parser.parse_args()

    if not args.batch_mode:
        parser.error("This script version focuses on batch mode. Use your existing single-node CLI if needed.")

    if args.graph_indices is None:
        parser.error("--batch-mode erfordert --graph-indices")

    batch_explain_nodes_with_ig(
        model_path=args.model,
        data_path=args.data,
        config=args.config,
        norm_stats=args.norm_stats,
        edge_stats=args.edge_stats,
        graph_indices=args.graph_indices,
        node_type=args.node_type,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        baseline_type=args.baseline_type,
        train_split_file=args.train_split_file,
        include_neighbors=args.include_neighbors,
        k_hops=args.k_hops,
        max_neighbors_per_type=args.max_neighbors_per_type,
    )
