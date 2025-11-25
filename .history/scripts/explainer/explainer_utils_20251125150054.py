import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from dataloader import ShiftDataset
from model import HeteroGNNModel

# Default dimensionalities that are used when instantiating the hetero model.
DEFAULT_IN_DIM_DICT: Dict[str, int] = {
    "H": 34,
    "C": 39,
    "Others": 16,
}
EDGE_IN_DIM = 10


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_pickle_file(path: Optional[str]) -> Any:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file at {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_config(config_path: str = "explain_this/config.pkl") -> Any:
    return load_pickle_file(config_path)


def load_stats(
    norm_stats_path: Optional[str],
    edge_stats_path: Optional[str],
) -> Tuple[Any, Optional[Dict[str, float]]]:
    norm_stats = load_pickle_file(norm_stats_path) if norm_stats_path else None
    edge_stats = load_pickle_file(edge_stats_path) if edge_stats_path else None
    return norm_stats, edge_stats


def build_dataset(
    data_path: str,
    config: Any,
    norm_stats: Optional[Any] = None,
    edge_stats: Optional[Dict[str, float]] = None,
) -> ShiftDataset:
    root_dir, file_name = os.path.split(data_path)
    kwargs = {
        "root_dir": root_dir or "data",
        "file_name": file_name,
        "normalize_node_features": _get_config_value(
            config, "normalize_node_features", True
        ),
        "normalize_edge_features": _get_config_value(
            config, "normalize_edge_features", True
        ),
    }
    if norm_stats is not None:
        kwargs["norm_stats"] = norm_stats
    if edge_stats is not None:
        kwargs.update(edge_stats)
    return ShiftDataset(**kwargs)


def build_model_from_config(
    config: Any,
    device: torch.device,
    in_dim_dict: Optional[Dict[str, int]] = None,
) -> HeteroGNNModel:
    in_dim_dict = in_dim_dict or DEFAULT_IN_DIM_DICT
    operator_kwargs = _get_config_value(config, "operator_kwargs", {}) or {}
    operator_type = _get_config_value(config, "operator_type", "SAGEConv")
    if operator_type in {"GATConv", "GATv2Conv"}:
        operator_kwargs = dict(operator_kwargs)
        operator_kwargs.setdefault("add_self_loops", False)
    model = HeteroGNNModel(
        in_dim_dict=in_dim_dict,
        hidden_dim=_get_config_value(config, "hidden_dim", 128),
        out_dim=_get_config_value(config, "out_dim", 128),
        encoder_dropout=_get_config_value(config, "encoder_dropout", 0.1),
        gnnlayer_dropout=_get_config_value(config, "gnnlayer_dropout", 0.1),
        num_gnn_layers=_get_config_value(config, "num_gnn_layers", 2),
        operator_type=operator_type,
        operator_kwargs=operator_kwargs,
        edge_in_dim=EDGE_IN_DIM,
    ).to(device)
    return model


def load_trained_model(
    model_path: str,
    config: Any,
    device: torch.device,
) -> HeteroGNNModel:
    model = build_model_from_config(config, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def heterodata_to_dicts(
    data,
) -> Tuple[
    Dict[str, Tensor],
    Dict[Tuple[str, str, str], Tensor],
    Optional[Dict[Tuple[str, str, str], Tensor]],
    Dict[str, Optional[Tensor]],
]:
    x_dict = {ntype: data[ntype].x for ntype in data.node_types}
    y_dict: Dict[str, Optional[Tensor]] = {}
    for ntype in data.node_types:
        y_val = getattr(data[ntype], "y", None)
        y_dict[ntype] = y_val

    edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], Tensor] = {}
    for store in data.edge_stores:
        key = store._key
        edge_index_dict[key] = store.edge_index
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            edge_attr_dict[key] = store.edge_attr

    if not edge_attr_dict:
        edge_attr = None
    else:
        edge_attr = edge_attr_dict

    return x_dict, edge_index_dict, edge_attr, y_dict


class NodeTypeRegressionWrapper(torch.nn.Module):
    """
    Wraps a hetero GNN so that only predictions of a single node type are
    returned. This is helpful because most explainers expect a tensor instead
    of a dictionary as model output.
    """

    def __init__(self, base_model: torch.nn.Module, node_type: str):
        super().__init__()
        self.base_model = base_model
        self.node_type = node_type

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Tensor:
        out = self.base_model(x_dict, edge_index_dict, edge_attr_dict)
        if self.node_type not in out or out[self.node_type] is None:
            raise ValueError(
                f"Base model does not return predictions for node type "
                f"'{self.node_type}'."
            )
        return out[self.node_type].squeeze(-1)


def summarize_node_mask(
    node_mask_dict: Dict[str, Tensor],
    node_type: str,
    node_idx: int,
    top_k: int,
) -> List[Dict[str, float]]:
    mask = node_mask_dict.get(node_type)
    if mask is None:
        return []
    if mask.dim() == 1:
        values = mask
    else:
        if node_idx >= mask.size(0):
            raise IndexError(
                f"Node index {node_idx} out of bounds for type {node_type} "
                f"with {mask.size(0)} nodes."
            )
        values = mask[node_idx]
    values = values.detach().cpu()
    k = min(top_k, values.numel())
    if k <= 0:
        return []
    scores, indices = torch.topk(values, k)
    return [
        {"feature_idx": int(idx), "importance": float(score)}
        for score, idx in zip(scores.tolist(), indices.tolist())
    ]


def summarize_incident_edges(
    edge_mask_dict: Dict[Tuple[str, str, str], Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    node_type: str,
    node_idx: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[float, Tuple[str, str, str], int]] = []
    for edge_type, mask in edge_mask_dict.items():
        if mask is None:
            continue
        src_type, _, dst_type = edge_type
        edge_index = edge_index_dict.get(edge_type)
        if edge_index is None:
            continue
        mask_vals = mask.view(-1).detach().cpu()
        rows = edge_index[0].detach().cpu()
        cols = edge_index[1].detach().cpu()
        for edge_pos in range(mask_vals.size(0)):
            if src_type == node_type and int(rows[edge_pos]) == node_idx:
                candidates.append((float(mask_vals[edge_pos]), edge_type, edge_pos))
            elif dst_type == node_type and int(cols[edge_pos]) == node_idx:
                candidates.append((float(mask_vals[edge_pos]), edge_type, edge_pos))
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_entries = candidates[: max(0, top_k)]
    summaries: List[Dict[str, Any]] = []
    for importance, edge_type, edge_pos in top_entries:
        edge_index = edge_index_dict[edge_type]
        summaries.append(
            {
                "edge_type": edge_type,
                "edge_position": edge_pos,
                "importance": importance,
                "src_index": int(edge_index[0, edge_pos]),
                "dst_index": int(edge_index[1, edge_pos]),
            }
        )
    return summaries


def validate_indices(length: int, index: int, label: str = "index") -> None:
    if index < 0 or index >= length:
        raise IndexError(f"{label} {index} out of bounds for length {length}.")
