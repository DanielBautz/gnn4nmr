
from __future__ import annotations

import json
import os
import random
import re
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from dataloader import ALL_ELEMENTS, ShiftDataset, create_dataloaders
from model import HeteroGNNModel

try:
    from feature_visualization import get_feature_names
except Exception:
    from notebooks.feature_visualization import get_feature_names

NODE_TYPES = ("H", "C", "Others")
TARGETS = ("H", "C")
DETERMINISTIC_METHODS = ("gnnexplainer", "integrated_gradients", "ablation")
ALL_METHODS = ("gnnexplainer", "integrated_gradients", "ablation", "random")

FEATURE_NAME_CACHE = {nt: list(get_feature_names(nt)) for nt in NODE_TYPES}
ELEMENT_MAP = {str(e).lower(): str(e) for e in ALL_ELEMENTS}


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_runtime_for_a100(
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
) -> None:
    """Enable fast CUDA defaults for Ampere/A100 style workloads."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision(str(matmul_precision))
    except Exception:
        pass


def default_num_workers() -> int:
    # Windows uses spawn and tends to be slower for this dataset.
    if os.name == "nt":
        return 0
    cpu_cnt = os.cpu_count() or 4
    return int(max(2, min(8, cpu_cnt // 2)))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _find_col_by_slug(columns: List[str], candidate_slugs: List[str]) -> Optional[str]:
    slug_to_col = {_slug(c): c for c in columns}
    for cand in candidate_slugs:
        if cand in slug_to_col:
            return slug_to_col[cand]
    return None


def normalize_node_type(value: Any) -> Optional[str]:
    v = str(value).strip().lower()
    if v in {"h", "hnode", "h_node", "h-node"}:
        return "H"
    if v in {"c", "cnode", "c_node", "c-node"}:
        return "C"
    if v in {"others", "other", "othersnode", "others_node", "others-node"}:
        return "Others"
    return None


def canonical_feature_name(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value).strip())
    if text.lower().startswith("nodetype_onehot") or text.lower().startswith("nodetypeonehot"):
        return "nodetype_onehot"
    if text.lower().startswith("element_"):
        sym_raw = text.split("_", 1)[1]
        sym = ELEMENT_MAP.get(sym_raw.lower(), sym_raw)
        return f"element_{sym}"
    if text in {"CN_X", "cn_x", "CNX", "cnx", "CN(x)", "cn(x)", "CN(X)"}:
        return "CN(X)"
    return text


def load_and_normalize_topk_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Top-k CSV not found: {path}")

    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = list(raw.columns)

    feature_col = _find_col_by_slug(cols, ["feature", "features"])
    node_col = _find_col_by_slug(cols, ["nodetype", "node_type"])

    method_col_candidates = {
        "gnnexplainer_H": ["gnnexplainerh", "gnnexh"],
        "gnnexplainer_C": ["gnnexplainerc", "gnnexc"],
        "integrated_gradients_H": ["integratedgradientsh", "intgradh"],
        "integrated_gradients_C": ["integratedgradientsc", "intgradc"],
        "ablation_H": ["ablationstudyh", "ablationh"],
        "ablation_C": ["ablationstudyc", "ablationc"],
    }

    resolved_method_cols: Dict[str, str] = {}
    for out_col, slugs in method_col_candidates.items():
        col = _find_col_by_slug(cols, slugs)
        if col is None:
            raise ValueError(f"Missing required column for {out_col}. Available columns: {cols}")
        resolved_method_cols[out_col] = col

    if feature_col is None or node_col is None:
        raise ValueError(f"Could not resolve feature/node_type columns. Available columns: {cols}")

    out = pd.DataFrame()
    out["feature"] = raw[feature_col].astype(str).str.strip()
    out["node_type"] = raw[node_col].apply(normalize_node_type)

    for out_col, src_col in resolved_method_cols.items():
        out[out_col] = pd.to_numeric(raw[src_col], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=["node_type"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Top-k CSV became empty after node_type normalization.")
    return out


def build_feature_lookup() -> Dict[str, Dict[str, int]]:
    lookup: Dict[str, Dict[str, int]] = {nt: {} for nt in NODE_TYPES}
    for nt in NODE_TYPES:
        names = FEATURE_NAME_CACHE[nt]
        mapping: Dict[str, int] = {}
        for idx, name in enumerate(names):
            token = canonical_feature_name(name)
            mapping[token] = idx
            if token == "CN(X)":
                mapping["CN_X"] = idx
        for idx, elem in enumerate(ALL_ELEMENTS):
            if idx < len(names):
                mapping[elem] = idx
                mapping[f"element_{elem}"] = idx
        lookup[nt] = mapping
    return lookup


def resolve_feature_to_indices(node_type: str, feature: Any, feature_lookup: Dict[str, Dict[str, int]]) -> Tuple[int, ...]:
    token = canonical_feature_name(feature)
    if token == "nodetype_onehot":
        return tuple(range(min(13, len(FEATURE_NAME_CACHE[node_type]))))
    if token.startswith("element_"):
        symbol = token.split("_", 1)[1]
        idx = feature_lookup[node_type].get(symbol)
        if idx is None:
            idx = feature_lookup[node_type].get(token)
        return (int(idx),) if idx is not None else tuple()
    idx = feature_lookup[node_type].get(token)
    return (int(idx),) if idx is not None else tuple()

def resolve_feature_selection(topk_df: pd.DataFrame, method: str, target: str, feature_lookup: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    value_col = f"{method}_{target}"
    rows = topk_df[topk_df[value_col] == 1]

    selected_items: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    keep = {nt: set() for nt in NODE_TYPES}
    seen = set()

    for _, row in rows.iterrows():
        nt = str(row["node_type"])
        raw_feat = str(row["feature"]).strip()
        feat_key = canonical_feature_name(raw_feat)
        key = (nt, feat_key)
        if key in seen:
            continue
        seen.add(key)

        idxs = resolve_feature_to_indices(nt, feat_key, feature_lookup)
        if len(idxs) == 0:
            unresolved.append({"node_type": nt, "feature": raw_feat, "method": method, "target": target})
            continue

        selected_items.append(
            {
                "node_type": nt,
                "feature_raw": raw_feat,
                "feature_key": feat_key,
                "indices": tuple(int(i) for i in idxs),
            }
        )
        keep[nt].update(int(i) for i in idxs)

    keep_indices_by_type = {nt: tuple(sorted(v)) for nt, v in keep.items()}
    return {
        "method": method,
        "target": target,
        "selected_items": selected_items,
        "unresolved": unresolved,
        "keep_indices_by_type": keep_indices_by_type,
        "n_items": len(selected_items),
        "n_dims_total": int(sum(len(v) for v in keep_indices_by_type.values())),
    }


def build_random_pool(topk_df: pd.DataFrame, target: str, feature_lookup: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    cols = [f"gnnexplainer_{target}", f"integrated_gradients_{target}", f"ablation_{target}"]
    candidates = topk_df[topk_df[cols].sum(axis=1) > 0]
    pool: List[Dict[str, Any]] = []
    seen = set()

    for _, row in candidates.iterrows():
        nt = str(row["node_type"])
        raw_feat = str(row["feature"]).strip()
        feat_key = canonical_feature_name(raw_feat)
        key = (nt, feat_key)
        if key in seen:
            continue
        seen.add(key)

        idxs = resolve_feature_to_indices(nt, feat_key, feature_lookup)
        if len(idxs) == 0:
            continue

        pool.append(
            {
                "node_type": nt,
                "feature_raw": raw_feat,
                "feature_key": feat_key,
                "indices": tuple(int(i) for i in idxs),
            }
        )
    return pool


def sample_random_selection(random_pool: List[Dict[str, Any]], target: str, k: int, seed: int) -> Dict[str, Any]:
    if len(random_pool) == 0:
        raise ValueError(f"Random pool is empty for target={target}")
    k_eff = int(max(1, min(int(k), len(random_pool))))
    rng = random.Random(int(seed) * 10007 + (0 if target == "H" else 1))
    picked = rng.sample(random_pool, k_eff)

    keep = {nt: set() for nt in NODE_TYPES}
    for item in picked:
        keep[item["node_type"]].update(item["indices"])
    keep_indices_by_type = {nt: tuple(sorted(v)) for nt, v in keep.items()}

    return {
        "method": "random",
        "target": target,
        "selected_items": picked,
        "unresolved": [],
        "keep_indices_by_type": keep_indices_by_type,
        "n_items": len(picked),
        "n_dims_total": int(sum(len(v) for v in keep_indices_by_type.values())),
    }


def feature_name_from_index(node_type: str, feature_idx: int) -> str:
    names = FEATURE_NAME_CACHE.get(node_type, [])
    if 0 <= int(feature_idx) < len(names):
        return str(names[int(feature_idx)])
    return f"feature_{int(feature_idx)}"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_topk_table(topk_df: pd.DataFrame, feature_lookup: Dict[str, Dict[str, int]]):
    deterministic: Dict[Tuple[str, str], Dict[str, Any]] = {}
    preflight_rows: List[Dict[str, Any]] = []

    for method in DETERMINISTIC_METHODS:
        for target in TARGETS:
            sel = resolve_feature_selection(topk_df, method, target, feature_lookup)
            deterministic[(method, target)] = sel
            preflight_rows.append(
                {
                    "method": method,
                    "target": target,
                    "selected_items": sel["n_items"],
                    "selected_dims_total": sel["n_dims_total"],
                    "unresolved_items": len(sel["unresolved"]),
                }
            )
            if sel["n_items"] <= 0:
                raise ValueError(
                    f"Preflight failed: method={method}, target={target} has no selected features. "
                    f"Please check top-k CSV content."
                )

    random_pool_by_target: Dict[str, List[Dict[str, Any]]] = {}
    random_k_by_target: Dict[str, int] = {}

    for target in TARGETS:
        pool = build_random_pool(topk_df, target, feature_lookup)
        random_pool_by_target[target] = pool

        ks = [deterministic[(m, target)]["n_items"] for m in DETERMINISTIC_METHODS]
        random_k = int(np.median(np.array(ks, dtype=float)))
        random_k = max(1, random_k)
        random_k_by_target[target] = random_k

        if len(pool) <= 0:
            raise ValueError(f"Preflight failed: random pool empty for target={target}")

        preflight_rows.append(
            {
                "method": "random",
                "target": target,
                "selected_items": random_k,
                "selected_dims_total": np.nan,
                "unresolved_items": 0,
                "pool_size": len(pool),
            }
        )

    preflight_df = pd.DataFrame(preflight_rows)
    if preflight_df["unresolved_items"].fillna(0).sum() > 0:
        warnings.warn("Some selected features could not be mapped to indices. Check preflight table.")

    return deterministic, random_pool_by_target, random_k_by_target, preflight_df

def _apply_feature_selection(data, keep_indices_by_type: Dict[str, Tuple[int, ...]]):
    for nt in NODE_TYPES:
        if nt not in data.node_types:
            continue

        x = data[nt].x
        keep = keep_indices_by_type.get(nt, tuple())

        if len(keep) == 0:
            data[nt].x = torch.zeros((x.size(0), 1), dtype=x.dtype)
            continue

        valid = [i for i in keep if 0 <= int(i) < x.size(1)]
        if len(valid) == 0:
            data[nt].x = torch.zeros((x.size(0), 1), dtype=x.dtype)
            continue

        data[nt].x = x[:, valid]
    return data


class FeatureSelectShiftDataset(Dataset):
    """View dataset that applies feature selection on top of a cached base ShiftDataset."""

    def __init__(self, base_dataset: ShiftDataset, keep_indices_by_type: Optional[Dict[str, Tuple[int, ...]]] = None):
        self.base_dataset = base_dataset
        keep_indices_by_type = keep_indices_by_type or {}
        self.keep_indices_by_type: Dict[str, Tuple[int, ...]] = {
            nt: tuple(sorted(int(i) for i in keep_indices_by_type.get(nt, tuple())))
            for nt in NODE_TYPES
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        return _apply_feature_selection(data, self.keep_indices_by_type)


def create_fixed_split_indices(
    seed: int,
    config: Dict[str, Any],
    data_dir: Path,
    base_dataset: Optional[ShiftDataset] = None,
) -> Dict[str, List[int]]:
    set_all_seeds(int(seed))

    if base_dataset is None:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=int(config["batch_size"]),
            root_dir=str(data_dir),
            file_name=str(config["data_file_name"]),
            split_ratio=tuple(config["split_ratio"]),
            normalize_node_features=bool(config["normalize_node_features"]),
            normalize_edge_features=bool(config["normalize_edge_features"]),
        )
        return {
            "train": list(train_loader.dataset.indices),
            "val": list(val_loader.dataset.indices),
            "test": list(test_loader.dataset.indices),
        }

    split_ratio = tuple(config["split_ratio"])
    compound_to_indices: Dict[str, List[int]] = {}
    for idx, nx_g in enumerate(base_dataset.nx_graphs):
        compound = str(nx_g.graph.get("compound", "unknown"))
        compound_to_indices.setdefault(compound, []).append(int(idx))

    compounds = list(compound_to_indices.keys())
    random.shuffle(compounds)

    num_compounds = len(compounds)
    train_end = int(split_ratio[0] * num_compounds)
    val_end = train_end + int(split_ratio[1] * num_compounds)

    train_compounds = compounds[:train_end]
    val_compounds = compounds[train_end:val_end]
    test_compounds = compounds[val_end:]

    train_indices: List[int] = []
    for comp in train_compounds:
        train_indices.extend(compound_to_indices[comp])
    val_indices: List[int] = []
    for comp in val_compounds:
        val_indices.extend(compound_to_indices[comp])
    test_indices: List[int] = []
    for comp in test_compounds:
        test_indices.extend(compound_to_indices[comp])

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def build_loaders_for_selection(
    split_indices: Dict[str, List[int]],
    keep_indices_by_type: Dict[str, Tuple[int, ...]],
    config: Dict[str, Any],
    base_dataset: ShiftDataset,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
):
    dataset = FeatureSelectShiftDataset(
        base_dataset=base_dataset,
        keep_indices_by_type=keep_indices_by_type,
    )

    train_subset = Subset(dataset, split_indices["train"])
    val_subset = Subset(dataset, split_indices["val"])
    test_subset = Subset(dataset, split_indices["test"])

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if int(max(0, num_workers)) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(max(2, prefetch_factor))

    train_loader = PyGDataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = PyGDataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = PyGDataLoader(test_subset, shuffle=False, **loader_kwargs)

    return dataset, train_loader, val_loader, test_loader


def build_run_config(base_config: Dict[str, Any], seed: int, in_dim_dict: Dict[str, int], num_epochs: Optional[int] = None):
    cfg = dict(base_config)
    cfg["seed"] = int(seed)
    cfg["in_dim_dict"] = dict(in_dim_dict)
    cfg["operator_kwargs"] = dict(cfg.get("operator_kwargs", {}))

    if num_epochs is not None:
        cfg["num_epochs"] = int(num_epochs)

    if cfg["operator_type"] in ["GATConv", "GATv2Conv"]:
        cfg["operator_kwargs"]["add_self_loops"] = False

    return SimpleNamespace(**cfg)


def _extract_batch_dicts(batch_data):
    x_dict, y_dict = {}, {}
    for ntype in batch_data.node_types:
        x_dict[ntype] = batch_data[ntype].x
        y_dict[ntype] = getattr(batch_data[ntype], "y", None)

    edge_index_dict, edge_attr_dict = {}, {}
    for store in batch_data.edge_stores:
        src, rel, dst = store._key
        edge_index_dict[(src, rel, dst)] = store.edge_index
        edge_attr_dict[(src, rel, dst)] = store.edge_attr

    return x_dict, y_dict, edge_index_dict, edge_attr_dict


def _resolve_amp_dtype(amp_dtype: str = "bf16"):
    amp_dtype = str(amp_dtype).lower().strip()
    if amp_dtype in {"bf16", "bfloat16"} and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _train_one_epoch_fast(
    model,
    dataloader,
    device,
    optimizer,
    config,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    non_blocking: bool = True,
):
    model.train()
    total_mse_H, total_mae_H, count_H = 0.0, 0.0, 0
    total_mse_C, total_mae_C, count_C = 0.0, 0.0, 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    dtype = _resolve_amp_dtype(amp_dtype)

    for batch_data in dataloader:
        batch_data = batch_data.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        x_dict, y_dict, edge_index_dict, edge_attr_dict = _extract_batch_dicts(batch_data)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp_enabled else nullcontext()

        with autocast_ctx:
            out_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            loss_terms = []

            if out_dict.get("H", None) is not None and y_dict.get("H", None) is not None:
                valid_H = ~torch.isnan(y_dict["H"])
                if int(valid_H.sum().item()) > 0:
                    pred_H = out_dict["H"][valid_H]
                    tgt_H = y_dict["H"][valid_H]
                    diff_H = pred_H - tgt_H
                    mse_H = torch.mean(diff_H * diff_H)
                    mae_H = torch.mean(torch.abs(diff_H))
                    loss_terms.append(mae_H * float(config.loss_weight_H))
                    total_mse_H += float(mse_H.item())
                    total_mae_H += float(mae_H.item())
                    count_H += 1

            if out_dict.get("C", None) is not None and y_dict.get("C", None) is not None:
                valid_C = ~torch.isnan(y_dict["C"])
                if int(valid_C.sum().item()) > 0:
                    pred_C = out_dict["C"][valid_C]
                    tgt_C = y_dict["C"][valid_C]
                    diff_C = pred_C - tgt_C
                    mse_C = torch.mean(diff_C * diff_C)
                    mae_C = torch.mean(torch.abs(diff_C))
                    loss_terms.append(mae_C * float(config.loss_weight_C))
                    total_mse_C += float(mse_C.item())
                    total_mae_C += float(mae_C.item())
                    count_C += 1

            if len(loss_terms) == 0:
                continue
            loss = torch.stack(loss_terms).mean()

        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

    train_mse_H = total_mse_H / count_H if count_H > 0 else 0.0
    train_mae_H = total_mae_H / count_H if count_H > 0 else 0.0
    train_mse_C = total_mse_C / count_C if count_C > 0 else 0.0
    train_mae_C = total_mae_C / count_C if count_C > 0 else 0.0
    return train_mse_H, train_mae_H, train_mse_C, train_mae_C


@torch.no_grad()
def _evaluate_fast(model, dataloader, device, config, use_amp: bool = True, amp_dtype: str = "bf16", non_blocking: bool = True):
    model.eval()
    total_mse_H, total_mae_H, count_H = 0.0, 0.0, 0
    total_mse_C, total_mae_C, count_C = 0.0, 0.0, 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    dtype = _resolve_amp_dtype(amp_dtype)

    for batch_data in dataloader:
        batch_data = batch_data.to(device, non_blocking=non_blocking)
        x_dict, y_dict, edge_index_dict, edge_attr_dict = _extract_batch_dicts(batch_data)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp_enabled else nullcontext()

        with autocast_ctx:
            out_dict = model(x_dict, edge_index_dict, edge_attr_dict)

            if out_dict.get("H", None) is not None and y_dict.get("H", None) is not None:
                valid_H = ~torch.isnan(y_dict["H"])
                if int(valid_H.sum().item()) > 0:
                    pred_H = out_dict["H"][valid_H]
                    tgt_H = y_dict["H"][valid_H]
                    diff_H = pred_H - tgt_H
                    mse_H = torch.mean(diff_H * diff_H)
                    mae_H = torch.mean(torch.abs(diff_H))
                    total_mse_H += float(mse_H.item())
                    total_mae_H += float(mae_H.item())
                    count_H += 1

            if out_dict.get("C", None) is not None and y_dict.get("C", None) is not None:
                valid_C = ~torch.isnan(y_dict["C"])
                if int(valid_C.sum().item()) > 0:
                    pred_C = out_dict["C"][valid_C]
                    tgt_C = y_dict["C"][valid_C]
                    diff_C = pred_C - tgt_C
                    mse_C = torch.mean(diff_C * diff_C)
                    mae_C = torch.mean(torch.abs(diff_C))
                    total_mse_C += float(mse_C.item())
                    total_mae_C += float(mae_C.item())
                    count_C += 1

    val_mse_H = total_mse_H / count_H if count_H > 0 else 0.0
    val_mae_H = total_mae_H / count_H if count_H > 0 else 0.0
    val_mse_C = total_mse_C / count_C if count_C > 0 else 0.0
    val_mae_C = total_mae_C / count_C if count_C > 0 else 0.0
    val_score = (val_mae_H * float(config.loss_weight_H) + val_mae_C * float(config.loss_weight_C)) / 2.0
    return val_mse_H, val_mae_H, val_mse_C, val_mae_C, float(val_score)


@torch.no_grad()
def compute_test_mad(model, dataloader, device, use_amp: bool = True, amp_dtype: str = "bf16", non_blocking: bool = True):
    model.eval()
    abs_sum = {"H": 0.0, "C": 0.0}
    count = {"H": 0, "C": 0}
    amp_enabled = bool(use_amp and device.type == "cuda")
    dtype = _resolve_amp_dtype(amp_dtype)

    for batch_data in dataloader:
        batch_data = batch_data.to(device, non_blocking=non_blocking)
        x_dict, y_dict, edge_index_dict, edge_attr_dict = _extract_batch_dicts(batch_data)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp_enabled else nullcontext()
        with autocast_ctx:
            out_dict = model(x_dict, edge_index_dict, edge_attr_dict)

        for nt in ["H", "C"]:
            pred = out_dict.get(nt, None)
            target = y_dict.get(nt, None)
            if pred is None or target is None:
                continue

            valid = ~torch.isnan(target)
            if int(valid.sum().item()) <= 0:
                continue

            err = (pred[valid] - target[valid]).abs().reshape(-1)
            abs_sum[nt] += float(err.sum().item())
            count[nt] += int(err.numel())

    mad_h = (abs_sum["H"] / count["H"]) if count["H"] > 0 else float("nan")
    mad_c = (abs_sum["C"] / count["C"]) if count["C"] > 0 else float("nan")
    return {
        "mad_H": float(mad_h),
        "mad_C": float(mad_c),
        "count_H": int(count["H"]),
        "count_C": int(count["C"]),
    }

def train_for_run(
    run_cfg,
    train_loader,
    val_loader,
    test_loader,
    method: str,
    target: str,
    seed: int,
    selected_items: List[Dict[str, Any]],
    use_wandb: bool = True,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    log_every_n_epochs: int = 1,
    compile_model: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeteroGNNModel(
        run_cfg.in_dim_dict,
        hidden_dim=int(run_cfg.hidden_dim),
        out_dim=int(run_cfg.out_dim),
        encoder_dropout=float(run_cfg.encoder_dropout),
        gnnlayer_dropout=float(run_cfg.gnnlayer_dropout),
        num_gnn_layers=int(run_cfg.num_gnn_layers),
        operator_type=str(run_cfg.operator_type),
        operator_kwargs=dict(run_cfg.operator_kwargs),
        edge_in_dim=10,
    ).to(device)

    if bool(compile_model) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[WARN] torch.compile skipped: {exc}")

    optimizer_name = str(run_cfg.optimizer).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(run_cfg.lr), weight_decay=float(run_cfg.weight_decay))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(run_cfg.lr), weight_decay=float(run_cfg.weight_decay))
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(run_cfg.lr), weight_decay=float(run_cfg.weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {run_cfg.optimizer}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(run_cfg.scheduler_factor),
        patience=int(run_cfg.scheduler_patience),
    )

    wandb_active = False
    wandb_error = ""

    if use_wandb and wandb is not None:
        wb_project = f"gnn_topk_{method}_{seed}"
        wb_name = f"{target}_topk"
        wb_config = {
            "seed": int(seed),
            "method": str(method),
            "target": str(target),
            "selected_items": int(len(selected_items)),
            "selected_item_keys": [f"{x['node_type']}::{x['feature_key']}" for x in selected_items],
            "in_dim_dict": dict(run_cfg.in_dim_dict),
            "num_epochs": int(run_cfg.num_epochs),
            "batch_size": int(run_cfg.batch_size),
            "lr": float(run_cfg.lr),
            "operator_type": str(run_cfg.operator_type),
        }
        try:
            wandb.init(project=wb_project, name=wb_name, config=wb_config, reinit=True)
            wandb_active = True
        except Exception as exc:
            wandb_error = f"wandb.init failed: {exc}"
            print(f"[WARN] {wandb_error}")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_score = float("inf")
    best_epoch = -1
    early_stop_patience = 10
    early_stop_count = 0
    history_rows: List[Dict[str, Any]] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_t = _resolve_amp_dtype(amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and amp_t == torch.float16))

    try:
        for epoch in range(int(run_cfg.num_epochs)):
            train_mse_H, train_mae_H, train_mse_C, train_mae_C = _train_one_epoch_fast(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                config=run_cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_scaler=scaler,
                non_blocking=bool(getattr(train_loader, "pin_memory", False)),
            )
            val_mse_H, val_mae_H, val_mse_C, val_mae_C, val_score = _evaluate_fast(
                model=model,
                dataloader=val_loader,
                device=device,
                config=run_cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                non_blocking=bool(getattr(val_loader, "pin_memory", False)),
            )
            scheduler.step(val_score)

            hist_row = {
                "epoch": int(epoch),
                "train_mse_H": float(train_mse_H),
                "train_mae_H": float(train_mae_H),
                "train_mse_C": float(train_mse_C),
                "train_mae_C": float(train_mae_C),
                "val_mse_H": float(val_mse_H),
                "val_mae_H": float(val_mae_H),
                "val_mse_C": float(val_mse_C),
                "val_mae_C": float(val_mae_C),
                "val_score": float(val_score),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history_rows.append(hist_row)

            do_log_epoch = (int(epoch) % int(max(1, log_every_n_epochs)) == 0) or (epoch == int(run_cfg.num_epochs) - 1)
            if do_log_epoch and wandb_active and wandb.run is not None:
                wandb.log({"epoch": int(epoch), **hist_row})

            if float(val_score) <= float(best_val_score):
                best_val_score = float(val_score)
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    break

        model.load_state_dict(best_state)
        test_mse_H, test_mae_H, test_mse_C, test_mae_C, test_score = _evaluate_fast(
            model=model,
            dataloader=test_loader,
            device=device,
            config=run_cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            non_blocking=bool(getattr(test_loader, "pin_memory", False)),
        )
        mad = compute_test_mad(
            model=model,
            dataloader=test_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            non_blocking=bool(getattr(test_loader, "pin_memory", False)),
        )

        final_metrics = {
            "status": "ok",
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val_score),
            "test_mse_H": float(test_mse_H),
            "test_mae_H": float(test_mae_H),
            "test_mse_C": float(test_mse_C),
            "test_mae_C": float(test_mae_C),
            "test_score": float(test_score),
            "mad_H": float(mad["mad_H"]),
            "mad_C": float(mad["mad_C"]),
            "mad_count_H": int(mad["count_H"]),
            "mad_count_C": int(mad["count_C"]),
            "error": "",
            "wandb_error": wandb_error,
        }

        if wandb_active and wandb.run is not None:
            wandb.log(
                {
                    "best_epoch": int(best_epoch),
                    "best_val_score": float(best_val_score),
                    "test_mse_H": float(test_mse_H),
                    "test_mae_H": float(test_mae_H),
                    "test_mse_C": float(test_mse_C),
                    "test_mae_C": float(test_mae_C),
                    "test_score": float(test_score),
                    "mad_H": float(mad["mad_H"]),
                    "mad_C": float(mad["mad_C"]),
                }
            )

        return final_metrics, pd.DataFrame(history_rows)

    except Exception as exc:
        return {
            "status": "failed",
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val_score) if np.isfinite(best_val_score) else float("nan"),
            "test_mse_H": float("nan"),
            "test_mae_H": float("nan"),
            "test_mse_C": float("nan"),
            "test_mae_C": float("nan"),
            "test_score": float("nan"),
            "mad_H": float("nan"),
            "mad_C": float("nan"),
            "mad_count_H": 0,
            "mad_count_C": 0,
            "error": str(exc),
            "wandb_error": wandb_error,
        }, pd.DataFrame(history_rows)

    finally:
        if wandb_active and wandb is not None and wandb.run is not None:
            wandb.run.summary["run_status"] = "finished"
            wandb.finish()

def _persist_table(path: Path, rows: List[Dict[str, Any]], key_cols: Optional[List[str]] = None):
    key_cols = key_cols or []
    new_df = pd.DataFrame(rows)

    if path.exists():
        old_df = pd.read_csv(path)
        full_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        full_df = new_df

    if key_cols:
        keep_cols = [c for c in key_cols if c in full_df.columns]
        if keep_cols:
            full_df = full_df.drop_duplicates(subset=keep_cols, keep="last")

    full_df.to_csv(path, index=False)


def run_all_topk_experiments(
    topk_csv_path: Path,
    output_dir: Path,
    seeds: List[int],
    base_config: Dict[str, Any],
    data_dir: Path,
    use_wandb: bool = True,
    skip_completed: bool = True,
    smoke_num_epochs: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 4,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
    log_every_n_epochs: int = 1,
    compile_model: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime_for_a100(
        enable_tf32=bool(enable_tf32),
        cudnn_benchmark=bool(cudnn_benchmark),
        matmul_precision=str(matmul_precision),
    )

    if num_workers is None:
        num_workers = default_num_workers()
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    run_results_path = output_dir / "run_results.csv"
    epoch_history_path = output_dir / "epoch_history.csv"
    feature_sets_path = output_dir / "feature_sets_resolved.csv"
    preflight_path = output_dir / "preflight_summary.csv"
    summary_path = output_dir / "summary_by_method_target.csv"

    topk_df = load_and_normalize_topk_csv(topk_csv_path)
    feature_lookup = build_feature_lookup()
    deterministic, random_pool_by_target, random_k_by_target, preflight_df = validate_topk_table(topk_df, feature_lookup)
    preflight_df.to_csv(preflight_path, index=False)

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topk_csv_path": str(topk_csv_path),
        "output_dir": str(output_dir),
        "seeds": [int(s) for s in seeds],
        "base_config": dict(base_config),
        "smoke_num_epochs": int(smoke_num_epochs) if smoke_num_epochs is not None else None,
        "random_k_by_target": {k: int(v) for k, v in random_k_by_target.items()},
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "prefetch_factor": int(prefetch_factor),
        "use_amp": bool(use_amp),
        "amp_dtype": str(amp_dtype),
        "enable_tf32": bool(enable_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "matmul_precision": str(matmul_precision),
        "log_every_n_epochs": int(log_every_n_epochs),
        "compile_model": bool(compile_model),
    }
    save_json(output_dir / "run_config.json", run_meta)

    # Cache the base dataset once. Feature subsets are lightweight views on this dataset.
    base_dataset = ShiftDataset(
        root_dir=str(data_dir),
        file_name=str(base_config["data_file_name"]),
        normalize_node_features=bool(base_config["normalize_node_features"]),
        normalize_edge_features=bool(base_config["normalize_edge_features"]),
    )

    completed_run_ids = set()
    if skip_completed and run_results_path.exists():
        prev = pd.read_csv(run_results_path)
        if "run_id" in prev.columns:
            completed_run_ids = set(prev["run_id"].astype(str).tolist())

    run_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        split_indices = create_fixed_split_indices(
            seed=int(seed),
            config=base_config,
            data_dir=data_dir,
            base_dataset=base_dataset,
        )
        save_json(output_dir / f"split_indices_seed{int(seed)}.json", split_indices)

        random_sel = {
            t: sample_random_selection(
                random_pool=random_pool_by_target[t],
                target=t,
                k=int(random_k_by_target[t]),
                seed=int(seed),
            )
            for t in TARGETS
        }

        for method in ALL_METHODS:
            for target in TARGETS:
                run_id = f"{method}__{target}__seed{int(seed)}"
                if run_id in completed_run_ids:
                    continue

                selection = random_sel[target] if method == "random" else deterministic[(method, target)]
                keep_indices_by_type = {
                    nt: tuple(selection["keep_indices_by_type"].get(nt, tuple()))
                    for nt in NODE_TYPES
                }
                in_dim_dict = {nt: max(1, len(keep_indices_by_type[nt])) for nt in NODE_TYPES}

                # Ensure fair, reproducible per-run initialization/order.
                set_all_seeds(int(seed))

                _, train_loader, val_loader, test_loader = build_loaders_for_selection(
                    split_indices=split_indices,
                    keep_indices_by_type=keep_indices_by_type,
                    config=base_config,
                    base_dataset=base_dataset,
                    num_workers=int(num_workers),
                    pin_memory=bool(pin_memory),
                    prefetch_factor=int(prefetch_factor),
                )

                run_cfg = build_run_config(
                    base_config=base_config,
                    seed=int(seed),
                    in_dim_dict=in_dim_dict,
                    num_epochs=smoke_num_epochs,
                )

                final_metrics, hist_df = train_for_run(
                    run_cfg=run_cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    method=method,
                    target=target,
                    seed=int(seed),
                    selected_items=selection["selected_items"],
                    use_wandb=bool(use_wandb),
                    use_amp=bool(use_amp),
                    amp_dtype=str(amp_dtype),
                    log_every_n_epochs=int(log_every_n_epochs),
                    compile_model=bool(compile_model),
                )

                mad_primary = float(final_metrics["mad_H"]) if target == "H" else float(final_metrics["mad_C"])

                run_row = {
                    "run_id": run_id,
                    "seed": int(seed),
                    "method": method,
                    "target": target,
                    "status": final_metrics["status"],
                    "error": final_metrics["error"],
                    "wandb_error": final_metrics["wandb_error"],
                    "project": f"gnn_topk_{method}_{int(seed)}",
                    "run_name": f"{target}_topk",
                    "selected_item_count": int(selection["n_items"]),
                    "selected_dim_H": int(len(keep_indices_by_type["H"])),
                    "selected_dim_C": int(len(keep_indices_by_type["C"])),
                    "selected_dim_Others": int(len(keep_indices_by_type["Others"])),
                    "in_dim_H": int(in_dim_dict["H"]),
                    "in_dim_C": int(in_dim_dict["C"]),
                    "in_dim_Others": int(in_dim_dict["Others"]),
                    "best_epoch": int(final_metrics["best_epoch"]),
                    "best_val_score": float(final_metrics["best_val_score"]),
                    "test_mse_H": float(final_metrics["test_mse_H"]),
                    "test_mae_H": float(final_metrics["test_mae_H"]),
                    "test_mse_C": float(final_metrics["test_mse_C"]),
                    "test_mae_C": float(final_metrics["test_mae_C"]),
                    "mad_H": float(final_metrics["mad_H"]),
                    "mad_C": float(final_metrics["mad_C"]),
                    "mad_primary": float(mad_primary),
                    "mad_count_H": int(final_metrics["mad_count_H"]),
                    "mad_count_C": int(final_metrics["mad_count_C"]),
                }
                run_rows.append(run_row)

                if not hist_df.empty:
                    hist_df = hist_df.copy()
                    hist_df["run_id"] = run_id
                    hist_df["seed"] = int(seed)
                    hist_df["method"] = method
                    hist_df["target"] = target
                    epoch_rows.extend(hist_df.to_dict("records"))

                for nt in NODE_TYPES:
                    for idx in keep_indices_by_type[nt]:
                        feature_rows.append(
                            {
                                "run_id": run_id,
                                "seed": int(seed),
                                "method": method,
                                "target": target,
                                "node_type": nt,
                                "feature_idx": int(idx),
                                "feature_name": feature_name_from_index(nt, int(idx)),
                            }
                        )

                _persist_table(run_results_path, run_rows, key_cols=["run_id"])
                run_rows = []

                if len(epoch_rows) > 0:
                    _persist_table(epoch_history_path, epoch_rows, key_cols=["run_id", "epoch"])
                    epoch_rows = []

                if len(feature_rows) > 0:
                    _persist_table(feature_sets_path, feature_rows, key_cols=["run_id", "node_type", "feature_idx"])
                    feature_rows = []

                print(
                    f"[{run_id}] status={run_row['status']} mad_primary={run_row['mad_primary']:.6f} "
                    f"(mad_H={run_row['mad_H']:.6f}, mad_C={run_row['mad_C']:.6f})"
                )

    if run_results_path.exists():
        run_df = pd.read_csv(run_results_path)
        summary_df = (
            run_df.groupby(["method", "target"], dropna=False)["mad_primary"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
            .sort_values(["target", "mean"], ascending=[True, True])
        )
        summary_df.to_csv(summary_path, index=False)
    else:
        run_df = pd.DataFrame()
        summary_df = pd.DataFrame()

    epoch_df = pd.read_csv(epoch_history_path) if epoch_history_path.exists() else pd.DataFrame()
    feat_df = pd.read_csv(feature_sets_path) if feature_sets_path.exists() else pd.DataFrame()

    return run_df, epoch_df, feat_df, summary_df


def load_results_for_visualization(run_dir: Path):
    run_dir = Path(run_dir)
    run_path = run_dir / "run_results.csv"
    epoch_path = run_dir / "epoch_history.csv"
    feat_path = run_dir / "feature_sets_resolved.csv"
    run_df = pd.read_csv(run_path) if run_path.exists() else pd.DataFrame()
    epoch_df = pd.read_csv(epoch_path) if epoch_path.exists() else pd.DataFrame()
    feat_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
    return run_df, epoch_df, feat_df

def plot_mad_distribution(run_df: pd.DataFrame):
    if run_df.empty:
        print("No run data available.")
        return

    ok_df = run_df[run_df["status"] == "ok"].copy()
    if ok_df.empty:
        print("No successful runs available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    for ax, target in zip(axes, TARGETS):
        sub = ok_df[ok_df["target"] == target].copy()
        if sub.empty:
            ax.set_title(f"target={target} (no data)")
            ax.axis("off")
            continue

        methods = sorted(sub["method"].unique().tolist())
        data = [sub[sub["method"] == m]["mad_primary"].astype(float).dropna().to_numpy() for m in methods]

        ax.boxplot(data, labels=methods, showmeans=True)
        ax.set_title(f"MAD distribution on test split ({target})")
        ax.set_xlabel("method")
        ax.set_ylabel("mad_primary")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_seed_stability(run_df: pd.DataFrame):
    if run_df.empty:
        print("No run data available.")
        return

    ok_df = run_df[run_df["status"] == "ok"].copy()
    if ok_df.empty:
        print("No successful runs available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    for ax, target in zip(axes, TARGETS):
        sub = ok_df[ok_df["target"] == target].copy()
        if sub.empty:
            ax.set_title(f"target={target} (no data)")
            ax.axis("off")
            continue

        for method in sorted(sub["method"].unique().tolist()):
            mdf = sub[sub["method"] == method].sort_values("seed")
            ax.plot(
                mdf["seed"],
                mdf["mad_primary"],
                marker="o",
                linewidth=1.5,
                alpha=0.9,
                label=method,
            )

        ax.set_title(f"Seed stability ({target})")
        ax.set_xlabel("seed")
        ax.set_ylabel("mad_primary")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def build_ranking_tables(run_df: pd.DataFrame):
    if run_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ok_df = run_df[run_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = {}
    for target in TARGETS:
        sub = ok_df[ok_df["target"] == target]
        rank_df = (
            sub.groupby("method", dropna=False)["mad_primary"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )
        out[target] = rank_df

    return out.get("H", pd.DataFrame()), out.get("C", pd.DataFrame())


def plot_learning_curves(epoch_df: pd.DataFrame):
    if epoch_df.empty:
        print("No epoch history available.")
        return

    req_cols = {"method", "target", "epoch", "train_mae_H", "train_mae_C", "val_mae_H", "val_mae_C"}
    if not req_cols.issubset(set(epoch_df.columns)):
        print("Epoch history missing required columns for learning curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    h_df = epoch_df[epoch_df["target"] == "H"].copy()
    for method in sorted(h_df["method"].unique().tolist()):
        m = h_df[h_df["method"] == method]
        grp = m.groupby("epoch", dropna=False).agg(train=("train_mae_H", "mean"), val=("val_mae_H", "mean")).reset_index()
        axes[0, 0].plot(grp["epoch"], grp["train"], label=method)
        axes[0, 1].plot(grp["epoch"], grp["val"], label=method)

    axes[0, 0].set_title("Train MAE_H (mean over seeds)")
    axes[0, 1].set_title("Val MAE_H (mean over seeds)")

    c_df = epoch_df[epoch_df["target"] == "C"].copy()
    for method in sorted(c_df["method"].unique().tolist()):
        m = c_df[c_df["method"] == method]
        grp = m.groupby("epoch", dropna=False).agg(train=("train_mae_C", "mean"), val=("val_mae_C", "mean")).reset_index()
        axes[1, 0].plot(grp["epoch"], grp["train"], label=method)
        axes[1, 1].plot(grp["epoch"], grp["val"], label=method)

    axes[1, 0].set_title("Train MAE_C (mean over seeds)")
    axes[1, 1].set_title("Val MAE_C (mean over seeds)")

    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()
