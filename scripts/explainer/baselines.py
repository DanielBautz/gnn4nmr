"""
Scientific baseline builder for Integrated Gradients explanations.

Layout indices are derived directly from the feature functions in dataloader.py:
  get_h_features, get_c_features, get_others_features.

Normalization note:
  Medians are computed in the *normalized* feature space (i.e., on dataset[idx].x,
  which already has the continuous features z-scored for indices >= 13).
  The OneHot and mass/charge fields at indices 0-12 are never normalized.
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_ELEMENTS: List[str] = [
    "H", "C", "Li", "B", "N", "O",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl",
]
E: int = len(ALL_ELEMENTS)  # 13 — OneHot length

# Default cache directory for baseline artefacts (medians, distributions, etc.)
# Previously this lived under "models"; moved to a dedicated folder for clarity.
BASELINE_CACHE_DIR = "baselines"


# ---------------------------------------------------------------------------
# Layout dataclass
# ---------------------------------------------------------------------------

@dataclass
class NodeLayout:
    """Holds the exact feature-index layout for one node type."""
    node_type: str
    n_features: int
    onehot_slice: slice          # slice(0, E)
    mass_idx: int                # E + 0

    formal_charge_idx: int       # E + 1
    degree_idx: int              # E + 2

    # Optional indices (None if the node type does not have this feature)
    shift_low_idx: Optional[int] = None
    cn_x_idx: Optional[int] = None
    shielding_dia_idx: Optional[int] = None
    shielding_para_idx: Optional[int] = None
    span_idx: Optional[int] = None
    skew_idx: Optional[int] = None
    asymmetry_idx: Optional[int] = None
    anisotropy_idx: Optional[int] = None
    at_charge_mull_idx: Optional[int] = None
    at_charge_loew_idx: Optional[int] = None
    orb_mull_s_idx: Optional[int] = None
    orb_mull_p_idx: Optional[int] = None
    orb_mull_d_idx: Optional[int] = None
    orb_stdev_mull_p_idx: Optional[int] = None
    orb_loew_s_idx: Optional[int] = None
    orb_loew_p_idx: Optional[int] = None
    orb_loew_d_idx: Optional[int] = None
    orb_stdev_loew_p_idx: Optional[int] = None
    bo_loew_idx: Optional[int] = None        # H: single BO_loew
    bo_mayer_idx: Optional[int] = None       # H: single BO_mayer
    bo_loew_sum_idx: Optional[int] = None    # C: BO_loew_sum
    bo_loew_av_idx: Optional[int] = None     # C
    bo_mayer_sum_idx: Optional[int] = None   # C
    bo_mayer_av_idx: Optional[int] = None    # C
    mayer_va_idx: Optional[int] = None


# ---------------------------------------------------------------------------
# get_layout
# ---------------------------------------------------------------------------

def get_layout(node_type: str) -> NodeLayout:
    """
    Return the exact feature-index layout for the given node_type.

    Derived from the feature functions in dataloader.py:
      - get_h_features   → 33 features (indices 0-32)
      - get_c_features   → 39 features (indices 0-38)
      - get_others_features → 16 features (indices 0-15)
    """
    if node_type == "H":
        # Feature order (all 33):
        # [0:13) one-hot  |  13: mass  |  14: formal_charge  |  15: degree
        # 16: shift_low  |  17: CN(X)
        # 18: shielding_dia  |  19: shielding_para
        # 20: span  |  21: skew  |  22: asymmetry  |  23: anisotropy
        # 24: at_charge_mull  |  25: at_charge_loew
        # 26: orb_charge_mull_s  |  27: orb_charge_mull_p
        # 28: orb_charge_loew_s  |  29: orb_charge_loew_p
        # 30: BO_loew  |  31: BO_mayer  |  32: mayer_VA
        return NodeLayout(
            node_type="H",
            n_features=33,
            onehot_slice=slice(0, E),
            mass_idx=E,            # 13
            formal_charge_idx=E+1, # 14
            degree_idx=E+2,        # 15
            shift_low_idx=E+3,     # 16
            cn_x_idx=E+4,          # 17
            shielding_dia_idx=E+5, # 18
            shielding_para_idx=E+6,# 19
            span_idx=E+7,          # 20
            skew_idx=E+8,          # 21
            asymmetry_idx=E+9,     # 22
            anisotropy_idx=E+10,   # 23
            at_charge_mull_idx=E+11,  # 24
            at_charge_loew_idx=E+12,  # 25
            orb_mull_s_idx=E+13,   # 26
            orb_mull_p_idx=E+14,   # 27
            orb_mull_d_idx=None,
            orb_stdev_mull_p_idx=None,
            orb_loew_s_idx=E+15,   # 28
            orb_loew_p_idx=E+16,   # 29
            orb_loew_d_idx=None,
            orb_stdev_loew_p_idx=None,
            bo_loew_idx=E+17,      # 30
            bo_mayer_idx=E+18,     # 31
            bo_loew_sum_idx=None,
            bo_loew_av_idx=None,
            bo_mayer_sum_idx=None,
            bo_mayer_av_idx=None,
            mayer_va_idx=E+19,     # 32
        )

    elif node_type == "C":
        # Feature order (all 39):
        # [0:13) one-hot  |  13: mass  |  14: formal_charge  |  15: degree
        # 16: shift_low  |  17: CN(X)
        # 18: shielding_dia  |  19: shielding_para
        # 20: span  |  21: skew  |  22: asymmetry  |  23: anisotropy
        # 24: at_charge_mull  |  25: at_charge_loew
        # 26: orb_charge_mull_s  |  27: orb_charge_mull_p  |  28: orb_charge_mull_d
        # 29: orb_stdev_mull_p
        # 30: orb_charge_loew_s  |  31: orb_charge_loew_p  |  32: orb_charge_loew_d
        # 33: orb_stdev_loew_p
        # 34: BO_loew_sum  |  35: BO_loew_av
        # 36: BO_mayer_sum  |  37: BO_mayer_av
        # 38: mayer_VA
        return NodeLayout(
            node_type="C",
            n_features=39,
            onehot_slice=slice(0, E),
            mass_idx=E,            # 13
            formal_charge_idx=E+1, # 14
            degree_idx=E+2,        # 15
            shift_low_idx=E+3,     # 16
            cn_x_idx=E+4,          # 17
            shielding_dia_idx=E+5, # 18
            shielding_para_idx=E+6,# 19
            span_idx=E+7,          # 20
            skew_idx=E+8,          # 21
            asymmetry_idx=E+9,     # 22
            anisotropy_idx=E+10,   # 23
            at_charge_mull_idx=E+11,  # 24
            at_charge_loew_idx=E+12,  # 25
            orb_mull_s_idx=E+13,   # 26
            orb_mull_p_idx=E+14,   # 27
            orb_mull_d_idx=E+15,   # 28
            orb_stdev_mull_p_idx=E+16,  # 29
            orb_loew_s_idx=E+17,   # 30
            orb_loew_p_idx=E+18,   # 31
            orb_loew_d_idx=E+19,   # 32
            orb_stdev_loew_p_idx=E+20,  # 33
            bo_loew_idx=None,
            bo_mayer_idx=None,
            bo_loew_sum_idx=E+21,  # 34
            bo_loew_av_idx=E+22,   # 35
            bo_mayer_sum_idx=E+23, # 36
            bo_mayer_av_idx=E+24,  # 37
            mayer_va_idx=E+25,     # 38
        )

    elif node_type == "Others":
        # Feature order (all 16):
        # [0:13) one-hot  |  13: mass  |  14: formal_charge  |  15: degree
        return NodeLayout(
            node_type="Others",
            n_features=16,
            onehot_slice=slice(0, E),
            mass_idx=E,            # 13
            formal_charge_idx=E+1, # 14
            degree_idx=E+2,        # 15
            # All NMR/orbital features are absent for "Others"
        )

    else:
        raise ValueError(f"Unknown node_type: '{node_type}'. Expected 'H', 'C', or 'Others'.")


# ---------------------------------------------------------------------------
# detect_element
# ---------------------------------------------------------------------------

def detect_element(x: torch.Tensor, onehot_slice: slice) -> str:
    """
    Determine element string from the OneHot block of a node-feature vector.

    Args:
        x: Feature tensor for a single node [F].
        onehot_slice: Slice into x that selects the one-hot block.

    Returns:
        Element symbol string, e.g. 'C'.
    """
    oh = x[onehot_slice]
    idx = int(oh.argmax().item())
    if idx < 0 or idx >= len(ALL_ELEMENTS):
        return "Unknown"
    return ALL_ELEMENTS[idx]


# ---------------------------------------------------------------------------
# Median computation helpers
# ---------------------------------------------------------------------------

def _collect_node_features(
    dataset,
    train_graph_indices: List[int],
    node_type: str,
) -> Optional[torch.Tensor]:
    """
    Collect all node-feature tensors of `node_type` from the given train graphs.

    Returns a tensor of shape [N_nodes, F], or None if no nodes were found.
    The dataset is expected to return HeteroData objects with `.x` per node type.
    """
    all_feats = []
    for g_idx in train_graph_indices:
        data = dataset[g_idx]
        if not hasattr(data, node_type) and node_type not in data.node_types:
            continue
        try:
            x = data[node_type].x  # [N, F]
            if x is not None and x.shape[0] > 0:
                all_feats.append(x.detach().cpu())
        except (AttributeError, KeyError):
            continue

    if not all_feats:
        return None
    return torch.cat(all_feats, dim=0)  # [total_nodes, F]


def compute_train_median_global(
    dataset,
    train_graph_indices: List[int],
    node_type: str,
) -> torch.Tensor:
    """
    Compute per-feature median across all nodes of `node_type` in train graphs.

    Args:
        dataset: ShiftDataset (or Subset) — returns normalized HeteroData.
        train_graph_indices: Graph indices that belong to the train split.
        node_type: 'H', 'C', or 'Others'.

    Returns:
        Tensor of shape [F] with median feature values.
    """
    all_feats = _collect_node_features(dataset, train_graph_indices, node_type)
    if all_feats is None:
        layout = get_layout(node_type)
        logger.warning(
            "compute_train_median_global: No nodes of type '%s' found in "
            "%d train graphs. Returning zeros.", node_type, len(train_graph_indices)
        )
        return torch.zeros(layout.n_features)
    median = torch.median(all_feats, dim=0).values  # [F]
    logger.info(
        "compute_train_median_global: computed median for %d %s nodes "
        "from %d train graphs.",
        all_feats.shape[0], node_type, len(train_graph_indices),
    )
    return median


def compute_train_median_by_element(
    dataset,
    train_graph_indices: List[int],
    node_type: str,
    min_count: int = 50,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-feature median separately for each element present in the
    train split.  Only elements with at least `min_count` nodes are included
    (others fall back to the global median).

    Args:
        dataset: ShiftDataset (or Subset).
        train_graph_indices: Graph indices for the train split.
        node_type: 'H', 'C', or 'Others'.
        min_count: Minimum number of nodes required to keep element-specific median.

    Returns:
        Dict mapping element symbol → median tensor [F].
        Elements below `min_count` are excluded (caller falls back to global).
    """
    all_feats = _collect_node_features(dataset, train_graph_indices, node_type)
    if all_feats is None:
        return {}

    layout = get_layout(node_type)
    oh = all_feats[:, layout.onehot_slice]  # [N, E]
    elem_indices = oh.argmax(dim=1)          # [N] — index into ALL_ELEMENTS

    result: Dict[str, torch.Tensor] = {}
    for e_idx, elem in enumerate(ALL_ELEMENTS):
        mask = elem_indices == e_idx
        count = int(mask.sum().item())
        if count < min_count:
            logger.debug(
                "compute_train_median_by_element: element '%s' has only %d nodes "
                "(< min_count=%d) — excluded from element-specific medians.",
                elem, count, min_count,
            )
            continue
        subset = all_feats[mask]  # [count, F]
        result[elem] = torch.median(subset, dim=0).values
        logger.info(
            "compute_train_median_by_element: %s[%s]: %d nodes.",
            node_type, elem, count,
        )
    return result


# ---------------------------------------------------------------------------
# Element distribution (for probabilistic OneHot baseline)
# ---------------------------------------------------------------------------

def compute_element_distribution(
    dataset,
    train_graph_indices: List[int],
) -> torch.Tensor:
    """
    Compute empirical element distribution p(element) over the train split.

    The result is a vector of length len(ALL_ELEMENTS) in the exact order of
    ALL_ELEMENTS. Elements with zero OneHot (padding/unknown) are ignored.
    """
    counts = torch.zeros(len(ALL_ELEMENTS), dtype=torch.float32)

    for node_type in ("H", "C", "Others"):
        feats = _collect_node_features(dataset, train_graph_indices, node_type)
        if feats is None:
            continue
        layout = get_layout(node_type)
        oh = feats[:, layout.onehot_slice]
        # ignore rows where onehot is all zeros (padding/unknown)
        mask = oh.sum(dim=1) > 0
        if mask.any():
            counts += oh[mask].sum(dim=0)

    total = float(counts.sum().item())
    if total <= 0:
        logger.warning(
            "compute_element_distribution: no valid one-hot entries found; returning uniform distribution."
        )
        return torch.full_like(counts, 1.0 / len(ALL_ELEMENTS))

    return counts / total


def load_or_compute_element_distribution(
    dataset,
    train_graph_indices: List[int],
    cache_dir: str = BASELINE_CACHE_DIR,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Cache-aware wrapper for compute_element_distribution."""
    os.makedirs(cache_dir, exist_ok=True)
    dist_path = os.path.join(cache_dir, "baseline_elem_distribution.pt")

    if not force_recompute and os.path.exists(dist_path):
        logger.info("Loading cached element distribution from %s", dist_path)
        return torch.load(dist_path, weights_only=True)

    logger.info(
        "Computing empirical element distribution over %d train graphs...",
        len(train_graph_indices),
    )
    dist = compute_element_distribution(dataset, train_graph_indices)
    torch.save(dist, dist_path)
    logger.info("Saved element distribution to %s", dist_path)
    return dist


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_or_compute_medians(
    dataset,
    train_graph_indices: List[int],
    node_type: str,
    cache_dir: str = BASELINE_CACHE_DIR,
    min_count: int = 50,
    force_recompute: bool = False,
):
    """
    Load cached medians from `cache_dir`, or compute and save them.

    Files:
        {cache_dir}/baseline_median_global_{node_type}.pt
        {cache_dir}/baseline_median_by_element_{node_type}.pt

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            (median_global [F], median_by_element dict)
    """
    os.makedirs(cache_dir, exist_ok=True)
    global_path = os.path.join(cache_dir, f"baseline_median_global_{node_type}.pt")
    by_elem_path = os.path.join(cache_dir, f"baseline_median_by_element_{node_type}.pt")

    if not force_recompute and os.path.exists(global_path) and os.path.exists(by_elem_path):
        logger.info("Loading cached medians from %s and %s", global_path, by_elem_path)
        median_global = torch.load(global_path, weights_only=True)
        median_by_element = torch.load(by_elem_path, weights_only=False)
        return median_global, median_by_element

    logger.info(
        "Computing medians for node_type='%s' over %d train graphs...",
        node_type, len(train_graph_indices),
    )
    median_global = compute_train_median_global(dataset, train_graph_indices, node_type)
    median_by_element = compute_train_median_by_element(
        dataset, train_graph_indices, node_type, min_count=min_count
    )

    torch.save(median_global, global_path)
    torch.save(median_by_element, by_elem_path)
    logger.info("Saved medians to %s and %s", global_path, by_elem_path)

    return median_global, median_by_element


# ---------------------------------------------------------------------------
# build_scientific_baseline — the main entry point
# ---------------------------------------------------------------------------

def build_scientific_baseline(
    target_features: torch.Tensor,
    node_type: str,
    median_global: torch.Tensor,
    median_by_element: Optional[Dict[str, torch.Tensor]] = None,
    elem_distribution: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build a feature-specific scientific baseline for IG.

    Strategy:
      1. Start from the element-conditional median (or global median as fallback).
      2. Identity overwrite: OneHot + mass (keep input values).
      3. Zero overwrite: formal_charge, shift_low, span/skew/asymmetry/anisotropy,
                          at_charge_mull/loew, orb_stdev_mull_p/loew_p.
      4. Median stays: degree, CN(X), shielding_dia/para, orbital s/p/d populations,
                        bond orders, mayer_VA.

    Args:
        target_features: Normalized feature tensor for the target node [F].
                         Must be on CPU.
        node_type: 'H', 'C', or 'Others'.
        median_global: Global per-feature median tensor [F] (CPU).
        median_by_element: Optional dict element -> median tensor [F] (CPU).

    Returns:
        Baseline tensor [F] on CPU.
    """
    layout = get_layout(node_type)
    element = detect_element(target_features, layout.onehot_slice)

    # --- Step 1: choose starting point ---
    if median_by_element is not None and element in median_by_element:
        baseline = median_by_element[element].clone().float()
        logger.debug(
            "build_scientific_baseline: node_type=%s element=%s → using element-specific median.",
            node_type, element,
        )
    else:
        baseline = median_global.clone().float()
        logger.debug(
            "build_scientific_baseline: node_type=%s element=%s → using global median (no element-specific found).",
            node_type, element,
        )

    # Safety: ensure shapes match
    if baseline.shape[0] != layout.n_features:
        raise ValueError(
            f"Median tensor has {baseline.shape[0]} features but layout for "
            f"'{node_type}' expects {layout.n_features}."
        )

    # --- Step 2: OneHot & mass ---
    # Atomtyp bleibt fix (OneHot aus den Eingabefeatures), keine probabilistische Verteilung.
    baseline[layout.onehot_slice] = target_features[layout.onehot_slice].float()

    # Masse bleibt element-konditional wie im Median (nicht auf Input überschreiben).
    # baseline[layout.mass_idx] stammt bereits aus median_global/median_by_element.

    # --- Step 3: Zero-reference features ---
    # formal_charge
    baseline[layout.formal_charge_idx] = 0.0

    # shift_low
    if layout.shift_low_idx is not None:
        baseline[layout.shift_low_idx] = 0.0

    # span, skew, asymmetry, anisotropy
    for idx_name in ("span_idx", "skew_idx", "asymmetry_idx", "anisotropy_idx"):
        idx = getattr(layout, idx_name, None)
        if idx is not None:
            baseline[idx] = 0.0

    # at_charge_mull, at_charge_loew
    for idx_name in ("at_charge_mull_idx", "at_charge_loew_idx"):
        idx = getattr(layout, idx_name, None)
        if idx is not None:
            baseline[idx] = 0.0

    # orb_stdev_mull_p, orb_stdev_loew_p (only C has these)
    for idx_name in ("orb_stdev_mull_p_idx", "orb_stdev_loew_p_idx"):
        idx = getattr(layout, idx_name, None)
        if idx is not None:
            baseline[idx] = 0.0

    # --- Step 4: Logging & Sanity checks ---
    _log_and_assert_baseline(
        baseline,
        target_features,
        layout,
        element,
        node_type,
        elem_distribution=elem_distribution,
    )

    return baseline


# ---------------------------------------------------------------------------
# Internal: logging + assertions
# ---------------------------------------------------------------------------

def _log_and_assert_baseline(
    baseline: torch.Tensor,
    target_features: torch.Tensor,
    layout: NodeLayout,
    element: str,
    node_type: str,
    elem_distribution: Optional[torch.Tensor] = None,
) -> None:
    """Log key baseline values (logger only) and run sanity assertions."""

    # Logging (only via logger; no stdout prints)
    logger.debug("=== Scientific Baseline: node_type=%s  element=%s ===", node_type, element)

    def _fmt(idx):
        return f"{baseline[idx].item():.6f}" if idx is not None else "N/A"

    logger.debug("  formal_charge  [%s] = %s  (expected 0.0)",
                 layout.formal_charge_idx, _fmt(layout.formal_charge_idx))
    if layout.shift_low_idx is not None:
        logger.debug("  shift_low      [%s] = %s  (expected 0.0)",
                     layout.shift_low_idx, _fmt(layout.shift_low_idx))
    if layout.span_idx is not None:
        logger.debug("  span           [%s] = %s  (expected 0.0)", layout.span_idx, _fmt(layout.span_idx))
    if layout.skew_idx is not None:
        logger.debug("  skew           [%s] = %s  (expected 0.0)", layout.skew_idx, _fmt(layout.skew_idx))
    if layout.asymmetry_idx is not None:
        logger.debug("  asymmetry      [%s] = %s  (expected 0.0)", layout.asymmetry_idx, _fmt(layout.asymmetry_idx))
    if layout.anisotropy_idx is not None:
        logger.debug("  anisotropy     [%s] = %s  (expected 0.0)", layout.anisotropy_idx, _fmt(layout.anisotropy_idx))

    # --- Assertions (soft) ---
    if elem_distribution is not None:
        expected = elem_distribution.to(baseline.device, dtype=baseline.dtype)
        # Accept either the probabilistic dist OR the original onehot (current policy keeps onehot fixed).
        onehot_match = (
            torch.allclose(baseline[layout.onehot_slice].float(), expected) or
            torch.allclose(baseline[layout.onehot_slice].float(), target_features[layout.onehot_slice].float())
        )
        if not onehot_match:
            logger.warning(
                "Scientific baseline onehot mismatch for %s[%s].\n  baseline=%s\n  expected(dist)=%s\n  input_onehot=%s",
                node_type,
                element,
                baseline[layout.onehot_slice].tolist(),
                expected.tolist(),
                target_features[layout.onehot_slice].tolist(),
            )
    else:
        onehot_match = torch.allclose(
            baseline[layout.onehot_slice].float(),
            target_features[layout.onehot_slice].float(),
        )
        if not onehot_match:
            logger.warning(
                "Scientific baseline onehot mismatch for %s[%s].\n  baseline=%s\n  input=%s",
                node_type,
                element,
                baseline[layout.onehot_slice].tolist(),
                target_features[layout.onehot_slice].tolist(),
            )

    mass_match = abs(
        baseline[layout.mass_idx].item() - target_features[layout.mass_idx].item()
    ) < 1e-5
    assert mass_match, (
        f"Scientific baseline assertion failed: mass mismatch for {node_type}[{element}].\n"
        f"  baseline mass = {baseline[layout.mass_idx].item()}\n"
        f"  input    mass = {target_features[layout.mass_idx].item()}"
    )

    if layout.shift_low_idx is not None:
        sl_val = baseline[layout.shift_low_idx].item()
        assert abs(sl_val) < 1e-6, (
            f"Scientific baseline assertion failed: shift_low is {sl_val} (expected 0.0) "
            f"for {node_type}[{element}]."
        )

    if layout.formal_charge_idx is not None:
        fc_val = baseline[layout.formal_charge_idx].item()
        assert abs(fc_val) < 1e-6, (
            f"Scientific baseline assertion failed: formal_charge is {fc_val} (expected 0.0) "
            f"for {node_type}[{element}]."
        )
    # retain hard checks only for mass/charge/shift_low to catch corruption
    mass_match = abs(
        baseline[layout.mass_idx].item() - target_features[layout.mass_idx].item()
    ) < 1e-5
    assert mass_match, (
        f"Scientific baseline assertion failed: mass mismatch for {node_type}[{element}].\n"
        f"  baseline mass = {baseline[layout.mass_idx].item()}\n"
        f"  input    mass = {target_features[layout.mass_idx].item()}"
    )

    if layout.shift_low_idx is not None:
        sl_val = baseline[layout.shift_low_idx].item()
        assert abs(sl_val) < 1e-6, (
            f"Scientific baseline assertion failed: shift_low is {sl_val} (expected 0.0) "
            f"for {node_type}[{element}]."
        )

    if layout.formal_charge_idx is not None:
        fc_val = baseline[layout.formal_charge_idx].item()
        assert abs(fc_val) < 1e-6, (
            f"Scientific baseline assertion failed: formal_charge is {fc_val} (expected 0.0) "
            f"for {node_type}[{element}]."
        )
