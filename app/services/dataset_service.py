"""Dataset loading, graph metadata, node-index maps and SDF lookup.

A "dataset entry" in the registry looks like:
    {
        "file_name": str,
        "dataset": ShiftDataset,        # normalized with dataset-wide stats
        "nx_graphs": list[nx.Graph],
        "summaries": list[dict],        # one per graph, for the molecule list
        "node_maps": dict,              # graph_idx -> node map (lazy)
        "num_compounds": int,
    }

Node map per graph (single source of truth for index translation):
    {
        "by_type": {"H": [atom_index, ...], "C": [...], "Others": [...]},
        "by_atom": {atom_index: (node_type, type_local_idx)},
    }
The dataloader iterates nx_g.nodes() in order and buckets nodes per element
class, so position i in by_type[nt] corresponds to row i of the HeteroData
tensors for that type. The nx node IDs themselves are the SDF atom indices.
"""
import math
import os
from collections import Counter

from . import pathsetup  # noqa: F401
from .. import config
from . import features

from dataloader import ShiftDataset  # noqa: E402

TARGET_ATTR = "shift_high-low"


def list_dataset_files():
    files = []
    for name in sorted(os.listdir(config.DATA_DIR)):
        if name.endswith(".pkl"):
            files.append(name)
    return files


def load_dataset(state, file_name, make_active=True):
    """Load a pkl into the registry (idempotent); optionally make it active.

    make_active=False is used by training so a model can be trained on any
    dataset without switching what the Explore view shows.
    """
    path = os.path.join(config.DATA_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with state.dataset_lock:
        if file_name in state.datasets:
            if make_active:
                state.active_dataset_file = file_name
            return state.datasets[file_name]

    # Heavy work outside the lock: ShiftDataset scans all graphs for stats.
    dataset = ShiftDataset(
        root_dir=config.DATA_DIR,
        file_name=file_name,
        normalize_node_features=True,
        normalize_edge_features=True,
    )
    summaries = [_graph_summary(idx, g) for idx, g in enumerate(dataset.nx_graphs)]
    compounds = {g.graph.get("compound", "unknown") for g in dataset.nx_graphs}
    entry = {
        "file_name": file_name,
        "dataset": dataset,
        "nx_graphs": dataset.nx_graphs,
        "summaries": summaries,
        "node_maps": {},
        "num_compounds": len(compounds),
    }

    with state.dataset_lock:
        state.datasets[file_name] = entry
        if make_active:
            state.active_dataset_file = file_name
    return entry


def get_node_map(entry, graph_idx):
    node_map = entry["node_maps"].get(graph_idx)
    if node_map is None:
        node_map = _build_node_map(entry["nx_graphs"][graph_idx])
        entry["node_maps"][graph_idx] = node_map
    return node_map


def _build_node_map(nx_g):
    by_type = {"H": [], "C": [], "Others": []}
    by_atom = {}
    for node in nx_g.nodes():
        ntype = features.node_type_for_element(nx_g.nodes[node]["element"])
        by_atom[node] = (ntype, len(by_type[ntype]))
        by_type[ntype].append(node)
    return {"by_type": by_type, "by_atom": by_atom}


def compound_str(compound):
    """Compounds are stored as ints (1) but SDF dirs/files use '001'."""
    try:
        return f"{int(str(compound)):03d}"
    except (TypeError, ValueError):
        return str(compound)


def _graph_summary(graph_idx, nx_g):
    compound = compound_str(nx_g.graph.get("compound", "unknown"))
    structure = nx_g.graph.get("structure", 0)
    counts = Counter(attrs["element"] for _, attrs in nx_g.nodes(data=True))
    return {
        "graph_idx": graph_idx,
        "compound": compound,
        "structure": structure,
        "label": _graph_label(compound, structure),
        "formula": _formula(counts),
        "num_atoms": nx_g.number_of_nodes(),
        "n_H": counts.get("H", 0),
        "n_C": counts.get("C", 0),
        "has_sdf": find_sdf_path(compound, structure) is not None,
    }


def _graph_label(compound, structure):
    try:
        return f"{compound}_{int(str(structure)):02d}"
    except (TypeError, ValueError):
        return f"{compound}_{structure}"


def _formula(counts):
    """Hill notation: C, H, then the rest alphabetically."""
    parts = []
    for el in ["C", "H"] + sorted(k for k in counts if k not in ("C", "H")):
        n = counts.get(el, 0)
        if n:
            parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def find_sdf_path(compound, structure):
    comp = compound_str(compound)
    try:
        sdf_name = f"{comp}_{int(str(structure)):02d}.sdf"
    except (TypeError, ValueError):
        sdf_name = f"{comp}_{structure}.sdf"
    candidates = [
        os.path.join(config.SDF_PRIMARY_DIR, comp, sdf_name),
        os.path.join(config.SDF_FALLBACK_DIR, sdf_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def molecule_detail(entry, graph_idx):
    nx_g = entry["nx_graphs"][graph_idx]
    node_map = get_node_map(entry, graph_idx)
    summary = entry["summaries"][graph_idx]

    sdf_text = None
    sdf_path = find_sdf_path(summary["compound"], summary["structure"])
    if sdf_path:
        with open(sdf_path, encoding="utf-8", errors="replace") as handle:
            sdf_text = handle.read()

    atoms = []
    for node in nx_g.nodes():
        attrs = nx_g.nodes[node]
        ntype, local_idx = node_map["by_atom"][node]
        gt = attrs.get(TARGET_ATTR, float("nan"))
        try:
            gt = float(gt)
        except (TypeError, ValueError):
            gt = float("nan")
        atoms.append(
            {
                "atom_index": int(node),
                "element": attrs["element"],
                "node_type": ntype,
                "type_local_idx": local_idx,
                "ground_truth": gt if math.isfinite(gt) else None,
                "feature_values": features.raw_values(attrs, ntype),
            }
        )

    bonds = []
    for u, v, battrs in nx_g.edges(data=True):
        bonds.append(
            {
                "a1": int(u),
                "a2": int(v),
                "bond_type": str(battrs.get("bond_type", "SINGLE")),
                "bond_order": battrs.get("bond_order"),
                "length": battrs.get("length"),
                "is_aromatic": bool(battrs.get("is_aromatic", False)),
            }
        )

    return {
        **summary,
        "sdf": sdf_text,
        "feature_names": features.FEATURE_NAMES,
        "atoms": atoms,
        "bonds": bonds,
    }


def make_normalized_view(nx_graphs, norm_stats, edge_stats):
    """A ShiftDataset over already-loaded graphs, normalized with *given* stats.

    Built via __new__ to skip the constructor's full-dataset stat scan and
    pickle re-read; sets exactly the attributes __getitem__/__len__ read
    (nx_graphs, normalize flags, norm_stats, edge mean/std scalars).
    """
    view = ShiftDataset.__new__(ShiftDataset)
    view.file_path = "<in-memory view>"
    view.nx_graphs = nx_graphs
    view.normalize_node_features = norm_stats is not None
    view.norm_stats = norm_stats
    view.normalize_edge_features = edge_stats is not None
    view.edge_length_mean = edge_stats["edge_length_mean"] if edge_stats else 0.0
    view.edge_length_std = edge_stats["edge_length_std"] if edge_stats else 1.0
    view.edge_order_mean = edge_stats["edge_order_mean"] if edge_stats else 0.0
    view.edge_order_std = edge_stats["edge_order_std"] if edge_stats else 1.0
    return view


def infer_in_dim_dict(dataset, max_probe=50):
    """Infer feature widths per node type from real samples (main.py's
    hardcoded H:34 is wrong; the dataloader produces 33)."""
    in_dims = {}
    for idx in range(min(len(dataset), max_probe)):
        data = dataset[idx]
        for ntype in data.node_types:
            if ntype not in in_dims:
                in_dims[ntype] = int(data[ntype].x.shape[1])
        if len(in_dims) == 3:
            break
    for ntype, dim in features.CANONICAL_IN_DIMS.items():
        in_dims.setdefault(ntype, dim)
    return in_dims
