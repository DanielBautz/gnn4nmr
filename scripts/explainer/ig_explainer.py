from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
import numpy as np
from captum.attr import IntegratedGradients

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.explainer.explainer_utils import NodeTypeRegressionWrapper, heterodata_to_dicts


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
):
    wrapped_model = NodeTypeRegressionWrapper(base_model, node_type)
    wrapped_model.to(device)
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

            out = wrapped_model(temp_x_dict, edge_index_dict, edge_attr_dict)  # [N]
            # Vorhersage für Zielknoten holen (Skalar)
            pred = out[node_idx]
            # WICHTIG: als [1, 1] speichern, NICHT [1]
            outputs.append(pred.view(1, 1))   # <--- HIER ÄNDERUNG

        # WICHTIG: Rückgabe-Shape [B, 1], NICHT [B]
        return torch.cat(outputs, dim=0)      # Shape: [batch, 1]

    target_features = x_dict[node_type][node_idx].clone().detach().to(device)

    # Baseline basierend auf Typ berechnen
    if baseline_type == "zero":
        baseline = torch.zeros_like(target_features)
    elif baseline_type == "mean":
        # Mittelwert über alle Knoten desselben Typs
        all_features = x_dict[node_type]  # [N, F]
        baseline = torch.mean(all_features, dim=0)  # [F]
    elif baseline_type == "random":
        # Zufällige Werte basierend auf der Verteilung der Features
        all_features = x_dict[node_type]  # [N, F]
        baseline = torch.mean(all_features, dim=0) + torch.std(all_features, dim=0) * torch.randn_like(target_features)
    elif baseline_type == "min":
        # Minimum über alle Knoten desselben Typs
        all_features = x_dict[node_type]  # [N, F]
        baseline = torch.min(all_features, dim=0)[0]  # [F]
    elif baseline_type == "max":
        # Maximum über alle Knoten desselben Typs
        all_features = x_dict[node_type]  # [N, F]
        baseline = torch.max(all_features, dim=0)[0]  # [F]
    else:
        # Default: zero baseline
        baseline = torch.zeros_like(target_features)

    # --- Gradient-Check: hängt der Output wirklich von `inputs` ab? ---
    inp = target_features.unsqueeze(0).clone().detach().requires_grad_(True)  # [1, F]
    out = forward_func(inp)  # [1, 1] (bei dir)
    grad = torch.autograd.grad(
        outputs=out.sum(),
        inputs=inp,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]

    if grad is None:
        raise RuntimeError(
            "Gradient-Check fehlgeschlagen: grad is None. "
            "forward_func erzeugt keinen Gradientenpfad von inputs -> output."
        )

    grad_norm = grad.abs().sum().item()
    if grad_norm == 0.0:
        print(
            "WARNUNG: Gradient-Check: Sum(|grad|) == 0. "
            "Entweder ist das Modell lokal konstant oder der Pfad ist effektiv gekappt."
        )
    else:
        print(f"Gradient-Check OK: Sum(|grad|) = {grad_norm:.6e}")
    # --- Ende Gradient-Check ---

    ig = IntegratedGradients(forward_func)

    attributions, delta = ig.attribute(
        inputs=target_features.unsqueeze(0),      # [1, F]
        baselines=baseline.unsqueeze(0),          # [1, F]
        n_steps=n_steps,
        target=0,                                 # passt, weil forward_func -> [B, 1]
        return_convergence_delta=True,
    )

    # Completeness: sum(IG) ~ f(x) - f(baseline)
    sum_ig = attributions.sum(dim=1)              # [1]

    with torch.no_grad():
        fx = forward_func(target_features.unsqueeze(0))  # [1, 1]
        f0 = forward_func(baseline.unsqueeze(0))         # [1, 1]
        fx_minus_f0 = (fx - f0).squeeze(-1)              # [1]

    print(f"f(x)             = {fx.item(): .6f}")
    print(f"f(baseline)      = {f0.item(): .6f}")
    print(f"f(x)-f(baseline) = {fx_minus_f0.item(): .6f}")
    print(f"sum(IG)          = {sum_ig.item(): .6f}")
    print(f"delta            = {delta.item(): .6f}  (sollte nahe 0 sein)")

    node_attr = attributions.squeeze(0)  # [F]
    node_mask_dict = {node_type: node_attr.unsqueeze(0)}  # [1, F]
    edge_mask_dict = {}

    return {
        "node_mask_dict": node_mask_dict,
        "edge_mask_dict": edge_mask_dict,
    }


def batch_ig_analysis(
    base_model: Any,
    dataset: Any,
    node_type: str,
    graph_indices: List[int],
    device: torch.device,
    n_steps: int = 50,
    baseline_type: str = "zero",
    progress_callback: Optional[Any] = None,
):
    """
    Compute IG explanations for all nodes of a given type across multiple graphs and aggregate results.

    Args:
        base_model: The trained model
        dataset: The dataset containing graphs
        node_type: Node type to analyze ('H', 'C', 'Others')
        graph_indices: List of graph indices to analyze
        device: Torch device
        n_steps: Number of IG integration steps
        baseline_type: Type of baseline for IG
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with aggregated results per node type: {'avg_importance': [...], 'std_importance': [...], 'node_count': N}
    """
    wrapped_model = NodeTypeRegressionWrapper(base_model, node_type)
    wrapped_model.to(device)
    wrapped_model.eval()

    all_attributions = []
    total_nodes = 0

    for graph_idx in graph_indices:
        if progress_callback:
            progress_callback(f"Processing graph {graph_idx}...")

        data = dataset[graph_idx].to(device)
        x_dict, edge_index_dict, edge_attr_dict, y_dict, atom_index_dict = heterodata_to_dicts(data)

        # Get all nodes of the specified type
        if node_type not in x_dict:
            continue

        node_features = x_dict[node_type]  # [N, F]
        num_nodes = node_features.size(0)
        total_nodes += num_nodes

        # Compute IG for each node of this type in the graph
        for node_idx in range(num_nodes):
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

                    out = wrapped_model(temp_x_dict, edge_index_dict, edge_attr_dict)
                    pred = out[node_idx]
                    outputs.append(pred.view(1, 1))

                return torch.cat(outputs, dim=0)

            target_features = node_features[node_idx].clone().detach().to(device)

            # Compute baseline
            if baseline_type == "zero":
                baseline = torch.zeros_like(target_features)
            elif baseline_type == "mean":
                all_features = x_dict[node_type]
                baseline = torch.mean(all_features, dim=0)
            elif baseline_type == "random":
                all_features = x_dict[node_type]
                baseline = torch.mean(all_features, dim=0) + torch.std(all_features, dim=0) * torch.randn_like(target_features)
            elif baseline_type == "min":
                all_features = x_dict[node_type]
                baseline = torch.min(all_features, dim=0)[0]
            elif baseline_type == "max":
                all_features = x_dict[node_type]
                baseline = torch.max(all_features, dim=0)[0]
            else:
                baseline = torch.zeros_like(target_features)

            ig = IntegratedGradients(forward_func)
            attributions = ig.attribute(
                inputs=target_features.unsqueeze(0),
                baselines=baseline.unsqueeze(0),
                n_steps=n_steps,
                target=0,
            )

            node_attr = attributions.squeeze(0).detach().cpu().numpy()
            all_attributions.append(node_attr)

    if not all_attributions:
        return {node_type: {'avg_importance': [], 'std_importance': [], 'node_count': 0}}

    # Aggregate attributions
    attributions_array = np.array(all_attributions)  # [num_nodes, num_features]
    avg_importance = np.mean(attributions_array, axis=0)
    avg_abs_importance = np.mean(np.abs(attributions_array), axis=0)
    std_importance = np.std(attributions_array, axis=0)

    return {
        node_type: {
            'avg_importance': avg_importance.tolist(),
            'avg_abs_importance': avg_abs_importance.tolist(),
            'std_importance': std_importance.tolist(),
            'node_count': total_nodes,
            'explainer_type': 'ig'
        }
    }



# Für Kompatibilität mit explain_this callable / CLI
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
):
    """
    Standalone-IG-Erklärungsfunktion, analog zum gnnexplainer-Skript.

    - Lädt Model + Config + Stats
    - Baut Dataset
    - Berechnet IG-Erklärung für einen Knoten
    - Speichert Ergebnis als .pt-Datei im GNNExplainer-ähnlichen Format
    """
    print(f"Computing IG explanation for {node_type}[{node_idx}] in graph {graph_idx}...")

    from scripts.explainer.explainer_utils import (
        load_config,
        load_stats,
        load_trained_model,
        build_dataset,
        heterodata_to_dicts,
        validate_indices,
        ensure_dir,
        get_device,
    )

    device = get_device(None)

    # Config laden:
    # - wenn dict: direkt verwenden
    # - wenn Pfad: load_config(Pfad)
    # - wenn None: config.pkl aus dem Modellpfad ableiten
    if config:
        if isinstance(config, dict):
            config_obj = config
        else:
            config_obj = load_config(config)
    else:
        # Annahme: Modell heißt SAGEConv_best_model.pt und config.pkl liegt daneben
        config_path = model_path.replace("SAGEConv_best_model.pt", "config.pkl")
        config_obj = load_config(config_path)

    # Modell laden
    model = load_trained_model(model_path, config_obj, device)

    # Stats laden (Pfad oder dict mit key 'path' zulassen, wie im Original)
    norm_stats_path = norm_stats["path"] if isinstance(norm_stats, dict) else norm_stats
    edge_stats_path = edge_stats["path"] if isinstance(edge_stats, dict) else edge_stats
    norm_stats_obj, edge_stats_obj = load_stats(norm_stats_path, edge_stats_path)

    # Dataset bauen und Graph auswählen
    dataset = build_dataset(
        data_path,
        config_obj,
        norm_stats=norm_stats_obj,
        edge_stats=edge_stats_obj,
    )

    validate_indices(len(dataset), graph_idx, "graph_idx")
    data = dataset[graph_idx].to(device)

    # HeteroData in dicts umwandeln
    x_dict, edge_index_dict, edge_attr_dict, y_dict = heterodata_to_dicts(data)

    # Optional: node_idx gegen y_dict validieren, wenn y vorhanden
    if y_dict.get(node_type) is not None:
        target_y = y_dict[node_type]
        validate_indices(target_y.size(0), node_idx, "node_idx")

    # IG-Erklärung berechnen
    explanation_result = compute_ig_explanation(
        model,
        data,
        node_type,
        node_idx,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        device,
        target=None,  # IG nutzt hier nur die Modellvorhersage, nicht die Ground-Truth
        baseline_type="zero",  # Default baseline
    )

    # Ergebnis speichern 
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
):
    """
    Batch IG analysis for all nodes of a given type across multiple graphs.

    - Loads model + config + stats
    - Builds dataset
    - Computes IG explanations for all nodes of the specified type
    - Aggregates results (mean/std importance per feature)
    - Saves aggregated results as .pt file
    """
    print(f"Computing batch IG analysis for {node_type} nodes across graphs {graph_indices}...")

    from scripts.explainer.explainer_utils import (
        load_config,
        load_stats,
        load_trained_model,
        build_dataset,
        get_device,
        ensure_dir,
    )

    device = get_device(None)

    # Load config
    if config:
        if isinstance(config, dict):
            config_obj = config
        else:
            config_obj = load_config(config)
    else:
        config_path = model_path.replace("SAGEConv_best_model.pt", "config.pkl")
        config_obj = load_config(config_path)

    # Load model
    model = load_trained_model(model_path, config_obj, device)

    # Load stats
    norm_stats_path = norm_stats["path"] if isinstance(norm_stats, dict) else norm_stats
    edge_stats_path = edge_stats["path"] if isinstance(edge_stats, dict) else edge_stats
    norm_stats_obj, edge_stats_obj = load_stats(norm_stats_path, edge_stats_path)

    # Build dataset
    dataset = build_dataset(
        data_path,
        config_obj,
        norm_stats=norm_stats_obj,
        edge_stats=edge_stats_obj,
    )

    # Validate graph indices
    for graph_idx in graph_indices:
        if graph_idx < 0 or graph_idx >= len(dataset):
            raise IndexError(f"Graph index {graph_idx} out of bounds for dataset of length {len(dataset)}.")

    # Perform batch analysis
    def progress_callback(msg):
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
    )

    # Save results
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create shorter filename to avoid path length issues
    if len(graph_indices) <= 5:
        graphs_str = "_".join(map(str, graph_indices))
    else:
        graphs_str = f"{graph_indices[0]}-{graph_indices[-1]}"

    filename = f"batch_ig_{node_type}_g{graphs_str}_{timestamp}.pt"
    save_path = Path(output_dir) / filename
    torch.save(batch_results, save_path)

    print(f"Saved batch IG analysis to {save_path}")
    print(f"Analyzed {batch_results[node_type]['node_count']} {node_type} nodes across {len(graph_indices)} graphs")
    return batch_results


if __name__ == "__main__":
    # Command line usage
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
                        choices=["zero", "mean", "random", "min", "max"],
                        help="Baseline-Typ für IG (default: zero)")

    # Single node mode (default)
    parser.add_argument("--graph-idx", type=int, default=0,
                        help="Graph-Index für Single-Node-Analyse (default: 0)")
    parser.add_argument("--node-idx", type=int, default=None,
                        help="Knoten-Index für Single-Node-Analyse")

    # Batch mode
    parser.add_argument("--batch-mode", action="store_true",
                        help="Batch-Modus aktivieren: Analysiert alle Knoten eines Typs über mehrere Graphen")
    parser.add_argument("--graph-indices", type=int, nargs="+", default=None,
                        help="Liste von Graph-Indizes für Batch-Analyse (z.B. --graph-indices 0 1 2)")

    args = parser.parse_args()

    if args.batch_mode:
        # Batch mode
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
        )
    else:
        # Single node mode (default)
        if args.node_idx is None:
            parser.error("Single-Node-Modus erfordert --node-idx")

        explain_node_with_ig(
            model_path=args.model,
            data_path=args.data,
            config=args.config,
            norm_stats=args.norm_stats,
            edge_stats=args.edge_stats,
            graph_idx=args.graph_idx,
            node_type=args.node_type,
            node_idx=args.node_idx,
            output_dir=args.output_dir,
        )
