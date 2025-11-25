import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.explain import Explainer, HeteroExplanation
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

# Allow running the script directly via `python src/explainer/gnnexplainer.py`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from explainer.explainer_utils import (  # noqa: E402
    NodeTypeRegressionWrapper,
    build_dataset,
    ensure_dir,
    get_device,
    heterodata_to_dicts,
    load_config,
    load_stats,
    load_trained_model,
    summarize_incident_edges,
    summarize_node_mask,
    validate_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heterograph node-level explanations via GNNExplainer"
    )
    parser.add_argument("--data", required=True, help="Path to the graph pickle file.")
    parser.add_argument(
        "--graph-idx",
        type=int,
        default=0,
        help="Index of the graph inside the dataset to explain.",
    )
    parser.add_argument(
        "--node-type",
        type=str,
        default="H",
        choices=["H", "C", "Others"],
        help="Node type for which explanations should be generated.",
    )
    parser.add_argument(
        "--node-idx",
        type=int,
        required=True,
        help="Index within the selected node type that should be explained.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained model weights. "
        "Defaults to {operator_type}_best_model.pt if omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="explain_this/config.pkl",
        help="Path to the pickled training config.",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="explain_this/norm_stats.pkl",
        help="Path to the pickled normalization statistics.",
    )
    parser.add_argument(
        "--edge-stats",
        type=str,
        default="explain_this/edge_stats.pkl",
        help="Path to the pickled edge normalization stats.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of optimization epochs for GNNExplainer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate used by GNNExplainer.",
    )
    parser.add_argument(
        "--explanation-type",
        type=str,
        choices=["phenomenon", "model"],
        default="phenomenon",
        help="Decide whether to explain the phenomenon (targets) or the model output.",
    )
    parser.add_argument(
        "--topk-features",
        type=int,
        default=5,
        help="Number of node features to summarize for the explained node.",
    )
    parser.add_argument(
        "--topk-edges",
        type=int,
        default=5,
        help="Number of incident edges to summarize for the explained node.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/explanations",
        help="Directory in which explanation artifacts are stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier (e.g. cuda, cuda:0, cpu). Detected automatically if unset.",
    )
    return parser.parse_args()


def _default_model_path(config: Any) -> str:
    operator_type = getattr(config, "operator_type", None)
    if operator_type is None and isinstance(config, dict):
        operator_type = config.get("operator_type")
    operator_type = operator_type or "SAGEConv"
    return f"explain_this/{operator_type}_best_model.pt"


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    config = load_config(args.config)
    model_path = args.model or _default_model_path(config)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model checkpoint at '{model_path}'. Use --model to supply a path."
        )

    norm_stats, edge_stats = load_stats(args.norm_stats, args.edge_stats)
    dataset = build_dataset(args.data, config, norm_stats=norm_stats, edge_stats=edge_stats)
    validate_indices(len(dataset), args.graph_idx, "graph_idx")

    data = dataset[args.graph_idx].to(device)
    x_dict, edge_index_dict, edge_attr_dict, y_dict = heterodata_to_dicts(data)

    base_model = load_trained_model(model_path, config, device)
    wrapped_model = NodeTypeRegressionWrapper(base_model, args.node_type)

    target = None
    target_tensor = y_dict.get(args.node_type)
    if target_tensor is not None:
        target_tensor = target_tensor.reshape(target_tensor.size(0), -1).squeeze(-1)
        validate_indices(target_tensor.size(0), args.node_idx, "node_idx")
        if args.explanation_type == "phenomenon":
            if torch.isnan(target_tensor[args.node_idx]):
                raise ValueError(
                    f"Target value for node {args.node_idx} of type "
                    f"{args.node_type} is NaN. Choose a different node or "
                    "switch to --explanation-type model."
                )
            target = target_tensor
    else:
        validate_indices(x_dict[args.node_type].size(0), args.node_idx, "node_idx")
        if args.explanation_type == "phenomenon":
            raise ValueError(
                f"No targets available for node type {args.node_type}; cannot explain phenomenon."
            )

    model_config = ModelConfig(
        mode=ModelMode.regression,
        task_level=ModelTaskLevel.node,
        return_type=ModelReturnType.raw,
    )

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=args.epochs, lr=args.lr),
        explanation_type=args.explanation_type,
        model_config=model_config,
        node_mask_type="attributes",
        edge_mask_type="object",
    )

    # --- WICHTIG: Hetero-Eingaben -> HeteroExplanation (entsprechend PyG GNNExplainer) ---
    explanation = explainer(
        x_dict,
        edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        target=target,
        index=args.node_idx,
    )

    # Laufzeit-Check + Typ-Hinweis: wir erwarten eine HeteroExplanation
    if not isinstance(explanation, HeteroExplanation):
        raise TypeError(
            f"Expected a HeteroExplanation, but got {type(explanation)}. "
            "Please ensure you are using a recent torch_geometric version where "
            "GNNExplainer supports heterogeneous graphs."
        )

    with torch.no_grad():
        predictions = base_model(x_dict, edge_index_dict, edge_attr_dict)
        node_prediction = float(predictions[args.node_type][args.node_idx].item())

    # HeteroExplanation stellt node_mask_dict und edge_mask_dict bereit
    feature_summary = summarize_node_mask(
        explanation.node_mask_dict,
        args.node_type,
        args.node_idx,
        top_k=max(args.topk_features, 0),
    )
    edge_summary = summarize_incident_edges(
        explanation.edge_mask_dict,
        edge_index_dict,
        args.node_type,
        args.node_idx,
        top_k=max(args.topk_edges, 0),
    )

    target_value = (
        float(target[args.node_idx].item()) if target is not None else None
    )
    summary = {
        "graph_idx": args.graph_idx,
        "node_type": args.node_type,
        "node_idx": args.node_idx,
        "prediction": node_prediction,
        "target": target_value,
        "top_features": feature_summary,
        "important_edges": edge_summary,
    }

    ensure_dir(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"gnnexplainer_{args.node_type}_n{args.node_idx}_g{args.graph_idx}_{timestamp}"
    torch.save(explanation, Path(args.output_dir) / f"{base_name}.pt")
    with open(Path(args.output_dir) / f"{base_name}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
