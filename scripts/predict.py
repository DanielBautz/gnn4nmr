import torch
import os
import pickle
import pandas as pd
from model import HeteroGNNModel
from dataloader import ShiftDataset
from operators import get_conv_operator

DEFAULT_IN_DIM_DICT = {
    "H": 33,
    "C": 39,
    "Others": 16,
}

def normalize_loaded_config(config):
    if isinstance(config, dict):
        return config
    if hasattr(config, 'as_dict') and callable(config.as_dict):
        as_dict = config.as_dict()
        if isinstance(as_dict, dict):
            return as_dict
    if hasattr(config, '__dict__'):
        return dict(vars(config))
    try:
        return dict(config)
    except Exception as exc:
        raise TypeError(f"Could not normalize config object of type {type(config)}") from exc


def predict(model_path, data_path, norm_stats_file='norm_stats.pkl', edge_stats_file='edge_stats.pkl', output_file='predictions.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open('config.pkl', 'rb') as f:
        config = normalize_loaded_config(pickle.load(f))

    # Load stats
    with open(norm_stats_file, 'rb') as f:
        norm_stats = pickle.load(f)
    with open(edge_stats_file, 'rb') as f:
        edge_stats = pickle.load(f)

    # Determine root_dir and file_name from data_path
    root_dir, file_name = os.path.split(data_path)

    # Create dataset with loaded stats
    dataset = ShiftDataset(
        root_dir=root_dir or 'data',
        file_name=file_name,
        normalize_node_features=config['normalize_node_features'],
        normalize_edge_features=config['normalize_edge_features'],
        norm_stats=norm_stats,
        **edge_stats
    )

    # Create model
    in_dim_dict = config.get('in_dim_dict', DEFAULT_IN_DIM_DICT)

    operator_kwargs = {}
    if config['operator_type'] == "GATConv" or config['operator_type'] == "GATv2Conv":
        operator_kwargs['add_self_loops'] = False

    model = HeteroGNNModel(
        in_dim_dict,
        hidden_dim=config['hidden_dim'],
        out_dim=config['out_dim'],
        encoder_dropout=config['encoder_dropout'],
        gnnlayer_dropout=config['gnnlayer_dropout'],
        num_gnn_layers=config['num_gnn_layers'],
        operator_type=config['operator_type'],
        operator_kwargs=operator_kwargs,
        edge_in_dim=10
    )

    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            nx_g = dataset.nx_graphs[idx]
            compound = nx_g.graph.get("compound", f"unknown_{idx}")
            structure = nx_g.graph.get("structure", "unknown")

            data = dataset[idx].to(device)

            x_dict = {}
            for ntype in data.node_types:
                x_dict[ntype] = data[ntype].x

            edge_index_dict = {}
            edge_attr_dict = {}
            for store in data.edge_stores:
                src, rel, dst = store._key
                edge_index_dict[(src, rel, dst)] = store.edge_index
                edge_attr_dict[(src, rel, dst)] = store.edge_attr

            out_dict = model(x_dict, edge_index_dict, edge_attr_dict)

            # Collect nodes
            h_nodes, c_nodes, o_nodes = [], [], []
            for node in nx_g.nodes():
                attrs = nx_g.nodes[node]
                element = attrs["element"]
                if element == "H":
                    h_nodes.append(node)
                elif element == "C":
                    c_nodes.append(node)
                else:
                    o_nodes.append(node)

            # H
            if 'H' in out_dict and out_dict['H'] is not None:
                for i, node in enumerate(h_nodes):
                    attrs = nx_g.nodes[node]
                    shift_low = attrs.get("shift_low", 0.0)
                    prediction = out_dict['H'][i].item() if i < out_dict['H'].shape[0] else float('nan')
                    results.append({
                        'compound': compound,
                        'structure': structure,
                        'atom_type': 'H',
                        'atom_idx': node,
                        'shift_low': shift_low,
                        'prediction': prediction
                    })

            # C
            if 'C' in out_dict and out_dict['C'] is not None:
                for i, node in enumerate(c_nodes):
                    attrs = nx_g.nodes[node]
                    shift_low = attrs.get("shift_low", 0.0)
                    prediction = out_dict['C'][i].item() if i < out_dict['C'].shape[0] else float('nan')
                    results.append({
                        'compound': compound,
                        'structure': structure,
                        'atom_type': 'C',
                        'atom_idx': node,
                        'shift_low': shift_low,
                        'prediction': prediction
                    })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions with trained GNN model")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model state dict (default: {operator_type}_best_model.pt)')
    parser.add_argument('--data', required=True, help='Path to data pickle file')
    parser.add_argument('--norm-stats', type=str, default='norm_stats.pkl',
                        help='Path to normalization stats pickle file')
    parser.add_argument('--edge-stats', type=str, default='edge_stats.pkl',
                        help='Path to edge stats pickle file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    # Load config to get operator_type if model not specified
    with open('config.pkl', 'rb') as f:
        config = normalize_loaded_config(pickle.load(f))

    if args.model is None:
        args.model = f"{config['operator_type']}_best_model.pt"

    predict(args.model, args.data, args.norm_stats, args.edge_stats, args.output)
