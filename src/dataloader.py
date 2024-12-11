import pickle
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import networkx as nx

def nx_to_pyg_data(nx_graph):
    node_features = []
    edge_index = []
    edge_features = []
    target = []

    # extrahiere Knoten-Features und Zielwerte
    for node, data in nx_graph.nodes(data=True):
        feature = [
            data.get('atomic_num', 0), data.get('degree', 0), data.get('shielding_dia', 0), data.get('shielding_para', 0),
            data.get('anisotropy', 0), data.get('at_charge_mull', 0), data.get('at_charge_loew', 0),
            data.get('orb_charge_mull_s', 0), data.get('orb_charge_mull_p', 0), data.get('orb_charge_loew_s', 0),
            data.get('orb_charge_loew_p', 0), data.get('orb_charge_mull_d', 0),
            data.get('orb_charge_loew_d', 0), data.get('BO_loew_av', 0),
            data.get('BO_mayer_av', 0)
        ]
        node_features.append(feature)

        # Zielwerte (nur Kohlenstoffatome relevant)
        target.append(data.get('shift_high-low', 0.0) if data.get('atomic_num') == 6 else float('nan'))

    # extrahiere Kanten
    for source, target, data in nx_graph.edges(data=True):
        edge_index.append([source, target])
        edge_feature = [
            data.get('bond_type', 0), data.get('is_conjugated', 0), data.get('is_aromatic', 0), data.get('bond_order', 0)
        ]
        edge_features.append(edge_feature)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float).view(-1, 1)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def get_data_loaders(data_path, batch_size=32):
    with open(data_path, 'rb') as f:
        nx_graphs = pickle.load(f)

    graph_data_list = [nx_to_pyg_data(g) for g in nx_graphs]
    data_list_with_targets = [data for data in graph_data_list if data.x.size(0) > 0]

    train_data = data_list_with_targets[:int(0.8 * len(data_list_with_targets))]
    val_data = data_list_with_targets[int(0.8 * len(data_list_with_targets)):]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader
