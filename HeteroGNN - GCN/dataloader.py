import os
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

# Liste aller möglichen Elemente, die wir als One-Hot differenzieren möchten
ALL_ELEMENTS = [
    "H", "C", "Li", "B", "N", "O", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl"
]

def get_element_onehot(elem: str):
    one_hot = [0.0] * len(ALL_ELEMENTS)
    if elem in ALL_ELEMENTS:
        idx = ALL_ELEMENTS.index(elem)
        one_hot[idx] = 1.0
    return one_hot

def get_h_features(attrs):
    feats = []
    elem = attrs.get('element', 'H')
    feats.extend(get_element_onehot(elem))
    feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    feats.append(attrs.get('shift_low', 0.0))
    feats.append(attrs.get('CN(X)', 0.0))
    feats.append(attrs.get('no_HCH', 0.0))
    feats.append(attrs.get('no_HYH', 0.0))
    feats.append(attrs.get('no_HYC', 0.0))
    feats.append(attrs.get('no_HYN', 0.0))
    feats.append(attrs.get('no_HYO', 0.0))
    feats.append(attrs.get('dist_HC', 0.0))
    feats.append(attrs.get('shift_low_neighbor_C', 0.0))
    feats.append(attrs.get('shielding_dia', 0.0))
    feats.append(attrs.get('shielding_para', 0.0))
    feats.append(attrs.get('span', 0.0))
    feats.append(attrs.get('skew', 0.0))
    feats.append(attrs.get('asymmetry', 0.0))
    feats.append(attrs.get('anisotropy', 0.0))
    feats.append(attrs.get('at_charge_mull', 0.0))
    feats.append(attrs.get('at_charge_loew', 0.0))
    feats.append(attrs.get('orb_charge_mull_s', 0.0))
    feats.append(attrs.get('orb_charge_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_s', 0.0))
    feats.append(attrs.get('orb_charge_loew_p', 0.0))
    feats.append(attrs.get('BO_loew', 0.0))
    feats.append(attrs.get('BO_mayer', 0.0))
    feats.append(attrs.get('mayer_VA', 0.0))
    return feats

def get_c_features(attrs):
    feats = []
    elem = attrs.get('element', 'C')
    feats.extend(get_element_onehot(elem))
    feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    feats.append(attrs.get('shift_low', 0.0))
    feats.append(attrs.get('CN(X)', 0.0))
    feats.append(attrs.get('no_CH', 0.0))
    feats.append(attrs.get('no_CC', 0.0))
    feats.append(attrs.get('no_CN', 0.0))
    feats.append(attrs.get('no_CO', 0.0))
    feats.append(attrs.get('no_CYH', 0.0))
    feats.append(attrs.get('no_CYC', 0.0))
    feats.append(attrs.get('no_CYN', 0.0))
    feats.append(attrs.get('no_CYO', 0.0))
    feats.append(attrs.get('shielding_dia', 0.0))
    feats.append(attrs.get('shielding_para', 0.0))
    feats.append(attrs.get('span', 0.0))
    feats.append(attrs.get('skew', 0.0))
    feats.append(attrs.get('asymmetry', 0.0))
    feats.append(attrs.get('anisotropy', 0.0))
    feats.append(attrs.get('at_charge_mull', 0.0))
    feats.append(attrs.get('at_charge_loew', 0.0))
    feats.append(attrs.get('orb_charge_mull_s', 0.0))
    feats.append(attrs.get('orb_charge_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_mull_d', 0.0))
    feats.append(attrs.get('orb_stdev_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_s', 0.0))
    feats.append(attrs.get('orb_charge_loew_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_d', 0.0))
    feats.append(attrs.get('orb_stdev_loew_p', 0.0))
    feats.append(attrs.get('BO_loew_sum', 0.0))
    feats.append(attrs.get('BO_loew_av', 0.0))
    feats.append(attrs.get('BO_mayer_sum', 0.0))
    feats.append(attrs.get('BO_mayer_av', 0.0))
    feats.append(attrs.get('mayer_VA', 0.0))
    return feats

def get_others_features(attrs):
    feats = []
    elem = attrs.get('element', 'X')  # 'X' = unbekannt
    feats.extend(get_element_onehot(elem))
    feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    return feats


class OldShiftDataset(Dataset):
    def __init__(self, root_dir="data", file_name="all_graphs.pkl"):
        super().__init__()
        self.file_path = os.path.join(root_dir, file_name)
        with open(self.file_path, "rb") as f:
            self.nx_graphs = pickle.load(f)

    def __len__(self):
        return len(self.nx_graphs)
    
    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        data = HeteroData()
        h_nodes, c_nodes, o_nodes = [], [], []
        h_features, c_features, o_features = [], [], []
        h_shifts, c_shifts, o_shifts = [], [], []
        node_idx_map = {}

        # (Feature-Extraktion analog – siehe Originalcode)
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            shift_val = attrs.get("shift_high-low", float('nan'))
            if element == "H":
                node_idx_map[node] = len(h_nodes)
                h_nodes.append(node)
                h_shifts.append(shift_val)
                feats = get_h_features(attrs)
                h_features.append(feats)
            elif element == "C":
                node_idx_map[node] = len(c_nodes)
                c_nodes.append(node)
                c_shifts.append(shift_val)
                feats = get_c_features(attrs)
                c_features.append(feats)
            else:
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                o_shifts.append(float('nan'))
                feats = get_others_features(attrs)
                o_features.append(feats)
        
        h_x = torch.tensor(h_features, dtype=torch.float)
        c_x = torch.tensor(c_features, dtype=torch.float)
        o_x = torch.tensor(o_features, dtype=torch.float)
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) else torch.empty((0,1))
        
        if len(h_nodes) > 0:
            data['H'].x = h_x
            data['H'].y = h_y
        if len(c_nodes) > 0:
            data['C'].x = c_x
            data['C'].y = c_y
        if len(o_nodes) > 0:
            data['Others'].x = o_x
            data['Others'].y = o_y
        
        edge_index_dict = {}
        edge_attr_dict = {}
        
        def add_edge(src_type, dst_type, src_id, dst_id, bond_feat):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]
                edge_attr_dict[rel] = []
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
            edge_attr_dict[rel].append(bond_feat)
        
        def get_bond_features(bond_data):
            bt = bond_data.get('bond_type', 'SINGLE')
            bt_val = 1.0 if bt == 'SINGLE' else 2.0 if bt == 'DOUBLE' else 3.0 if bt == 'TRIPLE' else 0.0
            is_arom = 1.0 if bond_data.get('is_aromatic', False) else 0.0
            bd = bond_data.get('bond_dir', 'NONE')
            bd_val = 0.0 if bd == 'NONE' else 1.0 if bd == 'ENDUPRIGHT' else 0.5
            bo = bond_data.get('bond_order', 1.0)
            return [bt_val, is_arom, bd_val, bo]
        
        for u, v in nx_g.edges():
            bond_data = nx_g[u][v]
            bond_feat = get_bond_features(bond_data)
            u_element = nx_g.nodes[u]["element"]
            v_element = nx_g.nodes[v]["element"]
            if u_element == "H":
                u_type = "H"
                u_idx = node_idx_map[u]
            elif u_element == "C":
                u_type = "C"
                u_idx = node_idx_map[u]
            else:
                u_type = "Others"
                u_idx = node_idx_map[u]
            if v_element == "H":
                v_type = "H"
                v_idx = node_idx_map[v]
            elif v_element == "C":
                v_type = "C"
                v_idx = node_idx_map[v]
            else:
                v_type = "Others"
                v_idx = node_idx_map[v]
            add_edge(u_type, v_type, u_idx, v_idx, bond_feat)
            add_edge(v_type, u_type, v_idx, u_idx, bond_feat)
        
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_feats = torch.tensor(edge_attr_dict[rel], dtype=torch.float)
            data[rel].edge_attr = edge_feats
        
        return data

class ShiftDataset(Dataset):
    def __init__(self, root_dir="data", file_name="all_graphs.pkl"):
        super().__init__()
        self.file_path = os.path.join(root_dir, file_name)
        with open(self.file_path, "rb") as f:
            self.nx_graphs = pickle.load(f)
        self.norm_stats = {}
        feat_collect = {'H': [], 'C': [], 'Others': []}
        
        for nx_g in self.nx_graphs:
            for node in nx_g.nodes():
                attrs = nx_g.nodes[node]
                element = attrs["element"]
                if element == "H":
                    feats = get_h_features(attrs)
                    feat_collect['H'].append(feats[13:])
                elif element == "C":
                    feats = get_c_features(attrs)
                    feat_collect['C'].append(feats[13:])
                else:
                    feats = get_others_features(attrs)
                    feat_collect['Others'].append(feats[13:])
        
        for ntype in feat_collect:
            if feat_collect[ntype]:
                arr = np.array(feat_collect[ntype])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                self.norm_stats[ntype] = (mean, std)
            else:
                self.norm_stats[ntype] = (None, None)
    
    def __len__(self):
        return len(self.nx_graphs)
    
    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        data = HeteroData()
        h_nodes, c_nodes, o_nodes = [], [], []
        h_features, c_features, o_features = [], [], []
        h_shifts, c_shifts, o_shifts = [], [], []
        node_idx_map = {}
    
        def get_h_features_local(attrs):
            return get_h_features(attrs)
        
        def get_c_features_local(attrs):
            return get_c_features(attrs)
        
        def get_others_features_local(attrs):
            return get_others_features(attrs)
        
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            shift_val = attrs.get("shift_high-low", float('nan'))
            if element == "H":
                node_idx_map[node] = len(h_nodes)
                h_nodes.append(node)
                h_shifts.append(shift_val)
                feats = get_h_features_local(attrs)
                h_features.append(feats)
            elif element == "C":
                node_idx_map[node] = len(c_nodes)
                c_nodes.append(node)
                c_shifts.append(shift_val)
                feats = get_c_features_local(attrs)
                c_features.append(feats)
            else:
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                o_shifts.append(float('nan'))
                feats = get_others_features_local(attrs)
                o_features.append(feats)
        
        if h_features:
            h_features = np.array(h_features, dtype=np.float32)
            mean, std = self.norm_stats['H']
            if mean is not None:
                h_features[:, 13:] = (h_features[:, 13:] - mean) / (std + 1e-6)
            h_x = torch.tensor(h_features, dtype=torch.float)
        else:
            h_x = torch.empty((0, 13))
        
        if c_features:
            c_features = np.array(c_features, dtype=np.float32)
            mean, std = self.norm_stats['C']
            if mean is not None:
                c_features[:, 13:] = (c_features[:, 13:] - mean) / (std + 1e-6)
            c_x = torch.tensor(c_features, dtype=torch.float)
        else:
            c_x = torch.empty((0, 13))
        
        if o_features:
            o_features = np.array(o_features, dtype=np.float32)
            mean, std = self.norm_stats['Others']
            if mean is not None:
                o_features[:, 13:] = (o_features[:, 13:] - mean) / (std + 1e-6)
            o_x = torch.tensor(o_features, dtype=torch.float)
        else:
            o_x = torch.empty((0, 13))
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) else torch.empty((0,1))
        
        if len(h_nodes) > 0:
            data['H'].x = h_x
            data['H'].y = h_y
        if len(c_nodes) > 0:
            data['C'].x = c_x
            data['C'].y = c_y
        if len(o_nodes) > 0:
            data['Others'].x = o_x
            data['Others'].y = o_y
        
        edge_index_dict = {}
        edge_attr_dict = {}
        
        def add_edge(src_type, dst_type, src_id, dst_id, bond_feat):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]
                edge_attr_dict[rel] = []
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
            edge_attr_dict[rel].append(bond_feat)
        
        def get_bond_features(bond_data):
            bt = bond_data.get('bond_type', 'SINGLE')
            bt_val = 1.0 if bt == 'SINGLE' else 2.0 if bt == 'DOUBLE' else 3.0 if bt == 'TRIPLE' else 0.0
            is_arom = 1.0 if bond_data.get('is_aromatic', False) else 0.0
            bd = bond_data.get('bond_dir', 'NONE')
            bd_val = 0.0 if bd == 'NONE' else 1.0 if bd == 'ENDUPRIGHT' else 0.5
            bo = bond_data.get('bond_order', 1.0)
            return [bt_val, is_arom, bd_val, bo]
        
        for u, v in nx_g.edges():
            bond_data = nx_g[u][v]
            bond_feat = get_bond_features(bond_data)
            u_element = nx_g.nodes[u]["element"]
            v_element = nx_g.nodes[v]["element"]
            if u_element == "H":
                u_type = "H"
                u_idx = node_idx_map[u]
            elif u_element == "C":
                u_type = "C"
                u_idx = node_idx_map[u]
            else:
                u_type = "Others"
                u_idx = node_idx_map[u]
            if v_element == "H":
                v_type = "H"
                v_idx = node_idx_map[v]
            elif v_element == "C":
                v_type = "C"
                v_idx = node_idx_map[v]
            else:
                v_type = "Others"
                v_idx = node_idx_map[v]
            add_edge(u_type, v_type, u_idx, v_idx, bond_feat)
            add_edge(v_type, u_type, v_idx, u_idx, bond_feat)
        
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_feats = torch.tensor(edge_attr_dict[rel], dtype=torch.float)
            data[rel].edge_attr = edge_feats
        
        return data

def create_dataloaders(batch_size=4, root_dir="data", file_name="all_graphs.pkl", split_ratio=(0.8, 0.1, 0.1)):
    dataset = ShiftDataset(root_dir=root_dir, file_name=file_name)
    
    # Gruppiere Graphen nach dem 'compound'-Attribut (in den Graph-Attributen)
    compound_to_indices = {}
    for idx, nx_g in enumerate(dataset.nx_graphs):
        compound = nx_g.graph.get("compound", None)
        if compound is None:
            compound = "unknown"
        if compound not in compound_to_indices:
            compound_to_indices[compound] = []
        compound_to_indices[compound].append(idx)
    
    compounds = list(compound_to_indices.keys())
    random.shuffle(compounds)
    
    num_compounds = len(compounds)
    train_end = int(split_ratio[0] * num_compounds)
    val_end = train_end + int(split_ratio[1] * num_compounds)
    
    train_compounds = compounds[:train_end]
    val_compounds = compounds[train_end:val_end]
    test_compounds = compounds[val_end:]
    
    train_indices = []
    for comp in train_compounds:
        train_indices.extend(compound_to_indices[comp])
    
    val_indices = []
    for comp in val_compounds:
        val_indices.extend(compound_to_indices[comp])
    
    test_indices = []
    for comp in test_compounds:
        test_indices.extend(compound_to_indices[comp])
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = PyGDataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
