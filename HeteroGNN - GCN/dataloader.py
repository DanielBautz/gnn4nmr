import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
import random

class ShiftDataset(Dataset):
    def __init__(self, root_dir="data", file_name="all_graphs.pkl"):
        super().__init__()
        self.file_path = os.path.join(root_dir, file_name)
        
        # Laden der Liste von NetworkX-Graphen
        with open(self.file_path, "rb") as f:
            self.nx_graphs = pickle.load(f)

    def __len__(self):
        return len(self.nx_graphs)
    
    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        
        data = HeteroData()
        
        # --- Listen für Knoten-IDs pro Typ ---
        h_nodes, c_nodes, o_nodes = [], [], []
        
        # --- Feature-Listen pro Typ ---
        h_features, c_features, o_features = [], [], []
        
        # --- Shift-Listen (Labels) ---
        h_shifts, c_shifts, o_shifts = [], [], []
        
        node_idx_map = {}
        
        # --------------------------------------------------------------------
        # HILFSFUNKTION: Node-Features generieren
        # --------------------------------------------------------------------
        def get_h_features(attrs):
            """
            Hier wird definiert, welche Felder für H als Feature genutzt werden.
            Gib am Ende eine (beliebig lange) Liste von floats zurück.
            """
            feats = []
            # Beispiel: Positionskoordinaten (3 floats)
            pos = attrs.get('pos', (0.0, 0.0, 0.0))
            feats.extend([pos[0], pos[1], pos[2]])
            
            # Mass, formale Ladung, Grad (Anzahl Bindungen)
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # Weitere Felder (hier nur exemplarisch)
            feats.append(attrs.get('CN(X)', 0.0))
            feats.append(attrs.get('no_HCH', 0.0))
            feats.append(attrs.get('no_HYO', 0.0))
            
            # Abschirmung (shielding_dia, shielding_para etc.)
            feats.append(attrs.get('shielding_dia', 0.0))
            feats.append(attrs.get('shielding_para', 0.0))
            feats.append(attrs.get('span', 0.0))
            feats.append(attrs.get('skew', 0.0))
            feats.append(attrs.get('asymmetry', 0.0))
            feats.append(attrs.get('anisotropy', 0.0))
            
            # Elektronen-Ladungen
            feats.append(attrs.get('at_charge_mull', 0.0))
            feats.append(attrs.get('at_charge_loew', 0.0))
            
            # Orbital-Aufteilung s/p
            feats.append(attrs.get('orb_charge_mull_s', 0.0))
            feats.append(attrs.get('orb_charge_mull_p', 0.0))
            feats.append(attrs.get('orb_charge_loew_s', 0.0))
            feats.append(attrs.get('orb_charge_loew_p', 0.0))
            
            # Bindungsordnungen
            feats.append(attrs.get('BO_loew', 0.0))
            feats.append(attrs.get('BO_mayer', 0.0))
            feats.append(attrs.get('mayer_VA', 0.0))
            
            return feats
        
        def get_c_features(attrs):
            """
            Beispiel für C-Features.
            Nutze ähnliche oder andere Felder wie bei H.
            """
            feats = []
            pos = attrs.get('pos', (0.0, 0.0, 0.0))
            feats.extend([pos[0], pos[1], pos[2]])
            
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # Shift_low und CN(X) ...
            feats.append(attrs.get('shift_low', 0.0))
            feats.append(attrs.get('CN(X)', 0.0))
            
            # Beispiel: no_CH, no_CO, ...
            feats.append(attrs.get('no_CH', 0.0))
            feats.append(attrs.get('no_CO', 0.0))
            
            # Shielding
            feats.append(attrs.get('shielding_dia', 0.0))
            feats.append(attrs.get('shielding_para', 0.0))
            feats.append(attrs.get('span', 0.0))
            feats.append(attrs.get('skew', 0.0))
            feats.append(attrs.get('asymmetry', 0.0))
            feats.append(attrs.get('anisotropy', 0.0))
            
            # Ladungen
            feats.append(attrs.get('at_charge_mull', 0.0))
            feats.append(attrs.get('at_charge_loew', 0.0))
            
            # Orbital Ladungen
            feats.append(attrs.get('orb_charge_mull_s', 0.0))
            feats.append(attrs.get('orb_charge_mull_p', 0.0))
            feats.append(attrs.get('orb_charge_mull_d', 0.0))
            feats.append(attrs.get('orb_charge_loew_s', 0.0))
            feats.append(attrs.get('orb_charge_loew_p', 0.0))
            feats.append(attrs.get('orb_charge_loew_d', 0.0))
            
            # Bindungsordnungen
            feats.append(attrs.get('BO_loew_sum', 0.0))
            feats.append(attrs.get('BO_loew_av', 0.0))
            feats.append(attrs.get('BO_mayer_sum', 0.0))
            feats.append(attrs.get('BO_mayer_av', 0.0))
            feats.append(attrs.get('mayer_VA', 0.0))
            
            return feats
        
        def get_others_features(attrs):
            """
            Beispiel für O oder andere 'Others'.
            Man kann hier ggf. weniger Felder nehmen.
            """
            feats = []
            pos = attrs.get('pos', (0.0, 0.0, 0.0))
            feats.extend([pos[0], pos[1], pos[2]])
            
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # Falls du auch 'shift_high-low' von Others willst (normalerweise NaN/None)
            # feats.append(attrs.get('shift_high-low', 0.0))
            
            return feats
        
        # --------------------------------------------------------------------
        # 1) Schleife über Knoten, Features sammeln
        # --------------------------------------------------------------------
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            
            # SHIFT für H/C (das ist dein Label)
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
                # Alle restlichen Elemente
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                
                # Im Normalfall haben wir kein shift => NaN
                o_shifts.append(float('nan'))
                
                feats = get_others_features(attrs)
                o_features.append(feats)
        
        # --- Konvertieren in Torch-Tensoren ---
        h_x = torch.tensor(h_features, dtype=torch.float)
        c_x = torch.tensor(c_features, dtype=torch.float)
        o_x = torch.tensor(o_features, dtype=torch.float)
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) else torch.empty((0,1))
        
        # --------------------------------------------------------------------
        # 2) Edges: Erstellen + Edge-Features
        # --------------------------------------------------------------------
        edge_index_dict = {}
        edge_attr_dict = {}  # Neu: hier speichern wir Edge-Features
        
        def add_edge(src_type, dst_type, src_id, dst_id, bond_feat):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]
                edge_attr_dict[rel] = []
            
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
            
            # Edge-Feature-Vector aufbauen
            edge_attr_dict[rel].append(bond_feat)
        
        # Hilfsfunktion, um aus NX-Bond-Attr ein Float-Feature-Array zu bauen
        def get_bond_features(bond_data):
            """
            Mach z.B. aus 'bond_type', 'is_aromatic', 'bond_dir', 'bond_order' 
            einen Float-Vektor. 
            
            Hier nur ein Beispiel, wie man es umsetzen könnte:
            - bond_type => one-hot oder Mapping (SINGLE=1, DOUBLE=2,...)
            - is_aromatic => 0/1
            - bond_dir => 0/1/2 (Mapping), etc.
            - bond_order => float
            """
            # Mapping bond_type
            bt = bond_data.get('bond_type', 'SINGLE')
            if bt == 'SINGLE':
                bt_val = 1.0
            elif bt == 'DOUBLE':
                bt_val = 2.0
            elif bt == 'TRIPLE':
                bt_val = 3.0
            else:
                bt_val = 0.0
            
            # is_aromatic => True/False
            is_arom = 1.0 if bond_data.get('is_aromatic', False) else 0.0
            
            # bond_dir => Mapping
            bd = bond_data.get('bond_dir', 'NONE')
            bd_val = 0.0
            if bd == 'NONE':
                bd_val = 0.0
            elif bd == 'ENDUPRIGHT':
                bd_val = 1.0
            # ggf. weitere Fälle
            
            # bond_order
            bo = bond_data.get('bond_order', 1.0)
            
            return [bt_val, is_arom, bd_val, bo]
        
        # NX edges iterieren
        for u, v in nx_g.edges():
            bond_data = nx_g[u][v]  # Kantenattribute in NetworkX
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
            
            # Hin- und Rückkante
            add_edge(u_type, v_type, u_idx, v_idx, bond_feat)
            add_edge(v_type, u_type, v_idx, u_idx, bond_feat)
        
        # --------------------------------------------------------------------
        # 3) HeteroData befüllen
        # --------------------------------------------------------------------
        if len(h_nodes) > 0:
            data['H'].x = h_x
            data['H'].y = h_y
        if len(c_nodes) > 0:
            data['C'].x = c_x
            data['C'].y = c_y
        if len(o_nodes) > 0:
            data['Others'].x = o_x
            data['Others'].y = o_y
        
        # Edges + Edge-Features
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
            
            # Edge-Feature array in FloatTensor
            edge_feats = edge_attr_dict[rel]
            edge_feats = torch.tensor(edge_feats, dtype=torch.float)
            data[rel].edge_attr = edge_feats
        
        return data


def create_dataloaders(batch_size=4, root_dir="data", file_name="all_graphs.pkl"):
    dataset = ShiftDataset(root_dir=root_dir, file_name=file_name)
    
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    random.shuffle(indices)
    
    train_end = int(0.8 * num_graphs)
    val_end = int(0.9 * num_graphs)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = PyGDataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
