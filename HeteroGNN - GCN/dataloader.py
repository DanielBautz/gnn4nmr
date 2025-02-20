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
    """
    Erzeugt einen One-Hot-Vektor der Länge len(ALL_ELEMENTS).
    An der Stelle des gefundenen Elements steht eine '1', sonst '0'.
    Falls ein Element nicht in ALL_ELEMENTS ist, wird ein Vektor mit nur 0 zurückgegeben.
    """
    one_hot = [0.0] * len(ALL_ELEMENTS)
    if elem in ALL_ELEMENTS:
        idx = ALL_ELEMENTS.index(elem)
        one_hot[idx] = 1.0
    return one_hot


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
        
        # Heterogenes Data-Objekt
        data = HeteroData()
        
        # -- Node-Indizes pro Typ --
        h_nodes, c_nodes, o_nodes = [], [], []
        
        # -- Feature-Listen pro Typ --
        h_features, c_features, o_features = [], [], []
        
        # -- Shift-Listen (Labels) --
        h_shifts, c_shifts, o_shifts = [], [], []
        
        node_idx_map = {}

        # -------------------------------------------------------
        # Hilfsfunktionen zum Extrahieren der Node-Features
        # -------------------------------------------------------
        def get_h_features(attrs):
            """
            Für Wasserstoff: Erst One-Hot für 'element',
            dann die restlichen Features. 
            (Beispielhaft, du kannst beliebig erweitern.)
            """
            feats = []
            
            # 1) One-Hot für Element
            elem = attrs.get('element', 'H')
            feats.extend(get_element_onehot(elem))
            
            # 2) Beispiel: Atom-Index
            feats.append(float(attrs.get('atom_idx', -1)))
            
            # 3) Position (x, y, z)
            #pos = attrs.get('pos', (0.0, 0.0, 0.0))
            #feats.extend([pos[0], pos[1], pos[2]])
            
            # 4) Mass, formale Ladung, Grad
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # 5) shift_low z.B.
            feats.append(attrs.get('shift_low', 0.0))
            
            # 6) CN(X), no_HCH, no_HYH, ...
            feats.append(attrs.get('CN(X)', 0.0))
            feats.append(attrs.get('no_HCH', 0.0))
            feats.append(attrs.get('no_HYH', 0.0))
            feats.append(attrs.get('no_HYC', 0.0))
            feats.append(attrs.get('no_HYN', 0.0))
            feats.append(attrs.get('no_HYO', 0.0))
            feats.append(attrs.get('dist_HC', 0.0))
            feats.append(attrs.get('shift_low_neighbor_C', 0.0))
            
            # 7) Abschirmungsdaten
            feats.append(attrs.get('shielding_dia', 0.0))
            feats.append(attrs.get('shielding_para', 0.0))
            feats.append(attrs.get('span', 0.0))
            feats.append(attrs.get('skew', 0.0))
            feats.append(attrs.get('asymmetry', 0.0))
            feats.append(attrs.get('anisotropy', 0.0))
            
            # 8) Ladungen
            feats.append(attrs.get('at_charge_mull', 0.0))
            feats.append(attrs.get('at_charge_loew', 0.0))
            
            # 9) Orbital-Aufteilung
            feats.append(attrs.get('orb_charge_mull_s', 0.0))
            feats.append(attrs.get('orb_charge_mull_p', 0.0))
            feats.append(attrs.get('orb_charge_loew_s', 0.0))
            feats.append(attrs.get('orb_charge_loew_p', 0.0))
            
            # 10) Bindungsordnungen
            feats.append(attrs.get('BO_loew', 0.0))
            feats.append(attrs.get('BO_mayer', 0.0))
            feats.append(attrs.get('mayer_VA', 0.0))
            
            return feats
        
        def get_c_features(attrs):
            """
            Für Kohlenstoff: Erst One-Hot, dann restliche Features.
            """
            feats = []
            
            # 1) One-Hot
            elem = attrs.get('element', 'C')
            feats.extend(get_element_onehot(elem))
            
            # 2) atom_idx
            feats.append(float(attrs.get('atom_idx', -1)))
            
            # 3) pos
            #pos = attrs.get('pos', (0.0, 0.0, 0.0))
            #feats.extend([pos[0], pos[1], pos[2]])
            
            # 4) mass, formal_charge, degree
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # 5) shift_low, CN(X)
            feats.append(attrs.get('shift_low', 0.0))
            feats.append(attrs.get('CN(X)', 0.0))
            
            # 6) no_CH, no_CC, no_CN, ...
            feats.append(attrs.get('no_CH', 0.0))
            feats.append(attrs.get('no_CC', 0.0))
            feats.append(attrs.get('no_CN', 0.0))
            feats.append(attrs.get('no_CO', 0.0))
            feats.append(attrs.get('no_CYH', 0.0))
            feats.append(attrs.get('no_CYC', 0.0))
            feats.append(attrs.get('no_CYN', 0.0))
            feats.append(attrs.get('no_CYO', 0.0))
            
            # 7) Abschirmungen
            feats.append(attrs.get('shielding_dia', 0.0))
            feats.append(attrs.get('shielding_para', 0.0))
            feats.append(attrs.get('span', 0.0))
            feats.append(attrs.get('skew', 0.0))
            feats.append(attrs.get('asymmetry', 0.0))
            feats.append(attrs.get('anisotropy', 0.0))
            
            # 8) Ladungen
            feats.append(attrs.get('at_charge_mull', 0.0))
            feats.append(attrs.get('at_charge_loew', 0.0))
            
            # 9) Orbital Ladungen
            feats.append(attrs.get('orb_charge_mull_s', 0.0))
            feats.append(attrs.get('orb_charge_mull_p', 0.0))
            feats.append(attrs.get('orb_charge_mull_d', 0.0))
            feats.append(attrs.get('orb_stdev_mull_p', 0.0))
            
            feats.append(attrs.get('orb_charge_loew_s', 0.0))
            feats.append(attrs.get('orb_charge_loew_p', 0.0))
            feats.append(attrs.get('orb_charge_loew_d', 0.0))
            feats.append(attrs.get('orb_stdev_loew_p', 0.0))
            
            # 10) BO
            feats.append(attrs.get('BO_loew_sum', 0.0))
            feats.append(attrs.get('BO_loew_av', 0.0))
            feats.append(attrs.get('BO_mayer_sum', 0.0))
            feats.append(attrs.get('BO_mayer_av', 0.0))
            feats.append(attrs.get('mayer_VA', 0.0))
            
            return feats
        
        def get_others_features(attrs):
            """
            Für alle restlichen Elemente (Li, B, N, O, ...).
            Auch hier: Erst One-Hot, dann ggf. Basis-Features.
            """
            feats = []
            
            # 1) One-Hot
            elem = attrs.get('element', 'X')  # 'X' = unbekannt
            feats.extend(get_element_onehot(elem))
            
            # 2) atom_idx
            feats.append(float(attrs.get('atom_idx', -1)))
            
            # 3) pos
            #pos = attrs.get('pos', (0.0, 0.0, 0.0))
            #feats.extend([pos[0], pos[1], pos[2]])
            
            # 4) mass, formal_charge, degree
            feats.append(attrs.get('mass', 0.0))
            feats.append(attrs.get('formal_charge', 0.0))
            feats.append(attrs.get('degree', 0.0))
            
            # Beliebig erweiterbar, wenn du mehr Felder für O/N etc. brauchst.
            return feats
        
        # -------------------------------------------------------
        # 1) Knoten-Schleife: H, C, Others
        # -------------------------------------------------------
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            
            # shift_high-low als Label (nur H und C)
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
                # -> Alle restlichen Elemente
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                
                o_shifts.append(float('nan'))  # i.d.R. kein Shift-Label
                feats = get_others_features(attrs)
                o_features.append(feats)
        
        # -- Tensor-Konvertierung --
        h_x = torch.tensor(h_features, dtype=torch.float)
        c_x = torch.tensor(c_features, dtype=torch.float)
        o_x = torch.tensor(o_features, dtype=torch.float)
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) else torch.empty((0,1))
        
        # -------------------------------------------------------
        # 2) Kanten + Edge-Features
        # -------------------------------------------------------
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
            """
            Einfaches Mapping für bond_type, is_aromatic, bond_dir, bond_order
            => 4 floats
            """
            bt = bond_data.get('bond_type', 'SINGLE')
            if bt == 'SINGLE':
                bt_val = 1.0
            elif bt == 'DOUBLE':
                bt_val = 2.0
            elif bt == 'TRIPLE':
                bt_val = 3.0
            else:
                bt_val = 0.0
            
            is_arom = 1.0 if bond_data.get('is_aromatic', False) else 0.0
            
            bd = bond_data.get('bond_dir', 'NONE')
            if bd == 'NONE':
                bd_val = 0.0
            elif bd == 'ENDUPRIGHT':
                bd_val = 1.0
            else:
                bd_val = 0.5  # Beispiel für unbekannte Fälle
            
            bo = bond_data.get('bond_order', 1.0)
            
            return [bt_val, is_arom, bd_val, bo]
        
        # NetworkX-Kanten durchgehen
        for u, v in nx_g.edges():
            bond_data = nx_g[u][v]  # {'bond_type': 'SINGLE', 'bond_order':1.0, ...}
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
            
            # Ungerichtete Kante => hin und zurück
            add_edge(u_type, v_type, u_idx, v_idx, bond_feat)
            add_edge(v_type, u_type, v_idx, u_idx, bond_feat)
        
        # -- HeteroData füllen --
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
            edge_feats = torch.tensor(edge_attr_dict[rel], dtype=torch.float)
            data[rel].edge_attr = edge_feats
        
        return data


def create_dataloaders(batch_size=4, root_dir="data", file_name="all_graphs.pkl"):
    dataset = ShiftDataset(root_dir=root_dir, file_name=file_name)
    
    # Gruppiere Graphen nach dem 'compound'-Attribut (das in den Graph-Attributen gespeichert ist)
    compound_to_indices = {}
    for idx, nx_g in enumerate(dataset.nx_graphs):
        compound = nx_g.graph.get("compound", None)
        if compound is None:
            compound = "unknown"  # Fallback, falls kein compound gesetzt ist
        if compound not in compound_to_indices:
            compound_to_indices[compound] = []
        compound_to_indices[compound].append(idx)
    
    # Erstelle eine Liste aller Compounds und mische diese zufällig
    compounds = list(compound_to_indices.keys())
    random.shuffle(compounds)
    
    num_compounds = len(compounds)
    train_end = int(0.8 * num_compounds)
    val_end = int(0.9 * num_compounds)
    
    train_compounds = compounds[:train_end]
    val_compounds = compounds[train_end:val_end]
    test_compounds = compounds[val_end:]
    
    # Sammle die Indizes für jeden Split basierend auf den Compounds
    train_indices = []
    for comp in train_compounds:
        train_indices.extend(compound_to_indices[comp])
    
    val_indices = []
    for comp in val_compounds:
        val_indices.extend(compound_to_indices[comp])
    
    test_indices = []
    for comp in test_compounds:
        test_indices.extend(compound_to_indices[comp])
    
    # Optional: Shuffle innerhalb der einzelnen Splits
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

def old_create_dataloaders(batch_size=4, root_dir="data", file_name="all_graphs.pkl"):
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
