import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import HeteroData

class ShiftDataset(Dataset):
    def __init__(self, root_dir="data", file_name="all_graphs.pkl"):
        super().__init__()
        self.file_path = os.path.join(root_dir, file_name)
        
        # Laden der Liste von NetworkX-Graphen
        with open(self.file_path, "rb") as f:
            self.nx_graphs = pickle.load(f)
        
        # Optional: Filtern oder Vorverarbeiten von Graphen
        # z.B.: Entfernen kaputter Graphen etc.
    
    def __len__(self):
        return len(self.nx_graphs)
    
    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        
        # Leeres HeteroData-Objekt
        data = HeteroData()
        
        # Wir möchten Knoten nach ihrem Typ "H", "C" oder "Others" separieren.
        # Node-Indizes pro Typ (Mapping NX->PyG).
        h_nodes, c_nodes, o_nodes = [], [], []
        
        # Feature-Listen (z.B. numerische und kategorische Features)
        h_features = []
        c_features = []
        o_features = []
        
        # Labels für shift_high-low (als Tensor)
        # Für "Others" gibt es keinen Label. Dort bleibt None oder NaN.
        h_shifts = []
        c_shifts = []
        o_shifts = []
        
        # Node-Zähler
        node_idx_map = {}  # nx_g node -> pytorch index (z.B. 0..N-1), pro Typ
        
        # 1) Knoten und ihre Features abfragen
        #    Wir gehen davon aus, dass die Node-Attribute in nx_g[node] wie folgt aussehen:
        #    {
        #       "element": "H" oder "C" oder "O"/"N"/...
        #       "shift_high-low": float (nur für H oder C)
        #       ... weitere Features ...
        #    }
        
        for i, node in enumerate(nx_g.nodes()):
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            
            # Extrahieren Sie weitere numerische oder boolsche Features, 
            # z.B. feats = [attrs["val1"], float(attrs["val2"]), attrs["is_aromatic"]*1.0, ...]
            # Im Minimalbeispiel nutzen wir nur shift und element. 
            
            if element == "H":
                node_idx_map[node] = len(h_nodes)
                h_nodes.append(node)
                
                shift_val = attrs.get("shift_high-low", 0.0)
                h_shifts.append(shift_val)
                
                # Beispiel: minimaler Featurevektor, 
                # z.B. [1.0, 0.0] => One-hot für "H" (nur Demo!)
                feats = [1.0]  # hier könnten Sie weitere numerische/cat Features anhängen
                h_features.append(feats)
                
            elif element == "C":
                node_idx_map[node] = len(c_nodes)
                c_nodes.append(node)
                
                shift_val = attrs.get("shift_high-low", 0.0)
                c_shifts.append(shift_val)
                
                # z.B. [0.0, 1.0] => One-hot für "C"
                feats = [1.0]
                c_features.append(feats)
            else:
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                
                # "Others" haben kein shift_high-low => Dummy-Wert (NaN)
                o_shifts.append(float('nan'))
                
                # z.B. [0.5] => Demo-Feature
                feats = [0.5]
                o_features.append(feats)
        
        # In Torch-Tensoren umwandeln
        h_x = torch.tensor(h_features, dtype=torch.float)
        c_x = torch.tensor(c_features, dtype=torch.float)
        o_x = torch.tensor(o_features, dtype=torch.float)
        
        # shift Arrays
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) > 0 else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) > 0 else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) > 0 else torch.empty((0,1))
        
        # 2) Edges anlegen: ungerichtete Kanten => in PyG muss man für heterogene Graphen
        #    pro (src_type, relation, dst_type) ein edge_index definieren.
        #    Wir traversieren alle Kanten und sortieren sie nach den Knotentypen.
        
        # Dictionary Edge-Index pro Relation
        edge_index_dict = {}
        
        # Hilfsfunktion: ein Dictionary-Eintrag pro (src_type, "bond", dst_type)
        def add_edge(src_type, dst_type, src_id, dst_id):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]  # 2 Listen für row, col
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
        
        for u, v in nx_g.edges():
            # Knoten-Typ herausfinden
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
            
            # ungerichtete Kante => wir brauchen hin und zurück
            add_edge(u_type, v_type, u_idx, v_idx)
            add_edge(v_type, u_type, v_idx, u_idx)
        
        # 3) HeteroData befüllen
        if len(h_nodes) > 0:
            data['H'].x = h_x
            data['H'].y = h_y  # shift label
        if len(c_nodes) > 0:
            data['C'].x = c_x
            data['C'].y = c_y
        if len(o_nodes) > 0:
            data['Others'].x = o_x
            data['Others'].y = o_y  # hier Dummy => NaNs
        
        # edge_index_dict nach Tensor konvertieren
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
            # Falls Edge-Features vorhanden: data[rel].edge_attr = ...
        
        return data

# dataloader.py
import random
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

def create_dataloaders(batch_size=4, root_dir="data", file_name="all_graphs.pkl"):
    dataset = ShiftDataset(root_dir=root_dir, file_name=file_name)
    
    # Beispiel: 80/10/10-Split
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
