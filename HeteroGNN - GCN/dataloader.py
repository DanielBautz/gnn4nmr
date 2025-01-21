import os
import pickle
import torch
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
        
        # Optional: Filtern oder Vorverarbeiten
    
    def __len__(self):
        return len(self.nx_graphs)
    
    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        
        # Leeres HeteroData
        data = HeteroData()
        
        h_nodes, c_nodes, o_nodes = [], [], []
        h_features, c_features, o_features = [], [], []
        h_shifts, c_shifts, o_shifts = [], [], []
        
        node_idx_map = {}
        
        # 1) Knoten & Features
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            
            if element == "H":
                node_idx_map[node] = len(h_nodes)
                h_nodes.append(node)
                
                shift_val = attrs.get("shift_high-low", 0.0)
                h_shifts.append(shift_val)
                h_features.append([1.0])  # Minimal-Feature
                
            elif element == "C":
                node_idx_map[node] = len(c_nodes)
                c_nodes.append(node)
                
                shift_val = attrs.get("shift_high-low", 0.0)
                c_shifts.append(shift_val)
                c_features.append([1.0])  # Minimal-Feature
            else:
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                
                o_shifts.append(float('nan'))
                o_features.append([0.5])  # z.B. 0.5
        
        # Torch-Tensoren
        h_x = torch.tensor(h_features, dtype=torch.float)
        c_x = torch.tensor(c_features, dtype=torch.float)
        o_x = torch.tensor(o_features, dtype=torch.float)
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if len(h_shifts) else torch.empty((0,1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if len(c_shifts) else torch.empty((0,1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if len(o_shifts) else torch.empty((0,1))
        
        # 2) Kanten
        edge_index_dict = {}
        def add_edge(src_type, dst_type, src_id, dst_id):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
        
        for u, v in nx_g.edges():
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
            add_edge(u_type, v_type, u_idx, v_idx)
            add_edge(v_type, u_type, v_idx, u_idx)
        
        # 3) HeteroData befüllen
        if len(h_nodes) > 0:
            data['H'].x = h_x
            data['H'].y = h_y
        if len(c_nodes) > 0:
            data['C'].x = c_x
            data['C'].y = c_y
        if len(o_nodes) > 0:
            data['Others'].x = o_x
            data['Others'].y = o_y
        
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
        
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
