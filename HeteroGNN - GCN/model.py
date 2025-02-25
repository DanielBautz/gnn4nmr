import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GraphConv

class MLPEncoder(nn.Module):
    """Einfaches MLP zur Transformation der Rohfeatures in d_model mit Dropout."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
    
    def forward(self, x):
        return self.net(x)

class HeteroGNNModel(nn.Module):
    def __init__(self, in_dim_dict, hidden_dim=32, out_dim=16, dropout_prob=0.1):
        """
        in_dim_dict: dict mit { 'H': <int>, 'C': <int>, 'Others': <int> } Feature-Dimensionen
        hidden_dim: Größe der MLP-Hidden und GNN-Hidden-Dimensionen
        out_dim: Größe der MLP-Output-Dimension, also Embedding-Dimension
        dropout_prob: Wahrscheinlichkeit für Dropout
        """
        super().__init__()
        
        # 1) Featureencoder pro Knotentyp (MLP) mit Dropout
        self.encoder_dict = nn.ModuleDict()
        for ntype, in_dim in in_dim_dict.items():
            self.encoder_dict[ntype] = MLPEncoder(in_dim, hidden_dim, out_dim, dropout_prob=dropout_prob)
        
        # Dropout-Layer für den GNN-Teil
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # 2) HeteroConv-Schichten (mit GraphConv)
        self.conv1 = HeteroConv({
            ('H', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('H', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('H', 'bond', 'Others'): GraphConv(out_dim, out_dim),
            
            ('C', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('C', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('C', 'bond', 'Others'): GraphConv(out_dim, out_dim),
            
            ('Others', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('Others', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('Others', 'bond', 'Others'): GraphConv(out_dim, out_dim),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('H', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('H', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('H', 'bond', 'Others'): GraphConv(out_dim, out_dim),
            
            ('C', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('C', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('C', 'bond', 'Others'): GraphConv(out_dim, out_dim),
            
            ('Others', 'bond', 'H'): GraphConv(out_dim, out_dim),
            ('Others', 'bond', 'C'): GraphConv(out_dim, out_dim),
            ('Others', 'bond', 'Others'): GraphConv(out_dim, out_dim),
        }, aggr='sum')
        
        # 3) Zwei Output-Köpfe für Shift-Vorhersage (Node-Regression) - einmal für H, einmal für C
        self.pred_heads = nn.ModuleDict({
            'H': nn.Sequential(nn.Linear(out_dim, 1)), 
            'C': nn.Sequential(nn.Linear(out_dim, 1))
        })
        
    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: Dict { 'H': [num_H, in_dim], 'C': [num_C, in_dim], 'Others': [num_O, in_dim] }
        edge_index_dict: Dict mit Kanten pro Relation
        """
        # (1) Rohfeatures -> MLP-Encoder (Dropout ist in den Encodern integriert)
        for ntype, x in x_dict.items():
            x_dict[ntype] = self.encoder_dict[ntype](x)
        
        # (2) Erste HeteroConv-Schicht + ReLU & Dropout
        x_dict = self.conv1(x_dict, edge_index_dict)
        for ntype in x_dict:
            x_dict[ntype] = self.dropout(nn.ReLU()(x_dict[ntype]))
        
        # (3) Zweite HeteroConv-Schicht + ReLU & Dropout
        x_dict = self.conv2(x_dict, edge_index_dict)
        for ntype in x_dict:
            x_dict[ntype] = self.dropout(nn.ReLU()(x_dict[ntype]))
        
        # (4) Vorhersagen für H und C
        out_dict = {}
        for ntype in x_dict:
            if ntype in self.pred_heads:
                out_dict[ntype] = self.pred_heads[ntype](x_dict[ntype])
            else:
                out_dict[ntype] = None
        
        return out_dict