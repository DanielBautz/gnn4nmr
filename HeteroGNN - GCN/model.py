import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv
from operators import get_conv_operator

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
    def __init__(
        self, 
        in_dim_dict, 
        hidden_dim=32, 
        out_dim=16, 
        encoder_dropout=0.1,
        gnnlayer_dropout=0.1,
        num_gnn_layers=2,
        operator_type="GraphConv", 
        operator_kwargs=None
    ):
        """
        in_dim_dict: dict mit { 'H': <int>, 'C': <int>, 'Others': <int> } Feature-Dimensionen
        hidden_dim: Größe der MLP-Hidden und GNN-Hidden-Dimensionen
        out_dim: Größe der MLP-Output-Dimension, also Embedding-Dimension
        encoder_dropout: Dropout-Wahrscheinlichkeit in den MLP-Encodern
        gnnlayer_dropout: Dropout-Wahrscheinlichkeit nach jeder HeteroConv-Schicht
        num_gnn_layers: Anzahl der HeteroConv-Schichten
        operator_type: Typ des GNN Operators (z.B. "GraphConv", "GCNConv", "GATConv", "SAGEConv", "GATv2Conv")
        operator_kwargs: zusätzliche Argumente für den GNN Operator
        """
        super().__init__()
        
        if operator_kwargs is None:
            operator_kwargs = {}
        
        # 1) Featureencoder pro Knotentyp mit encoder_dropout
        self.encoder_dict = nn.ModuleDict()
        for ntype, in_dim in in_dim_dict.items():
            self.encoder_dict[ntype] = MLPEncoder(in_dim, hidden_dim, out_dim, dropout_prob=encoder_dropout)
        
        # Dropout-Layer für GNN-Schichten
        self.gnn_dropout = nn.Dropout(p=gnnlayer_dropout)
        
        conv_class = get_conv_operator(operator_type)
        def create_conv_layers():
            relations = [
                ('H', 'bond', 'H'),
                ('H', 'bond', 'C'),
                ('H', 'bond', 'Others'),
                ('C', 'bond', 'H'),
                ('C', 'bond', 'C'),
                ('C', 'bond', 'Others'),
                ('Others', 'bond', 'H'),
                ('Others', 'bond', 'C'),
                ('Others', 'bond', 'Others')
            ]
            return {rel: conv_class(out_dim, out_dim, **operator_kwargs) for rel in relations}
        
        # Erstelle num_gnn_layers HeteroConv-Schichten
        self.convs = nn.ModuleList([HeteroConv(create_conv_layers(), aggr='sum') for _ in range(num_gnn_layers)])
        
        # Zwei Output-Köpfe für Shift-Vorhersage (Node-Regression)
        self.pred_heads = nn.ModuleDict({
            'H': nn.Linear(out_dim, 1), 
            'C': nn.Linear(out_dim, 1)
        })
        
    def forward(self, x_dict, edge_index_dict):
        # (1) Rohfeatures -> MLP-Encoder
        for ntype, x in x_dict.items():
            x_dict[ntype] = self.encoder_dict[ntype](x)
        
        # (2) Mehrere HeteroConv-Schichten
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for ntype in x_dict:
                x_dict[ntype] = self.gnn_dropout(nn.ReLU()(x_dict[ntype]))
        
        # (3) Vorhersagen für H und C
        out_dict = {}
        for ntype in x_dict:
            if ntype in self.pred_heads:
                out_dict[ntype] = self.pred_heads[ntype](x_dict[ntype])
            else:
                out_dict[ntype] = None
        
        return out_dict
