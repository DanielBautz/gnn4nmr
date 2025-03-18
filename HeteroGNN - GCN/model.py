import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv
from operators import get_conv_operator

class MLPEncoder(nn.Module):
    """Einfaches MLP zur Transformation der Rohfeatures in den d_model Raum mit Dropout."""
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
        operator_kwargs=None,
        edge_in_dim=10  # Dimension der rohen Kantenfeatures, z.B. 4
    ):
        """
        in_dim_dict: dict mit { 'H': <int>, 'C': <int>, 'Others': <int> } Feature-Dimensionen
        hidden_dim: Dimension der Hidden-Schichten im MLP und GNN
        out_dim: Dimension der Output-Einbettungen (nach dem MLP)
        encoder_dropout: Dropout-Wahrscheinlichkeit im MLP-Encoder
        gnnlayer_dropout: Dropout-Wahrscheinlichkeit nach jeder HeteroConv-Schicht
        num_gnn_layers: Anzahl der HeteroConv-Schichten
        operator_type: Typ des GNN Operators (z.B. "GraphConv", "GCNConv", "GATConv", "SAGEConv", "GATv2Conv", "GINEConv", "NNConv")
        operator_kwargs: zusätzliche Argumente für den GNN Operator
        edge_in_dim: Dimension der rohen Kantenfeatures, bevor sie projiziert werden (z.B. 4)
        """
        super().__init__()
        
        self.operator_type = operator_type  # Speichere den Typ für den Forward-Pfad
        
        if operator_kwargs is None:
            operator_kwargs = {}
        
        # Falls der Operator NNConv gewählt wird, setze automatisch 'edge_dim' in operator_kwargs.
        if operator_type == "NNConv":
            operator_kwargs.setdefault("edge_dim", edge_in_dim)
        
        # 1) Erstelle pro Knotentyp einen MLP-Encoder
        self.encoder_dict = nn.ModuleDict()
        for ntype, in_dim in in_dim_dict.items():
            self.encoder_dict[ntype] = MLPEncoder(in_dim, hidden_dim, out_dim, dropout_prob=encoder_dropout)
        
        # Für GINEConv wird ein zusätzlicher Edge-Projektions-Layer eingerichtet.
        if operator_type == "GINEConv":
            self.edge_proj = nn.Linear(edge_in_dim, out_dim)
        else:
            self.edge_proj = None
        
        # Dropout-Layer für die GNN-Schichten
        self.gnn_dropout = nn.Dropout(p=gnnlayer_dropout)
        
        conv_constructor = get_conv_operator(operator_type)
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
            # Hinweis: Wir verwenden out_dim als sowohl Input als auch Output der GNN-Schichten, da der MLP-Encoder bereits transformiert.
            return {rel: conv_constructor(out_dim, out_dim, **operator_kwargs) for rel in relations}
        
        # Erstelle num_gnn_layers HeteroConv-Schichten
        self.convs = nn.ModuleList([HeteroConv(create_conv_layers(), aggr='sum') for _ in range(num_gnn_layers)])
        
        # Zwei separate Vorhersageköpfe für Shift-Werte (für H und C)
        self.pred_heads = nn.ModuleDict({
            'H': nn.Linear(out_dim, 1), 
            'C': nn.Linear(out_dim, 1)
        })
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # (1) Transformiere die Knoteneingabefeatures mit den MLP-Encodern.
        for ntype, x in x_dict.items():
            x_dict[ntype] = self.encoder_dict[ntype](x)
        
        # Entscheide, ob Kantenfeatures weitergereicht werden sollen (nur bei GINEConv und NNConv)
        conv_kwargs = {}
        if self.operator_type in ["GINEConv", "NNConv"]:
            if edge_attr_dict is not None and self.edge_proj is not None:
                # Für GINEConv: projiziere die Kantenfeatures auf out_dim
                for key in edge_attr_dict:
                    edge_attr_dict[key] = self.edge_proj(edge_attr_dict[key])
            conv_kwargs["edge_attr_dict"] = edge_attr_dict
        
        # (2) Wende die HeteroConv-Schichten an.
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, **conv_kwargs)
            for ntype in x_dict:
                x_dict[ntype] = self.gnn_dropout(nn.ReLU()(x_dict[ntype]))
        
        # (3) Erzeuge die Vorhersagen für H und C.
        out_dict = {}
        for ntype in x_dict:
            if ntype in self.pred_heads:
                out_dict[ntype] = self.pred_heads[ntype](x_dict[ntype])
            else:
                out_dict[ntype] = None
        
        return out_dict
