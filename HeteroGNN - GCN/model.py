import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import HeteroConv
from operators import get_conv_operator

class MLPEncoder(nn.Module):
    """Einfaches MLP zur Transformation der Rohfeatures in den d_model Raum mit Dropout."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
    
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return x

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
        edge_in_dim=10,  # Dimension der rohen Kantenfeatures, z.B. 4
        use_activation_checkpointing=False
    ):
        """
        Verbesserte Version des HeteroGNN-Modells mit besserer Numerischer Stabilität.
        
        Args:
            use_activation_checkpointing: Wenn True, wird Activation Checkpointing verwendet.
                                         Dies reduziert Speichernutzung auf Kosten von Rechenzeit.
        """
        super().__init__()
        
        self.operator_type = operator_type
        self.use_checkpoint = use_activation_checkpointing
        
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
        self.act = nn.ReLU()
        
        conv_constructor = get_conv_operator(operator_type)
        
        # Liste der möglichen Beziehungen zwischen den Knotentypen
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
        
        # Erstelle num_gnn_layers HeteroConv-Schichten
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv_dict = {rel: conv_constructor(out_dim, out_dim, **operator_kwargs) for rel in relations}
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Zwei separate Vorhersageköpfe für Shift-Werte (für H und C)
        self.pred_heads = nn.ModuleDict({
            'H': nn.Linear(out_dim, 1), 
            'C': nn.Linear(out_dim, 1)
        })
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # (1) Transformiere die Knoteneingabefeatures mit den MLP-Encodern.
        enc_x_dict = {}
        for ntype, x in x_dict.items():
            enc_x_dict[ntype] = self.encoder_dict[ntype](x, use_checkpoint=self.use_checkpoint)
        
        # Entscheide, ob Kantenfeatures weitergereicht werden sollen (nur bei GINEConv und NNConv)
        conv_kwargs = {}
        if self.operator_type in ["GINEConv", "NNConv"]:
            if edge_attr_dict is not None and self.edge_proj is not None:
                # Für GINEConv: projiziere die Kantenfeatures auf out_dim
                edge_attr_dict_proj = {}
                for key in edge_attr_dict:
                    edge_attr_dict_proj[key] = self.edge_proj(edge_attr_dict[key])
                conv_kwargs["edge_attr_dict"] = edge_attr_dict_proj
            else:
                conv_kwargs["edge_attr_dict"] = edge_attr_dict
        
        # (2) Wende die HeteroConv-Schichten an.
        h_dict = enc_x_dict
        for conv in self.convs:
            # Hier geben wir die neue Version jedes Mal explizit weiter, um Speicher zu sparen
            new_h_dict = conv(h_dict, edge_index_dict, **conv_kwargs)
            
            # Anwenden von Aktivierung und Dropout
            for ntype in new_h_dict:
                new_h_dict[ntype] = self.act(new_h_dict[ntype])
                new_h_dict[ntype] = self.gnn_dropout(new_h_dict[ntype])
            
            h_dict = new_h_dict
        
        # (3) Erzeuge die Vorhersagen für H und C.
        out_dict = {}
        for ntype in h_dict:
            if ntype in self.pred_heads:
                out_dict[ntype] = self.pred_heads[ntype](h_dict[ntype])
            else:
                out_dict[ntype] = None
        
        return out_dict