# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GINEConv

def MLP(in_dim, out_dim, hidden_dim=64, num_layers=2):
    layers = []
    dim = in_dim
    for i in range(num_layers-1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)

class HeteroGNNModel(nn.Module):
    def __init__(self, c_in=38, h_in=32, o_in=6, e_in=4, hidden_dim=64, out_dim=1, num_layers=2, aggr='sum'):
        super().__init__()
        
        # MLP-Encoder für Knoten
        self.c_mlp = MLP(c_in, hidden_dim, hidden_dim=hidden_dim, num_layers=2)
        self.h_mlp = MLP(h_in, hidden_dim, hidden_dim=hidden_dim, num_layers=2)
        self.o_mlp = MLP(o_in, hidden_dim, hidden_dim=hidden_dim, num_layers=2)

        # MLP für Edge-Features (wird in GINEConv als edge_nn genutzt)
        self.edge_mlp = MLP(e_in, hidden_dim, hidden_dim=hidden_dim, num_layers=2)

        # MLP für GINEConv node_update
        # GINEConv benötigt ein nn-Modul, das auf Knotenfeatures angewendet wird (ohne Edge-Infos),
        # aber oft nimmt man hier einfach eine lineare Schicht oder ein MLP.
        self.node_mlp = MLP(hidden_dim, hidden_dim, hidden_dim=hidden_dim, num_layers=2)

        # Wir definieren die Edge-Typen:
        # Da wir ein heterogenes Graph haben, müssen wir für jede Kantenrelation einen GINEConv angeben.
        relations = [
            ('C','C'), ('C','H'), ('H','H'), ('C','O'), ('H','O'), ('O','O')
        ]
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for (src, dst) in relations:
                conv = GINEConv(nn=self.node_mlp, edge_dim=hidden_dim, train_eps=True)
                conv_dict[(src,dst)] = conv
            hetero_conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(hetero_conv)
        
        self.relu = nn.ReLU()
        
        # Output-Layer für C und H
        self.c_out = nn.Linear(hidden_dim, out_dim)
        self.h_out = nn.Linear(hidden_dim, out_dim)
        # O bekommt kein Output, da wir dort nicht vorhersagen.

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Eingangsprojektions-MLPs aufrufen
        if x_dict['C'].numel() > 0:
            x_dict['C'] = self.c_mlp(x_dict['C'])
        if x_dict['H'].numel() > 0:
            x_dict['H'] = self.h_mlp(x_dict['H'])
        if x_dict['O'].numel() > 0:
            x_dict['O'] = self.o_mlp(x_dict['O'])

        # Für die Convs müssen wir pro Relation edge_attr weitergeben.
        # GINEConv nimmt edge_attr übergeben. Wir transformieren edge_attr vorher mit edge_mlp.
        # Damit nicht bei jeder Iteration neu ein nn-Modul aufgerufen wird, machen wir das inline.
        
        # Edge-Attr durch edge_mlp schicken
        for et in edge_attr_dict:
            if edge_attr_dict[et].numel() > 0:
                edge_attr_dict[et] = self.edge_mlp(edge_attr_dict[et])

        # Mehrere Lagen von HeteroConv
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            # Aktivierung
            x_dict = {k: self.relu(v) for k,v in x_dict.items()}
        
        # Outputs für C und H
        out_c = self.c_out(x_dict['C']) if x_dict['C'].shape[0] > 0 else torch.empty(0,1)
        out_h = self.h_out(x_dict['H']) if x_dict['H'].shape[0] > 0 else torch.empty(0,1)
        
        return out_c, out_h
