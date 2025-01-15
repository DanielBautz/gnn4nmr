import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, num_layers=2):
        """
        out_channels=2 bedeutet:
          Channel 0 -> Shift für Kohlenstoff-Atome
          Channel 1 -> Shift für Wasserstoff-Atome
        """
        super(GNNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.lin(x)  # Ausgabe (N, 2)
        return x
