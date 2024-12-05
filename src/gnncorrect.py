import pickle
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import networkx as nx
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    # lade die Graph Daten
    with open('data/all_graphs.pkl', 'rb') as f:
        nx_graphs = pickle.load(f)

    # konvertiere NetworkX zu PyTorch Geometric Format
    # nur Kohlenstoff Atome werden für Vorhersage berücksichtigt
    def nx_to_pyg_data(nx_graph):
        node_features = []
        edge_index = []
        edge_features = []
        target = []
        valid_node_indices = []

        # extrahiere Knoten-Features und Zielwert (shift_high-low)
        for node, data in nx_graph.nodes(data=True):
            feature = [
                data.get('atomic_num', 0), data.get('degree', 0), data.get('shielding_dia', 0), data.get('shielding_para', 0),
                data.get('anisotropy', 0), data.get('at_charge_mull', 0), data.get('at_charge_loew', 0),
                data.get('orb_charge_mull_s', 0), data.get('orb_charge_mull_p', 0), data.get('orb_charge_loew_s', 0),
                data.get('orb_charge_loew_p', 0), data.get('orb_charge_mull_d', 0),
                data.get('orb_charge_loew_d', 0), data.get('BO_loew_av', 0),
                data.get('BO_mayer_av', 0)
            ]
            node_features.append(feature)

            if data.get('atomic_num') == 6:
                target.append(data.get('shift_high-low', 0.0))  # NaN durch 0.0 ersetzt
                valid_node_indices.append(node)
            else:
                target.append(float('nan'))

        # extrahiere Kanten Index und Kanten Features
        for source, target, data in nx_graph.edges(data=True):
            edge_index.append([source, target])
            edge_feature = [
                data.get('bond_type', 0), data.get('is_conjugated', 0), data.get('is_aromatic', 0), data.get('bond_order', 0)
            ]
            edge_features.append(edge_feature)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float).view(-1, 1)  # Zielwerte als (N, 1)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # konvertiere alle Graphen ins PyTorch Geometric-Format
    graph_data_list = [nx_to_pyg_data(g) for g in nx_graphs]

    # filtere Graphen ohne gültige Zielknoten heraus
    data_list_with_targets = [data for data in graph_data_list if data.x.size(0) > 0]

    # train/val Split 80/20
    train_data = data_list_with_targets[:int(0.8 * len(data_list_with_targets))]
    val_data = data_list_with_targets[int(0.8 * len(data_list_with_targets)):]

    # PyTorch Geometric DataLoader
    data_loader_train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    data_loader_val = DataLoader(val_data, batch_size=32, num_workers=4)

    # definiere Modell
    class GNNModel(pl.LightningModule):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
            self.save_hyperparameters()

        def forward(self, x, edge_index, edge_attr):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.fc(x)
            return x  # Ausgabe: (N, 1)

        def training_step(self, batch, batch_idx):
            out = self(batch.x, batch.edge_index, batch.edge_attr)
            target = batch.y
            loss = F.l1_loss(out, target)  # Verlustfunktion(MAE)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            out = self(batch.x, batch.edge_index, batch.edge_attr)
            target = batch.y
            loss = F.l1_loss(out, target)
            self.log('val_loss', loss)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

    # Weights & Biases für Logging
    wandb.init(project='gnn_shift_prediction')

    # trainiere Modell
    model = GNNModel(input_dim=15, hidden_dim=32, output_dim=1)
    trainer = pl.Trainer(max_epochs=50, devices=1, accelerator='auto', logger=pl.loggers.WandbLogger())
    trainer.fit(model, data_loader_train, data_loader_val)

if __name__ == "__main__":
    main()
