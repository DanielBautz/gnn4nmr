import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
import torch.nn as nn

class NodeSpecificMLPs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeSpecificMLPs, self).__init__()
        self.carbon_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.hydrogen_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.other_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, atomic_nums):
        outputs = torch.zeros_like(x[:, :1])

        # Kohlenstoff
        mask_carbon = (atomic_nums == 6)
        outputs[mask_carbon] = self.carbon_mlp(x[mask_carbon])

        # Wasserstoff
        mask_hydrogen = (atomic_nums == 1)
        outputs[mask_hydrogen] = self.hydrogen_mlp(x[mask_hydrogen])

        # Andere
        mask_other = ~mask_carbon & ~mask_hydrogen
        outputs[mask_other] = self.other_mlp(x[mask_other])

        return outputs


class GNNModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_mlps = NodeSpecificMLPs(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x, edge_index, edge_attr, atomic_nums):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.node_mlps(x, atomic_nums)
        return x

    def training_step(self, batch, batch_idx):
        atomic_nums = batch.x[:, 0].long()  # `atomic_num` ist in der ersten Spalte von x
        out = self(batch.x, batch.edge_index, batch.edge_attr, atomic_nums)
        target = batch.y
        loss = F.l1_loss(out, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        atomic_nums = batch.x[:, 0].long()
        out = self(batch.x, batch.edge_index, batch.edge_attr, atomic_nums)
        target = batch.y
        loss = F.l1_loss(out, target)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

