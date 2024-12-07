import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

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
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        target = batch.y
        loss = F.l1_loss(out, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        target = batch.y
        loss = F.l1_loss(out, target)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
