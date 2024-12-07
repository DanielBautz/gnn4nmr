import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import GNNModel
from dataloader import get_data_loaders
import wandb

def train_model(data_path, input_dim=15, hidden_dim=32, output_dim=1, max_epochs=50):
    wandb_logger = WandbLogger(project="gnn_shift_prediction")
    train_loader, val_loader = get_data_loaders(data_path)

    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator='auto', logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
