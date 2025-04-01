from model import HeteroGNNModel
from dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4
    )

example_data = next(iter(train_loader))
in_dim_dict = {}
for ntype in example_data.node_types:
    if example_data[ntype].x is not None:
        in_dim_dict[ntype] = example_data[ntype].x.size(-1)
    else:
        in_dim_dict[ntype] = 0
    
model = HeteroGNNModel(
    in_dim_dict, 
    hidden_dim=64, 
    out_dim=32
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Anzahl Parameter:", num_params)
