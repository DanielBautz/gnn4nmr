from model import GNNModel

model = GNNModel(
    in_channels=48, 
    hidden_channels= 64, 
    out_channels=1, 
    num_layers=2
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainierbare Parameter:", num_params)