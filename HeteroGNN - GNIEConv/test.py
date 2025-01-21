from model import HeteroGNNModel

model = HeteroGNNModel(
    c_in=38,
    h_in=32,
    o_in=6,
    e_in=4,
    hidden_dim=64,
    out_dim=1,
    num_layers=2
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Anzahl Parameter:", num_params)