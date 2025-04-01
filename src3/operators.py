from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv, GATv2Conv, GINEConv, NNConv, TransformerConv 
import torch.nn as nn

OPERATOR_MAP = {
    "GraphConv": GraphConv,
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "SAGEConv": SAGEConv,
    "GATv2Conv": GATv2Conv,
    "TransformerConv": TransformerConv
}

def get_conv_operator(operator_type: str):
    if operator_type == "GINEConv":
        def conv_constructor(in_dim, out_dim, **operator_kwargs):
            # Setze edge_dim auf out_dim, falls nicht anders angegeben.
            operator_kwargs.setdefault('edge_dim', out_dim)
            mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(mlp, **operator_kwargs)
        return conv_constructor
    elif operator_type == "NNConv":
        def conv_constructor(in_dim, out_dim, **operator_kwargs):
            # NNConv benötigt einen zusätzlichen Parameter 'nn'
            if "nn" not in operator_kwargs:
                # Hole 'edge_dim' und entferne ihn aus den kwargs, da er nicht an NNConv weitergegeben werden soll.
                edge_dim = operator_kwargs.pop("edge_dim", None)
                if edge_dim is None:
                    raise ValueError("For NNConv, please provide 'edge_dim' in operator_kwargs.")
                mlp = nn.Sequential(
                    nn.Linear(edge_dim, in_dim * out_dim),
                    nn.ReLU(),
                    nn.Linear(in_dim * out_dim, in_dim * out_dim)
                )
                operator_kwargs["nn"] = mlp
            else:
                # Falls 'nn' explizit gesetzt wurde, entferne 'edge_dim', falls vorhanden.
                operator_kwargs.pop("edge_dim", None)
            return NNConv(in_dim, out_dim, **operator_kwargs)
        return conv_constructor
    elif operator_type in OPERATOR_MAP:
        conv_class = OPERATOR_MAP[operator_type]
        def conv_constructor(in_dim, out_dim, **operator_kwargs):
            return conv_class(in_dim, out_dim, **operator_kwargs)
        return conv_constructor
    else:
        raise ValueError(f"Unbekannter Operator: {operator_type}. Verfügbare Optionen: {list(OPERATOR_MAP.keys()) + ['GINEConv', 'NNConv']}")
