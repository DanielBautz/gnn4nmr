from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv

OPERATOR_MAP = {
    "GraphConv": GraphConv,
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "SAGEConv": SAGEConv
}

def get_conv_operator(operator_type: str):
    if operator_type in OPERATOR_MAP:
        return OPERATOR_MAP[operator_type]
    else:
        raise ValueError(f"Unbekannter Operator: {operator_type}. Verfügbare Optionen: {list(OPERATOR_MAP.keys())}")
