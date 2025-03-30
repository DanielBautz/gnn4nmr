import torch
from torch_geometric.explain import GNNExplainer
import matplotlib.pyplot as plt

def explain_node(model, data, node_type, node_idx, epochs=200):
    """
    Erklärt einen Knoten (des Typs node_type) aus dem gegebenen HeteroData-Objekt
    mittels GNNExplainer.

    Parameter:
      - model: Das trainierte GNN-Modell.
      - data: Ein HeteroData-Objekt (z. B. ein Batch aus dem Testsplit).
      - node_type: Der Knotentyp als String (z. B. "H" oder "C").
      - node_idx: Der Index des zu erklärenden Knotens innerhalb des Knotentyps.
      - epochs: Anzahl der Epochen, die der Explainer trainiert (Default: 200).

    Rückgabe:
      - node_feat_mask: Maske für die Bedeutung der Knoteneingabefeatures.
      - edge_mask: Maske für die Wichtigkeit der Kanten im erklärten Subgraph.
    """
    # Wir verwenden als Subgraph die Kanten, die zwischen Knoten des gleichen Typs bestehen.
    relation = (node_type, "bond", node_type)
    if relation not in data:
        print(f"Relation {relation} nicht gefunden im Datenobjekt.")
        return None, None

    x = data[node_type].x
    edge_index = data[relation].edge_index
    # Falls vorhanden, kann auch edge_attr verwendet werden
    edge_attr = data[relation].edge_attr if hasattr(data[relation], "edge_attr") else None

    # Definiere einen Wrapper, der den Forward-Pfad nur für den gewünschten Knotentyp realisiert.
    def forward_wrapper(x_input, edge_index_input):
        x_dict = {node_type: x_input}
        edge_index_dict = {relation: edge_index_input}
        out = model(x_dict, edge_index_dict)
        # Rückgabe der Vorhersagen für den betrachteten Knotentyp
        return out[node_type]

    explainer = GNNExplainer(forward_wrapper, epochs=epochs)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)

    # Visualisierung des erklärten Subgraphen
    fig, ax = plt.subplots(figsize=(8, 6))
    explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=None, ax=ax)
    plt.title(f"Erklärung für Knoten {node_idx} (Typ: {node_type})")
    plt.show()

    return node_feat_mask, edge_mask

def explain_nodes(model, data, node_type, node_list, epochs=200):
    """
    Erklärt und visualisiert eine Liste von Knoten des gleichen Typs.

    Parameter:
      - model: Das trainierte GNN-Modell.
      - data: Ein HeteroData-Objekt (beispielsweise ein Batch aus dem Testsplit).
      - node_type: Der Knotentyp, der erklärt werden soll (z. B. "H" oder "C").
      - node_list: Liste von Knotenindizes (innerhalb des Knotentyps), die erklärt werden sollen.
      - epochs: Anzahl der Epochen für den GNNExplainer (Default: 200).

    Rückgabe:
      - explanations: Dictionary, das für jeden erklärten Knoten die Feature- und Edge-Masken enthält.
    """
    explanations = {}
    for node_idx in node_list:
        print(f"Erkläre Knoten {node_idx} vom Typ {node_type}...")
        feat_mask, edge_mask = explain_node(model, data, node_type, node_idx, epochs)
        explanations[node_idx] = {"feat_mask": feat_mask, "edge_mask": edge_mask}
    return explanations
