import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import numpy as np


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        # Wir gehen davon aus, dass in `root` eine `all_graphs.pkl` Datei liegt.
        # Die Datei sollte eine Liste von (G, labels) oder nur G enthalten.
        path = os.path.join(root, 'all_graphs.pkl')
        self.data, self.slices = torch.load(path)

    @staticmethod
    def from_pickle(pickle_path, root, transform=None, pre_transform=None):
        # Lädt die nx-Graphen und konvertiert sie in PyG Data-Objekte
        with open(pickle_path, 'rb') as f:
            graphs = pickle.load(f)

        # graphs sollte eine Liste von networkx Graphen sein.
        data_list = []
        for G in graphs:
            data_obj = MoleculeDataset._graph_to_data(G)
            data_list.append(data_obj)

        # InMemoryDataset-Format
        data, slices = MoleculeDataset.collate(data_list)
        torch.save((data, slices), os.path.join(root, 'all_graphs.pkl'))
        return MoleculeDataset(root, transform, pre_transform)

    @staticmethod
    def _graph_to_data(G: nx.Graph):
        # Extrahiert Knotenfeatures und Targets
        # Annahme: Jeder Knoten hat ein dict mit Keys wie 'element', 'pos', etc.

        # Elemente -> Eine einfache Kategorisierung: C=0, H=1, Other=2
        # Features unterscheiden sich je nach Atomtyp. Wir müssen sie auf eine gemeinsame Dimension bringen.
        # Hier: Wir sammeln alle möglichen Features und füllen Fehlendes mit Nullen auf.
        
        # Beispiel: Wir definieren ein gemeinsames Feature-Set (Vereinfachung)
        # Realistisch müsste man alle Features, die genannt wurden, in einen festen Vektor packen.
        # Hier machen wir beispielhaft nur einige wenige, um das Prinzip zu demonstrieren.
        # In der Realität sollten Sie den gesamten Feature-Katalog in ein sinnvolles Format bringen.

        # Beispiel Feature-Auswahl für Demonstration (bitte anpassen!):
        # Für C/H: element (numerisch), mass, formal_charge, degree, shift_high-low
        # Für Other: element, mass, formal_charge, degree (kein shift_high-low)
        # Dazu ggf. weitere Features in ein großes Vektorarray packen.
        
        node_features = []
        target_values = []
        for node, attrs in G.nodes(data=True):
            elem = attrs.get('element', 'X')
            if elem == 'C':
                elem_id = 0
            elif elem == 'H':
                elem_id = 1
            else:
                elem_id = 2

            # Beispiel-Features extrahieren:
            mass = attrs.get('mass', 0.0)
            formal_charge = attrs.get('formal_charge', 0.0)
            degree = attrs.get('degree', 0.0)
            
            # shift_high-low kann für C und H vorhanden sein:
            if elem_id in [0,1]:
                shift = attrs.get('shift_high-low', None)
            else:
                shift = None
            
            # Weitere Features können hier angefügt werden. In der Realität sollten alle genannten Features
            # in ein großes Feature-Array codiert werden. Hier nur ein Minimalbeispiel.
            
            # Beispiel: Node Feature Vektor
            # Feste Länge, z.B. 4: [elem_id, mass, formal_charge, degree]
            x = torch.tensor([float(elem_id), mass, formal_charge, degree], dtype=torch.float)
            node_features.append(x)
            
            # Target nur, wenn C oder H
            if shift is not None:
                target_values.append(float(shift))
            else:
                target_values.append(float('nan'))  # nan für "kein Target"

        x = torch.stack(node_features, dim=0)

        # Edges
        edges = []
        for u, v in G.edges():
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Targets als Tensor
        y = torch.tensor(target_values, dtype=torch.float)

        # Data-Objekt
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
