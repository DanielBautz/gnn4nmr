import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

# Neu: periodictable importieren
import periodictable

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        path = os.path.join(root, 'data.pkl')
        self.data, self.slices = torch.load(path)

    @staticmethod
    def from_pickle(pickle_path, root, transform=None, pre_transform=None):
        """Konvertiert die in pickle_path enthaltenen NetworkX-Graphen
           in PyTorch Geometric Data-Objekte und speichert sie als data.pkl."""
        with open(pickle_path, 'rb') as f:
            graphs = pickle.load(f)  # Liste von networkx.Graphen

        data_list = []
        for G in graphs:
            data_obj = MoleculeDataset._graph_to_data(G)
            data_list.append(data_obj)

        data, slices = MoleculeDataset.collate(data_list)
        os.makedirs(root, exist_ok=True)
        torch.save((data, slices), os.path.join(root, 'data.pkl'))
        return MoleculeDataset(root, transform, pre_transform)

    @staticmethod
    def _graph_to_data(G: nx.Graph):
        """
        Konvertiert einen NetworkX-Graphen G in ein PyG Data-Objekt.
        Für das Target y nutzen wir 2 Kanäle:
          y[i, 0] = Shift für Kohlenstoff-Atome (Z=6)
          y[i, 1] = Shift für Wasserstoff-Atome (Z=1)
        Andere Atome erhalten NaN in beiden Kanälen.
        """

        ALL_FEATURES = [
            "element",          # Ordnungszahl (z. B. 1=H, 6=C, 7=N...)
            "pos_x", "pos_y", "pos_z",
            "mass",
            "formal_charge",
            "degree",

            # Sowohl C als auch H haben shift_low
            "shift_low",
            "CN(X)",

            # Kohlenstoff-spezifische Count-Features
            "no_CH", "no_CC", "no_CN", "no_CO", "no_CYH", "no_CYC", "no_CYN", "no_CYO",

            # Wasserstoff-spezifische Count-Features
            "no_HCH", "no_HYH", "no_HYC", "no_HYN", "no_HYO", "dist_HC", "shift_low_neighbor_C",

            # NMR-Eigenschaften
            "shielding_dia",
            "shielding_para",
            "span",
            "skew",
            "asymmetry",
            "anisotropy",

            # Atomladungen
            "at_charge_mull",
            "at_charge_loew",

            # Orbital Ladungen (Mulliken)
            "orb_charge_mull_s",
            "orb_charge_mull_p",
            "orb_charge_mull_d",
            "orb_stdev_mull_p",

            # Orbital Ladungen (Loewdin)
            "orb_charge_loew_s",
            "orb_charge_loew_p",
            "orb_charge_loew_d",
            "orb_stdev_loew_p",

            # Bindungsordnungen
            "BO_loew_sum",
            "BO_loew_av",
            "BO_mayer_sum",
            "BO_mayer_av",
            "mayer_VA",

            # H-spezifische BO-Features
            "BO_loew",
            "BO_mayer",
        ]

        node_features = []
        target_values = []

        for node, attrs in G.nodes(data=True):
            # Lese das Elementsymbol, z. B. "C", "H", "N"
            elem_symbol = attrs.get('element', 'X')  # Fallback "X" falls nicht vorhanden

            # Versuche mithilfe periodictable die Ordnungszahl zu bekommen
            elem_symbol_cap = elem_symbol.capitalize()  # "C" -> "C", "h" -> "H", etc.
            try:
                # getattr(periodictable, "C") -> periodictable.c
                # und dann .number -> 6
                element_z = getattr(periodictable, elem_symbol_cap).number
            except AttributeError:
                element_z = 0  # Unbekanntes Symbol -> 0

            # SHIFT (Targets für C und H)
            shift_C = float('nan')
            shift_H = float('nan')

            if element_z == 6:  # Kohlenstoff
                shift_C = attrs.get('shift_high-low', float('nan'))
            elif element_z == 1:  # Wasserstoff
                shift_H = attrs.get('shift_high-low', float('nan'))

            # Features (x)
            fdict = {}
            # element -> Ordnungszahl
            fdict["element"] = float(element_z)

            # pos_x, pos_y, pos_z
            pos = attrs.get('pos', [0.0, 0.0, 0.0])
            fdict["pos_x"] = float(pos[0])
            fdict["pos_y"] = float(pos[1])
            fdict["pos_z"] = float(pos[2])

            # mass, formal_charge, degree
            fdict["mass"] = float(attrs.get('mass', 0.0))
            fdict["formal_charge"] = float(attrs.get('formal_charge', 0.0))
            fdict["degree"] = float(attrs.get('degree', 0.0))

            # shift_low, CN(X)
            fdict["shift_low"] = float(attrs.get('shift_low', 0.0))
            fdict["CN(X)"] = float(attrs.get('CN(X)', 0.0))

            # Kohlenstoff-spezifische Count-Features
            fdict["no_CH"] = float(attrs.get('no_CH', 0.0))
            fdict["no_CC"] = float(attrs.get('no_CC', 0.0))
            fdict["no_CN"] = float(attrs.get('no_CN', 0.0))
            fdict["no_CO"] = float(attrs.get('no_CO', 0.0))
            fdict["no_CYH"] = float(attrs.get('no_CYH', 0.0))
            fdict["no_CYC"] = float(attrs.get('no_CYC', 0.0))
            fdict["no_CYN"] = float(attrs.get('no_CYN', 0.0))
            fdict["no_CYO"] = float(attrs.get('no_CYO', 0.0))

            # Wasserstoff-spezifische Count-Features
            fdict["no_HCH"] = float(attrs.get('no_HCH', 0.0))
            fdict["no_HYH"] = float(attrs.get('no_HYH', 0.0))
            fdict["no_HYC"] = float(attrs.get('no_HYC', 0.0))
            fdict["no_HYN"] = float(attrs.get('no_HYN', 0.0))
            fdict["no_HYO"] = float(attrs.get('no_HYO', 0.0))
            fdict["dist_HC"] = float(attrs.get('dist_HC', 0.0))
            fdict["shift_low_neighbor_C"] = float(attrs.get('shift_low_neighbor_C', 0.0))

            # NMR-Eigenschaften
            fdict["shielding_dia"] = float(attrs.get('shielding_dia', 0.0))
            fdict["shielding_para"] = float(attrs.get('shielding_para', 0.0))
            fdict["span"] = float(attrs.get('span', 0.0))
            fdict["skew"] = float(attrs.get('skew', 0.0))
            fdict["asymmetry"] = float(attrs.get('asymmetry', 0.0))
            fdict["anisotropy"] = float(attrs.get('anisotropy', 0.0))

            # Atomladungen
            fdict["at_charge_mull"] = float(attrs.get('at_charge_mull', 0.0))
            fdict["at_charge_loew"] = float(attrs.get('at_charge_loew', 0.0))

            # Orbital Ladungen (Mulliken)
            fdict["orb_charge_mull_s"] = float(attrs.get('orb_charge_mull_s', 0.0))
            fdict["orb_charge_mull_p"] = float(attrs.get('orb_charge_mull_p', 0.0))
            fdict["orb_charge_mull_d"] = float(attrs.get('orb_charge_mull_d', 0.0))
            fdict["orb_stdev_mull_p"] = float(attrs.get('orb_stdev_mull_p', 0.0))

            # Orbital Ladungen (Loewdin)
            fdict["orb_charge_loew_s"] = float(attrs.get('orb_charge_loew_s', 0.0))
            fdict["orb_charge_loew_p"] = float(attrs.get('orb_charge_loew_p', 0.0))
            fdict["orb_charge_loew_d"] = float(attrs.get('orb_charge_loew_d', 0.0))
            fdict["orb_stdev_loew_p"] = float(attrs.get('orb_stdev_loew_p', 0.0))

            # Bindungsordnungen
            fdict["BO_loew_sum"] = float(attrs.get('BO_loew_sum', 0.0))
            fdict["BO_loew_av"] = float(attrs.get('BO_loew_av', 0.0))
            fdict["BO_mayer_sum"] = float(attrs.get('BO_mayer_sum', 0.0))
            fdict["BO_mayer_av"] = float(attrs.get('BO_mayer_av', 0.0))
            fdict["mayer_VA"] = float(attrs.get('mayer_VA', 0.0))

            # H-spezifische BO-Features
            fdict["BO_loew"] = float(attrs.get('BO_loew', 0.0))
            fdict["BO_mayer"] = float(attrs.get('BO_mayer', 0.0))

            # Feature-Vektor in der Reihenfolge ALL_FEATURES
            x_atom = []
            for feat_name in ALL_FEATURES:
                x_atom.append(fdict.get(feat_name, 0.0))
            x_atom = torch.tensor(x_atom, dtype=torch.float)
            node_features.append(x_atom)

            # Target [shift_C, shift_H]
            target_values.append([shift_C, shift_H])

        # x: Tensor [N, num_features]
        x = torch.stack(node_features, dim=0)

        # Edges extrahieren (bidirektional)
        edges = []
        for u, v in G.edges():
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Targets: [N, 2]
        y = torch.tensor(target_values, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
