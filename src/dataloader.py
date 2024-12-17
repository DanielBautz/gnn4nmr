import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        path = os.path.join(root, 'data.pkl')
        self.data, self.slices = torch.load(path)

    @staticmethod
    def from_pickle(pickle_path, root, transform=None, pre_transform=None):
        """ Konvertiert die in pickle_path enthaltenen NetworkX-Graphen
            in PyTorch Geometric Data-Objekte und speichert sie als data.pkl. """
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
        """ Konvertiert einen NetworkX-Graphen G in ein PyG Data-Objekt,
            wobei alle relevanten Features berücksichtigt werden. """

        # geordnete Liste aller Features, die im Feature-Vektor x landen sollen.
        # 'shift_high-low' als Target y
        # pos wird zerlegt in pos_x, pos_y, pos_z
        ALL_FEATURES = [
            "element_id",       # numerische Kodierung: C=0, H=1, other=2
            "atom_idx",
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

            # Gemeinsame NMR-Eigenschaften
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

            # Bindungsordnungen (Loewdin/Mayer)
            "BO_loew_sum",
            "BO_loew_av",
            "BO_mayer_sum",
            "BO_mayer_av",
            "mayer_VA",

            # H-spezifische BO-Features (einfach)
            "BO_loew",
            "BO_mayer",
        ]

        node_features = []
        target_values = []

        for node, attrs in G.nodes(data=True):
            # Bestimme element_id (0=C, 1=H, 2=Other)
            elem = attrs.get('element', 'X')
            if elem == 'C':
                elem_id = 0
            elif elem == 'H':
                elem_id = 1
            else:
                elem_id = 2

            # SHIFT (Target)
            if elem_id in [0, 1]:  # C oder H
                shift = attrs.get('shift_high-low', float('nan'))
            else:
                shift = float('nan')  # Andere Atome => kein shift

            # Baue Feature-Vektor x für diesen Knoten
            x_vec = []
            # Key-Value Paare in einem dictionary, um einfach nachzuschauen, ob ein Feature da ist. Sonst 0. 
            # pos separat in pos_x, pos_y, pos_z
            fdict = {}

            # element_id
            fdict["element_id"] = float(elem_id)
            # atom_idx
            fdict["atom_idx"] = float(attrs.get('atom_idx', 0))
            # pos_x, pos_y, pos_z
            pos = attrs.get('pos', [0.0, 0.0, 0.0])
            fdict["pos_x"] = float(pos[0])
            fdict["pos_y"] = float(pos[1])
            fdict["pos_z"] = float(pos[2])
            # mass, formal_charge, degree
            fdict["mass"] = float(attrs.get('mass', 0.0))
            fdict["formal_charge"] = float(attrs.get('formal_charge', 0.0))
            fdict["degree"] = float(attrs.get('degree', 0.0))

            # shift_low
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

            # BO (Loewdin/Mayer) - Carbon-spezifisch
            fdict["BO_loew_sum"] = float(attrs.get('BO_loew_sum', 0.0))
            fdict["BO_loew_av"] = float(attrs.get('BO_loew_av', 0.0))
            fdict["BO_mayer_sum"] = float(attrs.get('BO_mayer_sum', 0.0))
            fdict["BO_mayer_av"] = float(attrs.get('BO_mayer_av', 0.0))
            fdict["mayer_VA"] = float(attrs.get('mayer_VA', 0.0))

            # H-spezifische BO-Features (einfach)
            fdict["BO_loew"] = float(attrs.get('BO_loew', 0.0))
            fdict["BO_mayer"] = float(attrs.get('BO_mayer', 0.0))

            # Jetzt bauen wir den Feature-Vektor in der richtigen Reihenfolge zusammen:
            x_atom = []
            for feat_name in ALL_FEATURES:
                x_atom.append(fdict.get(feat_name, 0.0))

            x_atom = torch.tensor(x_atom, dtype=torch.float)
            node_features.append(x_atom)

            # shift_high-low als Target
            target_values.append(float(shift))

        x = torch.stack(node_features, dim=0)

        # Edges extrahieren (bidirektional)
        edges = []
        for u, v in G.edges():
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        y = torch.tensor(target_values, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
