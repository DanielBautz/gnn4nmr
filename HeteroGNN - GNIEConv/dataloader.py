# dataloader.py
import os
import pickle
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.loader import DataLoader

class MoleculeHeteroDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['all_graphs.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        with open(os.path.join(self.root, 'all_graphs.pkl'), 'rb') as f:
            all_nx_graphs = pickle.load(f)
        
        # Mapping für Elemente
        element_map = {
            'C': 0.0,
            'H': 1.0
            # Alle anderen Elemente -> 2.0 (z. B. O, N, ...), wir nutzen 2.0 als "Others"
        }

        def element_to_num(el):
            return element_map.get(el, 2.0)  # wenn unbekannt, 2.0 als "Other"

        # Mappings für bond_type
        # Passen Sie diese Mappings an Ihre Daten an.
        bond_type_map = {
            'SINGLE': 1.0,
            'DOUBLE': 2.0,
            'TRIPLE': 3.0,
            'AROMATIC': 4.0
            # fügen Sie weitere Typen hinzu falls nötig
        }

        # Mappings für bond_dir
        bond_dir_map = {
            'NONE': 0.0,
            'UP': 1.0,
            'DOWN': 2.0
            # fügen Sie weitere Richtungen hinzu falls nötig
        }

        def bool_to_float(b):
            return 1.0 if b else 0.0

        # Anpassungen in den Feature-Funktionen:
        def get_c_features(attrs):
            el = element_to_num(attrs['element'])
            idx = float(attrs['atom_idx'])
            x, y, z = attrs['pos']
            mass = float(attrs['mass'])
            fc = float(attrs['formal_charge'])
            deg = float(attrs['degree'])
            shl = float(attrs['shift_high-low'])
            sl = float(attrs['shift_low'])
            cnx = float(attrs['CN(X)'])
            return torch.tensor([
                el, idx, x, y, z, mass, fc, deg, shl, sl, cnx,
                float(attrs['no_CH']), float(attrs['no_CC']), float(attrs['no_CN']), float(attrs['no_CO']),
                float(attrs['no_CYH']), float(attrs['no_CYC']), float(attrs['no_CYN']), float(attrs['no_CYO']),
                float(attrs['shielding_dia']), float(attrs['shielding_para']), float(attrs['span']), float(attrs['skew']),
                float(attrs['asymmetry']), float(attrs['anisotropy']), float(attrs['at_charge_mull']), float(attrs['at_charge_loew']),
                float(attrs['orb_charge_mull_s']), float(attrs['orb_charge_mull_p']), float(attrs['orb_charge_mull_d']),
                float(attrs['orb_stdev_mull_p']),
                float(attrs['orb_charge_loew_s']), float(attrs['orb_charge_loew_p']), float(attrs['orb_charge_loew_d']),
                float(attrs['orb_stdev_loew_p']),
                float(attrs['BO_loew_sum']), float(attrs['BO_loew_av']), float(attrs['BO_mayer_sum']), float(attrs['BO_mayer_av']),
                float(attrs['mayer_VA'])
            ], dtype=torch.float)

        def get_h_features(attrs):
            el = element_to_num(attrs['element'])
            idx = float(attrs['atom_idx'])
            x, y, z = attrs['pos']
            mass = float(attrs['mass'])
            fc = float(attrs['formal_charge'])
            deg = float(attrs['degree'])
            shl = float(attrs['shift_high-low'])
            sl = float(attrs['shift_low'])
            cnx = float(attrs['CN(X)'])
            return torch.tensor([
                el, idx, x, y, z, mass, fc, deg, shl, sl, cnx,
                float(attrs['no_HCH']), float(attrs['no_HYH']), float(attrs['no_HYC']), float(attrs['no_HYN']), float(attrs['no_HYO']),
                float(attrs['dist_HC']), float(attrs['shift_low_neighbor_C']), float(attrs['shielding_dia']), float(attrs['shielding_para']),
                float(attrs['span']), float(attrs['skew']), float(attrs['asymmetry']), float(attrs['anisotropy']),
                float(attrs['at_charge_mull']), float(attrs['at_charge_loew']),
                float(attrs['orb_charge_mull_s']), float(attrs['orb_charge_mull_p']),
                float(attrs['orb_charge_loew_s']), float(attrs['orb_charge_loew_p']),
                float(attrs['BO_loew']), float(attrs['BO_mayer']), float(attrs['mayer_VA'])
            ], dtype=torch.float)

        def get_o_features(attrs):
            el = element_to_num(attrs['element'])
            idx = float(attrs['atom_idx'])
            x, y, z = attrs['pos']
            mass = float(attrs['mass'])
            fc = float(attrs['formal_charge'])
            deg = float(attrs['degree'])
            # wie zuvor vereinbart nehmen wir den radius statt x,y,z einzeln um auf 6 Features zu kommen
            radius = (x**2 + y**2 + z**2)**0.5
            return torch.tensor([el, idx, radius, mass, fc, deg], dtype=torch.float)

        def get_target_shift(attrs):
            return torch.tensor([float(attrs['shift_high-low'])], dtype=torch.float)

        def get_edge_features(attrs):
            # Hier Strings in Zahlen umwandeln
            bt = bond_type_map.get(attrs['bond_type'], 0.0)  # 0.0 falls unbekannt
            ia = bool_to_float(attrs['is_aromatic'])
            bd = bond_dir_map.get(attrs['bond_dir'], 0.0)  # 0.0 falls unbekannt
            bo = float(attrs['bond_order'])
            return torch.tensor([bt, ia, bd, bo], dtype=torch.float)

        data_list = []
        for g in all_nx_graphs:
            # Elemente aufteilen
            c_nodes = []
            h_nodes = []
            o_nodes = []
            for n, node_attrs in g.nodes(data=True):
                el = node_attrs['element']
                if el == 'C':
                    c_nodes.append(n)
                elif el == 'H':
                    h_nodes.append(n)
                else:
                    o_nodes.append(n)

            c_mapping = {old_id: i for i, old_id in enumerate(c_nodes)}
            h_mapping = {old_id: i for i, old_id in enumerate(h_nodes)}
            o_mapping = {old_id: i for i, old_id in enumerate(o_nodes)}

            c_x = [get_c_features(g.nodes[n]) for n in c_nodes]
            h_x = [get_h_features(g.nodes[n]) for n in h_nodes]
            o_x = [get_o_features(g.nodes[n]) for n in o_nodes]

            c_x = torch.stack(c_x) if len(c_x)>0 else torch.zeros((0,38))
            h_x = torch.stack(h_x) if len(h_x)>0 else torch.zeros((0,32))
            o_x = torch.stack(o_x) if len(o_x)>0 else torch.zeros((0,6))

            c_y = [get_target_shift(g.nodes[n]) for n in c_nodes]
            h_y = [get_target_shift(g.nodes[n]) for n in h_nodes]
            o_y = [torch.tensor([0.0]) for _ in o_nodes] # Dummy

            c_y = torch.stack(c_y) if len(c_y)>0 else torch.zeros((0,1))
            h_y = torch.stack(h_y) if len(h_y)>0 else torch.zeros((0,1))
            o_y = torch.stack(o_y) if len(o_y)>0 else torch.zeros((0,1))

            c_c_edges, c_c_eattr = [], []
            c_h_edges, c_h_eattr = [], []
            c_o_edges, c_o_eattr = [], []
            h_h_edges, h_h_eattr = [], []
            h_o_edges, h_o_eattr = [], []
            o_o_edges, o_o_eattr = [], []

            def add_edge(u,v,e_feat):
                u_type = 'C' if u in c_mapping else ('H' if u in h_mapping else 'O')
                v_type = 'C' if v in c_mapping else ('H' if v in h_mapping else 'O')
                if u_type=='C' and v_type=='C':
                    c_c_edges.append((c_mapping[u], c_mapping[v]))
                    c_c_eattr.append(e_feat)
                elif u_type=='C' and v_type=='H':
                    c_h_edges.append((c_mapping[u], h_mapping[v]))
                    c_h_eattr.append(e_feat)
                elif u_type=='H' and v_type=='C':
                    c_h_edges.append((c_mapping[v], h_mapping[u]))
                    c_h_eattr.append(e_feat)
                elif u_type=='C' and v_type=='O':
                    c_o_edges.append((c_mapping[u], o_mapping[v]))
                    c_o_eattr.append(e_feat)
                elif u_type=='O' and v_type=='C':
                    c_o_edges.append((c_mapping[v], o_mapping[u]))
                    c_o_eattr.append(e_feat)
                elif u_type=='H' and v_type=='H':
                    h_h_edges.append((h_mapping[u], h_mapping[v]))
                    h_h_eattr.append(e_feat)
                elif u_type=='H' and v_type=='O':
                    h_o_edges.append((h_mapping[u], o_mapping[v]))
                    h_o_eattr.append(e_feat)
                elif u_type=='O' and v_type=='H':
                    h_o_edges.append((h_mapping[v], o_mapping[u]))
                    h_o_eattr.append(e_feat)
                elif u_type=='O' and v_type=='O':
                    o_o_edges.append((o_mapping[u], o_mapping[v]))
                    o_o_eattr.append(e_feat)

            for u,v,edge_attrs in g.edges(data=True):
                e_feat = get_edge_features(edge_attrs)
                # (u->v)
                add_edge(u,v,e_feat)
                # (v->u)
                add_edge(v,u,e_feat)

            def to_tensor(edges, eattr):
                if len(edges)>0:
                    return torch.tensor(edges, dtype=torch.long).t(), torch.stack(eattr)
                else:
                    return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4), dtype=torch.float)

            c_c_edge_index, c_c_edge_attr = to_tensor(c_c_edges, c_c_eattr)
            c_h_edge_index, c_h_edge_attr = to_tensor(c_h_edges, c_h_eattr)
            c_o_edge_index, c_o_edge_attr = to_tensor(c_o_edges, c_o_eattr)
            h_h_edge_index, h_h_edge_attr = to_tensor(h_h_edges, h_h_eattr)
            h_o_edge_index, h_o_edge_attr = to_tensor(h_o_edges, h_o_eattr)
            o_o_edge_index, o_o_edge_attr = to_tensor(o_o_edges, o_o_eattr)

            data_hetero = HeteroData()
            data_hetero['C'].x = c_x
            data_hetero['H'].x = h_x
            data_hetero['O'].x = o_x

            data_hetero['C'].y = c_y
            data_hetero['H'].y = h_y
            data_hetero['O'].y = o_y

            if c_c_edge_index.size(1)>0:
                data_hetero['C','C'].edge_index = c_c_edge_index
                data_hetero['C','C'].edge_attr = c_c_edge_attr
            if c_h_edge_index.size(1)>0:
                data_hetero['C','H'].edge_index = c_h_edge_index
                data_hetero['C','H'].edge_attr = c_h_edge_attr
            if h_h_edge_index.size(1)>0:
                data_hetero['H','H'].edge_index = h_h_edge_index
                data_hetero['H','H'].edge_attr = h_h_edge_attr
            if c_o_edge_index.size(1)>0:
                data_hetero['C','O'].edge_index = c_o_edge_index
                data_hetero['C','O'].edge_attr = c_o_edge_attr
            if h_o_edge_index.size(1)>0:
                data_hetero['H','O'].edge_index = h_o_edge_index
                data_hetero['H','O'].edge_attr = h_o_edge_attr
            if o_o_edge_index.size(1)>0:
                data_hetero['O','O'].edge_index = o_o_edge_index
                data_hetero['O','O'].edge_attr = o_o_edge_attr

            data_list.append(data_hetero)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_dataloaders(root='data', batch_size=16, split_ratio=(0.8, 0.1, 0.1), shuffle=True):
    dataset = MoleculeHeteroDataset(root=root)
    n = len(dataset)
    train_end = int(split_ratio[0]*n)
    val_end = int((split_ratio[0]+split_ratio[1])*n)
    
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
