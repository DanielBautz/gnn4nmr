import os
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

# Liste aller möglichen Elemente für One-Hot-Encoding
ALL_ELEMENTS = [
    "H", "C", "Li", "B", "N", "O", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl"
]

def get_element_onehot(elem: str):
    one_hot = [0.0] * len(ALL_ELEMENTS)
    if elem in ALL_ELEMENTS:
        idx = ALL_ELEMENTS.index(elem)
        one_hot[idx] = 1.0
    return one_hot

def get_h_features(attrs):
    feats = []
    elem = attrs.get('element', 'H')
    feats.extend(get_element_onehot(elem))
    #feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    feats.append(attrs.get('shift_low', 0.0))
    feats.append(attrs.get('CN(X)', 0.0))
    #feats.append(attrs.get('no_HCH', 0.0))
    #feats.append(attrs.get('no_HYH', 0.0))
    #feats.append(attrs.get('no_HYC', 0.0))
    #feats.append(attrs.get('no_HYN', 0.0))
    #feats.append(attrs.get('no_HYO', 0.0))
    #feats.append(attrs.get('dist_HC', 0.0))
    #feats.append(attrs.get('shift_low_neighbor_C', 0.0))
    feats.append(attrs.get('shielding_dia', 0.0))
    feats.append(attrs.get('shielding_para', 0.0))
    feats.append(attrs.get('span', 0.0))
    feats.append(attrs.get('skew', 0.0))
    feats.append(attrs.get('asymmetry', 0.0))
    feats.append(attrs.get('anisotropy', 0.0))
    feats.append(attrs.get('at_charge_mull', 0.0))
    feats.append(attrs.get('at_charge_loew', 0.0))
    feats.append(attrs.get('orb_charge_mull_s', 0.0))
    feats.append(attrs.get('orb_charge_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_s', 0.0))
    feats.append(attrs.get('orb_charge_loew_p', 0.0))
    feats.append(attrs.get('BO_loew', 0.0))
    feats.append(attrs.get('BO_mayer', 0.0))
    feats.append(attrs.get('mayer_VA', 0.0))
    return feats

def get_c_features(attrs):
    feats = []
    elem = attrs.get('element', 'C')
    feats.extend(get_element_onehot(elem))
    #feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    feats.append(attrs.get('shift_low', 0.0))
    feats.append(attrs.get('CN(X)', 0.0))
    #feats.append(attrs.get('no_CH', 0.0))
    #feats.append(attrs.get('no_CC', 0.0))
    #feats.append(attrs.get('no_CN', 0.0))
    #feats.append(attrs.get('no_CO', 0.0))
    #feats.append(attrs.get('no_CYH', 0.0))
    #feats.append(attrs.get('no_CYC', 0.0))
    #feats.append(attrs.get('no_CYN', 0.0))
    #feats.append(attrs.get('no_CYO', 0.0))
    feats.append(attrs.get('shielding_dia', 0.0))
    feats.append(attrs.get('shielding_para', 0.0))
    feats.append(attrs.get('span', 0.0))
    feats.append(attrs.get('skew', 0.0))
    feats.append(attrs.get('asymmetry', 0.0))
    feats.append(attrs.get('anisotropy', 0.0))
    feats.append(attrs.get('at_charge_mull', 0.0))
    feats.append(attrs.get('at_charge_loew', 0.0))
    feats.append(attrs.get('orb_charge_mull_s', 0.0))
    feats.append(attrs.get('orb_charge_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_mull_d', 0.0))
    feats.append(attrs.get('orb_stdev_mull_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_s', 0.0))
    feats.append(attrs.get('orb_charge_loew_p', 0.0))
    feats.append(attrs.get('orb_charge_loew_d', 0.0))
    feats.append(attrs.get('orb_stdev_loew_p', 0.0))
    feats.append(attrs.get('BO_loew_sum', 0.0))
    feats.append(attrs.get('BO_loew_av', 0.0))
    feats.append(attrs.get('BO_mayer_sum', 0.0))
    feats.append(attrs.get('BO_mayer_av', 0.0))
    feats.append(attrs.get('mayer_VA', 0.0))
    return feats

def get_others_features(attrs):
    feats = []
    elem = attrs.get('element', 'X')  # 'X' = unbekannt
    feats.extend(get_element_onehot(elem))
    #feats.append(float(attrs.get('atom_idx', -1)))
    feats.append(attrs.get('mass', 0.0))
    feats.append(attrs.get('formal_charge', 0.0))
    feats.append(attrs.get('degree', 0.0))
    return feats

class ShiftDataset(Dataset):
    def __init__(self, root_dir="data", file_name="all_graphs_with_length.pkl",
                 normalize_node_features=True, normalize_edge_features=True,
                 norm_stats=None, edge_length_mean=None, edge_length_std=None,
                 edge_order_mean=None, edge_order_std=None):
        super().__init__()
        self.file_path = os.path.join(root_dir, file_name)
        with open(self.file_path, "rb") as f:
            self.nx_graphs = pickle.load(f)
        
        self.normalize_node_features = normalize_node_features
        self.normalize_edge_features = normalize_edge_features
        
        # Node Normalisierung: Berechne global Normalisierungsstatistiken für die kontinuierlichen Features (ab Index 13)
        if self.normalize_node_features:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = {}
                feat_collect = {'H': [], 'C': [], 'Others': []}
                for nx_g in self.nx_graphs:
                    for node in nx_g.nodes():
                        attrs = nx_g.nodes[node]
                        element = attrs["element"]
                        if element == "H":
                            feats = get_h_features(attrs)
                            feat_collect['H'].append(feats[13:])
                        elif element == "C":
                            feats = get_c_features(attrs)
                            feat_collect['C'].append(feats[13:])
                        else:
                            feats = get_others_features(attrs)
                            feat_collect['Others'].append(feats[13:])
                for ntype in feat_collect:
                    if feat_collect[ntype]:
                        arr = np.array(feat_collect[ntype])
                        mean = arr.mean(axis=0)
                        std = arr.std(axis=0)
                        self.norm_stats[ntype] = (mean, std)
                    else:
                        self.norm_stats[ntype] = (None, None)
        else:
            self.norm_stats = None
        
        # Edge Normalisierung: Für die ordinalen Features bond_order und length
        if self.normalize_edge_features:
            if edge_length_mean is not None and edge_length_std is not None and edge_order_mean is not None and edge_order_std is not None:
                self.edge_length_mean = edge_length_mean
                self.edge_length_std = edge_length_std
                self.edge_order_mean = edge_order_mean
                self.edge_order_std = edge_order_std
            else:
                self.edge_lengths = []
                self.edge_orders = []
                for nx_g in self.nx_graphs:
                    for u, v in nx_g.edges():
                        bond_data = nx_g[u][v]
                        self.edge_lengths.append(bond_data.get("length", 0.0))
                        self.edge_orders.append(bond_data.get("bond_order", 1.0))
                if self.edge_lengths:
                    self.edge_length_mean = np.mean(self.edge_lengths)
                    self.edge_length_std = np.std(self.edge_lengths)
                else:
                    self.edge_length_mean = 0.0
                    self.edge_length_std = 1.0
                if self.edge_orders:
                    self.edge_order_mean = np.mean(self.edge_orders)
                    self.edge_order_std = np.std(self.edge_orders)
                else:
                    self.edge_order_mean = 0.0
                    self.edge_order_std = 1.0
        else:
            self.edge_length_mean = 0.0
            self.edge_length_std = 1.0
            self.edge_order_mean = 0.0
            self.edge_order_std = 1.0

    def __len__(self):
        return len(self.nx_graphs)

    def __getitem__(self, idx):
        nx_g = self.nx_graphs[idx]
        data = HeteroData()
        h_nodes, c_nodes, o_nodes = [], [], []
        h_features, c_features, o_features = [], [], []
        h_shifts, c_shifts, o_shifts = [], [], []
        node_idx_map = {}
    
        def get_h_features_local(attrs):
            return get_h_features(attrs)
        
        def get_c_features_local(attrs):
            return get_c_features(attrs)
        
        def get_others_features_local(attrs):
            return get_others_features(attrs)
        
        # Extrahiere Knoten und ihre Zielwerte
        for node in nx_g.nodes():
            attrs = nx_g.nodes[node]
            element = attrs["element"]
            shift_val = attrs.get("shift_high-low", float('nan'))
            if element == "H":
                node_idx_map[node] = len(h_nodes)
                h_nodes.append(node)
                h_shifts.append(shift_val)
                feats = get_h_features_local(attrs)
                h_features.append(feats)
            elif element == "C":
                node_idx_map[node] = len(c_nodes)
                c_nodes.append(node)
                c_shifts.append(shift_val)
                feats = get_c_features_local(attrs)
                c_features.append(feats)
            else:
                node_idx_map[node] = len(o_nodes)
                o_nodes.append(node)
                o_shifts.append(float('nan'))
                feats = get_others_features_local(attrs)
                o_features.append(feats)
        
        # Wende optionale Normalisierung für Knotendaten an (nur die kontinuierlichen Features ab Index 13)
        if h_features:
            h_features = np.array(h_features, dtype=np.float32)
            if self.normalize_node_features:
                mean, std = self.norm_stats['H']
                if mean is not None:
                    h_features[:, 13:] = (h_features[:, 13:] - mean) / (std + 1e-6)
            h_x = torch.tensor(h_features, dtype=torch.float)
        else:
            h_x = torch.empty((0, 13))
        
        if c_features:
            c_features = np.array(c_features, dtype=np.float32)
            if self.normalize_node_features:
                mean, std = self.norm_stats['C']
                if mean is not None:
                    c_features[:, 13:] = (c_features[:, 13:] - mean) / (std + 1e-6)
            c_x = torch.tensor(c_features, dtype=torch.float)
        else:
            c_x = torch.empty((0, 13))
        
        if o_features:
            o_features = np.array(o_features, dtype=np.float32)
            if self.normalize_node_features:
                mean, std = self.norm_stats['Others']
                if mean is not None:
                    o_features[:, 13:] = (o_features[:, 13:] - mean) / (std + 1e-6)
            o_x = torch.tensor(o_features, dtype=torch.float)
        else:
            o_x = torch.empty((0, 13))
        
        h_y = torch.tensor(h_shifts, dtype=torch.float).view(-1, 1) if h_shifts else torch.empty((0, 1))
        c_y = torch.tensor(c_shifts, dtype=torch.float).view(-1, 1) if c_shifts else torch.empty((0, 1))
        o_y = torch.tensor(o_shifts, dtype=torch.float).view(-1, 1) if o_shifts else torch.empty((0, 1))
        
        if h_nodes:
            data['H'].x = h_x
            data['H'].y = h_y
        if c_nodes:
            data['C'].x = c_x
            data['C'].y = c_y
        if o_nodes:
            data['Others'].x = o_x
            data['Others'].y = o_y
        
        edge_index_dict = {}
        edge_attr_dict = {}
        
        def add_edge(src_type, dst_type, src_id, dst_id, bond_feat):
            rel = (src_type, "bond", dst_type)
            if rel not in edge_index_dict:
                edge_index_dict[rel] = [[], []]
                edge_attr_dict[rel] = []
            edge_index_dict[rel][0].append(src_id)
            edge_index_dict[rel][1].append(dst_id)
            edge_attr_dict[rel].append(bond_feat)
        
        def get_bond_features(bond_data):
            # Nominale Features werden per One-Hot kodiert.
            # bond_type: SINGLE, DOUBLE, TRIPLE
            bond_type = bond_data.get('bond_type', 'SINGLE')
            if bond_type == 'SINGLE':
                bond_type_onehot = [1, 0, 0]
            elif bond_type == 'DOUBLE':
                bond_type_onehot = [0, 1, 0]
            elif bond_type == 'TRIPLE':
                bond_type_onehot = [0, 0, 1]
            else:
                bond_type_onehot = [0, 0, 0]
            
            # bond_dir: NONE, ENDUPRIGHT, OTHER
            bond_dir = bond_data.get('bond_dir', 'NONE')
            if bond_dir == 'NONE':
                bond_dir_onehot = [1, 0, 0]
            elif bond_dir == 'ENDUPRIGHT':
                bond_dir_onehot = [0, 1, 0]
            else:
                bond_dir_onehot = [0, 0, 1]
            
            # is_aromatic: binär → One-Hot
            is_aromatic = bond_data.get('is_aromatic', False)
            is_aromatic_onehot = [0, 1] if is_aromatic else [1, 0]
            
            # Ordinale Features: bond_order und length
            bond_order = bond_data.get('bond_order', 1.0)
            bond_order_val = (bond_order - self.edge_order_mean) / (self.edge_order_std + 1e-6)
            
            length = bond_data.get('length', 0.0)
            length_val = (length - self.edge_length_mean) / (self.edge_length_std + 1e-6)
            
            return bond_type_onehot + bond_dir_onehot + is_aromatic_onehot + [bond_order_val, length_val]
        
        # Extrahiere Kanten (bidirektional)
        for u, v in nx_g.edges():
            bond_data = nx_g[u][v]
            bond_feat = get_bond_features(bond_data)
            u_element = nx_g.nodes[u]["element"]
            v_element = nx_g.nodes[v]["element"]
            if u_element == "H":
                u_type = "H"
                u_idx = node_idx_map[u]
            elif u_element == "C":
                u_type = "C"
                u_idx = node_idx_map[u]
            else:
                u_type = "Others"
                u_idx = node_idx_map[u]
            if v_element == "H":
                v_type = "H"
                v_idx = node_idx_map[v]
            elif v_element == "C":
                v_type = "C"
                v_idx = node_idx_map[v]
            else:
                v_type = "Others"
                v_idx = node_idx_map[v]
            add_edge(u_type, v_type, u_idx, v_idx, bond_feat)
            add_edge(v_type, u_type, v_idx, u_idx, bond_feat)
        
        for rel, (row, col) in edge_index_dict.items():
            data[rel].edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_feats = torch.tensor(edge_attr_dict[rel], dtype=torch.float)
            data[rel].edge_attr = edge_feats
        
        return data

def create_dataloaders(batch_size=4, root_dir=None, file_name="all_graphs_with_length.pkl", split_ratio=(0.8, 0.1, 0.1),
                       normalize_node_features=True, normalize_edge_features=True):
    if root_dir is None:
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        if 'notebooks' in caller_file or 'main.py' not in caller_file:
            root_dir = os.path.join(os.path.dirname(__file__), "../data")
        else:
            # If called from scripts/main.py or similar, use "data"
            root_dir = "data"
    dataset = ShiftDataset(root_dir=root_dir, file_name=file_name,
                           normalize_node_features=normalize_node_features,
                           normalize_edge_features=normalize_edge_features)
    
    # Gruppiere Graphen nach dem 'compound'-Attribut
    compound_to_indices = {}
    for idx, nx_g in enumerate(dataset.nx_graphs):
        compound = nx_g.graph.get("compound", None)
        if compound is None:
            compound = "unknown"
        if compound not in compound_to_indices:
            compound_to_indices[compound] = []
        compound_to_indices[compound].append(idx)
    
    compounds = list(compound_to_indices.keys())
    random.shuffle(compounds)
    
    num_compounds = len(compounds)
    train_end = int(split_ratio[0] * num_compounds)
    val_end = train_end + int(split_ratio[1] * num_compounds)
    
    train_compounds = compounds[:train_end]
    val_compounds = compounds[train_end:val_end]
    test_compounds = compounds[val_end:]

    print("Train")
    print(train_compounds)
    print("Val")
    print(val_compounds)
    print("Test")
    print(test_compounds)
    
    train_indices = []
    for comp in train_compounds:
        train_indices.extend(compound_to_indices[comp])
    
    val_indices = []
    for comp in val_compounds:
        val_indices.extend(compound_to_indices[comp])
    
    test_indices = []
    for comp in test_compounds:
        test_indices.extend(compound_to_indices[comp])
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    #train_indices = [14, 65, 699, 328, 66, 87, 522, 483, 196, 868, 478, 404, 161, 677, 494, 318, 467, 339, 6, 241, 890, 810, 869, 244, 575, 844, 434, 817, 70, 510, 621, 752, 499, 930, 641, 487, 600, 458, 53, 375, 423, 717, 393, 254, 649, 831, 143, 269, 923, 660, 745, 684, 325, 316, 558, 358, 312, 173, 305, 357, 192, 453, 586, 690, 904, 925, 495, 618, 149, 691, 349, 747, 311, 171, 786, 392, 813, 756, 791, 711, 843, 72, 144, 812, 346, 921, 615, 828, 694, 907, 629, 417, 461, 201, 264, 165, 79, 788, 251, 353, 793, 9, 795, 590, 193, 550, 876, 455, 928, 598, 355, 209, 373, 730, 515, 211, 401, 909, 408, 359, 63, 69, 262, 348, 276, 492, 464, 398, 875, 715, 212, 13, 238, 225, 888, 82, 661, 56, 336, 719, 697, 174, 693, 462, 519, 411, 931, 203, 570, 162, 274, 848, 258, 395, 589, 302, 794, 723, 303, 252, 120, 466, 645, 250, 215, 551, 497, 860, 920, 613, 582, 671, 606, 433, 887, 864, 800, 580, 378, 903, 830, 802, 170, 221, 167, 513, 867, 39, 782, 308, 197, 260, 877, 334, 472, 198, 83, 721, 616, 71, 479, 58, 900, 428, 194, 415, 122, 384, 321, 517, 451, 610, 186, 200, 381, 457, 460, 320, 419, 489, 236, 15, 54, 389, 906, 400, 686, 583, 614, 619, 758, 603, 622, 182, 759, 151, 450, 908, 474, 544, 740, 463, 319, 166, 780, 736, 156, 219, 268, 680, 744, 337, 665, 218, 750, 159, 818, 617, 724, 488, 577, 742, 881, 265, 16, 801, 67, 698, 253, 55, 938, 927, 430, 157, 825, 78, 297, 814, 124, 121, 796, 36, 811, 642, 725, 871, 178, 806, 315, 217, 526, 17, 822, 512, 861, 899, 905, 371, 718, 228, 437, 625, 695, 873, 807, 33, 886, 418, 278, 573, 792, 152, 548, 845, 541, 309, 183, 263, 307, 824, 220, 235, 584, 35, 231, 390, 728, 261, 223, 410, 493, 585, 870, 324, 429, 31, 663, 798, 89, 206, 7, 80, 471, 543, 465, 140, 3, 579, 332, 387, 299, 210, 327, 722, 847, 926, 331, 620, 486, 885, 733, 85, 674, 164, 313, 10, 862, 820, 8, 901, 624, 293, 84, 247, 273, 176, 737, 670, 294, 380, 889, 382, 572, 342, 425, 518, 397, 687, 689, 751, 898, 301, 498, 456, 377, 351, 688, 476, 292, 623, 710, 158, 667, 399, 832, 781, 514, 552, 878, 180, 432, 922, 592, 664, 344, 370, 427, 555, 678, 68, 52, 557, 322, 729, 525, 242, 155, 214, 379, 799, 540, 840, 270, 720, 239, 30, 511, 62, 277, 827, 712, 1, 295, 420, 153, 123, 732, 338, 347, 668, 816, 298, 216, 256, 190, 306, 884, 835, 809, 459, 475, 594, 924, 754, 431, 50, 929, 232, 147, 893, 77, 413, 481, 842, 597, 846, 837, 169, 581, 291, 356, 739, 757, 834, 662, 255, 682, 5, 609, 587, 406, 259, 240, 234, 823, 272, 473, 608, 148, 163, 588, 821, 88, 350, 4, 516, 527, 559, 436, 892, 731, 412, 683, 790, 542, 627, 496, 895, 626, 145, 490, 787, 329, 741, 933, 402, 396, 383, 602, 611, 12, 578, 175, 785, 146, 520, 34, 808, 484, 424, 125, 179, 805, 267, 160, 185, 2, 596, 243, 340, 612, 644, 150, 275, 86, 529, 797, 290, 477, 394, 546, 343, 271, 374, 310, 643, 468, 576, 388, 714, 195, 880, 202, 849, 333, 593, 199, 142, 815, 414, 640, 547, 51, 783, 833, 726, 675, 230, 141, 177, 866, 317, 934, 554, 672, 296, 601, 189, 330, 685, 354, 300, 485, 574, 323, 826, 734, 391, 896, 716, 753, 59, 936, 735, 491, 341, 681, 749, 556, 435, 172, 882, 154, 266, 480, 126, 246, 314, 595, 452, 81, 524, 894, 304, 647, 784, 205, 863, 224, 403, 422, 803, 19, 819, 935, 646, 439, 836, 128, 326, 405, 249, 204, 937, 227, 599, 438, 75, 279, 454, 841, 545, 74, 416, 713, 872, 0, 181, 897, 528, 604, 839, 187, 676, 229, 482, 207, 673, 345, 376, 743, 213, 628, 605, 521, 233, 372, 385, 352, 184, 426, 470, 168, 902, 553, 37, 696, 60, 838, 73, 865, 257, 789, 237, 755, 127, 669, 591, 61, 939, 76, 64, 571, 421, 932, 648, 666, 248, 386, 11, 222, 891, 469, 335, 549, 748, 746, 208, 727, 523, 874, 129, 245, 409, 829, 32, 191, 883, 407, 38, 57, 226, 738, 679, 692, 188, 804, 18, 607, 879]
    #val_indices = [441, 764, 775, 651, 443, 918, 656, 363, 446, 509, 705, 96, 113, 368, 917, 502, 448, 503, 779, 445, 773, 762, 444, 768, 777, 90, 763, 508, 98, 116, 447, 913, 659, 766, 365, 112, 99, 706, 704, 449, 505, 366, 97, 652, 772, 703, 95, 767, 776, 362, 657, 654, 111, 504, 658, 770, 91, 119, 369, 916, 910, 92, 919, 110, 507, 774, 650, 708, 500, 114, 914, 506, 94, 771, 655, 117, 709, 701, 440, 915, 93, 700, 761, 912, 702, 765, 367, 442, 364, 778, 760, 118, 115, 501, 769, 653, 707, 361, 911, 360]
    #test_indices = [282, 42, 43, 44, 137, 639, 566, 638, 132, 21, 565, 103, 23, 531, 100, 567, 48, 535, 289, 856, 636, 564, 104, 852, 634, 538, 25, 287, 854, 632, 537, 855, 530, 560, 631, 284, 630, 286, 857, 851, 27, 534, 539, 633, 858, 28, 850, 533, 49, 107, 29, 130, 105, 106, 40, 561, 859, 283, 288, 41, 285, 568, 563, 133, 281, 108, 24, 136, 532, 635, 45, 569, 101, 135, 853, 138, 20, 22, 134, 109, 536, 131, 280, 47, 139, 46, 26, 562, 637, 102]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)


    
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = PyGDataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
