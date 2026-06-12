"""Canonical feature names, mirroring scripts/dataloader.py exactly.

The order of these lists MUST match get_h_features / get_c_features /
get_others_features in scripts/dataloader.py. verify_against_sample() is
called once at startup to assert the lengths against real tensor shapes.
"""
from . import pathsetup  # noqa: F401  (sys.path side effect)

from dataloader import (  # noqa: E402
    ALL_ELEMENTS,
    get_c_features,
    get_h_features,
    get_others_features,
)

_ONEHOT_NAMES = [f"elem_{el}" for el in ALL_ELEMENTS]

FEATURE_NAMES = {
    "H": _ONEHOT_NAMES
    + [
        "mass",
        "formal_charge",
        "degree",
        "shift_low",
        "CN(X)",
        "shielding_dia",
        "shielding_para",
        "span",
        "skew",
        "asymmetry",
        "anisotropy",
        "at_charge_mull",
        "at_charge_loew",
        "orb_charge_mull_s",
        "orb_charge_mull_p",
        "orb_charge_loew_s",
        "orb_charge_loew_p",
        "BO_loew",
        "BO_mayer",
        "mayer_VA",
    ],
    "C": _ONEHOT_NAMES
    + [
        "mass",
        "formal_charge",
        "degree",
        "shift_low",
        "CN(X)",
        "shielding_dia",
        "shielding_para",
        "span",
        "skew",
        "asymmetry",
        "anisotropy",
        "at_charge_mull",
        "at_charge_loew",
        "orb_charge_mull_s",
        "orb_charge_mull_p",
        "orb_charge_mull_d",
        "orb_stdev_mull_p",
        "orb_charge_loew_s",
        "orb_charge_loew_p",
        "orb_charge_loew_d",
        "orb_stdev_loew_p",
        "BO_loew_sum",
        "BO_loew_av",
        "BO_mayer_sum",
        "BO_mayer_av",
        "mayer_VA",
    ],
    "Others": _ONEHOT_NAMES + ["mass", "formal_charge", "degree"],
}

EDGE_FEATURE_NAMES = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "dir_none",
    "dir_endupright",
    "dir_other",
    "not_aromatic",
    "aromatic",
    "bond_order",
    "length",
]

_RAW_GETTERS = {
    "H": get_h_features,
    "C": get_c_features,
    "Others": get_others_features,
}

# Fallback dims when a node type is absent from the training data entirely.
CANONICAL_IN_DIMS = {ntype: len(names) for ntype, names in FEATURE_NAMES.items()}


def node_type_for_element(element):
    return element if element in ("H", "C") else "Others"


def raw_values(node_attrs, node_type):
    """Unnormalized feature vector for a node, same order as FEATURE_NAMES."""
    return [float(v) for v in _RAW_GETTERS[node_type](node_attrs)]


def verify_against_sample(hetero_data):
    """Assert canonical name lists match the actual tensor widths."""
    for ntype in hetero_data.node_types:
        width = int(hetero_data[ntype].x.shape[1])
        expected = len(FEATURE_NAMES[ntype])
        if width != expected:
            raise RuntimeError(
                f"Feature name list for '{ntype}' has {expected} entries but "
                f"the dataset produces {width} features. Update app/services/features.py."
            )
