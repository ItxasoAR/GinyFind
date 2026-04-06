"""
residue_features.py
--------------------
Extracts per-residue physicochemical features from a BioPython structure.

Features computed:
    - One-hot encoding of amino acid type (20 standard AAs)
    - Hydrophobicity (Kyte-Doolittle scale)
    - Side-chain charge at physiological pH
    - H-bond donor / acceptor capacity
    - Side-chain volume (Å³)
    - Molecular weight of the residue
    - B-factor (mean over all heavy atoms)
    - Is the residue part of a known catalytic triad motif? (binary flag)
"""

import numpy as np
from Bio.PDB import PPBuilder

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "ALA":  1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS":  2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE":  4.5,
    "LEU":  3.8, "LYS": -3.9, "MET":  1.9, "PHE":  2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL":  4.2,
}

# Formal charge at pH 7 (-1 / 0 / +1)
CHARGE = {
    "ALA":  0, "ARG":  1, "ASN":  0, "ASP": -1, "CYS":  0,
    "GLN":  0, "GLU": -1, "GLY":  0, "HIS":  0, "ILE":  0,
    "LEU":  0, "LYS":  1, "MET":  0, "PHE":  0, "PRO":  0,
    "SER":  0, "THR":  0, "TRP":  0, "TYR":  0, "VAL":  0,
}

# H-bond donors (number of NH / OH groups on side chain)
HBOND_DONOR = {
    "ALA": 0, "ARG": 5, "ASN": 1, "ASP": 0, "CYS": 1,
    "GLN": 1, "GLU": 0, "GLY": 0, "HIS": 1, "ILE": 0,
    "LEU": 0, "LYS": 3, "MET": 0, "PHE": 0, "PRO": 0,
    "SER": 1, "THR": 1, "TRP": 1, "TYR": 1, "VAL": 0,
}

# H-bond acceptors (number of O / N lone pairs on side chain)
HBOND_ACCEPTOR = {
    "ALA": 0, "ARG": 0, "ASN": 1, "ASP": 2, "CYS": 0,
    "GLN": 1, "GLU": 2, "GLY": 0, "HIS": 1, "ILE": 0,
    "LEU": 0, "LYS": 0, "MET": 1, "PHE": 0, "PRO": 0,
    "SER": 1, "THR": 1, "TRP": 0, "TYR": 1, "VAL": 0,
}

# Side-chain volume (Å³) — Zamyatnin 1972 / Tsai 1999
SC_VOLUME = {
    "ALA": 88.6,  "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY":  60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER":  89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0,
}

# Residue molecular weight (Da)
MOL_WEIGHT = {
    "ALA":  89.1, "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2,
    "GLN": 146.2, "GLU": 147.1, "GLY":  75.1, "HIS": 155.2, "ILE": 131.2,
    "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1,
    "SER": 105.1, "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1,
}

# Residues commonly found in catalytic triads / active sites
CATALYTIC_AA = {"HIS", "CYS", "ASP", "GLU", "SER", "LYS", "ARG", "TYR"}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _get_resname(residue) -> str:
    """Return uppercase 3-letter residue name, clipped to known AAs."""
    return residue.get_resname().strip().upper()


def one_hot_aa(resname: str) -> np.ndarray:
    """20-dimensional one-hot vector for amino acid identity."""
    vec = np.zeros(20, dtype=np.float32)
    if resname in STANDARD_AA:
        vec[STANDARD_AA.index(resname)] = 1.0
    return vec


def mean_bfactor(residue) -> float:
    """Mean B-factor across all heavy atoms of the residue."""
    bfactors = [
        atom.get_bfactor()
        for atom in residue.get_atoms()
        if atom.element != "H"
    ]
    return float(np.mean(bfactors)) if bfactors else 0.0


def scalar_features(residue) -> np.ndarray:
    """
    Returns a 1-D array of scalar physicochemical features:
        [hydrophobicity, charge, hbond_donor, hbond_acceptor,
         sc_volume, mol_weight, mean_bfactor, is_catalytic]
    """
    resname = _get_resname(residue)
    features = np.array([
        HYDROPHOBICITY.get(resname, 0.0),
        float(CHARGE.get(resname, 0)),
        float(HBOND_DONOR.get(resname, 0)),
        float(HBOND_ACCEPTOR.get(resname, 0)),
        SC_VOLUME.get(resname, 100.0),
        MOL_WEIGHT.get(resname, 110.0),
        mean_bfactor(residue),
        float(resname in CATALYTIC_AA),
    ], dtype=np.float32)
    return features


def get_residue_features(residue) -> np.ndarray:
    """
    Full per-residue feature vector combining one-hot AA encoding
    and scalar physicochemical properties.

    Returns
    -------
    np.ndarray of shape (28,)
        [20 one-hot AA] + [8 scalar features]
    """
    resname = _get_resname(residue)
    return np.concatenate([one_hot_aa(resname), scalar_features(residue)])


def get_feature_names() -> list[str]:
    """Human-readable names for all 28 residue-level features."""
    aa_names = [f"aa_{aa}" for aa in STANDARD_AA]
    scalar_names = [
        "hydrophobicity", "charge", "hbond_donor", "hbond_acceptor",
        "sc_volume", "mol_weight", "mean_bfactor", "is_catalytic",
    ]
    return aa_names + scalar_names
