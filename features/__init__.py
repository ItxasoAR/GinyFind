"""
features/__init__.py
---------------------
Unified feature extraction pipeline.

Calling `extract_features(structure, pdb_path)` returns:
    - X        : np.ndarray  (n_residues, n_features=49)
    - residues : list of Bio.PDB.Residue objects (same order as X rows)

Total feature dimensionality:
    28  residue physicochemical features
  +  7  geometric / structural features
  + 14  neighbour-based features
  ────
    49  features per residue
"""

import numpy as np
from Bio.PDB import PDBParser

from features.residue_features import (
    get_residue_features,
    get_feature_names,
    STANDARD_AA,
)
from features.geometric_features import (
    compute_sasa,
    compute_residue_depths,
    compute_centroid,
    compute_dssp,
    get_geometric_features,
    get_geometric_feature_names,
)
from features.neighbor_features import (
    build_ca_index,
    get_neighbor_features,
    get_neighbor_feature_names,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(structure, pdb_path: str) -> tuple[np.ndarray, list]:
    """
    Extract the full 49-dimensional feature vector for every standard
    amino-acid residue in `structure`.

    Parameters
    ----------
    structure : Bio.PDB Structure object
    pdb_path  : path to the PDB file (needed for DSSP)

    Returns
    -------
    X         : np.ndarray of shape (n_residues, 49)
    residues  : list of Bio.PDB.Residue, aligned with rows of X
    """
    # --- Pre-compute structure-level quantities ----------------------------
    sasa_dict  = compute_sasa(structure)
    depth_dict = compute_residue_depths(structure)
    centroid   = compute_centroid(structure)
    dssp_dict  = compute_dssp(structure, pdb_path)
    tree, all_residues = build_ca_index(structure)

    feature_rows = []
    valid_residues = []

    for residue in structure.get_residues():
        resname = residue.get_resname().strip().upper()
        # Skip non-standard residues, waters, ligands
        if resname not in STANDARD_AA:
            continue
        if "CA" not in residue:
            continue

        f_res  = get_residue_features(residue)                    # (28,)
        f_geo  = get_geometric_features(                          # (7,)
            residue, sasa_dict, depth_dict, centroid, dssp_dict
        )
        f_nbr  = get_neighbor_features(                           # (14,)
            residue, tree, all_residues, sasa_dict
        )

        row = np.concatenate([f_res, f_geo, f_nbr])               # (49,)
        feature_rows.append(row)
        valid_residues.append(residue)

    if not feature_rows:
        raise ValueError("No standard residues found in structure.")

    X = np.vstack(feature_rows).astype(np.float32)
    return X, valid_residues


def all_feature_names() -> list[str]:
    """Return the 49 feature names in the same order as columns of X."""
    return (
        get_feature_names()            # 28
        + get_geometric_feature_names()  # 7
        + get_neighbor_feature_names()   # 14
    )


def load_structure(pdb_path: str):
    """Convenience wrapper: parse a PDB file and return the structure."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", pdb_path)
