"""
neighbor_features.py
---------------------
Computes per-residue features based on the 3D local environment
using a KD-tree for efficient spatial lookup.

Features computed:
    - Neighbour count at 5 Å, 8 Å, 12 Å (local density)
    - Mean and std of hydrophobicity of neighbours (8 Å shell)
    - Composition of neighbours: fraction of each charge class (-1/0/+1)
    - Count of catalytically relevant AAs (HIS,CYS,ASP,GLU,SER) within 8 Å
    - Concavity index: ratio of neighbour density to rSASA
      (high density + low SASA → likely buried cavity)
    - Mean B-factor of neighbours within 8 Å
"""

import numpy as np
from scipy.spatial import cKDTree

from features.residue_features import (
    HYDROPHOBICITY,
    CHARGE,
    CATALYTIC_AA,
    _get_resname,
    mean_bfactor,
)

# ---------------------------------------------------------------------------
# Build a spatial index over all Cα atoms in the structure
# ---------------------------------------------------------------------------

def build_ca_index(structure) -> tuple[cKDTree, list]:
    """
    Build a KD-tree over all Cα atom coordinates.

    Returns
    -------
    tree : cKDTree
    residue_list : list of Bio.PDB.Residue objects in the same order as tree
    """
    coords = []
    residues = []
    for residue in structure.get_residues():
        if "CA" in residue:
            coords.append(residue["CA"].get_vector().get_array())
            residues.append(residue)

    if not coords:
        raise ValueError("No Cα atoms found in structure.")

    tree = cKDTree(np.array(coords, dtype=np.float64))
    return tree, residues


# ---------------------------------------------------------------------------
# Per-residue neighbour feature computation
# ---------------------------------------------------------------------------

def get_neighbor_features(
    residue,
    tree: cKDTree,
    residue_list: list,
    sasa_dict: dict,
    radii: tuple[float, float, float] = (5.0, 8.0, 12.0),
) -> np.ndarray:
    """
    Compute neighbourhood-based features for a single residue.

    Parameters
    ----------
    residue      : Bio.PDB Residue object (must have a Cα atom)
    tree         : KD-tree built from all Cα coords (see build_ca_index)
    residue_list : residue list aligned with tree indices
    sasa_dict    : dict from compute_sasa(), used for concavity index
    radii        : tuple of three radii (Å) for density counts

    Returns
    -------
    np.ndarray of shape (14,)
    """
    if "CA" not in residue:
        return np.zeros(14, dtype=np.float32)

    ca_coord = residue["CA"].get_vector().get_array()

    # ---- Neighbour counts at each radius (excluding self) ----------------
    counts = []
    for r in radii:
        idx = tree.query_ball_point(ca_coord, r)
        # Exclude the residue itself
        n = sum(1 for i in idx if residue_list[i] is not residue)
        counts.append(float(n))
    n5, n8, n12 = counts

    # ---- Get neighbours within 8 Å (main analysis shell) -----------------
    idx_8 = tree.query_ball_point(ca_coord, radii[1])
    neighbours_8 = [
        residue_list[i] for i in idx_8 if residue_list[i] is not residue
    ]

    # ---- Hydrophobicity stats of neighbours ------------------------------
    hydro_vals = [
        HYDROPHOBICITY.get(_get_resname(nb), 0.0) for nb in neighbours_8
    ]
    if hydro_vals:
        hydro_mean = float(np.mean(hydro_vals))
        hydro_std = float(np.std(hydro_vals))
    else:
        hydro_mean, hydro_std = 0.0, 0.0

    # ---- Charge composition of neighbours --------------------------------
    charges = [CHARGE.get(_get_resname(nb), 0) for nb in neighbours_8]
    n_nb = max(len(neighbours_8), 1)
    frac_neg = sum(1 for c in charges if c < 0) / n_nb
    frac_neu = sum(1 for c in charges if c == 0) / n_nb
    frac_pos = sum(1 for c in charges if c > 0) / n_nb

    # ---- Catalytic residue count within 8 Å -----------------------------
    n_catalytic = sum(
        1 for nb in neighbours_8 if _get_resname(nb) in CATALYTIC_AA
    )

    # ---- Mean B-factor of neighbours within 8 Å -------------------------
    bfactor_vals = [mean_bfactor(nb) for nb in neighbours_8]
    mean_nbr_bfactor = float(np.mean(bfactor_vals)) if bfactor_vals else 0.0

    # ---- Concavity index: n8 / (rSASA + ε) ------------------------------
    from features.geometric_features import relative_sasa
    rsasa = relative_sasa(residue, sasa_dict)
    concavity = n8 / (rsasa + 1e-3)

    # ---- Assemble feature vector -----------------------------------------
    features = np.array([
        n5,
        n8,
        n12,
        hydro_mean,
        hydro_std,
        frac_neg,
        frac_neu,
        frac_pos,
        float(n_catalytic),
        mean_nbr_bfactor,
        concavity,
        # Normalised neighbour density (n8 / n12 — how tightly packed)
        n8 / (n12 + 1e-3),
        # Fraction of neighbours that are hydrophobic (hydro > 0)
        sum(1 for h in hydro_vals if h > 0) / n_nb,
        # Fraction of neighbours that are charged (|charge| > 0)
        sum(1 for c in charges if c != 0) / n_nb,
    ], dtype=np.float32)

    return features


def get_neighbor_feature_names() -> list[str]:
    """Names for the 14 neighbour-based features."""
    return [
        "n_neighbours_5A",
        "n_neighbours_8A",
        "n_neighbours_12A",
        "mean_hydro_8A",
        "std_hydro_8A",
        "frac_neg_charge_8A",
        "frac_neu_charge_8A",
        "frac_pos_charge_8A",
        "n_catalytic_8A",
        "mean_bfactor_8A",
        "concavity_index",
        "density_ratio_8_12",
        "frac_hydrophobic_8A",
        "frac_charged_8A",
    ]
