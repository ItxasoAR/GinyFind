"""
geometric_features.py
----------------------
Computes per-residue geometric / structural features from a BioPython
structure, including:

    - Relative Solvent Accessible Surface Area (rSASA) via FreeSASA
    - Residue depth (distance from Cα to the protein surface)
    - Mean residue depth
    - Distance from Cα to the protein geometric centroid
    - Secondary structure assignment (3-class one-hot) via DSSP

Dependencies:
    pip install biopython freesasa
    DSSP binary must be installed and available in PATH.
"""

import warnings
import numpy as np
from Bio.PDB import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth

try:
    import freesasa
    _FREESASA_AVAILABLE = True
except ImportError:
    _FREESASA_AVAILABLE = False


# ---------------------------------------------------------------------------
# One-time warning guards
# ---------------------------------------------------------------------------

_FREESASA_WARNING_SHOWN = False
_RESDEPTH_WARNING_SHOWN = False
_DSSP_WARNING_SHOWN = False


def _warn_once(flag_name: str, message: str, category=UserWarning) -> None:
    global _FREESASA_WARNING_SHOWN, _RESDEPTH_WARNING_SHOWN, _DSSP_WARNING_SHOWN

    if flag_name == "freesasa" and not _FREESASA_WARNING_SHOWN:
        warnings.warn(message, category)
        _FREESASA_WARNING_SHOWN = True
    elif flag_name == "resdepth" and not _RESDEPTH_WARNING_SHOWN:
        warnings.warn(message, category)
        _RESDEPTH_WARNING_SHOWN = True
    elif flag_name == "dssp" and not _DSSP_WARNING_SHOWN:
        warnings.warn(message, category)
        _DSSP_WARNING_SHOWN = True


if not _FREESASA_AVAILABLE:
    _warn_once(
        "freesasa",
        "freesasa not found — SASA features will be set to 0. "
        "Install with: pip install freesasa",
        ImportWarning,
    )

# ---------------------------------------------------------------------------
# Maximum ASA reference values (Å²) — extended tripeptide, Tien et al. 2013
# ---------------------------------------------------------------------------
MAX_ASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}

# Secondary structure 3-class mapping from DSSP 8-class codes
SS_MAP = {
    "H": 0,  # α-helix
    "G": 0,  # 3-10 helix
    "I": 0,  # π-helix
    "E": 1,  # β-strand
    "B": 1,  # β-bridge
    "T": 2,  # turn
    "S": 2,  # bend
    "-": 2,  # irregular
    " ": 2,
}

# ---------------------------------------------------------------------------
# SASA computation
# ---------------------------------------------------------------------------

def compute_sasa(structure) -> dict:
    """
    Run FreeSASA on the structure and return a dict mapping
    (chain_id, res_seq, icode) -> absolute ASA (Å²).

    Falls back to zeros if freesasa is unavailable or fails.
    """
    sasa_dict = {}
    if not _FREESASA_AVAILABLE:
        return sasa_dict

    try:
        from Bio.PDB import PDBIO
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_path)

        fs_structure = freesasa.Structure(tmp_path)
        fs_result    = freesasa.calc(fs_structure)
        os.unlink(tmp_path)

        for chain in structure.get_chains():
            for residue in chain.get_residues():
                key = (chain.id, residue.get_id()[1], residue.get_id()[2])
                seq = residue.get_id()[1]
                try:
                    sel = freesasa.selectArea(
                        [f"s, chain {chain.id} and resi {seq}"],
                        fs_structure, fs_result
                    )
                    asa = sel.get("s", 0.0)
                except Exception:
                    asa = 0.0
                sasa_dict[key] = asa

    except Exception as exc:
        _warn_once(
            "freesasa",
            f"FreeSASA calculation failed ({exc}). SASA features will be set to 0."
        )

    return sasa_dict


def relative_sasa(residue, sasa_dict: dict) -> float:
    """
    Normalise absolute ASA by the maximum reference value -> [0, 1].
    Values close to 0 indicate buried residues.
    """
    chain_id = residue.get_parent().id
    res_id = residue.get_id()
    key = (chain_id, res_id[1], res_id[2])
    abs_asa = sasa_dict.get(key, 0.0)
    resname = residue.get_resname().strip().upper()
    max_asa = MAX_ASA.get(resname, 200.0)
    return float(abs_asa / max_asa) if max_asa > 0 else 0.0

# ---------------------------------------------------------------------------
# Residue depth
# ---------------------------------------------------------------------------

def compute_residue_depths(structure, model_idx: int = 0) -> dict:
    """
    Compute Cα depth (Å) and mean residue depth for every residue using
    BioPython's ResidueDepth.

    Returns dict mapping residue full_id -> (ca_depth, mean_depth).
    Falls back to zeros on error.
    """
    depth_dict = {}
    try:
        model = list(structure.get_models())[model_idx]
        rd = ResidueDepth(model)
        for chain in model.get_chains():
            for residue in chain.get_residues():
                try:
                    key = (chain.id, residue.get_id())
                    val = rd.property_dict.get(key)
                    if val is not None:
                        ca_depth, mean_depth = float(val[0]), float(val[1])
                    else:
                        ca_depth, mean_depth = 0.0, 0.0
                    
                except (KeyError, TypeError, ValueError):
                    ca_depth, mean_depth = 0.0, 0.0
                depth_dict[residue.get_full_id()] = (ca_depth, mean_depth)
    except Exception as exc:
        _warn_once(
            "resdepth",
            "ResidueDepth failed. Depth features will be set to 0. "
            f"Underlying error: {exc}. Ensure MSMS is installed and in PATH."
        )
    return depth_dict

# ---------------------------------------------------------------------------
# Protein centroid distance
# ---------------------------------------------------------------------------

def compute_centroid(structure) -> np.ndarray:
    """Return the geometric centroid of all Cα atoms."""
    ca_coords = []
    for residue in structure.get_residues():
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_vector().get_array())
    if ca_coords:
        return np.mean(ca_coords, axis=0)
    return np.zeros(3, dtype=np.float32)


def distance_to_centroid(residue, centroid: np.ndarray) -> float:
    """Euclidean distance (Å) from residue Cα to the protein centroid."""
    if "CA" not in residue:
        return 0.0
    ca = residue["CA"].get_vector().get_array()
    return float(np.linalg.norm(ca - centroid))

# ---------------------------------------------------------------------------
# Secondary structure (DSSP)
# ---------------------------------------------------------------------------

def compute_dssp(structure, pdb_path: str) -> dict:
    """
    Run DSSP and return a dict mapping (chain_id, res_full_id) -> ss_onehot.
    ss_onehot is a 3-element array: [helix, strand, coil].

    Returns empty dict if DSSP is unavailable or fails.
    """
    dssp_dict = {}
    try:
        model = list(structure.get_models())[0]
        dssp = DSSP(model, pdb_path)
        for key in dssp.property_keys:
            chain_id, res_id = key
            ss_code = dssp[key][2]
            ss_class = SS_MAP.get(ss_code, 2)
            onehot = np.zeros(3, dtype=np.float32)
            onehot[ss_class] = 1.0
            dssp_dict[key] = onehot
    except Exception as exc:
        _warn_once(
            "dssp",
            "DSSP failed. Secondary-structure features will default to coil. "
            f"Underlying error: {exc}. Ensure DSSP/mkdssp is installed and in PATH."
        )
    return dssp_dict


def get_ss_onehot(residue, dssp_dict: dict) -> np.ndarray:
    """Retrieve the SS one-hot for a residue, defaulting to coil [0,0,1]."""
    chain_id = residue.get_parent().id
    res_id = residue.get_id()
    key = (chain_id, res_id)
    result = dssp_dict.get(key)
    if result is not None:
        return result
    coil = np.zeros(3, dtype=np.float32)
    coil[2] = 1.0
    return coil

# ---------------------------------------------------------------------------
# Master call: get all geometric features for one residue
# ---------------------------------------------------------------------------

def get_geometric_features(
    residue,
    sasa_dict: dict,
    depth_dict: dict,
    centroid: np.ndarray,
    dssp_dict: dict,
) -> np.ndarray:
    """
    Assemble all geometric features for a single residue.

    Returns
    -------
    np.ndarray of shape (7,)
        [rSASA, ca_depth, mean_depth, dist_to_centroid,
         ss_helix, ss_strand, ss_coil]
    """
    rsasa = relative_sasa(residue, sasa_dict)

    full_id = residue.get_full_id()
    ca_depth, mean_depth = depth_dict.get(full_id, (0.0, 0.0))

    dist_c = distance_to_centroid(residue, centroid)
    ss = get_ss_onehot(residue, dssp_dict)

    return np.array(
        [rsasa, ca_depth, mean_depth, dist_c, *ss],
        dtype=np.float32
    )


def get_geometric_feature_names() -> list[str]:
    """Names for the 7 geometric features."""
    return [
        "rSASA", "ca_depth", "mean_depth", "dist_to_centroid",
        "ss_helix", "ss_strand", "ss_coil",
    ]
