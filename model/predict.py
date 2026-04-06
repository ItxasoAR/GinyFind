"""
model/predict.py
-----------------
Loads a trained model and predicts ligand binding sites on a new PDB file.

Pipeline
--------
1. Extract features for every residue
2. Compute binding probability per residue
3. Threshold → positive residues
4. DBSCAN clustering on Cα coordinates of positive residues
   → each cluster = one predicted binding site
5. Return structured result dict

Usage
-----
python -m model.predict --pdb input.pdb --model model/trained_model.pkl
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from features import extract_features, load_structure
import tempfile

#Importing fuction from our modules
from scripts.extract_chains_and_sequences import (
    parse_pdb_file,
    write_chain_pdb,
)

from output.writer import write_all_outputs  

# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def extract_chain_files(pdb_path: str) -> list[Path]:
    """
    Use the SAME extraction logic as training to split a full PDB into chains.
    Returns list of temporary chain PDB files.
    """
    pdb_path = Path(pdb_path)

    chains = parse_pdb_file(
        pdb_path,
        min_length=1,           # no filtramos fuerte en inferencia
        keep_hetatm_aa=False,
    )

    if not chains:
        raise ValueError(f"No valid protein chains found in {pdb_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{pdb_path.stem}_chains_"))

    chain_paths = []
    for chain in chains:
        out_path = tmp_dir / f"{chain.sample_id}.pdb"
        write_chain_pdb(chain, out_path)
        chain_paths.append(out_path)

    return chain_paths


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BindingSite:
    """A predicted ligand binding site."""
    site_id: int
    residues: list          # list of Bio.PDB Residue objects
    probabilities: list     # per-residue binding probability
    center: np.ndarray      # geometric centroid of site (Å)
    mean_probability: float
    n_residues: int = field(init=False)

    def __post_init__(self):
        self.n_residues = len(self.residues)

    def residue_summary(self) -> list[dict]:
        """Return list of dicts summarising each residue in the site."""
        summary = []
        for res, prob in zip(self.residues, self.probabilities):
            summary.append({
                "chain":   res.get_parent().id,
                "resname": res.get_resname().strip(),
                "resseq":  res.get_id()[1],
                "prob":    round(float(prob), 3),
            })
        return sorted(summary, key=lambda d: -d["prob"])


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a serialised sklearn Pipeline from disk and return pipeline and threshold."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    pipeline = data["model"]           # extraer pipeline
    threshold = data.get("threshold", 0.5)  # fallback 0.5
    return pipeline, threshold


def predict_binding_probabilities(
    structure,
    pdb_path: str,
    pipeline,
) -> tuple[list, np.ndarray]:
    """
    Run feature extraction and model inference.

    Returns
    -------
    residues : list of Bio.PDB Residue (standard AA only)
    proba    : np.ndarray (n_residues,) — binding probability per residue
    """
    X, residues = extract_features(structure, pdb_path)
    proba = pipeline.predict_proba(X)[:, 1]
    return residues, proba


def cluster_binding_residues(
    residues: list,
    proba: np.ndarray,
    threshold: float = None,
    eps: float = 8.0,
    min_samples: int = 3,
) -> list[BindingSite]:
    """
    1. Filter residues above `threshold`
    2. DBSCAN cluster their Cα coordinates
    3. Return one BindingSite per cluster, sorted by mean probability (desc)

    Parameters
    ----------
    threshold   : probability cutoff for positive label
    eps         : DBSCAN epsilon in Å — max distance between Cα atoms to be
                  considered neighbours (default 8 Å)
    min_samples : DBSCAN minimum cluster size
    """
    # Filter positives
    pos_mask = proba >= threshold
    pos_residues = [r for r, m in zip(residues, pos_mask) if m]
    pos_proba    = proba[pos_mask]

    if len(pos_residues) == 0:
        return []

    # Gather Cα coordinates
    coords = []
    valid_res, valid_prob = [], []
    for res, prob in zip(pos_residues, pos_proba):
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())
            valid_res.append(res)
            valid_prob.append(prob)

    if not coords:
        return []

    coords_arr = np.array(coords, dtype=np.float64)

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_labels = db.fit_predict(coords_arr)

    # Build BindingSite objects (skip noise label -1)
    sites = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        mask = cluster_labels == cluster_id
        site_residues = [r for r, m in zip(valid_res, mask) if m]
        site_proba    = [p for p, m in zip(valid_prob, mask) if m]
        site_coords   = coords_arr[mask]
        center        = site_coords.mean(axis=0)

        sites.append(BindingSite(
            site_id          = cluster_id + 1,
            residues         = site_residues,
            probabilities    = site_proba,
            center           = center,
            mean_probability = float(np.mean(site_proba)),
        ))

    # Sort by mean probability descending → site 1 = most confident
    sites.sort(key=lambda s: -s.mean_probability)
    for i, site in enumerate(sites, start=1):
        site.site_id = i

    n_noise = int((cluster_labels == -1).sum())
    if n_noise > 0:
        print(f"  ⚠ {n_noise} residues above threshold not assigned to any site (DBSCAN noise)")

    return sites


def predict_full_structure(
    pdb_path: str,
    model_path: str,
    threshold: float = None,
    eps: float = 8.0,
    min_samples: int = 3,
):
    """
    Predict binding sites for ALL chains in a PDB.
    """
    chain_files = extract_chain_files(pdb_path)

    results = []

    for chain_file in chain_files:
        sites, residues, proba = predict(
            pdb_path=str(chain_file),
            model_path=model_path,
            threshold=threshold,
            eps=eps,
            min_samples=min_samples,
        )

        results.append({
            "chain_file": str(chain_file),
            "chain_name": chain_file.stem,
            "sites": sites,
            "residues": residues,
            "proba": proba,
        })

    return results


def predict(
    pdb_path: str,
    model_path: str,
    threshold: float = None,
    eps: float = 8.0,
    min_samples: int = 3,
) -> tuple[list[BindingSite], list, np.ndarray]:

    structure = load_structure(pdb_path)
    pipeline, model_threshold = load_model(model_path)

    # Usa el threshold pasado desde CLI si se especificó, si no, usa el guardado
    threshold = threshold if threshold is not None else model_threshold

    residues, proba = predict_binding_probabilities(structure, pdb_path, pipeline)
    sites = cluster_binding_residues(
        residues, proba,
        threshold=threshold, eps=eps, min_samples=min_samples,
    )
    print(f"Using threshold: {threshold:.3f}")
    return sites, residues, proba


def print_prediction_report(sites: list[BindingSite], pdb_path: str) -> None:
    """Pretty-print a summary of predicted binding sites to stdout."""
    pdb_id = Path(pdb_path).stem
    print(f"\n{'='*58}")
    print(f"  Predicted binding sites for: {pdb_id}")
    print(f"  Total sites found: {len(sites)}")
    print(f"{'='*58}")

    if not sites:
        print("  ⚠ No binding sites predicted above threshold.\n")
        return

    for site in sites:
        cx, cy, cz = site.center
        print(f"\n  ── Site {site.site_id} ─────────────────────────────────")
        print(f"     Residues      : {site.n_residues}")
        print(f"     Mean prob.    : {site.mean_probability:.3f}")
        print(f"     Center (Å)    : ({cx:.1f}, {cy:.1f}, {cz:.1f})")
        print(f"\n     {'Chain':>5} {'Res':>5} {'AA':>4} {'P(binding)':>10}")
        print(f"     {'-'*30}")
        for r in site.residue_summary():
            print(
                f"     {r['chain']:>5} {r['resseq']:>5} "
                f"{r['resname']:>4} {r['prob']:>10.3f}"
            )
    print(f"\n{'='*58}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict ligand binding sites from a PDB file."
    )
    parser.add_argument("--pdb",        required=True, help="Input PDB file")
    parser.add_argument("--model",      required=True, help="Trained model (.pkl)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Probability threshold (default: use it the save it in the model)")
    parser.add_argument("--eps",        type=float, default=8.0,
                        help="DBSCAN eps in Å (default: 8.0)")
    parser.add_argument("--min_samples",type=int,   default=3,
                        help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_dir", type=str,   default="output",
                        help="Directory for output files (default: output/)")
    args = parser.parse_args()

    results = predict_full_structure(
        pdb_path=args.pdb,
        model_path=args.model,
        threshold=args.threshold,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    print(f"\nProcessing structure: {args.pdb}")
    print(f"Detected chains: {len(results)}")


    all_residues = []
    all_proba = []
    all_sites = []

    site_counter = 1

    for result in results:
        chain_file = result["chain_file"]
        sites = result["sites"]
        residues = result["residues"]
        proba = result["proba"]

        print(f"\n>>> Predicting for chain: {Path(chain_file).name}")
        print_prediction_report(sites, chain_file)

        all_residues.extend(residues)
        all_proba.extend(proba)

        for site in sites:
            site.site_id = site_counter
            all_sites.append(site)
            site_counter += 1

    all_proba = np.array(all_proba)

    print(f"\n=== GLOBAL SUMMARY ===")
    print(f"Total residues: {len(all_residues)}")
    print(f"Total predicted sites: {len(all_sites)}")


    write_all_outputs(
        pdb_path=args.pdb,
        sites=all_sites,
        residues=all_residues,
        proba=all_proba,
        out_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
