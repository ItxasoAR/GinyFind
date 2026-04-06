"""
data/build_dataset.py
----------------------
Builds a residue-level dataset from PDB structures, either downloaded
from RCSB or read from a local directory. The script extracts features
for each residue and assigns binary labels indicating whether the residue
belongs to a ligand-binding site.

This script supports both:
    1) Standard dataset construction from a collection of PDB files
    2) Homology-aware dataset splits via an optional ids_file

Typical workflow
----------------
1. (Optional) Prepare PDBs:
   - Download from RCSB using a list of PDB IDs
   - OR use a local directory of PDB files
   - OR use preprocessed chain-level PDBs (e.g. 1ABC_A.pdb)

2. (Optional) Apply homology-based filtering:
   - Provide an ids_file (e.g. train/val/test splits)
   - Only structures whose filename stem matches the IDs are processed

3. For each PDB:
   - Parse structure
   - Extract residue-level features (via features/)
   - Detect ligand atoms (HETATM, excluding common solvents/additives)
   - Assign residue labels based on distance to ligand

4. Merge all structures into a single dataset and save to disk

Label rule
----------
A residue is labelled as binding site (1) if ANY of its heavy atoms is
within CONTACT_THRESHOLD Å of ANY ligand heavy atom.
Otherwise it is labelled as non-binding (0).

Ligand definition
-----------------
Ligands are identified from HETATM records, excluding common non-ligand
molecules such as water, ions, and crystallographic additives.

Supported input formats
-----------------------
- Standard PDB files:           1ABC.pdb
- Chain-level PDB files:        1ABC_A.pdb
- Any filename (ID inferred from stem)

Examples
--------
# 1. Build dataset from all PDBs in a directory
python -m data.build_dataset \
    --pdb_dir data/pdbs \
    --output data/dataset_all.npz

# 2. Build dataset using homology-based split (e.g. train set)
python -m data.build_dataset \
    --pdb_dir data/processed/chains \
    --ids_file splits/train_ids.txt \
    --output data/train_dataset.npz

# 3. Download PDBs from a list and build dataset
python -m data.build_dataset \
    --scpdb_list data/scpdb_ids.txt \
    --download_dir data/pdbs \
    --output data/dataset_downloaded.npz

# 4. Quick test (subset)
python -m data.build_dataset \
    --pdb_dir data/pdbs \
    --limit 10 \
    --output data/debug_dataset.npz

Output
------
Compressed .npz file containing:

    X      : float32 array (N_total_residues, n_features)
             Residue-level feature matrix

    y      : int8 array (N_total_residues,)
             Binary labels (1 = binding site, 0 = non-binding)

    ids    : object array (N_total_residues,)
             Unique residue identifiers:
             "PDBID_chainID_resSeq"

    pdbs   : object array (N_total_residues,)
             Source structure identifier (file stem)

Notes
-----
- Structures with no detected ligand or no positive residues are skipped
- Features are computed using modules in features/
- Designed to integrate with homology-based splitting pipelines
"""

import argparse
import gzip
import shutil
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from features import extract_features, load_structure

import csv
from collections import Counter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTACT_THRESHOLD = 4.5
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"

NON_LIGAND_RESNAMES = {
    "HOH", "WAT", "H2O",
    "SO4", "PO4", "GOL", "EDO", "PEG",
    "DMS", "ACT", "MPD", "FMT", "EOH",
    "BME", "MES", "TRS", "HED", "ACE",
    "NH2", "NH4", "CAC", "IOD", "CL",
    "NA", "MG", "ZN", "CA", "FE",
    "MN", "NI", "CU", "CO", "CD",
}

DEFAULT_CONTACT_THRESHOLD = 4.5 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, out_dir: Path) -> Path | None:
    out_path = out_dir / f"{pdb_id.upper()}.pdb"
    if out_path.exists():
        return out_path

    url = RCSB_URL.format(pdb_id=pdb_id.lower())
    gz_path = out_path.with_suffix(".pdb.gz")

    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)
        return out_path
    except Exception as exc:
        warnings.warn(f"Could not download {pdb_id}: {exc}")
        gz_path.unlink(missing_ok=True)
        return None


def get_ligand_atoms(structure) -> list:
    ligand_atoms = []
    for residue in structure.get_residues():
        hetflag = residue.get_id()[0]
        if isinstance(hetflag, str) and hetflag.startswith("H_"):
            resname = residue.get_resname().strip().upper()
            if resname in NON_LIGAND_RESNAMES:
                continue
            for atom in residue.get_atoms():
                if atom.element != "H":
                    ligand_atoms.append(atom)
    return ligand_atoms


def label_residues(residues: list, ligand_atoms: list, threshold=DEFAULT_CONTACT_THRESHOLD) -> np.ndarray:
    if not ligand_atoms:
        return np.zeros(len(residues), dtype=np.int8)

    lig_coords = np.array(
        [atom.get_vector().get_array() for atom in ligand_atoms],
        dtype=np.float64,
    )

    labels = np.zeros(len(residues), dtype=np.int8)

    for i, residue in enumerate(residues):
        res_coords = np.array(
            [a.get_vector().get_array() for a in residue.get_atoms() if a.element != "H"],
            dtype=np.float64,
        )
        if res_coords.size == 0:
            continue

        diffs = res_coords[:, None, :] - lig_coords[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        if dists.min() <= threshold:
            labels[i] = 1

    return labels


def infer_structure_id(pdb_path: str) -> str:
    """
    Accepts either:
      - 1ABC.pdb
      - 1ABC_A.pdb
      - any_other_name.pdb
    """
    return Path(pdb_path).stem


def process_pdb(pdb_path: str, contact_threshold: float = DEFAULT_CONTACT_THRESHOLD):
    pdb_name = infer_structure_id(pdb_path)

    try:
        structure = load_structure(pdb_path)
        ligand_atoms = get_ligand_atoms(structure)

        if not ligand_atoms:
            return None, "no_ligand"

        X, residues = extract_features(structure, pdb_path)
        y = label_residues(residues, ligand_atoms, threshold=contact_threshold)

        if y.sum() == 0:
            return None, "zero_positive"

        ids = []
        pdb_names = []

        for res in residues:
            chain = res.get_parent().id
            seq = res.get_id()[1]
            ids.append(f"{pdb_name}_{chain}_{seq}")
            pdb_names.append(pdb_name)

        return (X, y, ids, pdb_names), None

    except Exception as exc:
        return None, f"processing_failed: {exc}"

def read_ids_file(ids_file: str) -> set[str]:
    """
    Expects one sample id per line, typically matching file stem:
      1A0Q_A
      2RH1_B
    """
    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def filter_pdb_paths_by_ids(pdb_paths: list[str], allowed_ids: set[str]) -> list[str]:
    filtered = []
    for p in pdb_paths:
        stem = Path(p).stem
        if stem in allowed_ids:
            filtered.append(p)
    return filtered


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(pdb_paths: list[str], output_path: str, contact_threshold: float = DEFAULT_CONTACT_THRESHOLD, verbose: bool = True,) -> None:
    all_X, all_y, all_ids, all_pdbs = [], [], [], []
    n_ok = 0

    skip_reasons = Counter()
    skip_details = []

    for i, pdb_path in enumerate(pdb_paths):
        if verbose and i % 25 == 0:
            print(f"  [{i}/{len(pdb_paths)}] Processing {Path(pdb_path).name} ...")

        result, skip_reason = process_pdb(pdb_path, contact_threshold=contact_threshold)
        pdb_name = Path(pdb_path).stem

        if result is None:
            skip_reasons[skip_reason] += 1
            skip_details.append({
                "pdb_file": pdb_path,
                "pdb_name": pdb_name,
                "status": "skipped",
                "reason": skip_reason,
            })
            continue

        X, y, ids, pdb_names = result
        all_X.append(X)
        all_y.append(y)
        all_ids.extend(ids)
        all_pdbs.extend(pdb_names)
        n_ok += 1

        skip_details.append({
            "pdb_file": pdb_path,
            "pdb_name": pdb_name,
            "status": "processed",
            "reason": "",
        })

    if not all_X:
        raise RuntimeError("No structures were successfully processed.")

    X_all = np.vstack(all_X).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int8)
    ids_all = np.array(all_ids, dtype=object)
    pdbs_all = np.array(all_pdbs, dtype=object)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y_all,
        ids=ids_all,
        pdbs=pdbs_all,
    )

    # Save processing report
    report_path = Path(output_path).with_suffix(".report.tsv")
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_file", "pdb_name", "status", "reason"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(skip_details)

    pos = int(y_all.sum())
    total = len(y_all)

    print(f"\n✓ Dataset saved to {output_path}")
    print(f"  Structures processed : {n_ok} (skipped: {sum(skip_reasons.values())})")
    print(f"  Total residues       : {total}")
    print(f"  Binding-site (pos)   : {pos} ({100 * pos / total:.1f}%)")
    print(f"  Non-binding (neg)    : {total - pos} ({100 * (total - pos) / total:.1f}%)")

    if skip_reasons:
        print("\n  Skip summary:")
        for reason, count in skip_reasons.most_common():
            print(f"    - {reason}: {count}")

    print(f"\n  Report written to: {report_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build training dataset from PDB files.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb_dir", type=str, help="Directory containing .pdb files.")
    src.add_argument("--scpdb_list", type=str, help="Text file with one PDB ID per line to download from RCSB.")

    parser.add_argument(
        "--download_dir",
        type=str,
        default="data/pdbs",
        help="Directory where downloaded PDBs are stored."
    )
    parser.add_argument(
        "--ids_file",
        type=str,
        default=None,
        help="Optional text file with allowed file stems, one per line (e.g. 1A0Q_A)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.npz",
        help="Output .npz file path."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N structures."
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=DEFAULT_CONTACT_THRESHOLD,
        help="Distance threshold in Å to label a residue as binding (default: 4.5)."
    )

    args = parser.parse_args()

    if args.pdb_dir:
        pdb_paths = sorted(str(p) for p in Path(args.pdb_dir).glob("*.pdb"))
    else:
        dl_dir = Path(args.download_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)

        with open(args.scpdb_list, "r", encoding="utf-8") as f:
            pdb_ids = [line.strip() for line in f if line.strip()]

        print(f"Downloading {len(pdb_ids)} PDB files ...")
        pdb_paths = []
        for pdb_id in pdb_ids:
            p = download_pdb(pdb_id, dl_dir)
            if p is not None:
                pdb_paths.append(str(p))

    if args.ids_file:
        allowed_ids = read_ids_file(args.ids_file)
        before = len(pdb_paths)
        pdb_paths = filter_pdb_paths_by_ids(pdb_paths, allowed_ids)
        print(f"Filtered by ids_file: {before} -> {len(pdb_paths)} structures")

    if args.limit is not None:
        pdb_paths = pdb_paths[:args.limit]

    print(f"Building dataset from {len(pdb_paths)} PDB files ...")
    build_dataset(pdb_paths, args.output, contact_threshold=args.contact_threshold,)


if __name__ == "__main__":
    main()