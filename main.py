#!/usr/bin/env python3
"""
main.py
--------
Standalone command-line interface for the Ligand Binding Site Predictor.

Sub-commands
------------
  build     Build a training dataset from PDB files or scPDB IDs
  train     Train the Random Forest / Gradient Boosting model
  predict   Predict binding sites on a new PDB file

Examples
--------
# 1 — Build dataset from a directory of PDB files
python main.py build --pdb_dir data/pdbs --output data/dataset.npz

# 2 — Build dataset by downloading from RCSB (list of PDB IDs)
python main.py build --scpdb_list data/scpdb_ids.txt --output data/dataset.npz

# 3 — Train model
python main.py train --dataset data/dataset.npz --model RF

# 4 — Predict on a new structure
python main.py predict --pdb examples/1IEP.pdb --threshold 0.45

# 5 — Full pipeline in one go
python main.py predict --pdb examples/1IEP.pdb \
       --model model/trained_model.pkl --output_dir results/
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command: build
# ---------------------------------------------------------------------------

def cmd_build(args):
    from data.build_dataset import build_dataset, download_pdb
    from pathlib import Path

    if args.pdb_dir:
        pdb_paths = sorted(Path(args.pdb_dir).glob("*.pdb"))
        pdb_paths = [str(p) for p in pdb_paths]
    else:
        dl_dir = Path(args.download_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        with open(args.scpdb_list) as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
        print(f"Downloading {len(pdb_ids)} structures from RCSB …")
        pdb_paths = []
        for pdb_id in pdb_ids:
            p = download_pdb(pdb_id, dl_dir)
            if p:
                pdb_paths.append(str(p))

    if args.limit:
        pdb_paths = pdb_paths[:args.limit]

    print(f"Building dataset from {len(pdb_paths)} PDB files …")
    build_dataset(pdb_paths, args.output)


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def cmd_train(args):
    from model.train import train
    train(
        dataset_path = args.dataset,
        model_type   = args.model,
        output_path  = args.output,
        run_cv       = not args.no_cv,
    )


# ---------------------------------------------------------------------------
# Sub-command: predict
# ---------------------------------------------------------------------------

def cmd_predict(args):
    from model.predict import predict, print_prediction_report
    from output.writer import write_all_outputs
    from features import load_structure
    from Bio.PDB import PDBIO, Select
    from pathlib import Path
    import tempfile, os

    # Split into individual chains and predict each one
    structure = load_structure(args.pdb)
    chains = list(structure.get_chains())

    all_sites    = []
    all_residues = []
    all_proba    = []
    site_counter = 1

    tmp_files = []

    for chain in chains:
        # Write chain to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".pdb", delete=False,
            prefix=f"{Path(args.pdb).stem}_{chain.id}_"
        ) as tmp:
            tmp_path = tmp.name
        tmp_files.append(tmp_path)

        class ChainSelect(Select):
            def accept_chain(self, c):
                return c.id == chain.id

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_path, ChainSelect())

        print(f"\n>>> Chain {chain.id}")
        try:
            sites, residues, proba = predict(
                pdb_path    = tmp_path,
                model_path  = args.model,
                threshold   = args.threshold,
                eps         = args.eps,
                min_samples = args.min_samples,
            )
        except Exception as e:
            print(f"  Skipped: {e}")
            continue

        # Renumber sites globally
        for site in sites:
            site.site_id = site_counter
            site_counter += 1

        print_prediction_report(sites, tmp_path)
        all_sites.extend(sites)
        all_residues.extend(residues)
        all_proba.extend(list(proba))

    # Cleanup temp files
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    import numpy as np
    all_proba = np.array(all_proba)

    print(f"\n=== GLOBAL SUMMARY ===")
    print(f"Total sites predicted: {len(all_sites)}")

    write_all_outputs(
        pdb_path  = args.pdb,
        sites     = all_sites,
        residues  = all_residues,
        proba     = all_proba,
        out_dir   = args.output_dir,
    )

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Ligand Binding Site Predictor — ML-based structure analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ──────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Build training dataset")
    src = p_build.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb_dir",     help="Directory of .pdb files")
    src.add_argument("--scpdb_list",  help="File with PDB IDs to download")
    p_build.add_argument("--download_dir", default="data/pdbs")
    p_build.add_argument("--output",       default="data/dataset.npz")
    p_build.add_argument("--limit",        type=int, default=None)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the ML model")
    p_train.add_argument("--dataset",  required=True)
    p_train.add_argument("--model",    default="RF", choices=["RF", "GBT"],
                         help="RF = Random Forest | GBT = Gradient Boosting")
    p_train.add_argument("--output",   default="model/trained_model.pkl")
    p_train.add_argument("--no_cv",    action="store_true",
                         help="Skip cross-validation")

    # ── predict ────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predict binding sites on a PDB")
    p_pred.add_argument("--pdb",         required=True)
    p_pred.add_argument("--model",       default="model/trained_model.pkl")
    p_pred.add_argument("--threshold",   type=float, default=0.5,
                        help="Probability cutoff (default 0.5)")
    p_pred.add_argument("--eps",         type=float, default=8.0,
                        help="DBSCAN eps in Å (default 8.0)")
    p_pred.add_argument("--min_samples", type=int,   default=3,
                        help="DBSCAN min cluster size (default 3)")
    p_pred.add_argument("--output_dir",  default="output")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "build":   cmd_build,
        "train":   cmd_train,
        "predict": cmd_predict,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
