#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from data.build_dataset import download_pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Download PDB files only.")
    parser.add_argument(
        "--ids-file",
        required=True,
        help="Text file with one PDB ID per line (e.g. 1A0Q)"
    )
    parser.add_argument(
        "--out-dir",
        default="data/pdbs",
        help="Directory where downloaded PDB files will be stored"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the .pdb file already exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ids_file, "r", encoding="utf-8") as f:
        pdb_ids = [line.strip().replace(".pdb", "") for line in f if line.strip()]

    ok = 0
    fail = 0
    skipped = 0

    for pdb_id in pdb_ids:
        out_path = out_dir / f"{pdb_id.upper()}.pdb"

        if args.force and out_path.exists():
            out_path.unlink()

        if out_path.exists():
            skipped += 1
            continue

        p = download_pdb(pdb_id, out_dir)
        if p is None:
            fail += 1
        else:
            ok += 1

    print(f"Downloaded: {ok}")
    print(f"Already present: {skipped}")
    print(f"Failed: {fail}")


if __name__ == "__main__":
    main()