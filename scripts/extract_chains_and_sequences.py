#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "UNK": "X",
}

ATOM_RECORDS = {"ATOM  ", "HETATM"}
ALLOWED_ALTLOCS = {" ", "A", "1"}

# HETATM groups that should NOT be preserved as ligands
NON_LIGAND_RESNAMES = {
    "HOH", "WAT", "H2O",
    "SO4", "PO4", "GOL", "EDO", "PEG",
    "DMS", "ACT", "MPD", "FMT", "EOH",
    "BME", "MES", "TRS", "HED", "ACE",
    "NH2", "NH4", "CAC", "IOD", "CL",
    "NA", "MG", "ZN", "CA", "FE",
    "MN", "NI", "CU", "CO", "CD",
}


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    resseq: int
    icode: str


@dataclass
@dataclass
class ChainData:
    sample_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    length: int
    atom_lines: List[str]
    ligand_count: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract protein chains and ATOM-based sequences from PDB files."
    )
    parser.add_argument("--input-dir", default="data/pdbs", help="Folder with source PDB files.")
    parser.add_argument("--chains-dir", default="data/processed/chains", help="Output folder for per-chain PDB files.")
    parser.add_argument("--fasta-out", default="data/processed/sequences.fasta", help="Output FASTA path.")
    parser.add_argument("--metadata-out", default="data/processed/metadata.tsv", help="Output metadata TSV.")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum chain length to keep.")
    parser.add_argument(
        "--keep-hetatm-aa",
        action="store_true",
        help="Keep HETATM lines for amino-acid-like residues (e.g. MSE)."
    )
    return parser.parse_args()


def is_polymer_residue(record: str, resname: str, keep_hetatm_aa: bool) -> bool:
    """
    Decide whether a residue should count as part of the protein chain sequence.
    """
    if resname not in AA3_TO_AA1:
        return False

    if record == "ATOM  ":
        return True

    # HETATM amino-acid-like residues: keep only if requested, except a few useful defaults
    if record == "HETATM":
        if keep_hetatm_aa:
            return True
        return resname in {"MSE", "SEC", "PYL"}

    return False


def is_relevant_ligand(record: str, resname: str) -> bool:
    """
    Decide whether a HETATM residue should be preserved as ligand in chain PDBs.
    """
    if record != "HETATM":
        return False
    if resname in NON_LIGAND_RESNAMES:
        return False
    if resname in AA3_TO_AA1:
        return False  # polymer-like residue, not ligand
    return True


def parse_pdb_file(pdb_path: Path, min_length: int, keep_hetatm_aa: bool) -> List[ChainData]:
    pdb_id = pdb_path.stem.upper()

    residues_by_chain: Dict[str, Dict[Tuple[int, str], str]] = {}
    atoms_by_chain: Dict[str, List[str]] = {}
    ligand_lines: List[str] = []

    seen_atom_keys = set()
    seen_ligand_keys = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if len(raw_line) < 54:
                continue

            record = raw_line[:6]
            if record not in ATOM_RECORDS:
                continue

            altloc = raw_line[16]
            if altloc not in ALLOWED_ALTLOCS:
                continue

            resname = raw_line[17:20].strip().upper()
            chain_id = raw_line[21].strip() or "_"

            try:
                resseq = int(raw_line[22:26])
            except ValueError:
                continue

            icode = raw_line[26].strip() or " "

            atom_name = raw_line[12:16]
            xyz_key = raw_line[30:54]

            # Case 1: polymer residue kept in the extracted protein chain
            if is_polymer_residue(record, resname, keep_hetatm_aa):
                key = (resseq, icode)
                residues_by_chain.setdefault(chain_id, {})
                atoms_by_chain.setdefault(chain_id, [])

                if key not in residues_by_chain[chain_id]:
                    residues_by_chain[chain_id][key] = AA3_TO_AA1[resname]

                atom_key = ("polymer", chain_id, atom_name, resseq, icode, xyz_key)
                if atom_key in seen_atom_keys:
                    continue
                seen_atom_keys.add(atom_key)
                atoms_by_chain[chain_id].append(raw_line.rstrip("\n"))
                continue

            # Case 2: non-polymer ligand to preserve in ALL extracted chain files
            if is_relevant_ligand(record, resname):
                ligand_key = ("ligand", chain_id, resname, atom_name, resseq, icode, xyz_key)
                if ligand_key in seen_ligand_keys:
                    continue
                seen_ligand_keys.add(ligand_key)
                ligand_lines.append(raw_line.rstrip("\n"))

    chain_data: List[ChainData] = []

    for chain_id, residues in sorted(residues_by_chain.items()):
        ordered_keys = sorted(residues.keys(), key=lambda x: (x[0], x[1]))
        sequence = "".join(residues[k] for k in ordered_keys)

        if len(sequence) < min_length:
            continue

        sample_id = f"{pdb_id}_{chain_id}"

        # Important: keep polymer atoms + ligand lines
        combined_lines = list(atoms_by_chain.get(chain_id, [])) + ligand_lines
        ligand_count = len(ligand_lines)

        chain_data.append(
            ChainData(
                sample_id=sample_id,
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=sequence,
                length=len(sequence),
                atom_lines=combined_lines,
                ligand_count=ligand_count,
            )
        )

    return chain_data


def write_chain_pdb(chain: ChainData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        # Minimal valid PDB header so DSSP/Biopython recognizes the file as PDB
        handle.write(f"HEADER    EXTRACTED CHAIN {chain.sample_id}\n")
        handle.write(f"TITLE     {chain.pdb_id} CHAIN {chain.chain_id}\n")

        serial = 1
        for line in chain.atom_lines:
            if len(line) >= 11:
                line = f"{line[:6]}{serial:5d}{line[11:]}"
            handle.write(line + "\n")
            serial += 1

        handle.write("TER\nEND\n")

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    chains_dir = Path(args.chains_dir)
    fasta_out = Path(args.fasta_out)
    metadata_out = Path(args.metadata_out)

    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    chains_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        raise SystemExit(f"No PDB files found in {input_dir}")

    rows = []
    kept_chains = 0

    with fasta_out.open("w", encoding="utf-8") as fasta_handle:
        for pdb_path in pdb_files:
            chains = parse_pdb_file(
                pdb_path,
                min_length=args.min_length,
                keep_hetatm_aa=args.keep_hetatm_aa
            )

            for chain in chains:
                chain_pdb_path = chains_dir / f"{chain.sample_id}.pdb"
                write_chain_pdb(chain, chain_pdb_path)

                fasta_handle.write(f">{chain.sample_id}\n{chain.sequence}\n")

                rows.append({
                    "sample_id": chain.sample_id,
                    "pdb_file": str(pdb_path),
                    "chain_pdb_file": str(chain_pdb_path),
                    "pdb_id": chain.pdb_id,
                    "chain_id": chain.chain_id,
                    "length": chain.length,
                    "sequence": chain.sequence,
                    "extraction": "ATOM+HETATM",
                    "ligand_count": chain.ligand_count,
                })
                kept_chains += 1

    with metadata_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id", "pdb_file", "chain_pdb_file", "pdb_id", "chain_id",
                "length", "sequence", "extraction", "ligand_count",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(pdb_files)} PDB files")
    print(f"Kept {kept_chains} protein chains")
    print(f"FASTA written to {fasta_out}")
    print(f"Metadata written to {metadata_out}")


if __name__ == "__main__":
    main()