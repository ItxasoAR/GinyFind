import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

PDB_DIR = ROOT / "data" / "pdbs"
PROCESSED_DIR = ROOT / "data" / "processed"
CHAINS_DIR = PROCESSED_DIR / "chains"
FASTA_PATH = PROCESSED_DIR / "sequences.fasta"
METADATA_PATH = PROCESSED_DIR / "metadata.tsv"

CLUSTERS_DIR = PROCESSED_DIR / "clusters"
SPLITS_DIR = ROOT / "splits"

TRAIN_OUT = PROCESSED_DIR / "train_dataset.npz"
TEST_OUT = PROCESSED_DIR / "test_dataset.npz"


def run(cmd, cwd=ROOT):
    print("\n>>>", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def ensure_dirs():
    for d in [PROCESSED_DIR, CHAINS_DIR, CLUSTERS_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    # 1) Extract chains + sequences
    run([
        sys.executable, "-m", "scripts.extract_chains_and_sequences",
        "--input-dir", PDB_DIR,
        "--chains-dir", CHAINS_DIR,
        "--fasta-out", FASTA_PATH,
        "--metadata-out", METADATA_PATH,
    ])

    # 2) Homology clustering
    run([
        "bash", "scripts/run_mmseqs.sh"
    ])

    # 3) Split by homology (sin validation)
    run([
        sys.executable, "-m", "scripts.split_by_homology",
        "--metadata", METADATA_PATH,
        "--clusters", PROCESSED_DIR / "clusters" / "clusters.tsv",
        "--split-dir", SPLITS_DIR,
        "--train-frac", "0.8",
        "--val-frac", "0.0",
        "--test-frac", "0.2",
    ])

    # 4) Build TRAIN dataset
    run([
        sys.executable, "-m", "data.build_dataset",
        "--pdb_dir", CHAINS_DIR,
        "--ids_file", SPLITS_DIR / "train_ids.txt",
        "--output", TRAIN_OUT,
    ])

    # 5) Build TEST dataset
    run([
        sys.executable, "-m", "data.build_dataset",
        "--pdb_dir", CHAINS_DIR,
        "--ids_file", SPLITS_DIR / "test_ids.txt",
        "--output", TEST_OUT,
    ])

    print("\n✓ Full pipeline finished")
    print(f"  Train: {TRAIN_OUT}")
    print(f"  Test:  {TEST_OUT}")


if __name__ == "__main__":
    main()