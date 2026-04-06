import subprocess

cmd = [
    "python", "-m", "data.build_dataset",
    "--pdb_dir", "data/processed/chains",
    "--ids_file", "splits/train_ids.txt",
    "--output", "data/processed/train_dataset.npz"
]

subprocess.run(cmd)