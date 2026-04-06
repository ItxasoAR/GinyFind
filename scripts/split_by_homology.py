#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split samples by homology clusters.")
    parser.add_argument("--metadata", default="metadata/chains.tsv", help="TSV created by extract_chains_and_sequences.py")
    parser.add_argument("--clusters", default="data/processed/clusters/clusters.tsv", help="Representative-member TSV from MMseqs2")
    parser.add_argument("--split-dir", default="splits", help="Directory for split outputs")
    parser.add_argument("--output-metadata", default="splits/split_metadata.tsv", help="Output TSV with split assignment")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_clusters(path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    member_to_rep: Dict[str, str] = {}
    rep_to_members: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            member_to_rep[member] = rep
            rep_to_members[rep].append(member)
    return member_to_rep, rep_to_members


def greedy_assign(cluster_items: List[Tuple[str, List[str]]], targets: Dict[str, int]) -> Dict[str, str]:
    assigned: Dict[str, str] = {}
    current = {k: 0 for k in targets}
    order = sorted(cluster_items, key=lambda x: len(x[1]), reverse=True)

    for rep, members in order:
        # choose the split with the lowest fill ratio
        best_split = min(
            targets.keys(),
            key=lambda split: (current[split] / max(targets[split], 1), current[split], split),
        )
        assigned[rep] = best_split
        current[best_split] += len(members)
    return assigned


def main() -> None:
    args = parse_args()
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise SystemExit("train_frac + val_frac + test_frac must equal 1.0")

    metadata_path = Path(args.metadata)
    clusters_path = Path(args.clusters)
    split_dir = Path(args.split_dir)
    output_metadata = Path(args.output_metadata)

    split_dir.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(metadata_path)
    member_to_rep, rep_to_members = read_clusters(clusters_path)

    sample_ids = [row["sample_id"] for row in rows]
    missing = [sid for sid in sample_ids if sid not in member_to_rep]
    if missing:
        # singleton clusters for any sample absent from clusters.tsv
        for sid in missing:
            member_to_rep[sid] = sid
            rep_to_members[sid] = [sid]

    all_clusters = list(rep_to_members.items())
    rng = random.Random(args.seed)
    rng.shuffle(all_clusters)

    n_samples = len(sample_ids)
    targets = {
        "train": round(n_samples * args.train_frac),
        "val": round(n_samples * args.val_frac),
        "test": n_samples - round(n_samples * args.train_frac) - round(n_samples * args.val_frac),
    }

    cluster_to_split = greedy_assign(all_clusters, targets)

    for row in rows:
        rep = member_to_rep[row["sample_id"]]
        row["cluster_rep"] = rep
        row["split"] = cluster_to_split[rep]

    # write metadata
    fieldnames = list(rows[0].keys())
    with output_metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # write id lists
    for split in ("train", "val", "test"):
        ids = [row["sample_id"] for row in rows if row["split"] == split]
        with (split_dir / f"{split}_ids.txt").open("w", encoding="utf-8") as handle:
            for sample_id in ids:
                handle.write(sample_id + "\n")

    with (split_dir / "cluster_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["cluster_rep", "n_members", "split"])
        for rep, members in sorted(rep_to_members.items()):
            writer.writerow([rep, len(members), cluster_to_split[rep]])

    counts = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1

    print(f"Total samples: {n_samples}")
    for split in ("train", "val", "test"):
        print(f"{split}: {counts[split]}")
    print(f"Split metadata written to {output_metadata}")


if __name__ == "__main__":
    main()
