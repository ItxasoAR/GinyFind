#!/usr/bin/env bash
set -euo pipefail

FASTA_PATH="${1:-data/processed/sequences.fasta}"
OUT_DIR="${2:-data/processed/clusters}"
TMP_DIR="${3:-tmp_mmseqs}"
MIN_SEQ_ID="${MIN_SEQ_ID:-0.30}"
COVERAGE="${COVERAGE:-0.80}"
CLUSTER_MODE="${CLUSTER_MODE:-0}"
COV_MODE="${COV_MODE:-0}"

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "Error: mmseqs was not found in PATH. Install MMseqs2 first."
  exit 1
fi

mkdir -p "$OUT_DIR"

SEQ_DB="$OUT_DIR/seqDB"
CLU_DB="$OUT_DIR/cluDB"
CLUSTERS_TSV="$OUT_DIR/clusters.tsv"

# Clean previous outputs
rm -rf "$SEQ_DB"*
rm -rf "$CLU_DB"*
rm -rf "$CLUSTERS_TSV"
rm -rf "$TMP_DIR"

echo "Running MMseqs2 clustering..."
echo "MIN_SEQ_ID: $MIN_SEQ_ID"
echo "COVERAGE: $COVERAGE"
echo "CLUSTER_MODE: $CLUSTER_MODE"
echo "COV_MODE: $COV_MODE"

echo "Running MMseqs2 clustering..."
echo "Input FASTA: $FASTA_PATH"

mmseqs createdb "$FASTA_PATH" "$SEQ_DB"

mmseqs cluster "$SEQ_DB" "$CLU_DB" "$TMP_DIR" \
  --min-seq-id "$MIN_SEQ_ID" \
  -c "$COVERAGE" \
  --cov-mode "$COV_MODE" \
  --cluster-mode "$CLUSTER_MODE"

mmseqs createtsv "$SEQ_DB" "$SEQ_DB" "$CLU_DB" "$CLUSTERS_TSV"

echo "✓ Clusters written to $CLUSTERS_TSV"