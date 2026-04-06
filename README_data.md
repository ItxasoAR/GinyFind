# Data Processing Pipeline (GinyFind)

------------------------------------------------------------------------

## Overview

This document describes the full data processing pipeline used to
generate the datasets for training and evaluating the GinyFind model.

The pipeline transforms raw protein structures (PDB files) into
residue-level feature datasets suitable for machine learning. It
includes chain extraction, homology filtering, dataset splitting, and
feature computation.

All steps are fully automated and implemented in the `scripts/`
directory.

------------------------------------------------------------------------

## Pipeline Summary

The dataset construction follows these steps:

1.  Extraction of protein chains and sequences from an original PDB list
2.  Homology clustering (MMseqs2)
3.  Train/test split based on homology
4.  Residue-level feature computation
5.  Dataset generation for machine learning

------------------------------------------------------------------------

## Input Data

-   Source: Protein Data Bank (PDB)
-   Format: `.pdb` files
-   Initial input: list of PDB identifiers (`lista.txt`)

The dataset is automatically constructed from a predefined list of PDB
IDs.  
Protein structures are downloaded programmatically using the provided
script:

    python -m scripts.download_pdbs

Downloaded structures are stored in: `data/pdbs/`

------------------------------------------------------------------------

## Step 1 — Chain Extraction and Sequence Generation

**Script:** `scripts/extract_chains_and_sequences.py`

### Purpose

-   Split full PDB structures into individual protein chains
-   Extract amino acid sequences
-   Generate metadata for downstream processing

### Output

-   Chain PDB files → `data/processed/chains/`
-   FASTA sequences → `data/processed/sequences.fasta`
-   Metadata file → `data/processed/metadata.tsv`

### Notes

-   Non-protein components are removed
-   Chains with insufficient length or quality may be excluded

------------------------------------------------------------------------

## Step 2 — Homology Clustering

**Script:** `scripts/run_mmseqs.sh`

### Tool

-   MMseqs2 (Many-against-Many sequence searching)

### Purpose

-   Cluster protein sequences based on similarity
-   Identify homologous chains
-   Prevent data leakage between train and test sets

### Output

-   Cluster assignments → `data/processed/clusters/clusters.tsv`

### Notes

-   This step ensures a non-redundant dataset
-   Highly similar proteins are grouped together

------------------------------------------------------------------------

## Step 3 — Train/Test Split (Homology-aware)

**Script:** `scripts/split_by_homology.py`

### Purpose

-   Split the dataset into training and test sets
-   Ensure that homologous chains are not split across datasets

### Output

-   Train IDs → `splits/train_ids.txt`
-   Test IDs → `splits/test_ids.txt`
-   Metadata summary → `splits/split_metadata.tsv`

### Notes

-   This step avoids overfitting due to sequence similarity
-   Ensures realistic model evaluation

------------------------------------------------------------------------

## Step 4 — Feature Computation

**Module:** `features/`

### Purpose

Each residue is converted into a numerical feature vector describing its
structural and biochemical environment.

### Feature categories

**Structural features**

-   Relative solvent accessibility (rSASA)
-   Residue depth
-   Distance to centroid
-   Secondary structure (DSSP)

**Physicochemical features**

-   Hydrophobicity
-   Charge
-   Side-chain volume

**Neighbourhood features (8Å / 12Å)**

-   Number of neighbouring residues
-   Local density
-   Mean hydrophobicity and B-factor
-   Residue composition

### Notes

-   Features are computed per residue
-   Missing values are handled with default fallbacks

------------------------------------------------------------------------

## Step 5 — Dataset Construction

**Script:** `data/build_dataset.py`

### Purpose

-   Build final machine learning datasets
-   Assign labels (binding vs non-binding)
-   Filter invalid samples

### Output

-   Training dataset → `data/processed/train_dataset.npz`
-   Test dataset → `data/processed/test_dataset.npz`
-   Reports → `*.report.tsv`

------------------------------------------------------------------------

## Full Pipeline Execution

**Script:** `scripts/processing_full_pipeline.py`

### Run the complete pipeline:

    python -m scripts.processing_full_pipeline

This script performs all steps sequentially:

1.  Chain extraction
2.  Homology clustering
3.  Dataset splitting
4.  Dataset construction

------------------------------------------------------------------------

## Directory Structure (Generated Data)

    data/
      pdbs/
      processed/
        chains/
        sequences.fasta
        metadata.tsv
        clusters/
        train_dataset.npz
        test_dataset.npz

------------------------------------------------------------------------

## Reproducibility

The entire dataset generation process is reproducible using the provided
scripts. No manual intervention is required.

------------------------------------------------------------------------

## Notes

-   The preprocessing pipeline is not required for prediction
-   It is provided for reproducibility, model retraining and dataset
    extension

------------------------------------------------------------------------

## Summary

The GinyFind data processing pipeline ensures:

-   Non-redundant datasets (homology filtering)
-   Residue-level representation
-   Robust feature extraction
-   Reproducible dataset generation

This design enables reliable training and evaluation of machine learning
models for ligand-binding site prediction.
