# Overview

------------------------------------------------------------------------

GinyFind is a tool for predicting ligand binding sites in protein
structures using a trained machine learning model. Given a PDB file, the
program:

1.  Extracts protein chains automatically
2.  Computes structural and physicochemical features
3.  Predicts binding site probability per residue
4.  Groups residues into binding sites

# Installation

------------------------------------------------------------------------

### Recommended (install as package)

GinyFind can be installed as a Python package:

    pip install .

Alternatively, for development:

    pip install -e .

After installation, the following commands are available:

    ginyfind_predict --pdb input.pdb

    ginyfind_train --dataset data/processed/train_dataset.npz --model RF

    ginyfind_evaluate --dataset data/processed/test_dataset.npz

### Non-installed version

Biopython is required for GinyFind. To use GinyFind the following
installation needs to be carried out:

### 1 Python dependencies

    pip install -r requirements.txt

### 2 DSSP (secondary structure assignment)

**Used for the ss\_helix / ss\_strand / ss\_coil features. Without it
those features default to “coil” (zeros).**

-   **Linux (Debian/Ubuntu):**

<!-- -->

    sudo apt install dssp

-   **macOS:**

<!-- -->

    brew install dssp

-   **Windows: install via WSL (Ubuntu), then use the Linux command
    above.**

### 3 MSMS (residue depth — optional)

Used for ca\_depth and mean\_depth features. The model still runs
without it; those features are set to 0. However, prediction performance
may be affected. *Download the binary for your OS from:*

`https://ccsb.scripps.edu/msms/downloads/`

*Then add the extracted folder to your PATH:*

    export PATH="/path/to/msms:$PATH"   # add to ~/.bashrc or ~/.zshrc

### Verify everything is found by BioPython:

    python - <<'EOF'
    from Bio.PDB import DSSP
    from Bio.PDB.ResidueDepth import ResidueDepth
    print("DSSP and MSMS appear to be available")
    EOF

## How to run a prediction?

------------------------------------------------------------------------

Run prediction on a PDB file:

**Installed**

    ginyfind_predict --pdb input.pdb

**Non-installed**

    python -m model.predict [-h] --pdb input.pdb 

### Available Commands

GinyFind provides three main command-line tools:

-   **ginyfind\_predict** → Predict ligand binding sites from a PDB
    file  
-   **ginyfind\_train** → Train a new machine learning model  
-   **ginyfind\_evaluate** → Evaluate model performance on a dataset

Equivalent for non-installed version

    python -m model.predict 
    python -m model.train
    python -m model.evaluate

### Optional parameters for GinyFind

Example: prediction optional commands.

    python -m model.predict [-h] --pdb input.pdb [--model MODEL] [--threshold THRESHOLD] [--eps EPS]
                                   [--min_samples MIN_SAMPLES] [--output_dir OUTPUT_DIR]

-   *model*: Path to trained model (default: built-in model)
-   *threshold*: Probability threshold (default: use the saved one in
    the model)
-   *eps*: DBSCAN eps in Å (default: 8.0)
-   *min\_samples*: DBSCAN min\_samples (default: 3)
-   *output\_dir*: Directory for output files (default: output/)

**Full help for each command can be displayed with:**

    ginyfind_predict --help
    ginyfind_train --help
    ginyfind_evaluate --help

## Input

-   Input format: PDB file
-   The program accepts full PDB structures
-   Protein chains are automatically extracted internally

## Output

------------------------------------------------------------------------

The program generates:

-   Console report of predicted binding sites for the whole input
    structure.
-   Text report of predicted binding sites with probabilities per
    residue in:`output/`(By default).
-   Structured output files for visualization in Chimera and Pymol
    in:`output/`(By default).

## Model

Ligand binding site prediction is formulated as a binary classification
problem at the residue level. Each residue in a protein is classified
as:

-   Binding (1): part of a ligand-binding site.
-   Non-binding (0): not involved in ligand interaction.

The model outputs a probability score for each residue, indicating its
likelihood of being part of a binding site.

<center>
Model algorithm:<b> Random Forest</b>
</center>

### Feature Representation

Each residue is described by a 49-dimensional feature vector, combining:

-   <b> Structural features</b>
    1.  Relative solvent accessibility (rSASA)
    2.  Residue depth
    3.  Distance to protein centroid
    4.  Secondary structure (helix / strand / coil)
-   <b>Physicochemical properties</b>
    1.  Hydrophobicity
    2.  Charge
    3.  Side-chain volume
-   <b>Neighbourhood features</b>: Computed within spatial radii (e.g.,
    8Å and 12Å).
    1.  Number of neighbouring residues
    2.  Local density ratios
    3.  Mean hydrophobicity and B-factor
    4.  Fraction of charged / hydrophobic residues

These features capture both local geometry and biochemical environment.

<center>
→ In detail information about model training and used data in
`README_data.Rmd`
</center>

## Notes

-   No preprocessing required from the user
-   Works directly on raw PDB files
-   Robust to missing data (fallback values used)

## (Optional) Full Pipeline

The full dataset generation pipeline (clustering, feature extraction,
training) is available in: `scripts/`

Run:

    python -m scripts.processing_full_pipeline

**This step is not required for prediction, only for reproducibility.**
Read `README_data.Rmd` for more details.

\*\* IMPORTANT: The repository does not include raw or processed data to
keep it lightweight. All datasets can be fully reconstructed using the
provided pipeline.\*\*

------------------------------------------------------------------------

## Final remark

GinyFind is designed as a lightweight and user-friendly tool, allowing
direct prediction from raw PDB structures without requiring
preprocessing steps from the user.
