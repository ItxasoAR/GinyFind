"""
output/writer.py
-----------------
Generates all output files after prediction:

1. **Modified PDB** — same structure but with B-factor column replaced by
   the predicted binding probability × 100, so the protein can be coloured
   by "B-factor" in PyMOL or Chimera to visualise predicted sites.

2. **Text report** — human-readable summary: one section per site with a
   list of involved amino acids.

3. **PyMOL script** — .pml script that opens the coloured PDB and colours
   each site in a distinct colour with automatic selection objects.

4. **Chimera script** — .cxc script equivalent for UCSF Chimera X.
"""

from pathlib import Path
import numpy as np
from Bio.PDB import PDBIO, Select


# ---------------------------------------------------------------------------
# Colour palette (one per site, wraps if >10 sites)
# ---------------------------------------------------------------------------
SITE_COLOURS_PYMOL = [
    "red", "blue", "green", "magenta", "cyan",
    "yellow", "orange", "salmon", "lime", "violet",
]
SITE_COLOURS_CHIMERA = [
    "red", "blue", "lime green", "magenta", "cyan",
    "yellow", "orange", "salmon", "chartreuse", "violet",
]


# ---------------------------------------------------------------------------
# BioPython Select class for writing a single chain / model
# ---------------------------------------------------------------------------
class _AllSelect(Select):
    """Accepts all residues (used to write the full structure)."""
    def accept_residue(self, residue):
        return 1


# ---------------------------------------------------------------------------
# 1. Modified PDB (probability in B-factor column)
# ---------------------------------------------------------------------------

def write_probability_pdb(
    structure,
    residues: list,
    proba: np.ndarray,
    output_path: str,
) -> None:
    """
    Replace B-factor of every atom in each standard residue with the
    predicted binding probability scaled to [0, 100].
    Atoms in non-standard residues (ligands, water) keep their original B.
    """
    # Build a lookup: residue full_id → probability
    prob_lookup = {
        res.get_full_id(): float(p) * 100.0
        for res, p in zip(residues, proba)
    }

    # Mutate B-factors in place (they will be restored after writing)
    original_bfactors = {}
    for residue in structure.get_residues():
        fid = residue.get_full_id()
        if fid in prob_lookup:
            for atom in residue.get_atoms():
                original_bfactors[(fid, atom.get_id())] = atom.get_bfactor()
                atom.set_bfactor(prob_lookup[fid])

    # Write
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, _AllSelect())

    # Restore original B-factors
    for residue in structure.get_residues():
        fid = residue.get_full_id()
        for atom in residue.get_atoms():
            key = (fid, atom.get_id())
            if key in original_bfactors:
                atom.set_bfactor(original_bfactors[key])

    print(f"  ✓ Probability PDB written: {output_path}")


# ---------------------------------------------------------------------------
# 2. Text report
# ---------------------------------------------------------------------------

def write_text_report(
    sites: list,
    pdb_id: str,
    output_path: str,
) -> None:
    """Write a human-readable binding site report."""
    lines = [
        f"Ligand Binding Site Prediction Report",
        f"Structure : {pdb_id}",
        f"Sites found: {len(sites)}",
        "=" * 60,
        "",
    ]

    if not sites:
        lines.append("No binding sites predicted above threshold.")
    else:
        for site in sites:
            cx, cy, cz = site.center
            lines += [
                f"SITE {site.site_id}",
                f"  Residues      : {site.n_residues}",
                f"  Mean prob.    : {site.mean_probability:.3f}",
                f"  Center (Å)    : ({cx:.1f}, {cy:.1f}, {cz:.1f})",
                f"",
                f"  {'Chain':>5} {'ResSeq':>6} {'AA':>4}  {'P(binding)':>10}",
                f"  {'-'*35}",
            ]
            for r in site.residue_summary():
                lines.append(
                    f"  {r['chain']:>5} {r['resseq']:>6} {r['resname']:>4}  "
                    f"{r['prob']:>10.3f}"
                )
            lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"  ✓ Text report written   : {output_path}")


# ---------------------------------------------------------------------------
# 3. PyMOL script
# ---------------------------------------------------------------------------

def write_pymol_script(
    sites: list,
    pdb_filename: str,
    output_path: str,
) -> None:
    """
    Generate a PyMOL .pml script that:
      - Loads the probability PDB
      - Colours everything grey
      - Creates one named selection per site coloured distinctly
      - Colours the structure by B-factor (probability) with a gradient
    """
    lines = [
        f"# PyMOL script — Ligand Binding Site Prediction",
        f"load {pdb_filename}",
        f"",
        f"# Show as surface + cartoon",
        f"hide everything",
        f"show cartoon",
        f"color grey80",
        f"",
        f"# Colour by predicted probability (stored in B-factor × 100)",
        f"spectrum b, white_red, minimum=0, maximum=100",
        f"",
    ]

    for site in sites:
        colour = SITE_COLOURS_PYMOL[(site.site_id - 1) % len(SITE_COLOURS_PYMOL)]
        sel_parts = []
        for r in site.residue_summary():
            sel_parts.append(f"(chain {r['chain']} and resi {r['resseq']})")
        sel_str = " or ".join(sel_parts)
        sel_name = f"site_{site.site_id}"
        lines += [
            f"# Site {site.site_id} — {site.n_residues} residues, "
            f"mean p={site.mean_probability:.3f}",
            f"select {sel_name}, {sel_str}",
            f"show sticks, {sel_name}",
            f"color {colour}, {sel_name}",
            f"",
        ]

    lines += [
        "zoom",
        "deselect",
        "# End of script",
    ]

    Path(output_path).write_text("\n".join(lines))
    print(f"  ✓ PyMOL script written  : {output_path}")


# ---------------------------------------------------------------------------
# 4. Chimera (ChimeraX) script
# ---------------------------------------------------------------------------

def write_chimera_script(
    sites: list,
    pdb_filename: str,
    output_path: str,
) -> None:
    """
    Generate a ChimeraX .cxc script for visualisation.
    """
    lines = [
        f"# ChimeraX script — Ligand Binding Site Prediction",
        f"open {pdb_filename}",
        f"",
        f"# Show as surface",
        f"hide",
        f"show cartoons",
        f"color grey",
        f"",
        f"# Colour by binding probability (B-factor column × 100)",
        f"color bfactor palette 0,white:100,red",
        f"",
    ]

    for site in sites:
        colour = SITE_COLOURS_CHIMERA[(site.site_id - 1) % len(SITE_COLOURS_CHIMERA)]
        res_specs = []
        for r in site.residue_summary():
            res_specs.append(f"/{r['chain']}:{r['resseq']}")
        spec = " ".join(res_specs)
        lines += [
            f"# Site {site.site_id} — {site.n_residues} residues",
            f"color {spec} {colour}",
            f"show {spec} sticks",
            f"",
        ]

    lines += [
        "view",
        "# End of script",
    ]

    Path(output_path).write_text("\n".join(lines))
    print(f"  ✓ Chimera script written: {output_path}")


# ---------------------------------------------------------------------------
# Master output function
# ---------------------------------------------------------------------------

def write_all_outputs(
    pdb_path: str,
    sites: list,
    residues: list,
    proba: np.ndarray,
    out_dir: str = "output",
) -> None:
    """
    Write all four output files for a prediction run.

    Files created:
        <out_dir>/<pdb_id>_predicted.pdb
        <out_dir>/<pdb_id>_report.txt
        <out_dir>/<pdb_id>_pymol.pml
        <out_dir>/<pdb_id>_chimera.cxc
    """
    from features import load_structure

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdb_id = Path(pdb_path).stem
    structure = load_structure(pdb_path)

    prob_pdb  = str(out / f"{pdb_id}_predicted.pdb")
    report    = str(out / f"{pdb_id}_report.txt")
    pymol_scr = str(out / f"{pdb_id}_pymol.pml")
    chim_scr  = str(out / f"{pdb_id}_chimera.cxc")

    print(f"\nWriting output files to {out_dir}/")
    write_probability_pdb(structure, residues, proba, prob_pdb)
    write_text_report(sites, pdb_id, report)
    write_pymol_script(sites, prob_pdb, pymol_scr)
    write_chimera_script(sites, prob_pdb, chim_scr)
    print(f"\n✓ All outputs written for {pdb_id}\n")
