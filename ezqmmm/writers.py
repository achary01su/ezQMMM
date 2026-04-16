"""
Output writers for QM/MM input files, structures, and logs.

Each writer is a standalone function — no class state required.
"""

import shutil
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple

from ezqmmm.models import ChargeMod, SwitchRecord
from ezqmmm.geometry import remap_positions_by_residue


# ----------------------------------------------------
# QM/MM input writers
# ----------------------------------------------------

def write_orca(fname, coords, charges, method, basis,
               charge, mult, keywords, custom_blocks):
    """Write ORCA input with external point charge file."""
    with open(fname, 'w') as f:
        f.write(f"! {method} {basis}\n")
        for line in keywords.strip().split('\n'):
            if line.strip():
                f.write(f"! {line.strip()}\n")
        f.write("\n")
        if custom_blocks.strip():
            f.write(custom_blocks.strip() + "\n\n")
        if charges:
            pc = fname.replace('.inp', '_charges.pc')
            f.write(f'%pointcharges "{Path(pc).name}"\n\n')
            with open(pc, 'w') as pf:
                pf.write(f"{len(charges)}\n")
                for q, x, y, z in charges:
                    pf.write(f"{q:.6f}  {x:.6f}  {y:.6f}  {z:.6f}\n")
        f.write(f"* xyz {charge} {mult}\n")
        for elem, x, y, z in coords:
            f.write(f"{elem:<4s}  {x:.6f}  {y:.6f}  {z:.6f}\n")
        f.write("*\n")


def write_qchem(fname, coords, charges, method, basis,
                charge, mult, keywords, custom_blocks):
    """Write Q-Chem input with inline $external_charges block."""
    with open(fname, 'w') as f:
        f.write("$molecule\n")
        f.write(f"{charge} {mult}\n")
        for elem, x, y, z in coords:
            f.write(f"{elem:<4s}  {x:.6f}  {y:.6f}  {z:.6f}\n")
        f.write("$end\n\n")
        f.write("$rem\n")
        f.write("   jobtype              sp\n")
        f.write(f"   method               {method}\n")
        f.write(f"   basis                {basis}\n")
        if charges:
            f.write("   qm_mm                true\n")
        for line in keywords.strip().split('\n'):
            if line.strip():
                f.write(f"   {line.strip()}\n")
        f.write("$end\n")
        if custom_blocks.strip():
            f.write("\n" + custom_blocks.strip() + "\n")
        if charges:
            f.write("\n$external_charges\n")
            for q, x, y, z in charges:
                f.write(f"{x:.6f}  {y:.6f}  {z:.6f}  {q:.6f}\n")
            f.write("$end\n")


def write_psi4(fname, coords, charges, method, basis,
               charge, mult, keywords, custom_blocks):
    """
    Write Psi4 input.
    QM coords in Angstrom; point charges converted to Bohr.
    """
    ANG_TO_BOHR = 1.8897259886
    with open(fname, 'w') as f:
        f.write("memory 4 GB\n\n")
        f.write("molecule qmmm {\n")
        f.write(f"  {charge} {mult}\n")
        for elem, x, y, z in coords:
            f.write(f"  {elem:<4s}  {x:.6f}  {y:.6f}  {z:.6f}\n")
        if charges:
            f.write("  no_com\n  no_reorient\n")
        f.write("}\n\n")
        f.write("set {\n")
        f.write(f"  basis {basis}\n")
        for line in keywords.strip().split('\n'):
            if line.strip():
                f.write(f"  {line.strip()}\n")
        f.write("}\n\n")
        if custom_blocks.strip():
            f.write(custom_blocks.strip() + "\n\n")
        if charges:
            f.write("Chrgfield = QMMM()\n")
            for q, x, y, z in charges:
                xb = x * ANG_TO_BOHR
                yb = y * ANG_TO_BOHR
                zb = z * ANG_TO_BOHR
                f.write(f"Chrgfield.extern.addCharge"
                        f"({q:.6f}, {xb:.6f}, {yb:.6f}, {zb:.6f})\n")
            f.write("psi4.set_global_option_python('EXTERN', Chrgfield.extern)\n\n")
        f.write(f"energy('{method}')\n")


# ------------------------------------------------------------------
# Structure output
# ------------------------------------------------------------------

def write_structure(universe, frame: int, qm_atoms, mm_ag, base: Path,
                    qm_center: np.ndarray, box: np.ndarray):
    """
    Write a full-system PDB for this frame.
    MM atoms are temporarily remapped to minimum image positions
    near the QM centroid.  Remapping is done per-residue: all atoms
    in a residue receive the same displacement vector (computed from
    the residue's center of geometry) so that internal bond lengths
    are preserved even for residues that straddle the periodic boundary.
    Beta column: 9 = QM, 8 = MM within cutoff, 0 = outside cutoff.
    """
    qm_idx = set(qm_atoms.indices)
    mm_idx = set(mm_ag.indices)
    all_atoms = universe.select_atoms("all")

    orig_mm_pos = mm_ag.positions.copy()
    mm_ag.positions = remap_positions_by_residue(
        mm_ag, orig_mm_pos, qm_center, box
    )

    beta = np.zeros(len(all_atoms))
    if mm_idx:
        beta[np.array(list(mm_idx), dtype=int)] = 8.0
    if qm_idx:
        beta[np.array(list(qm_idx), dtype=int)] = 9.0

    orig_tf = all_atoms.tempfactors.copy()
    all_atoms.tempfactors = beta

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_atoms.write(str(base) + "_struct.pdb")

    all_atoms.tempfactors = orig_tf
    mm_ag.positions = orig_mm_pos


def write_topology(psf_source: str, output_dir: Path, prefix: str) -> Path:
    """
    Copy the original PSF once as the shared topology for all PDB frames.
    PSF is frame-independent (no coordinates, no box).
    """
    dest = output_dir / f"{prefix}_struct.psf"
    shutil.copy(psf_source, dest)
    return dest


# ------------------------------------------------------------------
# Log writers
# ------------------------------------------------------------------

def write_boundary_log(fh, all_mods: List[ChargeMod]):
    """Write detailed boundary charge modification log."""
    fh.write("=" * 72 + "\n")
    fh.write("ezQMMM 2.0  Boundary Charge Modification Detail\n")
    fh.write("=" * 72 + "\n\n")
    fh.write(f"{'Frame':<8} {'Type':<10} {'Atom / Site':<28} "
             f"{'PSF q':>8} {'Applied q':>10} {'Delta q':>10}  Reason\n")
    fh.write("-" * 72 + "\n")

    for m in sorted(all_mods, key=lambda x: (x.frame, x.mod_type)):
        tag = (f"{m.segid}:{m.resid} {m.resname} {m.name}"
               if m.atom_index is not None
               else f"virtual@{m.position[0]:.1f},{m.position[1]:.1f}")
        fh.write(
            f"{m.frame:<8d} {m.mod_type:<10s} {tag:<28s} "
            f"{m.psf_charge:+8.4f} {m.applied_charge:+10.4f} "
            f"{m.delta:+10.4f}  {m.reason}\n"
        )

    n_rem = sum(1 for m in all_mods if m.mod_type == 'removed')
    n_mod = sum(1 for m in all_mods if m.mod_type == 'modified')
    n_vir = sum(1 for m in all_mods if m.mod_type == 'virtual')
    fh.write("\n" + "-" * 72 + "\n")
    fh.write(f"Total: removed={n_rem}  modified={n_mod}  "
             f"virtual={n_vir}  total={len(all_mods)}\n")


def write_switching_log(fh, all_switch: List[SwitchRecord],
                        switchdist: float, cutoff: float,
                        expand: Tuple[bool, bool, bool]):
    """Write switching-function charge modification log."""
    supercell_on = any(expand)
    axis_labels = ','.join(l for l, e in zip(('x', 'y', 'z'), expand) if e)
    sw_label = (f"{switchdist:.2f} - {cutoff:.2f} Ang"
                if switchdist is not None else "disabled")
    fh.write("=" * 72 + "\n")
    fh.write("ezQMMM 2.0  Switching-Function Charge Modifications\n")
    fh.write(f"  Switching zone  : {sw_label}\n")
    fh.write(f"  Function        : quintic (NAMD-style)\n")
    fh.write(f"  Distance metric : minimum distance to any QM atom "
             f"(not center of mass)\n")
    if supercell_on:
        fh.write(f"  Image charges   : included (axes: {axis_labels}); "
                 f"marked in Src column as IMG\n")
    # Fixed the bug when the writer did not recognize None as a float
    #fh.write(f"  Recorded        : only charges with scale < 1 "
    #         f"(dist > {switchdist:.2f} Ang)\n")
    if switchdist is not None:
        fh.write(f"  Recorded        : only charges with scale < 1 "
               f"(dist > {switchdist:.2f} Ang)\n")
    fh.write("=" * 72 + "\n\n")

    src_col = "  Src" if supercell_on else ""
    fh.write(f"{'Frame':<8} {'Dist(Ang)':>9} {'Scale':>8} "
             f"{'q_orig':>10} {'q_scaled':>10} {'Delta q':>10}  "
             f"x(Ang)    y(Ang)    z(Ang){src_col}\n")
    fh.write("-" * 72 + "\n")

    prev_frame = None
    for r in sorted(all_switch, key=lambda x: (x.frame, x.dist)):
        if r.frame != prev_frame:
            fh.write(f"\n--- Frame {r.frame} ---\n")
            prev_frame = r.frame
        delta = r.scaled_charge - r.psf_charge
        src_tag = "  IMG" if (supercell_on and r.is_image) else "     "
        fh.write(
            f"{r.frame:<8d} {r.dist:>9.3f} {r.scale:>8.5f} "
            f"{r.psf_charge:>+10.4f} {r.scaled_charge:>+10.4f} "
            f"{delta:>+10.4f}  "
            f"{r.position[0]:>8.3f}  {r.position[1]:>8.3f}  "
            f"{r.position[2]:>8.3f}{src_tag}\n"
        )

    n_prim = sum(1 for r in all_switch if not r.is_image)
    n_img = sum(1 for r in all_switch if r.is_image)
    fh.write("\n" + "-" * 72 + "\n")
    if supercell_on:
        fh.write(f"Total switching-zone events: {len(all_switch)}  "
                 f"(primary={n_prim}, image={n_img})\n")
    else:
        fh.write(f"Total switching-zone charge events: {len(all_switch)}\n")
