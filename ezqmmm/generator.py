"""
QMMMGenerator — thin orchestrator for QM/MM input file generation.

All scientific logic (boundary schemes, switching, geometry) and I/O
(writers, logs) live in their own modules.  This class holds the
MDAnalysis Universe and coordinates the per-frame pipeline.
"""

import warnings
from pathlib import Path
from typing import Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import distances

from ezqmmm import writers
from ezqmmm.boundary import (
    apply_boundary_scheme,
    build_charge_mods,
    find_boundary_bonds,
    place_link_atom,
)
from ezqmmm.config import parse_axes, parse_pdb_stride, validate_config
from ezqmmm.elements import get_element_from_mass
from ezqmmm.geometry import (
    image_shells,
    tile_images,
)
from ezqmmm.models import ChargeMod, SwitchRecord
from ezqmmm.switching import apply_switching



class QMMMGenerator:
    """Generate QM/MM input files from MD trajectories."""

    def __init__(self, psf_file: str, dcd_file: str):
        print("Loading trajectory...")
        print(f"  PSF: {psf_file}")
        print(f"  DCD: {dcd_file}")
        self.universe = mda.Universe(psf_file, dcd_file)
        print(f"  Atoms: {len(self.universe.atoms)}")
        print(f"  Frames: {len(self.universe.trajectory)}")

        # Cache PSF charges — topology reference
        self._psf_charges: dict[int, float] = {
            atom.index: float(atom.charge) for atom in self.universe.atoms
        }

        # PSF files do not carry tempfactors — initialise to zero
        try:
            _ = self.universe.atoms.tempfactors
        except AttributeError:
            self.universe.add_TopologyAttr(
                'tempfactors', np.zeros(len(self.universe.atoms))
            )
            print("  Note: tempfactors not found in PSF — initialized to 0")

    # -----------------------------------------------------------
    # Coordinate extraction
    # -----------------------------------------------------------

    def extract_coordinates(self, qm_selection: str, frame: int):
        """
        Extract QM atom coordinates and place capping hydrogen link atoms
        at 1.09 Å along each QM-MM bond vector.
        Returns list of (element, x, y, z).
        """
        self.universe.trajectory[frame]
        qm_atoms = self.universe.select_atoms(qm_selection)
        coords = [
            (get_element_from_mass(m), p[0], p[1], p[2])
            for m, p in zip(qm_atoms.masses, qm_atoms.positions)
        ]
        for qm_idx, mm_idx in find_boundary_bonds(qm_atoms):
            try:
                lp = place_link_atom(self.universe, qm_idx, mm_idx, frame)
                coords.append(('H', lp[0], lp[1], lp[2]))
            except ValueError:
                pass
        return coords

    # ------------------------------------------------------------------
    # Point-charge extraction
    # ------------------------------------------------------------------

    def extract_point_charges(self, qm_selection: str, cutoff: float,
                              frame: int, boundary_scheme: str,
                              switchdist: Optional[float] = None,
                              expand: tuple[bool, bool, bool] = (False, False, False),
                              target_mm_charge: float = 0.0,
                              neutralize: bool = True,
                              neutralization_shell_fraction: float = 0.1):
        """
        Extract MM point charges for a given frame.

        Returns
        -------
        charges, mods, switch_recs, image_info, mm_ag, qm_center, box
        """
        self.universe.trajectory[frame]
        qm_atoms = self.universe.select_atoms(qm_selection)
        all_atoms = self.universe.select_atoms("all")
        qm_idx_set = set(qm_atoms.indices)
        mm_atoms = [a for a in all_atoms if a.index not in qm_idx_set]
        qm_pos = qm_atoms.positions
        box = self.universe.dimensions

        if not mm_atoms:
            return ([], [], [], {},
                    self.universe.atoms[[]], qm_pos.mean(axis=0), box)

        mm_positions = np.array([a.position for a in mm_atoms])

        # PBC-aware distance for primary MM selection
        dist_matrix = distances.distance_array(qm_pos, mm_positions, box=box)
        min_distances = dist_matrix.min(axis=0)

        # Whole-residue inclusion
        residues_in = set()
        for i, atom in enumerate(mm_atoms):
            if min_distances[i] <= cutoff:
                residues_in.add((atom.segid, atom.resid))

        mm_cut = [a for a in mm_atoms if (a.segid, a.resid) in residues_in]
        mm_ag = self.universe.atoms[np.array([a.index for a in mm_cut])]

        boundary_bonds = find_boundary_bonds(qm_atoms)

        if not boundary_bonds or boundary_scheme == 'NONE':
            primary_charges = [(a.charge, *a.position) for a in mm_cut]
            raw_mods = []
        else:
            primary_charges, raw_mods = apply_boundary_scheme(
                self.universe, mm_cut, boundary_bonds, boundary_scheme
            )

        # Remap primary charges to minimum image
        qm_center = qm_pos.mean(axis=0)
        lx, ly, lz = box[0], box[1], box[2]
        primary_charges = [
            (q,
             x - np.floor((x - qm_center[0]) / lx + 0.5) * lx,
             y - np.floor((y - qm_center[1]) / ly + 0.5) * ly,
             z - np.floor((z - qm_center[2]) / lz + 0.5) * lz)
            for q, x, y, z in primary_charges
        ]

        # Tile periodic images
        image_info = {}
        if any(expand):
            image_charges, shells, n_cand = tile_images(
                primary_charges, qm_pos, cutoff, box, expand
            )
            nx, ny, nz = shells
            image_info = {
                'nx': nx, 'ny': ny, 'nz': nz,
                'n_images': len(image_charges),
                'n_candidates': n_cand,
                'lx': box[0], 'ly': box[1], 'lz': box[2],
            }
            all_charges = primary_charges + image_charges
        else:
            all_charges = primary_charges

        # Switching
        switch_recs = []
        if switchdist is not None:
            all_charges, switch_recs = apply_switching(
                all_charges, qm_pos, switchdist, cutoff,
                box=None, frame=frame, n_primary=len(primary_charges),
            )

        # Charge neutralization — runs LAST, after tiling and switching,
        # so it sees the final charge set and corrects to the exact target.
        if neutralize and all_charges:
            total_q = sum(q for q, x, y, z in all_charges)
            residual = total_q - target_mm_charge
            if abs(residual) > 1e-6:
                positions = np.array([[x, y, z] for _, x, y, z in all_charges])
                all_dists = distances.distance_array(
                    qm_pos, positions, box=None
                ).min(axis=0)
                sorted_idx = np.argsort(all_dists)[::-1]
                qs_arr = np.array([q for q, x, y, z in all_charges])
                nonzero = np.where(np.abs(qs_arr) > 1e-4)[0]
                outer_pool = [i for i in sorted_idx if i in set(nonzero.tolist())]
                n_outer = max(1, int(len(outer_pool) * neutralization_shell_fraction))
                outer_idx = set(outer_pool[:n_outer])
                correction = -residual / n_outer
                all_charges = [
                    (q + correction, x, y, z) if i in outer_idx else (q, x, y, z)
                    for i, (q, x, y, z) in enumerate(all_charges)
                ]

        # Sanity check: verify final MM charge matches target
        if neutralize and all_charges:
            final_q = sum(q for q, x, y, z in all_charges)
            deviation = abs(final_q - target_mm_charge)
            if deviation > 0.01:
                warnings.warn(
                    f"Frame {frame}: MM charge after neutralization "
                    f"({final_q:+.4f} e) deviates from target "
                    f"({target_mm_charge:+.4f} e) by {deviation:.4f} e. "
                    f"This may indicate a bug in the charge pipeline.",
                    stacklevel=2,
                )

        mods = build_charge_mods(raw_mods, frame, qm_center, box,
                                 self._psf_charges)
        return all_charges, mods, switch_recs, image_info, mm_ag, qm_center, box

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(self, config: dict):
        """Run the full QM/MM input generation pipeline."""
        # --- Parse config ---
        qm_sel = config['qm_selection']
        mm_cutoff = config.get('mm_cutoff', 40.0)
        mm_switchdist = config.get('mm_switchdist')
        expand = parse_axes(config.get('supercell_axes', []))
        supercell_on = any(expand)
        neutralize_mm = config.get('neutralize_mm_charge', True)
        target_mm_charge = config.get('target_mm_charge', 0.0)
        pdb_stride = parse_pdb_stride(config.get('pdb_stride'))
        neutral_frac = config.get('neutralization_shell_fraction', 0.1)

        first = config.get('first_frame', 0)
        last = config.get('last_frame', -1)
        if last == -1 or last >= len(self.universe.trajectory):
            last = len(self.universe.trajectory) - 1
        stride = config.get('stride', 1)

        method = config.get('method', 'B3LYP')
        basis = config.get('basis', '6-31G*')
        charge = config.get('charge', 0)
        mult = config.get('multiplicity', 1)
        bscheme = config.get('boundary_scheme', 'RCD').upper()

        output_dir = Path(config.get('output_dir', '.'))
        prefix = config.get('output_prefix', 'qmmm')
        program = config['program'].lower()

        keywords = config.get(f'{program}_keywords', '') or ''
        custom_blocks = config.get(f'{program}_blocks', '') or ''

        # --- Validate ---
        validate_config(config, len(self.universe.trajectory))

        # Dry-run QM selection
        self.universe.trajectory[first]
        try:
            qm_test = self.universe.select_atoms(qm_sel)
        except Exception as e:
            raise ValueError(
                f"qm_selection '{qm_sel}' is invalid: {e}"
            ) from e
        if len(qm_test) == 0:
            raise ValueError(
                f"qm_selection '{qm_sel}' matched 0 atoms on frame {first}. "
                f"Check the selection string — an empty QM region will "
                f"produce NaN coordinates and meaningless output."
            )
        print(f"\n  QM selection validated: {len(qm_test)} atoms on frame {first}")

        # Automatic charge suggestion from topology values
        # This will be only printed on console, not in the log file since the log file is not open yet
        # The same raw sum of charges are also printed as summary for each frame 
        qm_psf_charge = sum(self._psf_charges.get(a.index, 0.0) for a in qm_test)
        suggested = round(qm_psf_charge)
        print(f"  QM charge sum from force field: {qm_psf_charge:+.4f} -> suggested charge: {suggested}")
        print(f"  Note: Double check your selection in case of non-interger values")

        if suggested != charge:
            print(f"  WARNING: Config charge ({charge}) differs too much from force field sum ({suggested})")

        test_bonds = find_boundary_bonds(qm_test)

        if test_bonds:
            print(f"\n  Boundary bonds ({len(test_bonds)} QM-MM cuts):")
            for qm_idx, mm_idx in test_bonds:
                qm_a = self.universe.atoms[qm_idx]
                mm_a = self.universe.atoms[mm_idx]
                qm_elem = get_element_from_mass(qm_a.mass)
                mm_elem = get_element_from_mass(mm_a.mass)
                print(f"    {qm_a.segid}:{qm_a.resname}{qm_a.resid}:{qm_a.name}"
                      f" -- {mm_a.segid}:{mm_a.resname}{mm_a.resid}:{mm_a.name}"
                      f"  ({qm_elem}-{mm_elem})")
                if qm_elem in ('N', 'O', 'S') or mm_elem in ('N', 'O', 'S'):
                      print(f"    WARNING: Polar bond cut -- only C-C cuts are tested")
                      print(f"    WARNING: The input will still be created. The user should be careful before using them")
        else:
            print(f"\n  Boundary bonds: none")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Open run log — mirrors console output to a file
        log_path = output_dir / f"{prefix}_run.log"
        #tell ruff to ignore the open() block error
        log_fh = open(log_path, 'w') # noqa: SIM115

        def log(msg=''):
            """Print to console and write to log file."""
            print(msg)
            log_fh.write(msg + '\n')

        # --- Print settings ---
        frames = list(range(first, last + 1, stride))
        log("\nSettings:")
        log(f"  Program     : {program.upper()}")
        log(f"  QM          : {method}/{basis}  charge={charge}  mult={mult}")
        log(f"  Boundary    : {bscheme}")
        log(f"  MM cutoff   : {mm_cutoff} Ang")
        if mm_switchdist is not None:
            log(f"  Switching   : {mm_switchdist} Ang -> {mm_cutoff} Ang")
        else:
            log("  Switching   : disabled")
        if supercell_on:
            self.universe.trajectory[first]
            box = self.universe.dimensions
            nx, ny, nz = image_shells(mm_cutoff, box, expand)
            axis_labels = [l for l, e in zip(('x', 'y', 'z'), expand) if e]
            log(f"  Supercell   : axes={','.join(axis_labels)}  |  "
                f"shells x={nx} (Lx={box[0]:.2f} Ang), "
                f"y={ny} (Ly={box[1]:.2f} Ang), "
                f"z={nz} (Lz={box[2]:.2f} Ang)  [first frame]")
        if neutralize_mm:
            log(f"  MM charge   : neutralized to {target_mm_charge:+.4f} e "
                f"(outermost {neutral_frac*100:.0f}% of charges adjusted)")
        else:
            log("  MM charge   : no neutralization (raw PSF charges used)")
        if pdb_stride:
            log(f"  PDB/PSF     : every {pdb_stride} frame(s)")
            psf_dest = writers.write_topology(config['psf_file'], output_dir, prefix)
            log(f"  Topology    : {psf_dest}")
        log(f"  Frames      : {len(frames)}")
        log(f"  Log file    : {log_path}")

        # --- Frame loop ---
        all_mods: list[ChargeMod] = []
        all_switch: list[SwitchRecord] = []
        generated = []

        # Charge tracking across frames
        qm_charges_per_frame = []
        mm_charges_per_frame = []

        writer_fn = {
            'orca': ('_orca.inp', writers.write_orca),
            'qchem': ('_qchem.in', writers.write_qchem),
            'psi4': ('_psi4.dat', writers.write_psi4),
        }

        log(f"\n{'Frame':>7}  {'QM atoms':>8}  {'MM charges':>10}  "
            f"{'QM q(PSF)':>10}  {'MM q(final)':>11}  "
            f"{'mods':>4}  {'switched':>8}"
            + ("  " + "images".rjust(8) if supercell_on else ""))
        log("-" * (76 + (10 if supercell_on else 0)))

        for i, frame in enumerate(frames, 1):
            coords = self.extract_coordinates(qm_sel, frame)

            charges, mods, sw_recs, img_inf, mm_ag, qm_center, box = \
                self.extract_point_charges(
                    qm_sel, mm_cutoff, frame, bscheme, mm_switchdist, expand,
                    target_mm_charge, neutralize_mm, neutral_frac
                )

            all_mods.extend(mods)
            all_switch.extend(sw_recs)

            # Compute charges
            qm_atoms = self.universe.select_atoms(qm_sel)
            qm_q = sum(self._psf_charges.get(a.index, 0.0) for a in qm_atoms)
            mm_q = sum(q for q, x, y, z in charges)
            qm_charges_per_frame.append(qm_q)
            mm_charges_per_frame.append(mm_q)

            img_str = (f"  {img_inf.get('n_images', 0):8d}"
                       if supercell_on else "")
            log(f"  {frame:5d}  {len(coords):8d}  {len(charges):10d}  "
                f"{qm_q:+10.4f}  {mm_q:+11.4f}  "
                f"{len(mods):4d}  {len(sw_recs):8d}{img_str}")

            base = output_dir / f"{prefix}_frame{frame}"
            suffix, write_fn = writer_fn[program]
            fname = str(base) + suffix
            write_fn(fname, coords, charges, method, basis,
                     charge, mult, keywords, custom_blocks)
            generated.append(fname)

            if pdb_stride and (i % pdb_stride == 0 or i == 1):
                writers.write_structure(self.universe, frame, qm_atoms,
                                        mm_ag, base, qm_center, box)

        # --- Charge summary ---
        log("\nCharge summary:")
        qm_arr = np.array(qm_charges_per_frame)
        mm_arr = np.array(mm_charges_per_frame)
        log(f"  QM PSF charge  :  mean={qm_arr.mean():+.4f}  "
            f"min={qm_arr.min():+.4f}  max={qm_arr.max():+.4f}  "
            f"std={qm_arr.std():.4f}")
        log(f"  MM final charge:  mean={mm_arr.mean():+.4f}  "
            f"min={mm_arr.min():+.4f}  max={mm_arr.max():+.4f}  "
            f"std={mm_arr.std():.4f}")
        if neutralize_mm:
            log(f"  MM target      :  {target_mm_charge:+.4f} e")

        # --- Write logs ---
        if all_mods:
            bpath = output_dir / f"{prefix}_boundary.log"
            with open(bpath, 'w') as fh:
                writers.write_boundary_log(fh, all_mods)
            log(f"\n  Boundary log  -> {bpath}")

        spath = output_dir / f"{prefix}_switching.log"
        with open(spath, 'w') as fh:
            writers.write_switching_log(fh, all_switch, mm_switchdist,
                                        mm_cutoff, expand)
        log(f"  Switching log -> {spath}")
        log(f"  Run log       -> {log_path}")

        log(f"\nGenerated {len(generated)} input files")
        log_fh.close()
        return generated
