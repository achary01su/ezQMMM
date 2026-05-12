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
    remap_positions_by_compound,
    tile_images,
)
from ezqmmm.models import ChargeMod, SwitchRecord
from ezqmmm.switching import apply_switching
from ezqmmm import __version__


import time
import multiprocessing as mp
from functools import partial

# --- Parallel worker state (module-level, multiprocessing) ---
_worker_universe = None
_worker_psf_charges = None

def _init_worker(psf_file, dcd_file):
    """Called once per worker process. Loads its own Universe."""
    global _worker_universe, _worker_psf_charges
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module="MDAnalysis")
    _worker_universe = mda.Universe(psf_file, dcd_file)
    _worker_psf_charges = {
        atom.index: float(atom.charge)
        for atom in _worker_universe.atoms
    }
    try:
        _ = _worker_universe.atoms.tempfactors
    except AttributeError:
        _worker_universe.add_TopologyAttr(
            'tempfactors', np.zeros(len(_worker_universe.atoms))
        )

def _process_frame(args):
    """
    Process a single frame in a worker process.
    Writes QM input file (and optionally PDB) directly.
    Returns small metadata dict — no large arrays cross the process boundary.
    """
    (frame, qm_sel, mm_cutoff, bscheme, mm_switchdist, expand,
     cs_bond_frac, pbc_compound, target_mm_charge, neutralize_mm,
     neutral_frac, method, basis, charge, mult, program,
     keywords, custom_blocks, output_dir, prefix, pdb_stride,
     frame_index) = args

    u = _worker_universe
    psf_charges = _worker_psf_charges

    # --- Extract QM coordinates ---
    u.trajectory[frame]
    qm_atoms = u.select_atoms(qm_sel)
    coords = [
        (get_element_from_mass(m), p[0], p[1], p[2])
        for m, p in zip(qm_atoms.masses, qm_atoms.positions)
    ]
    for qm_idx, mm_idx in find_boundary_bonds(qm_atoms):
        try:
            lp = place_link_atom(u, qm_idx, mm_idx, frame)
            coords.append(('H', lp[0], lp[1], lp[2]))
        except ValueError:
            pass

    # --- Extract point charges ---
    # (This replicates extract_point_charges but uses the worker's universe)
    qm_idx_set = set(qm_atoms.indices)
    all_atoms = u.select_atoms("all")
    mm_atoms = [a for a in all_atoms if a.index not in qm_idx_set]
    qm_pos = qm_atoms.positions
    box = u.dimensions

    if not mm_atoms:
        return {
            'frame': frame, 'fname': None,
            'n_coords': len(coords), 'n_charges': 0,
            'qm_q': 0.0, 'mm_q': 0.0,
            'n_mods': 0, 'n_switch': 0, 'n_images': 0,
            'mods': [], 'switch_recs': [],
        }

    mm_positions = np.array([a.position for a in mm_atoms])
    dist_matrix = distances.distance_array(qm_pos, mm_positions, box=box)
    min_distances = dist_matrix.min(axis=0)

    residues_in = set()
    for i, atom in enumerate(mm_atoms):
        if min_distances[i] <= mm_cutoff:
            residues_in.add((atom.segid, atom.resid))

    mm_cut = [a for a in mm_atoms if (a.segid, a.resid) in residues_in]
    mm_ag = u.atoms[np.array([a.index for a in mm_cut])]

    boundary_bonds = find_boundary_bonds(qm_atoms)
    qm_center = qm_pos.mean(axis=0)

    # Remap
    orig_positions = mm_ag.positions.copy()
    mm_ag.positions = remap_positions_by_compound(
        mm_ag, orig_positions, qm_center, box,
        compound=pbc_compound,
    )

    try:
        if not boundary_bonds or bscheme == 'NONE':
            primary_charges = [(a.charge, *a.position) for a in mm_cut]
            raw_mods = []
        else:
            primary_charges, raw_mods = apply_boundary_scheme(
                u, mm_cut, boundary_bonds, bscheme,
                cs_bond_fraction=cs_bond_frac,
            )

        # Tile periodic images
        image_info = {}
        if any(expand):
            image_charges, shells, n_cand = tile_images(
                primary_charges, qm_pos, mm_cutoff, box, expand
            )
            image_info = {
                'nx': shells[0], 'ny': shells[1], 'nz': shells[2],
                'n_images': len(image_charges), 'n_candidates': n_cand,
            }
            all_charges = primary_charges + image_charges
        else:
            all_charges = primary_charges

        # Switching
        switch_recs = []
        if mm_switchdist is not None:
            all_charges, switch_recs = apply_switching(
                all_charges, qm_pos, mm_switchdist, mm_cutoff,
                box=None, frame=frame, n_primary=len(primary_charges),
            )

        # Neutralization
        if neutralize_mm and all_charges:
            total_q = sum(q for q, x, y, z in all_charges)
            residual = total_q - target_mm_charge
            if abs(residual) > 1e-6:
                positions_arr = np.array([[x, y, z] for _, x, y, z in all_charges])
                all_dists = distances.distance_array(
                    qm_pos, positions_arr, box=None
                ).min(axis=0)
                sorted_idx = np.argsort(all_dists)[::-1]
                qs_arr = np.array([q for q, x, y, z in all_charges])
                nonzero = np.where(np.abs(qs_arr) > 1e-4)[0]
                outer_pool = [i for i in sorted_idx if i in set(nonzero.tolist())]
                n_outer = max(1, int(len(outer_pool) * neutral_frac))
                outer_idx = set(outer_pool[:n_outer])
                correction = -residual / n_outer
                all_charges = [
                    (q + correction, x, y, z) if i in outer_idx else (q, x, y, z)
                    for i, (q, x, y, z) in enumerate(all_charges)
                ]

        # Build charge mods as plain dicts (picklable)
        mods_dicts = []
        for d in raw_mods:
            if d['type'] == 'virtual':
                mods_dicts.append({
                    'frame': frame, 'mod_type': 'virtual',
                    'reason': d['reason'], 'psf_charge': 0.0,
                    'applied_charge': d['charge'],
                    'position': list(d['position']),
                })
            else:
                atom = d['atom']
                psf_q = psf_charges.get(atom.index, d['old_charge'])
                mods_dicts.append({
                    'frame': frame, 'mod_type': d['type'],
                    'reason': d['reason'], 'psf_charge': psf_q,
                    'applied_charge': d['new_charge'],
                    'position': list(d['position']),
                    'atom_index': atom.index, 'segid': atom.segid,
                    'resid': atom.resid, 'resname': atom.resname,
                    'name': atom.name,
                })

        # Switch records as plain dicts
        switch_dicts = []
        for r in switch_recs:
            switch_dicts.append({
                'frame': r.frame, 'psf_charge': r.psf_charge,
                'scaled_charge': r.scaled_charge, 'scale': r.scale,
                'dist': r.dist, 'position': list(r.position),
                'is_image': r.is_image,
            })

        # --- Write QM input file ---
        writer_fn = {
            'orca': ('_orca.inp', writers.write_orca),
            'qchem': ('_qchem.in', writers.write_qchem),
            'psi4': ('_psi4.dat', writers.write_psi4),
        }
        base = Path(output_dir) / f"{prefix}_frame{frame}"
        suffix, write_fn = writer_fn[program]
        fname = str(base) + suffix
        write_fn(fname, coords, all_charges, method, basis,
                 charge, mult, keywords, custom_blocks)

        # --- Optionally write PDB ---
        wrote_pdb = False
        if pdb_stride and (frame_index % pdb_stride == 0 or frame_index == 1):
            writers.write_structure(u, frame, qm_atoms, mm_ag,
                                   base, qm_center, box, pbc_compound)
            wrote_pdb = True

        # --- Compute charge stats ---
        qm_q = sum(psf_charges.get(a.index, 0.0) for a in qm_atoms)
        mm_q = sum(q for q, x, y, z in all_charges)

    finally:
        mm_ag.positions = orig_positions

    return {
        'frame': frame,
        'fname': fname,
        'n_coords': len(coords),
        'n_charges': len(all_charges),
        'qm_q': qm_q,
        'mm_q': mm_q,
        'n_mods': len(mods_dicts),
        'n_switch': len(switch_dicts),
        'n_images': image_info.get('n_images', 0),
        'image_info': image_info,
        'mods': mods_dicts,
        'switch_recs': switch_dicts,
        'wrote_pdb': wrote_pdb,
    }


def _dicts_to_mods(mod_dicts):
    """Reconstruct ChargeMod objects from plain dicts."""
    return [
        ChargeMod(
            frame=d['frame'], mod_type=d['mod_type'], reason=d['reason'],
            psf_charge=d['psf_charge'], applied_charge=d['applied_charge'],
            position=np.array(d['position']),
            atom_index=d.get('atom_index'), segid=d.get('segid', ''),
            resid=d.get('resid', 0), resname=d.get('resname', ''),
            name=d.get('name', ''),
        )
        for d in mod_dicts
    ]


def _dicts_to_switch(switch_dicts):
    """Reconstruct SwitchRecord objects from plain dicts."""
    return [
        SwitchRecord(
            frame=d['frame'], psf_charge=d['psf_charge'],
            scaled_charge=d['scaled_charge'], scale=d['scale'],
            dist=d['dist'], position=np.array(d['position']),
            is_image=d['is_image'],
        )
        for d in switch_dicts
    ]

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
                              cs_bond_fraction: float = 0.06,
                              pbc_compound: str = 'residue',
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

        # Remap primary charges to minimum image
        qm_center = qm_pos.mean(axis=0)
        lx, ly, lz = box[0], box[1], box[2]

        # Remap MM positions to minimum image relative to QM.
        # Positions must be in the correct periodic image.
        orig_positions = mm_ag.positions.copy()
        mm_ag.positions = remap_positions_by_compound(
            mm_ag, orig_positions, qm_center, box,
            compound=pbc_compound,
        )

        try:
            if not boundary_bonds or boundary_scheme == 'NONE':
                primary_charges = [(a.charge, *a.position) for a in mm_cut]
                raw_mods = []
            else:
                primary_charges, raw_mods = apply_boundary_scheme(
                    self.universe, mm_cut, boundary_bonds, boundary_scheme,
                    cs_bond_fraction=cs_bond_fraction,
            )


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

            mods = build_charge_mods(raw_mods, frame, self._psf_charges)
            return all_charges, mods, switch_recs, image_info, mm_ag, qm_center, box
        finally:
            mm_ag.positions = orig_positions 

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(self, config: dict):
        """Run the full QM/MM input generation pipeline."""
        start_time = time.time()
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
        #cs-scaling
        cs_bond_frac = config.get('cs_bond_fraction', 0.06) if bscheme == 'CS' else None
        pbc_compound = config.get('pbc_compound', 'residue') 
        
        #Support parellel execution
        n_workers = config.get('n_workers', 1)


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
        if bscheme == 'CS':                                          
            log(f"  CS bond frac: {cs_bond_frac:.3f}  "              
            f"(virtual charges at MM2 +/- {cs_bond_frac:.3f} x MM1-MM2 bond length)")  
        log(f"  MM cutoff   : {mm_cutoff} Ang")
        log(f"  PBC compound: {pbc_compound}") 

        #Support parellel execution
        if n_workers > 1:
            log(f"  Workers     : {n_workers}")

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
        
        # start the time
        loop_start = time.time()

        # Build argument tuples — same for serial and parallel
        frame_args = [
            (frame, qm_sel, mm_cutoff, bscheme, mm_switchdist, expand,
             cs_bond_frac, pbc_compound, target_mm_charge, neutralize_mm,
             neutral_frac, method, basis, charge, mult, program,
             keywords, custom_blocks, str(output_dir), prefix,
             pdb_stride, i)
            for i, frame in enumerate(frames, 1)
        ]
        if n_workers > 1:
            # ---- PARALLEL PATH ----
            log(f"\nProcessing {len(frames)} frames with {n_workers} workers...\n")

            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(config['psf_file'], config['dcd_file']),
            ) as pool:
                results = []
                for result in pool.imap_unordered(_process_frame, frame_args):
                    results.append(result)
                    n_done = len(results)
                    elapsed = time.time() - loop_start
                    fps = n_done / elapsed if elapsed > 0 else 0
                    remaining = (len(frames) - n_done) / fps if fps > 0 else 0
                    if remaining < 60:
                        eta_str = f"{remaining:.0f}s"
                    elif remaining < 3600:
                        eta_str = f"{remaining/60:.1f}m"
                    else:
                        eta_str = f"{remaining/3600:.1f}h"
                    spf = elapsed / n_done
                    print(f"\r  Completed {n_done}/{len(frames)} frames "
                          f"({spf:.1f} s/frame, ETA: {eta_str})  ",
                          end='', flush=True)
                print()  # newline after progress bar

            # Sort by frame for deterministic log output
            results.sort(key=lambda r: r['frame'])

            # Print the per-frame table (same format as serial path)
            for r in results:
                img_str = (f"  {r['n_images']:8d}" if supercell_on else "")
                log(f"  {r['frame']:5d}  {r['n_coords']:8d}  "
                    f"{r['n_charges']:10d}  "
                    f"{r['qm_q']:+10.4f}  {r['mm_q']:+11.4f}  "
                    f"{r['n_mods']:4d}  {r['n_switch']:8d}{img_str}")

            # Collect into accumulators
            for r in results:
                all_mods.extend(_dicts_to_mods(r['mods']))
                all_switch.extend(_dicts_to_switch(r['switch_recs']))
                qm_charges_per_frame.append(r['qm_q'])
                mm_charges_per_frame.append(r['mm_q'])
                generated.append(r['fname'])
            log_fh.flush()    # force write to disk. Python may buffer output otherwise
        else:
            # ---- SERIAL PATH (original code, unchanged) ----
            for i, frame in enumerate(frames, 1):
                coords = self.extract_coordinates(qm_sel, frame)

                charges, mods, sw_recs, img_inf, mm_ag, qm_center, box = \
                    self.extract_point_charges(
                        qm_sel, mm_cutoff, frame, bscheme, mm_switchdist,
                        expand, cs_bond_frac, pbc_compound,
                        target_mm_charge, neutralize_mm, neutral_frac
                    )

                all_mods.extend(mods)
                all_switch.extend(sw_recs)

                qm_atoms = self.universe.select_atoms(qm_sel)
                qm_q = sum(self._psf_charges.get(a.index, 0.0)
                           for a in qm_atoms)
                mm_q = sum(q for q, x, y, z in charges)
                qm_charges_per_frame.append(qm_q)
                mm_charges_per_frame.append(mm_q)

                img_str = (f"  {img_inf.get('n_images', 0):8d}"
                           if supercell_on else "")
                if i == 1:
                    eta_str = "estimating..."
                else:
                    elapsed = time.time() - loop_start
                    fps = i / elapsed
                    remaining = (len(frames) - i) / fps
                    if remaining < 60:
                        eta_str = f"{remaining:.0f}s"
                    elif remaining < 3600:
                        eta_str = f"{remaining/60:.1f}m"
                    else:
                        eta_str = f"{remaining/3600:.1f}h"
                log(f"  {frame:5d}  {len(coords):8d}  {len(charges):10d}  "
                    f"{qm_q:+10.4f}  {mm_q:+11.4f}  "
                    f"{len(mods):4d}  {len(sw_recs):8d}{img_str}"
                    f"  ETA: {eta_str}")

                base = output_dir / f"{prefix}_frame{frame}"
                suffix, write_fn = writer_fn[program]
                fname = str(base) + suffix
                write_fn(fname, coords, charges, method, basis,
                         charge, mult, keywords, custom_blocks)
                generated.append(fname)

                if pdb_stride and (i % pdb_stride == 0 or i == 1):
                    writers.write_structure(self.universe, frame, qm_atoms,
                                           mm_ag, base, qm_center, box,
                                           pbc_compound)

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

        # Print summary in the log file.
        elapsed = time.time() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}m"
        else:
            time_str = f"{elapsed/3600:.1f}h"

        log(f"\n  {'='*60}")
        log(f"  ezQMMM {__version__} | {len(frames)} frames | "
            f"{program.upper()} | {bscheme} | {method}/{basis}")
        log(f"  MM charge: {mm_arr.mean():+.4f} e | "
            f"  Wall time: {time_str} ")
        log(f"  {'='*60}")
        log_fh.close()
        return generated
