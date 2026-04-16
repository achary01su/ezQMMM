#!/usr/bin/env python3
"""
ezQMMM 2.0
Generates QM/MM single point calculation input files from MD trajectories
Supports: Q-Chem and ORCA

Requirements:
    pip install MDAnalysis numpy pyyaml

Usage:
    python ezQMMM2.py --example
    python ezQMMM2.py config.yaml

References:
    NAMD QM/MM: https://www.ks.uiuc.edu/Research/qmmm/
    Melo et al., Nature Methods 15, 351-354 (2018)
    Lin & Truhlar, J. Phys. Chem. A 109, 3991-4004 (2005)
"""

import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import MDAnalysis as mda
import numpy as np
import yaml
from MDAnalysis.analysis import distances

# Data records
# ---------------------------------------------------------------------------

class ChargeMod:
    """Record of a single charge modification at the QM/MM boundary."""

    __slots__ = (
        'frame', 'mod_type',
        'atom_index', 'segid', 'resid', 'resname', 'name',
        'psf_charge', 'applied_charge', 'delta',
        'position', 'reason',
    )

    def __init__(self, frame: int, mod_type: str, reason: str,
                 psf_charge: float, applied_charge: float,
                 position: np.ndarray,
                 atom_index: Optional[int] = None,
                 segid: str = '', resid: int = 0,
                 resname: str = '', name: str = ''):
        self.frame          = frame
        self.mod_type       = mod_type
        self.atom_index     = atom_index
        self.segid          = segid
        self.resid          = resid
        self.resname        = resname
        self.name           = name
        self.psf_charge     = psf_charge
        self.applied_charge = applied_charge
        self.delta          = applied_charge - psf_charge
        self.position       = np.array(position)
        self.reason         = reason


class SwitchRecord:
    """Record of a charge scaled by the switching function."""

    __slots__ = ('frame', 'psf_charge', 'scaled_charge', 'scale',
                 'dist', 'position', 'is_image')

    def __init__(self, frame: int, psf_charge: float, scaled_charge: float,
                 scale: float, dist: float, position: np.ndarray,
                 is_image: bool = False):
        self.frame         = frame
        self.psf_charge    = psf_charge
        self.scaled_charge = scaled_charge
        self.scale         = scale
        self.dist          = dist
        self.position      = np.array(position)
        self.is_image      = is_image


# ----------------------------------------------------------------
# Main generator
# ----------------------------------------------------------------

class QMMMGenerator:
    """Generate QM/MM input files from MD trajectories."""

    # If the simulations used hydrogen mass repartition,
    # then this mass-to-element conversion will not work.
    # This is only intended for the standard mass of each element.
    MASS_TO_ELEMENT = {
        (0.9,  1.2):  'H',  (1.9,  2.2):  'D',  (11.9, 12.2): 'C',
        (13.9, 14.2): 'N',  (15.9, 16.2): 'O',  (22.9, 23.2): 'Na',
        (24.0, 24.6): 'Mg', (30.8, 31.5): 'P',  (31.9, 32.2): 'S',
        (35.2, 35.8): 'Cl', (38.9, 39.5): 'K',  (39.9, 40.5): 'Ca',
        (55.6, 56.1): 'Fe', (63.2, 63.8): 'Cu', (65.1, 65.7): 'Zn',
    }

    def __init__(self, psf_file: str, dcd_file: str):
        print("Loading trajectory...")
        print(f"  PSF: {psf_file}")
        print(f"  DCD: {dcd_file}")
        self.universe = mda.Universe(psf_file, dcd_file)
        print(f"  Atoms: {len(self.universe.atoms)}")
        print(f"  Frames: {len(self.universe.trajectory)}")

        # Cache PSF charges before any frame is loaded — topology reference
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

    # ------------------------------------------------------------------
    # Element helpers
    # ------------------------------------------------------------------

    def _get_element_from_mass(self, mass: float) -> str:
        """
        Infer element symbol from atomic mass via lookup table.
        Returns 'X' for any mass not matched by the table — check the
        QM= atom count in the console output if unexpected elements appear.
        Note: hydrogen mass repartition (HMR) will break this lookup.
        """
        for (lo, hi), elem in self.MASS_TO_ELEMENT.items():
            if lo <= mass <= hi:
                return elem
        return 'X'

    # ------------------------------------------------------------------
    # Config parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_axes(axes_config) -> tuple[bool, bool, bool]:
        """
        Parse supercell_axes config into (expand_x, expand_y, expand_z).
        Accepts a list or comma-separated string: x/a, y/b, z/c.
        Examples: ['x','y'], 'x,y,z', 'z'
        """
        if not axes_config:
            return False, False, False
        if isinstance(axes_config, str):
            tokens = [t.strip().lower() for t in axes_config.split(',')]
        else:
            tokens = [str(t).strip().lower() for t in axes_config]
        return (
            any(t in ('x', 'a') for t in tokens),
            any(t in ('y', 'b') for t in tokens),
            any(t in ('z', 'c') for t in tokens),
        )

    @staticmethod
    def _parse_pdb_stride(value) -> Optional[int]:
        """
        Parse pdb_stride config value.
          'all'   -> 1  (every frame)
          'half'  -> 2  (every other frame)
          'tenth' -> 10 (every 10th frame)
          integer -> that integer
          null / absent -> None (disabled)
        """
        if value is None:
            return None
        if isinstance(value, int):
            return value
        mapping = {'all': 1, 'half': 2, 'tenth': 10}
        v = str(value).strip().lower()
        if v in mapping:
            return mapping[v]
        try:
            return int(v)
        # removed lint error
        #except ValueError:
        #    raise ValueError(
        except ValueError as e:
            raise ValueError(
                f"pdb_stride '{value}' not recognised. "
                f"Use: all, half, tenth, or an integer."
            ) from e

    # -----------------------------------------------------------
    # Coordinate / link-atom extraction
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
            (self._get_element_from_mass(m), p[0], p[1], p[2])
            for m, p in zip(qm_atoms.masses, qm_atoms.positions)
        ]
        for qm_idx, mm_idx in self._find_boundary_bonds(qm_atoms):
            try:
                lp = self._place_link_atom(qm_idx, mm_idx, frame)
                coords.append(('H', lp[0], lp[1], lp[2]))
            except ValueError:
                pass
        return coords

    # --------------------------------------------------------------
    # Minimum image remapping
    # -------------------------------------------------------------

    def _remap_position(self, pos: np.ndarray, qm_center: np.ndarray,
                        box: np.ndarray) -> np.ndarray:
        """
        Remap a single (3,) position to minimum image relative to QM centroid.
        Uses floor-based rounding (round-half-up) rather than np.round
        (banker's rounding) for consistency with standard MD conventions.
        """
        lx, ly, lz = box[0], box[1], box[2]
        return np.array([
            pos[0] - np.floor((pos[0] - qm_center[0]) / lx + 0.5) * lx,
            pos[1] - np.floor((pos[1] - qm_center[1]) / ly + 0.5) * ly,
            pos[2] - np.floor((pos[2] - qm_center[2]) / lz + 0.5) * lz,
        ])

    def _remap_positions_array(self, pos: np.ndarray, qm_center: np.ndarray,
                               box: np.ndarray) -> np.ndarray:
        """
        Vectorised remap of an (N, 3) position array to minimum image.
        Uses floor-based rounding (round-half-up) rather than np.round
        (banker's rounding) for consistency with standard MD conventions.
        """
        lx, ly, lz = box[0], box[1], box[2]
        out = pos.copy()
        out[:, 0] -= np.floor((pos[:, 0] - qm_center[0]) / lx + 0.5) * lx
        out[:, 1] -= np.floor((pos[:, 1] - qm_center[1]) / ly + 0.5) * ly
        out[:, 2] -= np.floor((pos[:, 2] - qm_center[2]) / lz + 0.5) * lz
        return out

    def _remap_positions_by_residue(self, mm_ag, orig_pos: np.ndarray,
                                     qm_center: np.ndarray,
                                     box: np.ndarray) -> np.ndarray:
        """
        Remap MM atom positions to minimum image, preserving residue
        internal geometry.  Two-step process per residue:

        1. **Unwrap**: remap every atom to minimum image relative to the
           residue's reference atom (first atom).  This reassembles
           residues that straddle the periodic boundary.
        2. **Shift**: remap the reference atom to minimum image relative
           to the QM centroid, and apply the same displacement to every
           atom in the residue.

        Both steps use standard floor-based minimum image arithmetic,
        but step 1 keeps the residue whole and step 2 places it near QM.
        """
        lx, ly, lz = box[0], box[1], box[2]
        new_pos = orig_pos.copy()

        # Map universe atom index → position array row
        idx_to_row = {idx: i for i, idx in enumerate(mm_ag.indices)}

        for res in mm_ag.residues:
            rows = [idx_to_row[a.index] for a in res.atoms
                    if a.index in idx_to_row]
            if not rows:
                continue

            ref = rows[0]
            ref_pos = new_pos[ref]

            # Step 1: unwrap all atoms to be near the reference atom
            for r in rows[1:]:
                new_pos[r, 0] -= np.floor((new_pos[r, 0] - ref_pos[0]) / lx + 0.5) * lx
                new_pos[r, 1] -= np.floor((new_pos[r, 1] - ref_pos[1]) / ly + 0.5) * ly
                new_pos[r, 2] -= np.floor((new_pos[r, 2] - ref_pos[2]) / lz + 0.5) * lz

            # Step 2: shift the whole residue to minimum image of QM center
            shift = np.array([
                -np.floor((ref_pos[0] - qm_center[0]) / lx + 0.5) * lx,
                -np.floor((ref_pos[1] - qm_center[1]) / ly + 0.5) * ly,
                -np.floor((ref_pos[2] - qm_center[2]) / lz + 0.5) * lz,
            ])

            for r in rows:
                new_pos[r] += shift

        return new_pos

    # -------------------------------------------------------------------------
    # Supercell image tiling
    # -------------------------------------------------------------------------

    def _image_shells(self, cutoff: float, box: np.ndarray,
                      expand: tuple[bool, bool, bool]) -> tuple[int, int, int]:
        """
        Number of image shells per axis: ceil(cutoff / L) for active axes,
        0 for suppressed axes.
        """
        return tuple(
            int(np.ceil(cutoff / box[i])) if do_expand else 0
            for i, do_expand in enumerate(expand)
        )

    def _tile_images(self, charges: list, qm_pos: np.ndarray,
                     cutoff: float, box: np.ndarray,
                     expand: tuple[bool, bool, bool]
                     ) -> tuple[list, tuple[int, int, int], int]:
        """
        Generate periodic images of primary charges along requested axes.
        Primary charges must already be remapped to minimum image positions
        relative to the QM centroid so the (0,0,0) shell skip correctly
        corresponds to what is already in the primary charge list.
        Returns (image_charges, shells, n_candidates).
        """
        if not charges:
            return [], (0, 0, 0), 0

        lx, ly, lz    = box[0], box[1], box[2]
        nx, ny, nz    = self._image_shells(cutoff, box, expand)
        image_charges = []
        n_candidates  = 0

        rq       = np.array([[x, y, z] for _, x, y, z in charges])  # (N, 3)
        rcharges = np.array([q for q, *_ in charges])                 # (N,)

        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(-nz, nz + 1):
                    if ix == 0 and iy == 0 and iz == 0:
                        continue
                    shifted      = rq + np.array([ix*lx, iy*ly, iz*lz])
                    n_candidates += len(shifted)
                    dists        = distances.distance_array(
                        qm_pos, shifted, box=None
                    ).min(axis=0)
                    for idx in np.where(dists <= cutoff)[0]:
                        image_charges.append((
                            float(rcharges[idx]),
                            float(shifted[idx, 0]),
                            float(shifted[idx, 1]),
                            float(shifted[idx, 2]),
                        ))

        return image_charges, (nx, ny, nz), n_candidates

    # -------------------------------------------------------------
    # Point-charge extraction
    # -------------------------------------------------------------

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
        charges     : list of (q, x, y, z)
        mods        : list of ChargeMod
        switch_recs : list of SwitchRecord
        image_info  : dict with image tiling statistics
        mm_ag       : MDAnalysis AtomGroup of MM atoms within cutoff
        qm_center   : np.ndarray (3,) QM centroid used for remapping
        box         : np.ndarray (6,) box dimensions at this frame
        """
        self.universe.trajectory[frame]
        qm_atoms   = self.universe.select_atoms(qm_selection)
        all_atoms  = self.universe.select_atoms("all")
        qm_idx_set = set(qm_atoms.indices)
        mm_atoms   = [a for a in all_atoms if a.index not in qm_idx_set]
        qm_pos     = qm_atoms.positions
        box        = self.universe.dimensions   # per-frame box

        if not mm_atoms:
            return [], [], [], {}, self.universe.atoms[[]], qm_pos.mean(axis=0), box

        mm_positions  = np.array([a.position for a in mm_atoms])

        # PBC-aware distance calculation for primary MM atom selection
        dist_matrix   = distances.distance_array(qm_pos, mm_positions, box=box)
        min_distances = dist_matrix.min(axis=0)

        # Whole-residue inclusion
        residues_in = set()
        for i, atom in enumerate(mm_atoms):
            if min_distances[i] <= cutoff:
                residues_in.add((atom.segid, atom.resid))

        mm_cut = [a for a in mm_atoms if (a.segid, a.resid) in residues_in]
        mm_ag  = self.universe.atoms[np.array([a.index for a in mm_cut])]

        boundary_bonds = self._find_boundary_bonds(qm_atoms)

        if not boundary_bonds or boundary_scheme == 'NONE':
            primary_charges = [(a.charge, *a.position) for a in mm_cut]
            raw_mods        = []
        else:
            primary_charges, raw_mods = self._apply_boundary_scheme(
                mm_cut, boundary_bonds, boundary_scheme
            )

        # Remap primary charges to minimum image positions relative to QM centroid.
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
            image_charges, shells, n_cand = self._tile_images(
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

        # Apply switching; box=None since positions are already explicit
        switch_recs = []
        if switchdist is not None:
            all_charges, switch_recs = self._apply_switching_to_charges(
                all_charges, qm_pos, switchdist, cutoff,
                box=None, frame=frame, n_primary=len(primary_charges),
            )

        # Charge neutralization — runs LAST, after tiling and switching,
        # so it sees the final charge set and corrects to the exact target.
        # Adjusts only the outermost non-zero charges (by distance from QM),
        # leaving charges near the QM region untouched.
        if neutralize and all_charges:
            total_q  = sum(q for q, x, y, z in all_charges)
            residual = total_q - target_mm_charge
            if abs(residual) > 1e-6:
                positions = np.array([[x, y, z] for _, x, y, z in all_charges])
                all_dists = distances.distance_array(
                    qm_pos, positions, box=None
                ).min(axis=0)
                sorted_idx = np.argsort(all_dists)[::-1]
                qs_arr     = np.array([q for q, x, y, z in all_charges])
                nonzero    = np.where(np.abs(qs_arr) > 1e-4)[0]
                outer_pool = [i for i in sorted_idx if i in set(nonzero.tolist())]
                n_outer    = max(1, int(len(outer_pool) * neutralization_shell_fraction))
                outer_idx  = set(outer_pool[:n_outer])
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

        mods = self._build_charge_mods(raw_mods, frame, qm_center, box)
        return all_charges, mods, switch_recs, image_info, mm_ag, qm_center, box

    # -----------------------------------------------
    # Boundary helpers
    # -----------------------------------------------

    def _find_boundary_bonds(self, qm_atoms):
        """Return list of (qm_atom_idx, mm_atom_idx)
           pairs at the QM/MM boundary."""
        qm_idx = set(qm_atoms.indices)
        boundary = []
        for atom in qm_atoms:
            if hasattr(atom, 'bonds'):
                for bond in atom.bonds:
                    other = bond.partner(atom)
                    if other.index not in qm_idx:
                        boundary.append((atom.index, other.index))
        return boundary

    def _place_link_atom(self, qm_idx: int, mm_idx: int, frame: int):
        """
        Place a capping hydrogen along the QM-MM bond at 1.09 Å from the QM atom.
        Raises ValueError if the bond vector is degenerate (< 0.1 Å).
        """
        self.universe.trajectory[frame]
        qm_pos = self.universe.atoms[qm_idx].position
        mm_pos = self.universe.atoms[mm_idx].position
        vec    = mm_pos - qm_pos
        vlen   = np.linalg.norm(vec)
        if vlen < 0.1:
            raise ValueError(f"Bond between atoms {qm_idx} and {mm_idx} "
                             f"is too short ({vlen:.3f} Å)")
        return qm_pos + (vec / vlen) * 1.09

    def _get_bonded_atoms(self, atom_idx: int) -> list:
        """Return indices of all atoms bonded to atom_idx."""
        atom = self.universe.atoms[atom_idx]
        return [bond.partner(atom).index for bond in atom.bonds] \
               if hasattr(atom, 'bonds') else []

    # ------------------------------------------------------------------
    # Switching function
    # ------------------------------------------------------------------

    def _apply_switching_to_charges(self, charges, qm_pos, sw, cut, box,
                                    frame, n_primary: int = None):
        """
        Vectorised NAMD-style quintic switching:
            S(r) = 1 - 10t^3 + 15t^4 - 6t^5,  t = (r - sw) / (cut - sw)
            S(r) = 1 for r <= sw
            S(r) = 0 for r >= cut
        Distance r = minimum distance from charge to any QM atom.
        box=None when image charges are present (explicit positions).
        n_primary marks the primary/image boundary for SwitchRecord.is_image.
        Only charges with scale < 1 are recorded.
        """
        if not charges:
            return [], []

        positions = np.array([[x, y, z] for _, x, y, z in charges])  # (N, 3)
        qs        = np.array([q for q, *_ in charges])                 # (N,)

        all_dists = distances.distance_array(
            qm_pos, positions, box=box
        ).min(axis=0)                                                  # (N,)

        sw_range = cut - sw
        t        = np.clip((all_dists - sw) / sw_range, 0.0, 1.0)
        scales   = np.where(all_dists <= sw, 1.0,
                   np.where(all_dists >= cut, 0.0,
                   1.0 - 10*t**3 + 15*t**4 - 6*t**5))

        scaled_qs = qs * scales
        scaled = [
            (float(scaled_qs[i]), float(positions[i, 0]),
             float(positions[i, 1]), float(positions[i, 2]))
            for i in range(len(charges))
        ]

        recs = []
        for i in np.where(scales < 1.0)[0]:
            is_img = (n_primary is not None) and (int(i) >= n_primary)
            recs.append(SwitchRecord(
                frame         = frame,
                psf_charge    = float(qs[i]),
                scaled_charge = float(scaled_qs[i]),
                scale         = float(scales[i]),
                dist          = float(all_dists[i]),
                position      = positions[i],
                is_image      = is_img,
            ))
        return scaled, recs

    # ------------------------------------------------------------------
    # Boundary scheme
    # ------------------------------------------------------------------

    def _apply_boundary_scheme(self, mm_atoms, boundary_bonds, scheme):
        """
        Apply boundary charge scheme to primary MM atoms.
        MM1 is directly bonded to the QM atom, MM2 is bonded to MM1,
        MM3 is bonded to MM2 (consecutive atoms along the bond chain).
        Returns (charges, raw_mod_dicts).
        """
        atom_map         = {a.index: a for a in mm_atoms}
        modified_charges = {idx: a.charge for idx, a in atom_map.items()}
        virtual_charges  = []
        removed_atoms    = set()
        charge_mods      = []

        for _qm_idx, mm1_idx in boundary_bonds:
            if mm1_idx not in atom_map:
                continue

            mm1_atom   = self.universe.atoms[mm1_idx]
            mm1_charge = mm1_atom.charge
            mm1_pos    = mm1_atom.position
            mm2_atoms  = self._get_bonded_atoms(mm1_idx)
            mm2_cut    = [i for i in mm2_atoms if i in atom_map]

            if scheme == 'Z1':
                # Zero the first consecutive MM atom
                removed_atoms.add(mm1_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm1_atom,
                    'old_charge': mm1_charge, 'new_charge': 0.0,
                    'reason': 'MM1 removed (Z1)',
                })

            elif scheme == 'Z2':
                # Zero the first two consecutive MM atoms
                removed_atoms.add(mm1_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm1_atom,
                    'old_charge': mm1_charge, 'new_charge': 0.0,
                    'reason': 'MM1 removed (Z2)',
                })
                for mm2_idx in mm2_cut:
                    mm2_atom = self.universe.atoms[mm2_idx]
                    old_q    = mm2_atom.charge
                    removed_atoms.add(mm2_idx)
                    charge_mods.append({
                        'type': 'removed', 'atom': mm2_atom,
                        'old_charge': old_q, 'new_charge': 0.0,
                        'reason': 'MM2 removed (Z2)',
                    })

            elif scheme == 'Z3':
                # Zero three consecutive MM atoms along the bond chain
                removed_atoms.add(mm1_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm1_atom,
                    'old_charge': mm1_charge, 'new_charge': 0.0,
                    'reason': 'MM1 removed (Z3)',
                })
                seen_mm3 = set()
                for mm2_idx in mm2_cut:
                    mm2_atom = self.universe.atoms[mm2_idx]
                    old_q    = mm2_atom.charge
                    removed_atoms.add(mm2_idx)
                    charge_mods.append({
                        'type': 'removed', 'atom': mm2_atom,
                        'old_charge': old_q, 'new_charge': 0.0,
                        'reason': 'MM2 removed (Z3)',
                    })
                    mm3_atoms = self._get_bonded_atoms(mm2_idx)
                    mm3_cut   = [i for i in mm3_atoms
                                 if i in atom_map
                                 and i != mm1_idx
                                 and i not in mm2_cut
                                 and i not in seen_mm3]
                    for mm3_idx in mm3_cut:
                        seen_mm3.add(mm3_idx)
                        mm3_atom = self.universe.atoms[mm3_idx]
                        old_q3   = mm3_atom.charge
                        removed_atoms.add(mm3_idx)
                        charge_mods.append({
                            'type': 'removed', 'atom': mm3_atom,
                            'old_charge': old_q3, 'new_charge': 0.0,
                            'reason': 'MM3 removed (Z3)',
                        })

            elif scheme == 'RCD':
                removed_atoms.add(mm1_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm1_atom,
                    'old_charge': mm1_charge, 'new_charge': 0.0,
                    'reason': 'MM1 removed (RCD)',
                })
                n = len(mm2_cut)
                if n:
                    for mm2_idx in mm2_cut:
                        mm2_pos  = self.universe.atoms[mm2_idx].position
                        midpoint = (mm1_pos + mm2_pos) * 0.5
                        # Factor of 2 preserves MM1-MM2 bond dipole: q*d == 2q*(d/2)
                        vq = 2.0 * mm1_charge / n
                        virtual_charges.append((vq, *midpoint))
                        charge_mods.append({
                            'type': 'virtual', 'position': midpoint,
                            'charge': vq, 'reason': 'Virtual RCD midpoint',
                        })
                        mm2_atom = self.universe.atoms[mm2_idx]
                        old_q    = mm2_atom.charge
                        modified_charges[mm2_idx] -= mm1_charge / n
                        charge_mods.append({
                            'type': 'modified', 'atom': mm2_atom,
                            'old_charge': old_q,
                            'new_charge': modified_charges[mm2_idx],
                            'reason': 'MM2 adjusted (RCD)',
                        })

            elif scheme == 'CS':
                removed_atoms.add(mm1_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm1_atom,
                    'old_charge': mm1_charge, 'new_charge': 0.0,
                    'reason': 'MM1 removed (CS)',
                })
                n = len(mm2_cut)
                if n:
                    split = mm1_charge / n
                    for mm2_idx in mm2_cut:
                        mm2_pos = self.universe.atoms[mm2_idx].position
                        vec     = mm2_pos - mm1_pos
                        vn      = np.linalg.norm(vec)
                        if vn > 0.1:
                            u = vec / vn
                            # Dipole correction pair placed ±0.3 Å around MM2
                            virtual_charges.append((split,  *(mm2_pos - u * 0.3)))
                            virtual_charges.append((-split, *(mm2_pos + u * 0.3)))
                        mm2_atom = self.universe.atoms[mm2_idx]
                        old_q    = mm2_atom.charge
                        modified_charges[mm2_idx] += split
                        charge_mods.append({
                            'type': 'modified', 'atom': mm2_atom,
                            'old_charge': old_q,
                            'new_charge': modified_charges[mm2_idx],
                            'reason': 'MM2 shifted (CS)',
                        })

        charges = [
            (modified_charges[idx], *atom_map[idx].position)
            for idx in modified_charges if idx not in removed_atoms
        ]
        charges.extend(virtual_charges)
        return charges, charge_mods

    # ------------------------------------------------------------------
    # Build typed ChargeMod objects from raw dicts
    # ------------------------------------------------------------------

    def _build_charge_mods(self, raw_mods: list, frame: int,
                           qm_center: np.ndarray,
                           box: np.ndarray) -> list[ChargeMod]:
        """Convert raw boundary scheme dicts to typed ChargeMod records."""
        out = []
        for d in raw_mods:
            if d['type'] == 'virtual':
                out.append(ChargeMod(
                    frame          = frame,
                    mod_type       = 'virtual',
                    reason         = d['reason'],
                    psf_charge     = 0.0,
                    applied_charge = d['charge'],
                    position       = self._remap_position(
                        np.array(d['position']), qm_center, box),
                ))
            else:
                atom  = d['atom']
                psf_q = self._psf_charges.get(atom.index, d['old_charge'])
                out.append(ChargeMod(
                    frame          = frame,
                    mod_type       = d['type'],
                    reason         = d['reason'],
                    psf_charge     = psf_q,
                    applied_charge = d['new_charge'],
                    position       = self._remap_position(
                        atom.position.copy(), qm_center, box),
                    atom_index     = atom.index,
                    segid          = atom.segid,
                    resid          = atom.resid,
                    resname        = atom.resname,
                    name           = atom.name,
                ))
        return out

    # ------------------------------------------------------------------
    # Structure writer (PDB per frame, PSF once)
    # ------------------------------------------------------------------

    def _write_structure(self, frame: int, qm_atoms, mm_ag, base: Path,
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
        qm_idx    = set(qm_atoms.indices)
        mm_idx    = set(mm_ag.indices)
        all_atoms = self.universe.select_atoms("all")

        orig_mm_pos     = mm_ag.positions.copy()
        mm_ag.positions = self._remap_positions_by_residue(
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
        mm_ag.positions       = orig_mm_pos

    def _write_topology(self, psf_source: str, output_dir: Path,
                        prefix: str) -> Path:
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

    def _write_boundary_log(self, fh, all_mods: list[ChargeMod]):
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

    def _write_switching_log(self, fh, all_switch: list[SwitchRecord],
                             switchdist: float, cutoff: float,
                             expand: tuple[bool, bool, bool]):
        supercell_on = any(expand)
        axis_labels  = ','.join(l for l, e in zip(('x', 'y', 'z'), expand) if e)
        sw_label = f"{switchdist:.2f} - {cutoff:.2f} Ang" \
                   if switchdist is not None else "disabled"
        fh.write("=" * 72 + "\n")
        fh.write("ezQMMM 2.0  Switching-Function Charge Modifications\n")
        fh.write(f"  Switching zone  : {sw_label}\n")
        fh.write("  Function        : quintic (NAMD-style)\n")
        fh.write("  Distance metric : minimum distance to any QM atom "
                 "(not center of mass)\n")
        if supercell_on:
            fh.write(f"  Image charges   : included (axes: {axis_labels}); "
                     f"marked in Src column as IMG\n")
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
            delta   = r.scaled_charge - r.psf_charge
            src_tag = "  IMG" if (supercell_on and r.is_image) else "     "
            fh.write(
                f"{r.frame:<8d} {r.dist:>9.3f} {r.scale:>8.5f} "
                f"{r.psf_charge:>+10.4f} {r.scaled_charge:>+10.4f} "
                f"{delta:>+10.4f}  "
                f"{r.position[0]:>8.3f}  {r.position[1]:>8.3f}  "
                f"{r.position[2]:>8.3f}{src_tag}\n"
            )

        n_prim = sum(1 for r in all_switch if not r.is_image)
        n_img  = sum(1 for r in all_switch if r.is_image)
        fh.write("\n" + "-" * 72 + "\n")
        if supercell_on:
            fh.write(f"Total switching-zone events: {len(all_switch)}  "
                     f"(primary={n_prim}, image={n_img})\n")
        else:
            fh.write(f"Total switching-zone charge events: {len(all_switch)}\n")

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(self, config: dict):
        qm_sel        = config['qm_selection']
        mm_cutoff     = config.get('mm_cutoff', 40.0)
        # Switching is disabled by default. Set mm_switchdist explicitly
        # in the config to enable it. If null or absent, no switching is applied.
        mm_switchdist = config.get('mm_switchdist')
        expand        = self._parse_axes(config.get('supercell_axes', []))
        supercell_on  = any(expand)
        neutralize_mm   = config.get('neutralize_mm_charge', True)
        target_mm_charge = config.get('target_mm_charge', 0.0)
        pdb_stride       = self._parse_pdb_stride(config.get('pdb_stride'))
        neutral_frac     = config.get('neutralization_shell_fraction', 0.1)

        first  = config.get('first_frame', 0)
        last   = config.get('last_frame', -1)
        if last == -1 or last >= len(self.universe.trajectory):
            last = len(self.universe.trajectory) - 1
        stride = config.get('stride', 1)

        method  = config.get('method', 'B3LYP')
        basis   = config.get('basis', '6-31G*')
        charge  = config.get('charge', 0)
        mult    = config.get('multiplicity', 1)
        bscheme = config.get('boundary_scheme', 'RCD').upper()

        output_dir = Path(config.get('output_dir', '.'))
        prefix     = config.get('output_prefix', 'qmmm')

        if 'program' not in config:
            raise ValueError("'program' required (orca/qchem)")

        valid_schemes = {'RCD', 'CS', 'Z1', 'Z2', 'Z3', 'NONE'}
        if bscheme not in valid_schemes:
            raise ValueError(
                f"boundary_scheme '{bscheme}' not recognised. "
                f"Valid options: {', '.join(sorted(valid_schemes))}"
            )

        program = config['program'].lower()
        valid_programs = {'orca', 'qchem', 'psi4'}
        if program not in valid_programs:
            raise ValueError(
                f"program '{program}' not recognised. "
                f"Valid options: {', '.join(sorted(valid_programs))}"
            )

        keywords      = config.get(f'{program}_keywords', '') or ''
        custom_blocks = config.get(f'{program}_blocks',   '') or ''

        # ----------------------------------------------------------
        # Fail-fast validation — catch bad inputs before the frame
        # loop so minutes of processing aren't wasted on a typo.
        # ----------------------------------------------------------

        # Stride must be positive
        if stride <= 0:
            raise ValueError(
                f"stride must be a positive integer, got {stride}"
            )

        # Frame range sanity
        if first < 0:
            raise ValueError(
                f"first_frame must be >= 0, got {first}"
            )
        if first > last:
            raise ValueError(
                f"first_frame ({first}) is after last_frame ({last}). "
                f"Trajectory has {len(self.universe.trajectory)} frames."
            )

        # Switching window: switchdist must be strictly less than cutoff
        if mm_switchdist is not None and mm_switchdist >= mm_cutoff:
            raise ValueError(
                f"mm_switchdist ({mm_switchdist}) must be less than "
                f"mm_cutoff ({mm_cutoff}). The switching function "
                f"attenuates charges between switchdist and cutoff."
            )

        # Neutralization shell fraction must be in (0, 1]
        if neutralize_mm and not (0.0 < neutral_frac <= 1.0):
            raise ValueError(
                f"neutralization_shell_fraction must be in (0, 1], "
                f"got {neutral_frac}"
            )

        # Dry-run the QM selection on the first frame to catch typos
        # before entering the frame loop.
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

        output_dir.mkdir(parents=True, exist_ok=True)

        # Open run log — mirrors console output to a file
        log_path = output_dir / f"{prefix}_run.log"
        #tell ruff to ignore the open() block error
        log_fh = open(log_path, 'w') # noqa: SIM115

        def log(msg=''):
            """Print to console and write to log file."""
            print(msg)
            log_fh.write(msg + '\n')

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
            nx, ny, nz  = self._image_shells(mm_cutoff, box, expand)
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
            psf_dest = self._write_topology(config['psf_file'], output_dir, prefix)
            log(f"  Topology    : {psf_dest}")
        log(f"  Frames      : {len(frames)}")
        log(f"  Log file    : {log_path}")

        all_mods:   list[ChargeMod]    = []
        all_switch: list[SwitchRecord] = []
        generated = []

        # Charge tracking across frames
        qm_charges_per_frame = []
        mm_charges_per_frame = []

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

            if program == 'orca':
                fname = str(base) + "_orca.inp"
                self._write_orca(fname, coords, charges, method, basis,
                                 charge, mult, keywords, custom_blocks)
            elif program == 'qchem':
                fname = str(base) + "_qchem.in"
                self._write_qchem(fname, coords, charges, method, basis,
                                  charge, mult, keywords, custom_blocks)
            elif program == 'psi4':
                fname = str(base) + "_psi4.dat"
                self._write_psi4(fname, coords, charges, method, basis,
                                 charge, mult, keywords, custom_blocks)
            generated.append(fname)

            if pdb_stride and (i % pdb_stride == 0 or i == 1):
                self._write_structure(frame, qm_atoms, mm_ag, base,
                                      qm_center, box)

        # Charge summary
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

        if all_mods:
            bpath = output_dir / f"{prefix}_boundary.log"
            with open(bpath, 'w') as fh:
                self._write_boundary_log(fh, all_mods)
            log(f"\n  Boundary log  -> {bpath}")

        spath = output_dir / f"{prefix}_switching.log"
        with open(spath, 'w') as fh:
            self._write_switching_log(fh, all_switch, mm_switchdist,
                                      mm_cutoff, expand)
        log(f"  Switching log -> {spath}")
        log(f"  Run log       -> {log_path}")

        log(f"\nGenerated {len(generated)} input files")
        log_fh.close()
        return generated

    # ---------------------------------------------------------
    # QM/MM input writers
    # ---------------------------------------------------------

    def _write_orca(self, fname, coords, charges, method, basis,
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

    def _write_qchem(self, fname, coords, charges, method, basis,
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

    def _write_psi4(self, fname, coords, charges, method, basis,
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


# ---------------------------------------
# CLI helpers
# ----------------------------------------

def create_example_config():
    config = """# ezQMMM 2.0 Configuration
psf_file: system.psf
dcd_file: trajectory.dcd
qm_selection: "resid 100 and not backbone"
mm_cutoff: 40.0
# mm_switchdist: 35.0   # uncomment to enable switching; disabled by default
first_frame: 0
last_frame: 100
stride: 10
method: B3LYP
basis: 6-31G*
charge: 0
multiplicity: 1
boundary_scheme: RCD

# MM charge neutralization (default: true).
# Distributes any residual MM charge evenly across all point charges each
# frame to enforce target_mm_charge. Eliminates frame-to-frame fluctuations
# from ions drifting in/out of the cutoff. Set to false for benchmarking
# or to reproduce raw PSF charge behaviour.
neutralize_mm_charge: true
target_mm_charge: 0.0
neutralization_shell_fraction: 0.1  # outermost 10% of charges absorb the correction
output_dir: ./qmmm_calculations
output_prefix: qmmm
program: qchem

# Axes along which to tile periodic images (any combo of x/y/z or a/b/c).
# Leave empty or omit to disable. Examples:
#   supercell_axes: [x, y]       # membrane protein, normal along z
#   supercell_axes: [x, y, z]    # small solvated active site
#   supercell_axes: [z]          # elongated system (DNA, fibril)
supercell_axes: []

# Write a PDB per frame (shared PSF written once).
# Options: all, half, tenth, or any integer stride. null = disabled.
pdb_stride: null

qchem_keywords: |
  scf_convergence    8
  mem_total          8000

qchem_blocks: |
  $basis
  C 0
  cc-pVDZ
  ****
  H 0
  cc-pVDZ
  ****
  $end
"""
    with open('config_example.yaml', 'w') as f:
        f.write(config)
    print("Created: config_example.yaml")


LOGO = r"""
--------------------------------------------------------
               ___  __  __ __  __ __  __   ____     ___
   ___  ____  / _ \|  \/  |  \/  |  \/  | |___ \   / _ \
  / _ \|_  / | | | | |\/| | |\/| | |\/| |   __) | | | | |
 |  __/ / /  | |_| | |  | | |  | | |  | |  / __/ _| |_| |
  \___|/___|  \__\_\_|  |_|_|  |_|_|  |_| |_____(_)\___/
             Easy QM/MM Input File Generator
                     Q-Chem · Orca
--------------------------------------------------------
"""


def main():
    print(LOGO)
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ezQMMM2.py config.yaml")
        print("  python ezQMMM2.py --example")
        sys.exit(1)

    if sys.argv[1] == '--example':
        create_example_config()
        return

    try:
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        gen = QMMMGenerator(config['psf_file'], config['dcd_file'])
        gen.generate(config)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
