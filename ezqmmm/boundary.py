"""
QM/MM boundary schemes: link-atom placement and charge redistribution.

Supported schemes:
  RCD  — Redistributed Charge and Dipole (recommended)
  CS   — Charge Shift
  Z1   — Zero MM1
  Z2   — Zero MM1 + MM2
  Z3   — Zero MM1 + MM2 + MM3
  NONE — No boundary treatment

MM1 is directly bonded to the QM atom, MM2 is bonded to MM1,
MM3 is bonded to MM2 — consecutive atoms along the covalent bond chain.

References:
    Lin & Truhlar, J. Phys. Chem. A 109, 3991-4004 (2005)
"""

import numpy as np
from typing import Dict, List

from ezqmmm.models import ChargeMod
from ezqmmm.geometry import remap_position


def find_boundary_bonds(qm_atoms) -> list:
    """Return list of (qm_atom_idx, mm_atom_idx) pairs at the QM/MM boundary."""
    qm_idx = set(qm_atoms.indices)
    boundary = []
    for atom in qm_atoms:
        if hasattr(atom, 'bonds'):
            for bond in atom.bonds:
                other = bond.partner(atom)
                if other.index not in qm_idx:
                    boundary.append((atom.index, other.index))
    return boundary


def place_link_atom(universe, qm_idx: int, mm_idx: int, frame: int):
    """
    Place a capping hydrogen along the QM-MM bond at 1.09 Å from the QM atom.
    Raises ValueError if the bond vector is degenerate (< 0.1 Å).
    """
    universe.trajectory[frame]
    qm_pos = universe.atoms[qm_idx].position
    mm_pos = universe.atoms[mm_idx].position
    vec = mm_pos - qm_pos
    vlen = np.linalg.norm(vec)
    if vlen < 0.1:
        raise ValueError(f"Bond between atoms {qm_idx} and {mm_idx} "
                         f"is too short ({vlen:.3f} Å)")
    return qm_pos + (vec / vlen) * 1.09


def get_bonded_atoms(universe, atom_idx: int) -> list:
    """Return indices of all atoms bonded to *atom_idx*."""
    atom = universe.atoms[atom_idx]
    return [bond.partner(atom).index for bond in atom.bonds] \
           if hasattr(atom, 'bonds') else []


def apply_boundary_scheme(universe, mm_atoms, boundary_bonds, scheme):
    """
    Apply boundary charge scheme to primary MM atoms.
    Returns (charges, raw_mod_dicts).
    """
    atom_map = {a.index: a for a in mm_atoms}
    modified_charges = {idx: a.charge for idx, a in atom_map.items()}
    virtual_charges = []
    removed_atoms = set()
    charge_mods = []

    for qm_idx, mm1_idx in boundary_bonds:
        if mm1_idx not in atom_map:
            continue

        mm1_atom = universe.atoms[mm1_idx]
        mm1_charge = mm1_atom.charge
        mm1_pos = mm1_atom.position
        mm2_atoms = get_bonded_atoms(universe, mm1_idx)
        mm2_cut = [i for i in mm2_atoms if i in atom_map]

        if scheme == 'Z1':
            removed_atoms.add(mm1_idx)
            charge_mods.append({
                'type': 'removed', 'atom': mm1_atom,
                'old_charge': mm1_charge, 'new_charge': 0.0,
                'reason': 'MM1 removed (Z1)',
            })

        elif scheme == 'Z2':
            removed_atoms.add(mm1_idx)
            charge_mods.append({
                'type': 'removed', 'atom': mm1_atom,
                'old_charge': mm1_charge, 'new_charge': 0.0,
                'reason': 'MM1 removed (Z2)',
            })
            for mm2_idx in mm2_cut:
                mm2_atom = universe.atoms[mm2_idx]
                old_q = mm2_atom.charge
                removed_atoms.add(mm2_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm2_atom,
                    'old_charge': old_q, 'new_charge': 0.0,
                    'reason': 'MM2 removed (Z2)',
                })

        elif scheme == 'Z3':
            removed_atoms.add(mm1_idx)
            charge_mods.append({
                'type': 'removed', 'atom': mm1_atom,
                'old_charge': mm1_charge, 'new_charge': 0.0,
                'reason': 'MM1 removed (Z3)',
            })
            seen_mm3 = set()
            for mm2_idx in mm2_cut:
                mm2_atom = universe.atoms[mm2_idx]
                old_q = mm2_atom.charge
                removed_atoms.add(mm2_idx)
                charge_mods.append({
                    'type': 'removed', 'atom': mm2_atom,
                    'old_charge': old_q, 'new_charge': 0.0,
                    'reason': 'MM2 removed (Z3)',
                })
                mm3_atoms = get_bonded_atoms(universe, mm2_idx)
                mm3_cut = [i for i in mm3_atoms
                           if i in atom_map
                           and i != mm1_idx
                           and i not in mm2_cut
                           and i not in seen_mm3]
                for mm3_idx in mm3_cut:
                    seen_mm3.add(mm3_idx)
                    mm3_atom = universe.atoms[mm3_idx]
                    old_q3 = mm3_atom.charge
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
                    mm2_pos = universe.atoms[mm2_idx].position
                    midpoint = (mm1_pos + mm2_pos) * 0.5
                    vq = 2.0 * mm1_charge / n
                    virtual_charges.append((vq, *midpoint))
                    charge_mods.append({
                        'type': 'virtual', 'position': midpoint,
                        'charge': vq, 'reason': 'Virtual RCD midpoint',
                    })
                    mm2_atom = universe.atoms[mm2_idx]
                    old_q = mm2_atom.charge
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
                    mm2_pos = universe.atoms[mm2_idx].position
                    vec = mm2_pos - mm1_pos
                    vn = np.linalg.norm(vec)
                    if vn > 0.1:
                        u = vec / vn
                        virtual_charges.append((split,  *(mm2_pos - u * 0.3)))
                        virtual_charges.append((-split, *(mm2_pos + u * 0.3)))
                    mm2_atom = universe.atoms[mm2_idx]
                    old_q = mm2_atom.charge
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


def build_charge_mods(raw_mods: list, frame: int,
                      qm_center: np.ndarray, box: np.ndarray,
                      psf_charges: Dict[int, float]) -> List[ChargeMod]:
    """Convert raw boundary scheme dicts to typed ChargeMod records."""
    out = []
    for d in raw_mods:
        if d['type'] == 'virtual':
            out.append(ChargeMod(
                frame=frame,
                mod_type='virtual',
                reason=d['reason'],
                psf_charge=0.0,
                applied_charge=d['charge'],
                position=remap_position(
                    np.array(d['position']), qm_center, box),
            ))
        else:
            atom = d['atom']
            psf_q = psf_charges.get(atom.index, d['old_charge'])
            out.append(ChargeMod(
                frame=frame,
                mod_type=d['type'],
                reason=d['reason'],
                psf_charge=psf_q,
                applied_charge=d['new_charge'],
                position=remap_position(
                    atom.position.copy(), qm_center, box),
                atom_index=atom.index,
                segid=atom.segid,
                resid=atom.resid,
                resname=atom.resname,
                name=atom.name,
            ))
    return out
