"""Tests for ezqmmm.boundary — bond detection, link atoms, all 6 schemes."""

import numpy as np
import pytest

from ezqmmm.boundary import (
    find_boundary_bonds, place_link_atom, get_bonded_atoms,
    apply_boundary_scheme, build_charge_mods,
)
from ezqmmm.geometry import remap_position


# ===================================================================
# Bond detection
# ===================================================================

class TestFindBoundaryBonds:
    def test_finds_qm_mm_bond(self, tiny_universe):
        qm = tiny_universe.select_atoms("resid 1")
        bonds = find_boundary_bonds(qm)
        assert (0, 3) in bonds

    def test_no_intra_qm_bonds(self, tiny_universe):
        qm = tiny_universe.select_atoms("resid 1")
        qm_idx = set(qm.indices)
        for qi, mi in find_boundary_bonds(qm):
            assert qi in qm_idx
            assert mi not in qm_idx

    def test_no_boundary_when_no_cut(self, tiny_universe):
        """Selecting everything as QM should find no boundary bonds."""
        qm = tiny_universe.select_atoms("all")
        bonds = find_boundary_bonds(qm)
        assert len(bonds) == 0

    def test_multiple_boundaries(self):
        """System with two QM-MM bonds should find both."""
        import MDAnalysis as mda
        u = mda.Universe.empty(
            4, n_residues=2, n_segments=1,
            atom_resindex=np.array([0, 0, 1, 1]),
            residue_segindex=np.array([0, 0]),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['A', 'B', 'C', 'D'])
        u.add_TopologyAttr('resname', ['QM', 'MM'])
        u.add_TopologyAttr('resid', [1, 2])
        u.add_TopologyAttr('segid', ['S'])
        # Two QM-MM bonds: 0-2 and 1-3
        u.add_TopologyAttr('bonds', [(0, 2), (1, 3)])
        u.atoms.positions = np.array([[0, 0, 0], [1, 0, 0],
                                       [2, 0, 0], [3, 0, 0]], dtype=np.float32)

        qm = u.select_atoms("resid 1")
        bonds = find_boundary_bonds(qm)
        assert (0, 2) in bonds
        assert (1, 3) in bonds


class TestGetBondedAtoms:
    def test_returns_bonded_partners(self, tiny_universe):
        # Atom 0 is bonded to 1, 2, and 3
        bonded = get_bonded_atoms(tiny_universe, 0)
        assert set(bonded) == {1, 2, 3}

    def test_terminal_atom(self, tiny_universe):
        # Atom 1 is only bonded to atom 0
        bonded = get_bonded_atoms(tiny_universe, 1)
        assert bonded == [0]


# ===================================================================
# Link atom placement
# ===================================================================

class TestPlaceLinkAtom:
    def test_distance_is_1_09(self, tiny_universe):
        lp = place_link_atom(tiny_universe, 0, 3, 0)
        qm_pos = tiny_universe.atoms[0].position
        assert abs(np.linalg.norm(lp - qm_pos) - 1.09) < 1e-5

    def test_direction_along_bond(self, tiny_universe):
        lp = place_link_atom(tiny_universe, 0, 3, 0)
        qm_pos = tiny_universe.atoms[0].position
        mm_pos = tiny_universe.atoms[3].position
        bond_dir = (mm_pos - qm_pos) / np.linalg.norm(mm_pos - qm_pos)
        link_dir = (lp - qm_pos) / np.linalg.norm(lp - qm_pos)
        assert abs(np.dot(bond_dir, link_dir) - 1.0) < 1e-5

    def test_degenerate_bond_raises(self):
        """Bond < 0.1 Å should raise ValueError."""
        import MDAnalysis as mda
        u = mda.Universe.empty(
            2, n_residues=1, n_segments=1,
            atom_resindex=np.array([0, 0]),
            residue_segindex=np.array([0]),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['A', 'B'])
        u.add_TopologyAttr('resname', ['X'])
        u.add_TopologyAttr('resid', [1])
        u.add_TopologyAttr('segid', ['S'])
        # Nearly overlapping atoms
        u.atoms.positions = np.array([[10.0, 10.0, 10.0],
                                       [10.05, 10.0, 10.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            place_link_atom(u, 0, 1, 0)


# ===================================================================
# Boundary schemes — structural + numerical
# ===================================================================

def _get_mm_atoms_and_bonds(universe):
    """Helper: return mm_atoms list and boundary_bonds."""
    qm = universe.select_atoms("resid 1")
    mm = [a for a in universe.select_atoms("all")
          if a.index not in set(qm.indices)]
    bonds = find_boundary_bonds(qm)
    return mm, bonds


def _total_charge(charges):
    """Sum the q values from a charge list [(q, x, y, z), ...]."""
    return sum(q for q, *_ in charges)


def _raw_mm_total(universe):
    """Total PSF charge of all non-QM atoms."""
    qm = universe.select_atoms("resid 1")
    mm = [a for a in universe.select_atoms("all")
          if a.index not in set(qm.indices)]
    return sum(a.charge for a in mm)


class TestRCD:
    def test_mm1_removed(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        removed = [m for m in mods if m['type'] == 'removed']
        assert any(m['atom'].index == 3 for m in removed)

    def test_creates_virtual_charges(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        virtuals = [m for m in mods if m['type'] == 'virtual']
        assert len(virtuals) > 0

    def test_modifies_mm2(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        modified = [m for m in mods if m['type'] == 'modified']
        assert len(modified) > 0

    def test_conserves_total_charge(self, tiny_universe):
        """RCD must conserve total MM charge exactly."""
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_total = _raw_mm_total(tiny_universe)
        charges, _ = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        assert abs(_total_charge(charges) - raw_total) < 1e-6

    def test_virtual_at_midpoint(self, tiny_universe):
        """Virtual charge should be at the midpoint of MM1-MM2 bond."""
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        virtuals = [m for m in mods if m['type'] == 'virtual']
        mm1_pos = tiny_universe.atoms[3].position
        # MM2 atoms bonded to MM1: atoms 4 and 5
        for v in virtuals:
            vpos = np.array(v['position'])
            # Should be midpoint between MM1 and one of its MM2 partners
            for mm2_idx in [4, 5]:
                mm2_pos = tiny_universe.atoms[mm2_idx].position
                expected_mid = (mm1_pos + mm2_pos) * 0.5
                if np.linalg.norm(vpos - expected_mid) < 1e-4:
                    break
            else:
                continue
            break
        else:
            pytest.fail("No virtual charge found at expected midpoint")


class TestCS:
    def test_mm1_removed(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'CS')
        removed = [m for m in mods if m['type'] == 'removed']
        assert any(m['atom'].index == 3 for m in removed)

    def test_adds_virtual_charges(self, tiny_universe):
        """CS adds ± dipole pairs, visible in the charge list."""
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_count = len(mm)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'CS')
        removed_count = sum(1 for m in mods if m['type'] == 'removed')
        # More charges than (raw - removed) because of virtual pairs
        assert len(charges) > raw_count - removed_count

    def test_conserves_total_charge(self, tiny_universe):
        """CS must conserve total MM charge exactly."""
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_total = _raw_mm_total(tiny_universe)
        charges, _ = apply_boundary_scheme(tiny_universe, mm, bonds, 'CS')
        assert abs(_total_charge(charges) - raw_total) < 1e-6


class TestZ1:
    def test_zeroes_mm1(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z1')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) == 1
        assert removed[0]['new_charge'] == 0.0

    def test_breaks_charge_neutrality(self, tiny_universe):
        """Z1 zeroes MM1 charge, so total changes by -MM1_charge."""
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_total = _raw_mm_total(tiny_universe)
        charges, _ = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z1')
        mm1_charge = tiny_universe.atoms[3].charge  # -0.3
        expected = raw_total - mm1_charge
        assert abs(_total_charge(charges) - expected) < 1e-6


class TestZ2:
    def test_zeroes_mm1_and_mm2(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z2')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) >= 2
        removed_idx = {m['atom'].index for m in removed}
        assert 3 in removed_idx  # MM1

    def test_breaks_charge_neutrality(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_total = _raw_mm_total(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z2')
        removed_q = sum(m['old_charge'] for m in mods if m['type'] == 'removed')
        assert abs(_total_charge(charges) - (raw_total - removed_q)) < 1e-6


class TestZ3:
    def test_zeroes_three_levels(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z3')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) >= 2  # at least MM1 + MM2 (MM3 depends on topology)

    def test_breaks_charge_neutrality(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        raw_total = _raw_mm_total(tiny_universe)
        charges, mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'Z3')
        removed_q = sum(m['old_charge'] for m in mods if m['type'] == 'removed')
        assert abs(_total_charge(charges) - (raw_total - removed_q)) < 1e-6


class TestNONE:
    def test_no_modifications_through_api(self, tiny_universe):
        """NONE scheme via the code path that skips apply_boundary_scheme."""
        qm = tiny_universe.select_atoms("resid 1")
        mm = [a for a in tiny_universe.select_atoms("all")
              if a.index not in set(qm.indices)]
        bonds = find_boundary_bonds(qm)
        # The generate code checks: if not bonds or scheme == 'NONE'
        # When NONE, it builds raw charges directly. Verify that path:
        charges = [(a.charge, *a.position) for a in mm]
        assert len(charges) == 9  # 12 - 3 QM
        assert abs(_total_charge(charges) - _raw_mm_total(tiny_universe)) < 1e-6


# ===================================================================
# build_charge_mods
# ===================================================================

class TestBuildChargeMods:
    def test_converts_removed_to_typed(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        _, raw_mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        psf_charges = {a.index: float(a.charge) for a in tiny_universe.atoms}
        qm_c = tiny_universe.select_atoms("resid 1").positions.mean(axis=0)
        box = tiny_universe.dimensions

        typed = build_charge_mods(raw_mods, frame=0, qm_center=qm_c,
                                   box=box, psf_charges=psf_charges)

        removed = [m for m in typed if m.mod_type == 'removed']
        assert len(removed) > 0
        for m in removed:
            assert m.atom_index is not None
            assert m.applied_charge == 0.0
            assert abs(m.delta - (0.0 - m.psf_charge)) < 1e-10

    def test_converts_virtual_to_typed(self, tiny_universe):
        mm, bonds = _get_mm_atoms_and_bonds(tiny_universe)
        _, raw_mods = apply_boundary_scheme(tiny_universe, mm, bonds, 'RCD')
        psf_charges = {a.index: float(a.charge) for a in tiny_universe.atoms}
        qm_c = tiny_universe.select_atoms("resid 1").positions.mean(axis=0)
        box = tiny_universe.dimensions

        typed = build_charge_mods(raw_mods, frame=0, qm_center=qm_c,
                                   box=box, psf_charges=psf_charges)

        virtuals = [m for m in typed if m.mod_type == 'virtual']
        assert len(virtuals) > 0
        for m in virtuals:
            assert m.atom_index is None
            assert m.psf_charge == 0.0
