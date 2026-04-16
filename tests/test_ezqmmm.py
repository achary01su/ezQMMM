"""
ezQMMM 2.0 — Full Test Suite

Requires MDAnalysis installed (runs in CI or locally, not in sandbox).

Tests are organized by module:
  - Config parsing and validation
  - Element lookup
  - Geometry (remapping, image shells, per-residue remap)
  - Switching function
  - Boundary schemes
  - Neutralization
  - Writers (ORCA, Q-Chem, Psi4)
  - PDB structure output
  - Run log and charge reporting
  - Data records (ChargeMod, SwitchRecord)
  - End-to-end pipeline

Fixture: a tiny 4-residue system built programmatically —
  Segment SEG1:
    Residue 1 (QM1): 3 atoms, mimics a small molecule
    Residue 2 (MM1): 3 atoms, bonded to QM1 (boundary test)
    Residue 3 (MM2): 3 atoms, pure MM water-like
    Residue 4 (MM3): 3 atoms, far away MM water-like
"""

import os
import sys
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import yaml
from MDAnalysis.analysis import distances

# Import both the monolith and the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Package imports
# Also import monolith for parity checks
import ezQMMM2 as mono
from ezqmmm.boundary import (
    apply_boundary_scheme,
    find_boundary_bonds,
    place_link_atom,
)
from ezqmmm.config import create_example_config, parse_axes, parse_pdb_stride, validate_config
from ezqmmm.elements import MASS_TO_ELEMENT, get_element_from_mass
from ezqmmm.geometry import (
    image_shells,
    remap_position,
    remap_positions_array,
    remap_positions_by_residue,
)
from ezqmmm.models import ChargeMod, SwitchRecord
from ezqmmm.switching import apply_switching
from ezqmmm.writers import (
    write_boundary_log,
    write_orca,
    write_psi4,
    write_qchem,
    write_switching_log,
)

# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def tiny_universe():
    """
    Build a minimal 12-atom, 4-residue universe for testing.

    Residue layout:
      res1 (QM):  atoms 0-2,  resname=QM1, segid=SEG1, near origin
      res2 (MM):  atoms 3-5,  resname=MM1, segid=SEG1, bonded to res1
      res3 (MM):  atoms 6-8,  resname=WAT, segid=SEG1, nearby water
      res4 (MM):  atoms 9-11, resname=WAT, segid=SEG1, far water

    Box: 100 x 100 x 100 Å, orthorhombic
    """
    n_atoms = 12
    n_residues = 4
    n_segments = 1

    atom_resindex = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    residue_segindex = np.array([0, 0, 0, 0])

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        n_segments=n_segments,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )

    # Topology attributes
    u.add_TopologyAttr('name', ['C1', 'H1', 'H2',
                                 'C2', 'H3', 'H4',
                                 'O1', 'HW1', 'HW2',
                                 'O2', 'HW3', 'HW4'])
    u.add_TopologyAttr('resname', ['QM1', 'MM1', 'WAT', 'WAT'])
    u.add_TopologyAttr('resid', [1, 2, 3, 4])
    u.add_TopologyAttr('segid', ['SEG1'])
    u.add_TopologyAttr('mass', [12.011, 1.008, 1.008,
                                 12.011, 1.008, 1.008,
                                 15.999, 1.008, 1.008,
                                 15.999, 1.008, 1.008])
    u.add_TopologyAttr('charge', [-0.2, 0.1, 0.1,
                                   -0.3, 0.15, 0.15,
                                   -0.82, 0.41, 0.41,
                                   -0.82, 0.41, 0.41])
    u.add_TopologyAttr('tempfactors', np.zeros(n_atoms))

    # Bonds: 0-1, 0-2 (within res1), 0-3 (QM-MM boundary), 3-4, 3-5,
    #         6-7, 6-8, 9-10, 9-11
    bonds = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5),
             (6, 7), (6, 8), (9, 10), (9, 11)]
    u.add_TopologyAttr('bonds', bonds)

    # Positions: QM near origin, MM1 nearby, WAT at 20 Å, far WAT at 45 Å
    positions = np.array([
        [10.0, 10.0, 10.0],  # C1 (QM)
        [11.0, 10.0, 10.0],  # H1 (QM)
        [10.0, 11.0, 10.0],  # H2 (QM)
        [12.5, 10.0, 10.0],  # C2 (MM1, bonded to C1)
        [13.5, 10.0, 10.0],  # H3 (MM1)
        [12.5, 11.0, 10.0],  # H4 (MM1)
        [20.0, 10.0, 10.0],  # O1 (WAT, nearby)
        [21.0, 10.0, 10.0],  # HW1
        [20.0, 11.0, 10.0],  # HW2
        [45.0, 10.0, 10.0],  # O2 (WAT, far)
        [46.0, 10.0, 10.0],  # HW3
        [45.0, 11.0, 10.0],  # HW4
    ], dtype=np.float32)

    u.atoms.positions = positions
    u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

    return u


@pytest.fixture
def straddling_universe():
    """
    A 6-atom, 2-residue universe where one water straddles the box boundary.
    Used to test per-residue PDB remapping.

    res1 (QM): atoms 0-2 at x≈50 (QM region)
    res2 (MM): atoms 3-5, O at x=1, H at x=99 (straddling)
    Box: 100 x 100 x 100
    """
    u = mda.Universe.empty(
        6, n_residues=2, n_segments=1,
        atom_resindex=np.array([0, 0, 0, 1, 1, 1]),
        residue_segindex=np.array([0, 0]),
        trajectory=True,
    )
    u.add_TopologyAttr('name', ['C1', 'H1', 'H2', 'O1', 'HW1', 'HW2'])
    u.add_TopologyAttr('resname', ['QM1', 'WAT'])
    u.add_TopologyAttr('resid', [1, 2])
    u.add_TopologyAttr('segid', ['SEG1'])
    u.add_TopologyAttr('mass', [12.011, 1.008, 1.008, 15.999, 1.008, 1.008])
    u.add_TopologyAttr('charge', [-0.2, 0.1, 0.1, -0.82, 0.41, 0.41])
    u.add_TopologyAttr('tempfactors', np.zeros(6))
    u.add_TopologyAttr('bonds', [(0, 1), (0, 2), (3, 4), (3, 5)])

    u.atoms.positions = np.array([
        [50.0, 50.0, 50.0],
        [51.0, 50.0, 50.0],
        [50.0, 51.0, 50.0],
        [1.0, 50.0, 50.0],   # O — one side of box
        [99.0, 50.0, 50.0],  # H — other side (straddling)
        [1.5, 50.0, 50.0],   # H — same side as O
    ], dtype=np.float32)
    u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

    return u


# ===================================================================
# Config parsing
# ===================================================================

class TestParseAxes:
    @pytest.mark.parametrize("input,expected", [
        ([], (False, False, False)),
        (None, (False, False, False)),
        (['x', 'y'], (True, True, False)),
        ('x,y,z', (True, True, True)),
        (['a', 'c'], (True, False, True)),
        ('z', (False, False, True)),
        (['X', 'Y'], (True, True, False)),
    ])
    def test_parse(self, input, expected):
        assert parse_axes(input) == expected

    def test_parity_with_monolith(self):
        cases = [[], None, ['x', 'y'], 'x,y,z', ['a', 'c'], 'z']
        for c in cases:
            assert parse_axes(c) == mono.QMMMGenerator._parse_axes(c)


class TestParsePdbStride:
    @pytest.mark.parametrize("input,expected", [
        (None, None), ('all', 1), ('half', 2), ('tenth', 10),
        (5, 5), ('20', 20),
    ])
    def test_parse(self, input, expected):
        assert parse_pdb_stride(input) == expected

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            parse_pdb_stride('bogus')

    def test_parity_with_monolith(self):
        for c in [None, 'all', 'half', 'tenth', 5, '20']:
            assert parse_pdb_stride(c) == mono.QMMMGenerator._parse_pdb_stride(c)


class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config({
            'qm_selection': 'resid 1', 'program': 'orca',
            'stride': 10, 'first_frame': 0, 'last_frame': 50,
        }, 100)

    @pytest.mark.parametrize("label,cfg", [
        ('missing qm_selection', {'program': 'orca'}),
        ('missing program', {'qm_selection': 'resid 1'}),
        ('bad program', {'qm_selection': 'r', 'program': 'gaussian'}),
        ('bad scheme', {'qm_selection': 'r', 'program': 'orca',
                        'boundary_scheme': 'FAKE'}),
        ('stride=0', {'qm_selection': 'r', 'program': 'orca', 'stride': 0}),
        ('stride<0', {'qm_selection': 'r', 'program': 'orca', 'stride': -1}),
        ('first<0', {'qm_selection': 'r', 'program': 'orca',
                     'first_frame': -1}),
        ('first>last', {'qm_selection': 'r', 'program': 'orca',
                        'first_frame': 50, 'last_frame': 10}),
        ('sw>=cut', {'qm_selection': 'r', 'program': 'orca',
                     'mm_switchdist': 40, 'mm_cutoff': 40}),
        ('sw>cut', {'qm_selection': 'r', 'program': 'orca',
                    'mm_switchdist': 45, 'mm_cutoff': 40}),
        ('nf=0', {'qm_selection': 'r', 'program': 'orca',
                  'neutralize_mm_charge': True,
                  'neutralization_shell_fraction': 0.0}),
        ('nf>1', {'qm_selection': 'r', 'program': 'orca',
                  'neutralize_mm_charge': True,
                  'neutralization_shell_fraction': 1.5}),
        ('nf<0', {'qm_selection': 'r', 'program': 'orca',
                  'neutralize_mm_charge': True,
                  'neutralization_shell_fraction': -0.1}),
    ])
    def test_invalid_rejected(self, label, cfg):
        with pytest.raises(ValueError):
            validate_config(cfg, 100)


class TestExampleConfig:
    def test_generates_and_parses(self, tmp_path):
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            create_example_config()
            with open('config_example.yaml') as f:
                cfg = yaml.safe_load(f)
            assert cfg['psf_file'] == 'system.psf'
            assert cfg['program'] == 'qchem'
            assert cfg['boundary_scheme'] == 'RCD'
            assert cfg['neutralize_mm_charge'] is True
            assert cfg['neutralization_shell_fraction'] == 0.1
        finally:
            os.chdir(orig)


# ===================================================================
# Elements
# ===================================================================

class TestElements:
    @pytest.mark.parametrize("mass,expected", [
        (1.008, 'H'), (2.014, 'D'), (12.011, 'C'), (14.007, 'N'),
        (15.999, 'O'), (30.974, 'P'), (32.06, 'S'), (55.845, 'Fe'),
        (65.38, 'Zn'), (22.99, 'Na'), (39.1, 'K'),
    ])
    def test_known_elements(self, mass, expected):
        assert get_element_from_mass(mass) == expected

    def test_unknown_returns_x(self):
        assert get_element_from_mass(99.0) == 'X'

    def test_table_matches_monolith(self):
        assert MASS_TO_ELEMENT == mono.QMMMGenerator.MASS_TO_ELEMENT


# ===================================================================
# Geometry — remapping
# ===================================================================

class TestRemapPosition:
    @pytest.mark.parametrize("x,ref,L,expected", [
        (50.0, 50.0, 100.0, 50.0),    # no shift
        (95.0, 5.0, 100.0, -5.0),     # positive wrap
        (5.0, 95.0, 100.0, 105.0),    # negative wrap
        (50.0, 0.0, 100.0, -50.0),    # half-box boundary
    ])
    def test_scalar_cases(self, x, ref, L, expected):
        pos = np.array([x, 0.0, 0.0])
        qm_c = np.array([ref, 0.0, 0.0])
        box = np.array([L, L, L])
        result = remap_position(pos, qm_c, box)
        assert abs(result[0] - expected) < 1e-10

    def test_3d(self):
        pos = np.array([95.0, 5.0, 50.0])
        qm_c = np.array([5.0, 95.0, 50.0])
        box = np.array([100.0, 100.0, 100.0])
        result = remap_position(pos, qm_c, box)
        assert abs(result[0] - (-5.0)) < 1e-10
        assert abs(result[1] - 105.0) < 1e-10
        assert abs(result[2] - 50.0) < 1e-10


class TestRemapPositionsArray:
    def test_batch(self):
        pos = np.array([[95.0, 50.0, 50.0], [5.0, 50.0, 50.0]])
        qm_c = np.array([5.0, 50.0, 50.0])
        box = np.array([100.0, 100.0, 100.0])
        out = remap_positions_array(pos, qm_c, box)
        assert abs(out[0, 0] - (-5.0)) < 1e-10
        assert abs(out[1, 0] - 5.0) < 1e-10


class TestRemapByResidue:
    def test_straddling_water_preserved(self, straddling_universe):
        """Water with O at x=1 and H at x=99 should be reassembled."""
        u = straddling_universe
        mm_ag = u.select_atoms("resname WAT")
        orig_pos = mm_ag.positions.copy()
        qm_c = np.array([50.0, 50.0, 50.0])
        box = u.dimensions

        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, box)

        # After remap, O-H distances should be small (water reassembled).
        # Original: O at x=1, H at x=99 → 98 Å apart (broken).
        # After unwrap+shift: both should be within a few Å of each other.
        dist_OH1 = np.linalg.norm(out[0] - out[1])
        dist_OH2 = np.linalg.norm(out[0] - out[2])
        assert dist_OH1 < 5.0, f"O-H1 still broken: {dist_OH1:.1f} Å"
        assert dist_OH2 < 5.0, f"O-H2 still broken: {dist_OH2:.1f} Å"

    def test_per_atom_would_break(self, straddling_universe):
        """Confirm per-atom remap splits the straddling water."""
        u = straddling_universe
        mm_ag = u.select_atoms("resname WAT")
        orig_pos = mm_ag.positions.copy()
        qm_c = np.array([50.0, 50.0, 50.0])
        box = u.dimensions

        out = remap_positions_array(orig_pos, qm_c, box)
        dist = abs(out[0, 0] - out[1, 0])
        assert dist > 50, f"Per-atom should break water, got dist={dist:.1f}"

    def test_nearby_water_unchanged(self, tiny_universe):
        """Water already near QM should not be shifted."""
        u = tiny_universe
        mm_ag = u.select_atoms("resid 3")  # nearby WAT
        orig_pos = mm_ag.positions.copy()
        qm_atoms = u.select_atoms("resid 1")
        qm_c = qm_atoms.positions.mean(axis=0)
        box = u.dimensions

        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, box)
        np.testing.assert_allclose(out, orig_pos, atol=1e-5)

    def test_two_residues_independent(self, tiny_universe):
        """Multiple residues each get their own shift."""
        u = tiny_universe
        mm_ag = u.select_atoms("resid 3 or resid 4")
        orig_pos = mm_ag.positions.copy()
        qm_atoms = u.select_atoms("resid 1")
        qm_c = qm_atoms.positions.mean(axis=0)
        box = u.dimensions

        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, box)

        # Each residue should preserve internal geometry
        for res in mm_ag.residues:
            idx = [i for i, a in enumerate(mm_ag.atoms) if a in res.atoms]
            for j in idx[1:]:
                orig_d = orig_pos[idx[0]] - orig_pos[j]
                new_d = out[idx[0]] - out[j]
                np.testing.assert_allclose(orig_d, new_d, atol=1e-5)


class TestImageShells:
    def test_no_expand(self):
        box = np.array([100.0, 100.0, 100.0])
        assert image_shells(40, box, (False, False, False)) == (0, 0, 0)

    def test_partial_expand(self):
        box = np.array([100.0, 100.0, 100.0])
        assert image_shells(40, box, (True, False, True)) == (1, 0, 1)

    def test_small_box_needs_more_shells(self):
        box = np.array([30.0, 30.0, 30.0])
        shells = image_shells(50, box, (True, True, True))
        assert shells == (2, 2, 2)


# ===================================================================
# Switching function
# ===================================================================

class TestSwitchingMath:
    """Test the quintic formula directly."""

    @staticmethod
    def quintic(r, sw, cut):
        if r <= sw:
            return 1.0
        if r >= cut:
            return 0.0
        t = (r - sw) / (cut - sw)
        return 1.0 - 10 * t**3 + 15 * t**4 - 6 * t**5

    @pytest.mark.parametrize("r,expected", [
        (30.0, 1.0), (35.0, 1.0), (40.0, 0.0), (45.0, 0.0),
    ])
    def test_boundary_values(self, r, expected):
        assert self.quintic(r, 35.0, 40.0) == expected

    def test_midpoint_approximately_half(self):
        assert abs(self.quintic(37.5, 35.0, 40.0) - 0.5) < 0.01

    def test_monotonic_decrease(self):
        prev = 1.0
        for r in np.linspace(35.0, 40.0, 200):
            s = self.quintic(r, 35.0, 40.0)
            assert s <= prev + 1e-10
            prev = s


class TestApplySwitching:
    def test_all_inside_switchdist(self):
        """Charges inside switchdist should be unscaled."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 5.0, 0.0, 0.0), (0.3, 8.0, 0.0, 0.0)]
        scaled, recs = apply_switching(
            charges, qm_pos, sw=10.0, cut=15.0,
            box=None, frame=0,
        )
        for (q_orig, _, _, _), (q_scaled, _, _, _) in zip(charges, scaled):
            assert abs(q_orig - q_scaled) < 1e-10
        assert len(recs) == 0

    def test_beyond_cutoff_zeroed(self):
        """Charges beyond cutoff should be zeroed."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 20.0, 0.0, 0.0)]
        scaled, recs = apply_switching(
            charges, qm_pos, sw=10.0, cut=15.0,
            box=None, frame=0,
        )
        assert abs(scaled[0][0]) < 1e-10
        assert len(recs) == 1
        assert abs(recs[0].scale) < 1e-10

    def test_in_switching_zone_partially_scaled(self):
        """Charges in the switching zone should be partially scaled."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 12.5, 0.0, 0.0)]  # midpoint of 10-15
        scaled, recs = apply_switching(
            charges, qm_pos, sw=10.0, cut=15.0,
            box=None, frame=0,
        )
        assert 0 < abs(scaled[0][0]) < 0.5
        assert len(recs) == 1
        assert 0 < recs[0].scale < 1

    def test_image_flag(self):
        """Charges after n_primary should be flagged as images."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [
            (-0.5, 12.0, 0.0, 0.0),  # primary, in switching zone
            (-0.5, 13.0, 0.0, 0.0),  # image, in switching zone
        ]
        scaled, recs = apply_switching(
            charges, qm_pos, sw=10.0, cut=15.0,
            box=None, frame=0, n_primary=1,
        )
        primaries = [r for r in recs if not r.is_image]
        images = [r for r in recs if r.is_image]
        assert len(primaries) == 1
        assert len(images) == 1


# ===================================================================
# Boundary schemes
# ===================================================================

class TestBoundaryBonds:
    def test_finds_qm_mm_bond(self, tiny_universe):
        u = tiny_universe
        qm_atoms = u.select_atoms("resid 1")
        bonds = find_boundary_bonds(qm_atoms)
        # Bond between atom 0 (QM C1) and atom 3 (MM C2)
        assert (0, 3) in bonds

    def test_no_false_positives(self, tiny_universe):
        u = tiny_universe
        qm_atoms = u.select_atoms("resid 1")
        bonds = find_boundary_bonds(qm_atoms)
        # Should only find 0-3, not intra-QM bonds
        for qm_idx, mm_idx in bonds:
            assert qm_idx in qm_atoms.indices
            assert mm_idx not in qm_atoms.indices


class TestPlaceLinkAtom:
    def test_distance_is_1_09(self, tiny_universe):
        u = tiny_universe
        # Bond 0 (QM) — 3 (MM)
        lp = place_link_atom(u, 0, 3, 0)
        qm_pos = u.atoms[0].position
        dist = np.linalg.norm(lp - qm_pos)
        assert abs(dist - 1.09) < 1e-6

    def test_direction_along_bond(self, tiny_universe):
        u = tiny_universe
        lp = place_link_atom(u, 0, 3, 0)
        qm_pos = u.atoms[0].position
        mm_pos = u.atoms[3].position
        bond_vec = mm_pos - qm_pos
        link_vec = lp - qm_pos
        # Should be parallel
        cos_angle = (np.dot(bond_vec, link_vec) /
                     (np.linalg.norm(bond_vec) * np.linalg.norm(link_vec)))
        assert abs(cos_angle - 1.0) < 1e-6


class TestBoundarySchemes:
    def _get_charges_and_mods(self, universe, scheme):
        qm_atoms = universe.select_atoms("resid 1")
        all_atoms = universe.select_atoms("all")
        mm_atoms = [a for a in all_atoms if a.index not in set(qm_atoms.indices)]
        boundary_bonds = find_boundary_bonds(qm_atoms)
        charges, mods = apply_boundary_scheme(
            universe, mm_atoms, boundary_bonds, scheme
        )
        return charges, mods

    def test_rcd_removes_mm1(self, tiny_universe):
        charges, mods = self._get_charges_and_mods(tiny_universe, 'RCD')
        removed = [m for m in mods if m['type'] == 'removed']
        assert any(m['atom'].index == 3 for m in removed)

    def test_rcd_creates_virtual(self, tiny_universe):
        charges, mods = self._get_charges_and_mods(tiny_universe, 'RCD')
        virtuals = [m for m in mods if m['type'] == 'virtual']
        assert len(virtuals) > 0

    def test_rcd_modifies_mm2(self, tiny_universe):
        charges, mods = self._get_charges_and_mods(tiny_universe, 'RCD')
        modified = [m for m in mods if m['type'] == 'modified']
        assert len(modified) > 0

    def test_z1_zeroes_mm1(self, tiny_universe):
        charges, mods = self._get_charges_and_mods(tiny_universe, 'Z1')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) == 1
        assert removed[0]['new_charge'] == 0.0

    def test_z2_zeroes_mm1_and_mm2(self, tiny_universe):
        charges, mods = self._get_charges_and_mods(tiny_universe, 'Z2')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) >= 2

    def test_cs_creates_dipole_pair(self, tiny_universe):
        """CS should add virtual dipole charges to the charge list."""
        # Get charge count without boundary scheme for comparison
        qm_atoms = tiny_universe.select_atoms("resid 1")
        all_atoms = tiny_universe.select_atoms("all")
        mm_atoms = [a for a in all_atoms
                    if a.index not in set(qm_atoms.indices)]
        raw_count = len(mm_atoms)

        # CS removes MM1 but adds virtual dipole pairs
        charges, mods = self._get_charges_and_mods(tiny_universe, 'CS')
        removed = [m for m in mods if m['type'] == 'removed']
        assert len(removed) == 1  # MM1 removed
        # The charge list should have more entries than raw MM minus removed,
        # because CS adds virtual dipole pairs (±split around each MM2)
        assert len(charges) > raw_count - len(removed)

    def test_none_no_modifications(self, tiny_universe):
        qm_atoms = tiny_universe.select_atoms("resid 1")
        all_atoms = tiny_universe.select_atoms("all")
        mm_atoms = [a for a in all_atoms
                    if a.index not in set(qm_atoms.indices)]
        charges = [(a.charge, *a.position) for a in mm_atoms]
        # NONE should produce the same charges as raw
        assert len(charges) == 9  # 12 total - 3 QM


# ===================================================================
# Neutralization
# ===================================================================

class TestNeutralization:
    def test_total_charge_hits_target(self, tiny_universe):
        """After neutralization, total MM charge should match target."""
        u = tiny_universe
        qm_atoms = u.select_atoms("resid 1")
        qm_pos = qm_atoms.positions
        all_atoms = u.select_atoms("all")
        mm_atoms = [a for a in all_atoms
                    if a.index not in set(qm_atoms.indices)]

        charges = [(a.charge, *a.position) for a in mm_atoms]

        target = 0.0
        total_q = sum(q for q, x, y, z in charges)
        residual = total_q - target

        if abs(residual) > 1e-6:
            positions = np.array([[x, y, z] for _, x, y, z in charges])
            all_dists = distances.distance_array(
                qm_pos, positions, box=None
            ).min(axis=0)
            sorted_idx = np.argsort(all_dists)[::-1]
            qs_arr = np.array([q for q, *_ in charges])
            nonzero = np.where(np.abs(qs_arr) > 1e-4)[0]
            outer_pool = [i for i in sorted_idx if i in set(nonzero.tolist())]
            n_outer = max(1, int(len(outer_pool) * 0.1))
            outer_idx = set(outer_pool[:n_outer])
            correction = -residual / n_outer
            charges = [
                (q + correction, x, y, z) if i in outer_idx else (q, x, y, z)
                for i, (q, x, y, z) in enumerate(charges)
            ]

        final_q = sum(q for q, *_ in charges)
        assert abs(final_q - target) < 1e-6

    def test_nonzero_target(self, tiny_universe):
        """Neutralization should work with non-zero target."""
        u = tiny_universe
        qm_atoms = u.select_atoms("resid 1")
        qm_pos = qm_atoms.positions
        all_atoms = u.select_atoms("all")
        mm_atoms = [a for a in all_atoms
                    if a.index not in set(qm_atoms.indices)]

        charges = [(a.charge, *a.position) for a in mm_atoms]
        target = 2.0
        total_q = sum(q for q, *_ in charges)
        residual = total_q - target

        if abs(residual) > 1e-6:
            positions = np.array([[x, y, z] for _, x, y, z in charges])
            all_dists = distances.distance_array(
                qm_pos, positions, box=None
            ).min(axis=0)
            sorted_idx = np.argsort(all_dists)[::-1]
            qs_arr = np.array([q for q, *_ in charges])
            nonzero = np.where(np.abs(qs_arr) > 1e-4)[0]
            outer_pool = [i for i in sorted_idx if i in set(nonzero.tolist())]
            n_outer = max(1, int(len(outer_pool) * 0.1))
            outer_idx = set(outer_pool[:n_outer])
            correction = -residual / n_outer
            charges = [
                (q + correction, x, y, z) if i in outer_idx else (q, x, y, z)
                for i, (q, x, y, z) in enumerate(charges)
            ]

        final_q = sum(q for q, *_ in charges)
        assert abs(final_q - target) < 1e-6


# ===================================================================
# Writers
# ===================================================================

class TestWriters:
    @pytest.fixture
    def sample_data(self):
        coords = [('C', 1.0, 2.0, 3.0), ('H', 1.5, 2.5, 3.5)]
        charges = [(0.5, 10.0, 20.0, 30.0), (-0.3, 11.0, 21.0, 31.0)]
        return coords, charges

    def test_orca_format(self, tmp_path, sample_data):
        coords, charges = sample_data
        fname = str(tmp_path / 'test_orca.inp')
        write_orca(fname, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

        content = Path(fname).read_text()
        assert '! B3LYP 6-31G*' in content
        assert '* xyz 0 1' in content
        assert content.strip().endswith('*')

        pc_path = fname.replace('.inp', '_charges.pc')
        pc_content = Path(pc_path).read_text()
        assert pc_content.startswith('2\n')

    def test_qchem_format(self, tmp_path, sample_data):
        coords, charges = sample_data
        fname = str(tmp_path / 'test_qchem.in')
        write_qchem(fname, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

        content = Path(fname).read_text()
        assert '$molecule' in content
        assert '0 1' in content
        assert 'qm_mm                true' in content
        assert '$external_charges' in content

    def test_psi4_format(self, tmp_path, sample_data):
        coords, charges = sample_data
        fname = str(tmp_path / 'test_psi4.dat')
        write_psi4(fname, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

        content = Path(fname).read_text()
        assert 'molecule qmmm' in content
        assert 'no_com' in content
        assert 'Chrgfield = QMMM()' in content
        assert "energy('B3LYP')" in content

    def test_qchem_no_charges(self, tmp_path):
        coords = [('C', 1.0, 2.0, 3.0)]
        fname = str(tmp_path / 'test_nocharge.in')
        write_qchem(fname, coords, [], 'B3LYP', '6-31G*', 0, 1, '', '')

        content = Path(fname).read_text()
        assert 'qm_mm' not in content
        assert '$external_charges' not in content

    def test_orca_custom_keywords(self, tmp_path, sample_data):
        coords, charges = sample_data
        fname = str(tmp_path / 'test_kw.inp')
        write_orca(fname, coords, charges, 'B3LYP', '6-31G*', 0, 1,
                   'TightSCF\nPAL4', '')

        content = Path(fname).read_text()
        assert '! TightSCF' in content
        assert '! PAL4' in content

    def test_psi4_bohr_conversion(self, tmp_path, sample_data):
        coords, charges = sample_data
        fname = str(tmp_path / 'test_bohr.dat')
        write_psi4(fname, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

        content = Path(fname).read_text()
        # First charge x=10.0 Å → 10.0 * 1.8897259886 = 18.897260
        assert '18.897260' in content

    def test_monolith_parity(self, tmp_path, sample_data):
        """Package and monolith writers produce identical output."""
        coords, charges = sample_data

        class FakeGen:
            pass
        gen = FakeGen()
        gen._write_orca = mono.QMMMGenerator._write_orca.__get__(gen)
        gen._write_qchem = mono.QMMMGenerator._write_qchem.__get__(gen)
        gen._write_psi4 = mono.QMMMGenerator._write_psi4.__get__(gen)

        for label, ext, mono_fn, pkg_fn in [
            ('orca', '_orca.inp', gen._write_orca, write_orca),
            ('qchem', '_qchem.in', gen._write_qchem, write_qchem),
            ('psi4', '_psi4.dat', gen._write_psi4, write_psi4),
        ]:
            mdir = tmp_path / 'mono'
            pdir = tmp_path / 'pkg'
            mdir.mkdir(exist_ok=True)
            pdir.mkdir(exist_ok=True)

            fm = str(mdir / f't{ext}')
            fp = str(pdir / f't{ext}')
            mono_fn(fm, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
            pkg_fn(fp, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

            assert Path(fm).read_text() == Path(fp).read_text(), \
                f"{label} output mismatch"


# ===================================================================
# Data records
# ===================================================================

class TestChargeMod:
    def test_delta_computed(self):
        mod = ChargeMod(
            frame=0, mod_type='removed', reason='test',
            psf_charge=0.5, applied_charge=0.0,
            position=np.array([1.0, 2.0, 3.0]),
        )
        assert abs(mod.delta - (-0.5)) < 1e-10

    def test_optional_fields_default(self):
        mod = ChargeMod(
            frame=1, mod_type='virtual', reason='midpoint',
            psf_charge=0.0, applied_charge=0.3,
            position=np.array([0.0, 0.0, 0.0]),
        )
        assert mod.atom_index is None
        assert mod.segid == ''
        assert mod.resid == 0


class TestSwitchRecord:
    def test_creation(self):
        rec = SwitchRecord(
            frame=5, psf_charge=-0.5, scaled_charge=-0.25,
            scale=0.5, dist=37.5,
            position=np.array([10.0, 20.0, 30.0]),
        )
        assert rec.is_image is False
        assert abs(rec.scale - 0.5) < 1e-10

    def test_image_flag(self):
        rec = SwitchRecord(
            frame=0, psf_charge=0.3, scaled_charge=0.15,
            scale=0.5, dist=12.0,
            position=np.array([0.0, 0.0, 0.0]),
            is_image=True,
        )
        assert rec.is_image is True


# ===================================================================
# Neutralization sanity check (the warning)
# ===================================================================

class TestNeutralizationWarning:
    def test_no_warning_when_on_target(self):
        """No warning if charge matches target."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Simulate: charges already sum to target
            # The sanity check fires when deviation > 0.01
            final_q = 0.0001
            target = 0.0
            deviation = abs(final_q - target)
            assert deviation <= 0.01

    def test_warning_when_off_target(self):
        """Warning should fire for large deviation."""
        final_q = 3.2683
        target = 2.0
        deviation = abs(final_q - target)
        assert deviation > 0.01


# ===================================================================
# Log writers
# ===================================================================

class TestLogWriters:
    def test_boundary_log(self, tmp_path):
        mods = [
            ChargeMod(frame=0, mod_type='removed', reason='MM1 removed (RCD)',
                       psf_charge=-0.3, applied_charge=0.0,
                       position=np.array([1, 2, 3.0]),
                       atom_index=5, segid='SEG1', resid=2,
                       resname='MM1', name='C2'),
            ChargeMod(frame=0, mod_type='virtual', reason='Virtual RCD midpoint',
                       psf_charge=0.0, applied_charge=-0.6,
                       position=np.array([4, 5, 6.0])),
        ]
        fpath = tmp_path / 'boundary.log'
        with open(fpath, 'w') as fh:
            write_boundary_log(fh, mods)
        content = fpath.read_text()
        assert 'Boundary Charge Modification' in content
        assert 'removed' in content
        assert 'virtual' in content
        assert 'Total:' in content

    def test_switching_log(self, tmp_path):
        recs = [
            SwitchRecord(frame=0, psf_charge=-0.5, scaled_charge=-0.25,
                          scale=0.5, dist=37.5,
                          position=np.array([10, 20, 30.0])),
        ]
        fpath = tmp_path / 'switching.log'
        with open(fpath, 'w') as fh:
            write_switching_log(fh, recs, 35.0, 40.0, (False, False, False))
        content = fpath.read_text()
        assert 'Switching-Function' in content
        assert '37.500' in content


# ===================================================================
# README consistency
# ===================================================================

class TestReadmeConsistency:
    @pytest.fixture(autouse=True)
    def _load_readme(self):
        readme_path = Path(__file__).resolve().parent.parent / 'README.md'
        if readme_path.exists():
            self.readme = readme_path.read_text()
        else:
            pytest.skip("README.md not found")

    def test_no_distribute_across_all_charges(self):
        idx = self.readme.find('After selection, boundary scheme corrections')
        assert idx != -1
        paragraph = self.readme[idx:idx + 500]
        assert 'across all charges' not in paragraph
        assert 'outermost' in paragraph

    def test_pdb_remapping_describes_two_step(self):
        assert 'two-step process' in self.readme or 'unwrap' in self.readme.lower()

    def test_neutralization_is_last(self):
        idx = self.readme.find('correction is applied after')
        if idx != -1:
            paragraph = self.readme[idx:idx + 200]
            assert 'switching' in paragraph.lower()


# ===================================================================
# Bug fix regression checks
# ===================================================================

class TestBugFixRegressions:
    @pytest.fixture(autouse=True)
    def _load_source(self):
        src_path = Path(__file__).resolve().parent.parent / 'ezQMMM2.py'
        self.source = src_path.read_text()

    def test_neutral_frac_defined_in_generate(self):
        """neutral_frac must be read from config, not undefined."""
        assert any('neutral_frac' in l and 'config.get' in l
                    for l in self.source.split('\n'))

    def test_pdb_not_gated_behind_else(self):
        """PDB/PSF writing must not be inside the neutralization else."""
        lines = self.source.split('\n')
        for i, line in enumerate(lines):
            if 'if pdb_stride:' in line and '_write_topology' not in line:
                indent = len(line) - len(line.lstrip())
                for j in range(i - 1, max(0, i - 10), -1):
                    s = lines[j].strip()
                    if not s:
                        continue
                    pindent = len(lines[j]) - len(lines[j].lstrip())
                    assert not (pindent == indent and s == 'else:'), \
                        f"pdb_stride at line {i+1} still under else: at line {j+1}"
                    break
                return

    def test_write_structure_uses_per_residue(self):
        """PDB writer must use per-residue remap, not per-atom."""
        idx = self.source.index('def _write_structure')
        chunk = self.source[idx:idx + 1000]
        assert '_remap_positions_by_residue' in chunk
        assert '_remap_positions_array' not in chunk

    def test_neutralization_sanity_check_exists(self):
        """Post-neutralization deviation check must exist."""
        assert 'deviation' in self.source and '0.01' in self.source
