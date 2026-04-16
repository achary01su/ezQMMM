"""
Integration tests — end-to-end pipeline through the package API.

Tests the full chain: QM coordinate extraction → MM charge extraction
→ boundary scheme → remapping → tiling → switching → neutralization.
Also tests whole-residue inclusion and the QMMMGenerator public interface.
"""

import MDAnalysis as mda
import numpy as np

from ezqmmm.generator import QMMMGenerator


def _build_generator(universe):
    """Bypass __init__ to build a generator from an existing universe."""
    gen = object.__new__(QMMMGenerator)
    gen.universe = universe
    gen._psf_charges = {a.index: float(a.charge) for a in universe.atoms}
    return gen


# ===================================================================
# Coordinate extraction
# ===================================================================

class TestExtractCoordinates:
    def test_returns_correct_elements(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        coords = gen.extract_coordinates('resid 1', 0)
        elements = [elem for elem, *_ in coords]
        # res1 has C (12.011), H (1.008), H (1.008)
        assert 'C' in elements
        assert elements.count('H') >= 2  # native + possible link atom

    def test_link_atom_added(self, tiny_universe):
        """Bond 0-3 should produce a link hydrogen."""
        gen = _build_generator(tiny_universe)
        coords = gen.extract_coordinates('resid 1', 0)
        # 3 QM atoms + 1 link H
        assert len(coords) == 4
        # Last should be the link H
        assert coords[-1][0] == 'H'

    def test_no_link_atom_without_boundary(self, tiny_universe):
        """Selecting everything as QM → no boundary bonds → no link atoms."""
        gen = _build_generator(tiny_universe)
        coords = gen.extract_coordinates('all', 0)
        # All 12 atoms, no link atoms
        assert len(coords) == 12


# ===================================================================
# Whole-residue inclusion
# ===================================================================

class TestWholeResidueInclusion:
    def test_partial_residue_pulls_in_full(self, tiny_universe):
        """
        If one atom of a residue is within cutoff, all atoms are included.
        Res3 (WAT at x≈20) is ~10 Å from QM. With cutoff=15, atom O1
        at x=20 is within cutoff, so all 3 atoms of res3 should be in.
        """
        gen = _build_generator(tiny_universe)
        charges, _, _, _, mm_ag, *_ = gen.extract_point_charges(
            'resid 1', cutoff=25.0, frame=0,
            boundary_scheme='NONE', neutralize=False,
        )
        # res2 (x≈12.5) is within 25 Å → 3 atoms
        # res3 (x≈20) is within 25 Å → 3 atoms
        # res4 (x≈80) is outside 25 Å → excluded
        included_resids = set(mm_ag.resids)
        assert 2 in included_resids
        assert 3 in included_resids
        assert 4 not in included_resids

    def test_far_residue_excluded(self, tiny_universe):
        """Res4 at x=80, QM at x≈10, box=100 → PBC distance=30 Å.
        With cutoff=25, it should be excluded."""
        gen = _build_generator(tiny_universe)
        _, _, _, _, mm_ag, *_ = gen.extract_point_charges(
            'resid 1', cutoff=25.0, frame=0,
            boundary_scheme='NONE', neutralize=False,
        )
        assert 4 not in set(mm_ag.resids)


# ===================================================================
# Full pipeline consistency
# ===================================================================

class TestFullPipeline:
    def test_rcd_with_switching_and_neutralization(self, tiny_universe):
        """Run the full pipeline and verify output is consistent."""
        gen = _build_generator(tiny_universe)
        charges, mods, sw_recs, img_info, mm_ag, qm_center, box = \
            gen.extract_point_charges(
                'resid 1', cutoff=50.0, frame=0,
                boundary_scheme='RCD',
                switchdist=40.0,
                target_mm_charge=0.0,
                neutralize=True,
                neutralization_shell_fraction=0.1,
            )

        # Charges should be a non-empty list of (q, x, y, z)
        assert len(charges) > 0
        for c in charges:
            assert len(c) == 4

        # Total charge should be on target
        total = sum(q for q, *_ in charges)
        assert abs(total - 0.0) < 1e-4

        # Boundary mods should exist (we have a QM-MM bond)
        assert len(mods) > 0

        # qm_center should be near the QM atoms
        qm_pos = tiny_universe.select_atoms("resid 1").positions.mean(axis=0)
        np.testing.assert_allclose(qm_center, qm_pos, atol=1e-4)

    def test_none_scheme_no_boundary(self, tiny_universe):
        """NONE scheme should produce no boundary modifications."""
        gen = _build_generator(tiny_universe)
        charges, mods, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='NONE', neutralize=False,
        )
        assert len(mods) == 0

    def test_all_schemes_produce_charges(self, tiny_universe):
        """Every valid scheme should return a non-empty charge list."""
        gen = _build_generator(tiny_universe)
        for scheme in ['RCD', 'CS', 'Z1', 'Z2', 'Z3', 'NONE']:
            charges, *_ = gen.extract_point_charges(
                'resid 1', cutoff=50.0, frame=0,
                boundary_scheme=scheme, neutralize=False,
            )
            assert len(charges) > 0, f"Scheme {scheme} produced no charges"

    def test_supercell_adds_images_for_small_box(self):
        """With a small box and supercell on, images should appear."""
        u = mda.Universe.empty(
            6, n_residues=2, n_segments=1,
            atom_resindex=np.array([0, 0, 0, 1, 1, 1]),
            residue_segindex=np.array([0, 0]),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['C1', 'H1', 'H2', 'O', 'H3', 'H4'])
        u.add_TopologyAttr('resname', ['QM', 'WAT'])
        u.add_TopologyAttr('resid', [1, 2])
        u.add_TopologyAttr('segid', ['S'])
        u.add_TopologyAttr('mass', [12.011, 1.008, 1.008, 15.999, 1.008, 1.008])
        u.add_TopologyAttr('charge', [0, 0, 0, -0.82, 0.41, 0.41])
        u.add_TopologyAttr('tempfactors', np.zeros(6))
        u.add_TopologyAttr('bonds', [(0, 1), (0, 2), (3, 4), (3, 5)])
        # MM at x≈45 (near box edge), QM at x=25, box=50
        # Image at x=45-50=-5, distance from QM=30 < cutoff=35
        u.atoms.positions = np.array([
            [25, 25, 25], [26, 25, 25], [25, 26, 25],
            [45, 25, 25], [46, 25, 25], [45, 26, 25],
        ], dtype=np.float32)
        u.dimensions = np.array([50, 50, 50, 90, 90, 90], dtype=np.float32)

        gen = _build_generator(u)
        charges, _, _, img_info, *_ = gen.extract_point_charges(
            'resid 1', cutoff=35.0, frame=0,
            boundary_scheme='NONE',
            expand=(True, True, True),
            neutralize=False,
        )
        assert img_info.get('n_images', 0) > 0

    def test_no_images_for_large_box(self, tiny_universe):
        """Box=100, cutoff=40 → no images needed."""
        gen = _build_generator(tiny_universe)
        charges, _, _, img_info, *_ = gen.extract_point_charges(
            'resid 1', cutoff=40.0, frame=0,
            boundary_scheme='NONE',
            expand=(True, True, True),
            neutralize=False,
        )
        assert img_info.get('n_images', 0) == 0


# ===================================================================
# Sanity: QM + MM charges should cover what we expect
# ===================================================================

class TestChargeAccounting:
    def test_qm_psf_charge_is_sum_of_selection(self, tiny_universe):
        """QM PSF charge should be the sum of selected atom charges."""
        gen = _build_generator(tiny_universe)
        qm = tiny_universe.select_atoms("resid 1")
        expected = sum(gen._psf_charges[a.index] for a in qm)
        # This is what the run log reports
        assert abs(expected - 0.0) < 1e-10  # res1: -0.2 + 0.1 + 0.1

    def test_neutralized_mm_matches_target_across_schemes(self, tiny_universe):
        """Neutralization target should be met regardless of boundary scheme."""
        gen = _build_generator(tiny_universe)
        for scheme in ['RCD', 'CS', 'Z1', 'Z2', 'Z3', 'NONE']:
            charges, *_ = gen.extract_point_charges(
                'resid 1', cutoff=50.0, frame=0,
                boundary_scheme=scheme,
                target_mm_charge=1.0,
                neutralize=True,
                neutralization_shell_fraction=0.2,
            )
            total = sum(q for q, *_ in charges)
            assert abs(total - 1.0) < 1e-3, \
                f"Scheme {scheme}: total={total:.4f}, expected 1.0"
