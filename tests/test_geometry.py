"""Tests for ezqmmm.geometry — remapping, image shells, tiling."""

import numpy as np
import pytest

from ezqmmm.geometry import (
    image_shells,
    remap_position,
    remap_positions_array,
    remap_positions_by_residue,
    tile_images,
)

# ===================================================================
# Minimum image remapping
# ===================================================================

class TestRemapPosition:
    @pytest.mark.parametrize("x,ref,L,expected", [
        (50.0, 50.0, 100.0, 50.0),
        (95.0, 5.0, 100.0, -5.0),
        (5.0, 95.0, 100.0, 105.0),
        (50.0, 0.0, 100.0, -50.0),
        (0.0, 0.0, 100.0, 0.0),
        (99.99, 0.01, 100.0, -0.01),
    ])
    def test_scalar(self, x, ref, L, expected):
        result = remap_position(
            np.array([x, 0.0, 0.0]),
            np.array([ref, 0.0, 0.0]),
            np.array([L, L, L]),
        )
        assert abs(result[0] - expected) < 1e-6

    def test_3d(self):
        result = remap_position(
            np.array([95.0, 5.0, 50.0]),
            np.array([5.0, 95.0, 50.0]),
            np.array([100.0, 100.0, 100.0]),
        )
        assert abs(result[0] - (-5.0)) < 1e-6
        assert abs(result[1] - 105.0) < 1e-6
        assert abs(result[2] - 50.0) < 1e-6


class TestRemapPositionsArray:
    def test_batch(self):
        pos = np.array([[95.0, 50.0, 50.0], [5.0, 50.0, 50.0]])
        out = remap_positions_array(
            pos, np.array([5.0, 50.0, 50.0]), np.array([100.0, 100.0, 100.0])
        )
        assert abs(out[0, 0] - (-5.0)) < 1e-6
        assert abs(out[1, 0] - 5.0) < 1e-6

    def test_preserves_input(self):
        """Input array should not be mutated."""
        pos = np.array([[95.0, 50.0, 50.0]])
        original = pos.copy()
        remap_positions_array(pos, np.array([5.0, 50.0, 50.0]),
                               np.array([100.0, 100.0, 100.0]))
        np.testing.assert_array_equal(pos, original)


# ===================================================================
# Per-residue remapping (PDB writer)
# ===================================================================

class TestRemapByResidue:
    def test_straddling_water_reassembled(self, straddling_universe):
        """O at x=1, H at x=99 should end up within bond distance."""
        u = straddling_universe
        mm_ag = u.select_atoms("resname WAT")
        orig_pos = mm_ag.positions.copy()
        qm_c = np.array([50.0, 50.0, 50.0])

        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, u.dimensions)

        dist_OH1 = np.linalg.norm(out[0] - out[1])
        dist_OH2 = np.linalg.norm(out[0] - out[2])
        assert dist_OH1 < 5.0, f"O-H1 broken: {dist_OH1:.1f} Å"
        assert dist_OH2 < 5.0, f"O-H2 broken: {dist_OH2:.1f} Å"

    def test_per_atom_would_break_straddling_water(self, straddling_universe):
        """Confirm per-atom remap leaves O and H 98 Å apart."""
        u = straddling_universe
        mm_ag = u.select_atoms("resname WAT")
        orig_pos = mm_ag.positions.copy()
        out = remap_positions_array(orig_pos, np.array([50.0, 50.0, 50.0]),
                                     u.dimensions)
        assert abs(out[0, 0] - out[1, 0]) > 50

    def test_distant_water_shifts_as_unit(self):
        """Water at x≈95, QM at x=5 — whole residue shifts by -100."""
        # Build a minimal mock
        import MDAnalysis as mda
        u = mda.Universe.empty(
            3, n_residues=1, n_segments=1,
            atom_resindex=np.array([0, 0, 0]),
            residue_segindex=np.array([0]),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['O', 'H1', 'H2'])
        u.add_TopologyAttr('resname', ['WAT'])
        u.add_TopologyAttr('resid', [1])
        u.add_TopologyAttr('segid', ['S'])
        pos = np.array([[95.0, 50.0, 50.0],
                         [94.0, 50.0, 50.0],
                         [95.5, 50.0, 50.0]], dtype=np.float32)
        u.atoms.positions = pos
        u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

        mm_ag = u.select_atoms("all")
        out = remap_positions_by_residue(mm_ag, pos.copy(),
                                          np.array([5.0, 50.0, 50.0]),
                                          u.dimensions)
        # All shift by -100
        assert abs(out[0, 0] - (-5.0)) < 1e-4
        assert abs(out[1, 0] - (-6.0)) < 1e-4
        assert abs(out[2, 0] - (-4.5)) < 1e-4
        # Bond lengths preserved
        for j in [1, 2]:
            orig_d = pos[0, 0] - pos[j, 0]
            new_d = out[0, 0] - out[j, 0]
            assert abs(orig_d - new_d) < 1e-4

    def test_nearby_water_unchanged(self, tiny_universe):
        u = tiny_universe
        mm_ag = u.select_atoms("resid 3")
        orig_pos = mm_ag.positions.copy()
        qm_c = u.select_atoms("resid 1").positions.mean(axis=0)
        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, u.dimensions)
        np.testing.assert_allclose(out, orig_pos, atol=1e-4)

    def test_two_residues_preserve_geometry(self, tiny_universe):
        u = tiny_universe
        mm_ag = u.select_atoms("resid 3 or resid 4")
        orig_pos = mm_ag.positions.copy()
        qm_c = u.select_atoms("resid 1").positions.mean(axis=0)
        out = remap_positions_by_residue(mm_ag, orig_pos, qm_c, u.dimensions)

        for res in mm_ag.residues:
            idx = [i for i, a in enumerate(mm_ag.atoms) if a in res.atoms]
            for j in idx[1:]:
                orig_d = orig_pos[idx[0]] - orig_pos[j]
                new_d = out[idx[0]] - out[j]
                np.testing.assert_allclose(orig_d, new_d, atol=1e-4)


# ===================================================================
# Image shells and tiling
# ===================================================================

class TestImageShells:
    def test_no_expand(self):
        assert image_shells(40, np.array([100.0, 100.0, 100.0]),
                            (False, False, False)) == (0, 0, 0)

    def test_partial(self):
        assert image_shells(40, np.array([100.0, 100.0, 100.0]),
                            (True, False, True)) == (1, 0, 1)

    def test_small_box(self):
        assert image_shells(50, np.array([30.0, 30.0, 30.0]),
                            (True, True, True)) == (2, 2, 2)


class TestTileImages:
    def test_no_charges_returns_empty(self):
        images, shells, n = tile_images(
            [], np.array([[0, 0, 0.0]]), 40.0,
            np.array([100.0, 100.0, 100.0]), (True, True, True),
        )
        assert images == []
        assert n == 0

    def test_large_box_no_images(self):
        """Box=140, cutoff=50 → nearest image at 90 Å, outside cutoff."""
        primary = [(-0.5, 10.0, 10.0, 10.0)]
        qm_pos = np.array([[5.0, 10.0, 10.0]])
        images, shells, n = tile_images(
            primary, qm_pos, 50.0,
            np.array([140.0, 140.0, 140.0]), (True, True, True),
        )
        assert len(images) == 0

    def test_small_box_produces_images(self):
        """Primary near box edge → image wraps to near QM, within cutoff."""
        # Primary at x=55, QM at x=5, box=60 → image at x=-5, dist=10 < cutoff=40
        primary = [(-0.5, 55.0, 5.0, 5.0)]
        qm_pos = np.array([[5.0, 5.0, 5.0]])
        images, shells, n = tile_images(
            primary, qm_pos, 40.0,
            np.array([60.0, 60.0, 60.0]), (True, True, True),
        )
        assert len(images) > 0
        # All images should be within cutoff of QM
        for _q, x, y, z in images:
            d = np.linalg.norm(np.array([x, y, z]) - qm_pos[0])
            assert d <= 40.0 + 0.01

    def test_images_have_same_charge_as_primary(self):
        """Image charges must equal their primary source charge."""
        primary = [(-0.82, 55.0, 5.0, 5.0), (0.41, 56.0, 5.0, 5.0)]
        qm_pos = np.array([[5.0, 5.0, 5.0]])
        images, _, _ = tile_images(
            primary, qm_pos, 40.0,
            np.array([60.0, 60.0, 60.0]), (True, True, True),
        )
        assert len(images) > 0
        primary_qs = {round(q, 6) for q, *_ in primary}
        for q, *_ in images:
            assert round(q, 6) in primary_qs

    def test_single_axis_only_tiles_that_axis(self):
        """supercell_axes=[x] should only produce images along x."""
        primary = [(-0.5, 5.0, 50.0, 50.0)]
        qm_pos = np.array([[5.0, 50.0, 50.0]])
        images, shells, _ = tile_images(
            primary, qm_pos, 40.0,
            np.array([60.0, 60.0, 60.0]), (True, False, False),
        )
        assert shells[1] == 0
        assert shells[2] == 0
        for _, _x, y, z in images:
            # y and z should be identical to primary
            assert abs(y - 50.0) < 1e-6
            assert abs(z - 50.0) < 1e-6
