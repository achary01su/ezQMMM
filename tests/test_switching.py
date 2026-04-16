"""Tests for ezqmmm.switching — quintic switching function."""

import numpy as np

from ezqmmm.switching import apply_switching


class TestQuinticBehavior:
    """Test apply_switching at known distance regimes."""

    def test_inside_switchdist_unscaled(self):
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 5.0, 0.0, 0.0), (0.3, 8.0, 0.0, 0.0)]
        scaled, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        for orig, new in zip(charges, scaled):
            assert abs(orig[0] - new[0]) < 1e-10
        assert len(recs) == 0

    def test_beyond_cutoff_zeroed(self):
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 20.0, 0.0, 0.0)]
        scaled, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        assert abs(scaled[0][0]) < 1e-10
        assert len(recs) == 1
        assert abs(recs[0].scale) < 1e-10

    def test_midpoint_partially_scaled(self):
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 12.5, 0.0, 0.0)]
        scaled, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        assert 0 < abs(scaled[0][0]) < 0.5
        assert 0 < recs[0].scale < 1

    def test_midpoint_approximately_half(self):
        """S(midpoint) ≈ 0.5 for the quintic."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-1.0, 12.5, 0.0, 0.0)]
        scaled, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        assert abs(recs[0].scale - 0.5) < 0.01

    def test_monotonic_across_zone(self):
        """Charges at increasing distances should have decreasing scale."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        dists = np.linspace(10.5, 14.5, 20)
        charges = [(-1.0, d, 0.0, 0.0) for d in dists]
        scaled, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        scales = [abs(s[0]) for s in scaled]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i + 1] - 1e-10

    def test_positions_unchanged(self):
        """Switching should modify charges, not positions."""
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [(-0.5, 12.0, 3.0, 7.0)]
        scaled, _ = apply_switching(charges, qm_pos, 10.0, 15.0,
                                     box=None, frame=0)
        assert abs(scaled[0][1] - 12.0) < 1e-10
        assert abs(scaled[0][2] - 3.0) < 1e-10
        assert abs(scaled[0][3] - 7.0) < 1e-10

    def test_empty_charges(self):
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        scaled, recs = apply_switching([], qm_pos, 10.0, 15.0,
                                        box=None, frame=0)
        assert scaled == []
        assert recs == []


class TestImageFlag:
    def test_primary_vs_image(self):
        qm_pos = np.array([[0.0, 0.0, 0.0]])
        charges = [
            (-0.5, 12.0, 0.0, 0.0),  # primary
            (-0.5, 13.0, 0.0, 0.0),  # image
        ]
        _, recs = apply_switching(charges, qm_pos, 10.0, 15.0,
                                   box=None, frame=0, n_primary=1)
        primaries = [r for r in recs if not r.is_image]
        images = [r for r in recs if r.is_image]
        assert len(primaries) == 1
        assert len(images) == 1
