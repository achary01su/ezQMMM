"""Tests for charge neutralization — tested through the package API."""

import warnings
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from ezqmmm.generator import QMMMGenerator
from ezqmmm.boundary import find_boundary_bonds, apply_boundary_scheme
from ezqmmm.switching import apply_switching


def _build_generator(universe):
    """
    Build a QMMMGenerator-like object from an existing universe,
    bypassing __init__ which requires PSF/DCD files.
    """
    gen = object.__new__(QMMMGenerator)
    gen.universe = universe
    gen._psf_charges = {a.index: float(a.charge) for a in universe.atoms}
    return gen


class TestNeutralizationHitsTarget:
    def test_target_zero(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='RCD', target_mm_charge=0.0,
            neutralize=True, neutralization_shell_fraction=0.1,
        )
        total = sum(q for q, *_ in charges)
        assert abs(total - 0.0) < 1e-4

    def test_target_positive(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='RCD', target_mm_charge=2.0,
            neutralize=True, neutralization_shell_fraction=0.1,
        )
        total = sum(q for q, *_ in charges)
        assert abs(total - 2.0) < 1e-4

    def test_target_negative(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='RCD', target_mm_charge=-1.0,
            neutralize=True, neutralization_shell_fraction=0.1,
        )
        total = sum(q for q, *_ in charges)
        assert abs(total - (-1.0)) < 1e-4


class TestNeutralizationDisabled:
    def test_raw_charges_when_off(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='NONE', neutralize=False,
        )
        # Without boundary or neutralization, total should be raw PSF sum
        qm = tiny_universe.select_atoms("resid 1")
        mm = [a for a in tiny_universe.select_atoms("all")
              if a.index not in set(qm.indices)]
        # Only atoms within cutoff=50 (res2 at ~12.5, res3 at ~20 are in;
        # res4 at ~80 is out). Whole-residue: 6 atoms
        raw_in_cutoff = sum(a.charge for a in mm if a.resid in [2, 3])
        total = sum(q for q, *_ in charges)
        assert abs(total - raw_in_cutoff) < 1e-4


class TestNeutralizationOnlyTouchesOuterShell:
    def test_inner_charges_unchanged(self, tiny_universe):
        """Charges near QM should not be modified by neutralization."""
        gen = _build_generator(tiny_universe)

        # Get charges without neutralization
        charges_raw, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='NONE', neutralize=False,
        )

        # Get charges with neutralization
        charges_neut, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='NONE', target_mm_charge=0.0,
            neutralize=True, neutralization_shell_fraction=0.5,
        )

        # Sort both by position for comparison
        raw_sorted = sorted(charges_raw, key=lambda c: (c[1], c[2], c[3]))
        neut_sorted = sorted(charges_neut, key=lambda c: (c[1], c[2], c[3]))

        # Find the charges closest to QM — they should be identical
        qm_c = tiny_universe.select_atoms("resid 1").positions.mean(axis=0)
        raw_dists = [np.linalg.norm(np.array([x, y, z]) - qm_c)
                     for _, x, y, z in raw_sorted]

        # Inner 50% should be untouched (we set fraction=0.5)
        median_d = np.median(raw_dists)
        for (q_raw, x, y, z), (q_neut, *_) in zip(raw_sorted, neut_sorted):
            d = np.linalg.norm(np.array([x, y, z]) - qm_c)
            if d < median_d:
                assert abs(q_raw - q_neut) < 1e-10, \
                    f"Inner charge at d={d:.1f} was modified"


class TestPipelineOrder:
    def test_neutralization_after_switching(self, tiny_universe):
        """
        With switching enabled, neutralization should still hit the target.
        This verifies the order: boundary → remap → tile → switch → neutralize.
        If neutralization ran before switching, the total would drift.
        """
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='RCD',
            switchdist=40.0,
            target_mm_charge=0.0,
            neutralize=True, neutralization_shell_fraction=0.1,
        )
        total = sum(q for q, *_ in charges)
        assert abs(total - 0.0) < 1e-4

    def test_neutralization_after_switching_nonzero_target(self, tiny_universe):
        gen = _build_generator(tiny_universe)
        charges, *_ = gen.extract_point_charges(
            'resid 1', cutoff=50.0, frame=0,
            boundary_scheme='RCD',
            switchdist=40.0,
            target_mm_charge=3.0,
            neutralize=True, neutralization_shell_fraction=0.1,
        )
        total = sum(q for q, *_ in charges)
        assert abs(total - 3.0) < 1e-4


class TestSanityCheckWarning:
    def test_no_warning_on_good_pipeline(self, tiny_universe):
        """Normal pipeline should not trigger the sanity warning."""
        gen = _build_generator(tiny_universe)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen.extract_point_charges(
                'resid 1', cutoff=50.0, frame=0,
                boundary_scheme='RCD', target_mm_charge=0.0,
                neutralize=True, neutralization_shell_fraction=0.1,
            )
            charge_warnings = [x for x in w
                               if 'deviates from target' in str(x.message)]
            assert len(charge_warnings) == 0
