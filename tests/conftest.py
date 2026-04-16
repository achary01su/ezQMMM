"""
Shared test fixtures for ezQMMM.

tiny_universe: 12-atom, 4-residue system with a QM-MM boundary bond.
straddling_universe: 6-atom system with a water straddling the box boundary.
outside_cutoff_universe: 12-atom system where residue 4 is genuinely outside cutoff.
"""

import MDAnalysis as mda
import numpy as np
import pytest


@pytest.fixture
def tiny_universe():
    """
    12-atom, 4-residue system:
      res1 (QM1):  atoms 0-2,  x≈10, QM region
      res2 (MM1):  atoms 3-5,  x≈12.5, bonded to res1 (boundary)
      res3 (WAT):  atoms 6-8,  x≈20, nearby MM water
      res4 (WAT):  atoms 9-11, x≈80, far MM water (outside typical cutoff)
    Box: 100x100x100  |  Bond 0–3 is the QM-MM boundary.
    """
    u = mda.Universe.empty(
        12, n_residues=4, n_segments=1,
        atom_resindex=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        residue_segindex=np.array([0, 0, 0, 0]),
        trajectory=True,
    )
    u.add_TopologyAttr('name', ['C1', 'H1', 'H2', 'C2', 'H3', 'H4',
                                 'O1', 'HW1', 'HW2', 'O2', 'HW3', 'HW4'])
    u.add_TopologyAttr('resname', ['QM1', 'MM1', 'WAT', 'WAT'])
    u.add_TopologyAttr('resid', [1, 2, 3, 4])
    u.add_TopologyAttr('segid', ['SEG1'])
    u.add_TopologyAttr('mass', [12.011, 1.008, 1.008, 12.011, 1.008, 1.008,
                                 15.999, 1.008, 1.008, 15.999, 1.008, 1.008])
    u.add_TopologyAttr('charge', [-0.2, 0.1, 0.1, -0.3, 0.15, 0.15,
                                   -0.82, 0.41, 0.41, -0.82, 0.41, 0.41])
    u.add_TopologyAttr('tempfactors', np.zeros(12))
    u.add_TopologyAttr('bonds', [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5),
                                  (6, 7), (6, 8), (9, 10), (9, 11)])
    u.atoms.positions = np.array([
        [10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0],
        [12.5, 10.0, 10.0], [13.5, 10.0, 10.0], [12.5, 11.0, 10.0],
        [20.0, 10.0, 10.0], [21.0, 10.0, 10.0], [20.0, 11.0, 10.0],
        [80.0, 10.0, 10.0], [81.0, 10.0, 10.0], [80.0, 11.0, 10.0],
    ], dtype=np.float32)
    u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    return u


@pytest.fixture
def straddling_universe():
    """
    6-atom, 2-residue system where one water straddles the box boundary.
      res1 (QM1): atoms 0-2 at x≈50
      res2 (WAT): O at x=1, H at x=99, H at x=1.5 (straddling)
    Box: 100x100x100
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
        [50.0, 50.0, 50.0], [51.0, 50.0, 50.0], [50.0, 51.0, 50.0],
        [1.0, 50.0, 50.0], [99.0, 50.0, 50.0], [1.5, 50.0, 50.0],
    ], dtype=np.float32)
    u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    return u
