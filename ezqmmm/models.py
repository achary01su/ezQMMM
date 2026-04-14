"""Data records for charge modifications and switching events."""

import numpy as np
from typing import Optional


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
