"""
ezQMMM 2.0

Generates QM/MM single-point calculation input files from CHARMM/NAMD MD
trajectories.  Supports ORCA, Q-Chem, and Psi4.
"""

__version__ = "2.0.1"

from ezqmmm.config import create_example_config  # noqa: F401
from ezqmmm.generator import QMMMGenerator  # noqa: F401
from ezqmmm.models import ChargeMod, SwitchRecord  # noqa: F401
