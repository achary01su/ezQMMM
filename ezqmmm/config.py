"""Configuration parsing, validation, and example generation."""

from typing import Optional


def parse_axes(axes_config) -> tuple[bool, bool, bool]:
    """
    Parse supercell_axes config into (expand_x, expand_y, expand_z).
    Accepts a list or comma-separated string: x/a, y/b, z/c.
    Examples: ['x','y'], 'x,y,z', 'z'
    """
    if not axes_config:
        return False, False, False
    if isinstance(axes_config, str):
        tokens = [t.strip().lower() for t in axes_config.split(',')]
    else:
        tokens = [str(t).strip().lower() for t in axes_config]
    return (
        any(t in ('x', 'a') for t in tokens),
        any(t in ('y', 'b') for t in tokens),
        any(t in ('z', 'c') for t in tokens),
    )


def parse_pdb_stride(value) -> Optional[int]:
    """
    Parse pdb_stride config value.
      'all'   -> 1  (every frame)
      'half'  -> 2  (every other frame)
      'tenth' -> 10 (every 10th frame)
      integer -> that integer
      null / absent -> None (disabled)
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    mapping = {'all': 1, 'half': 2, 'tenth': 10}
    v = str(value).strip().lower()
    if v in mapping:
        return mapping[v]
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(
            f"pdb_stride '{value}' not recognised. "
            f"Use: all, half, tenth, or an integer."
        ) from e


def validate_config(config: dict, n_frames: int):
    """
    Validate a parsed config dict.  Raises ValueError with a precise
    message for any invalid or scientifically dangerous input.

    Parameters
    ----------
    config   : dict loaded from YAML
    n_frames : total number of frames in the trajectory
    """
    # --- Required keys ---
    if 'qm_selection' not in config:
        raise ValueError("'qm_selection' is required in the config file")
    if 'program' not in config:
        raise ValueError("'program' required (orca/qchem/psi4)")

    # --- Program ---
    program = config['program'].lower()
    valid_programs = {'orca', 'qchem'}
    if program not in valid_programs:
        raise ValueError(
            f"program '{program}' not recognised. "
            f"Valid options: {', '.join(sorted(valid_programs))}"
        )

    # --- Boundary scheme ---
    bscheme = config.get('boundary_scheme', 'RCD').upper()
    valid_schemes = {'RCD', 'CS', 'Z1', 'Z2', 'Z3', 'NONE'}
    if bscheme not in valid_schemes:
        raise ValueError(
            f"boundary_scheme '{bscheme}' not recognised. "
            f"Valid options: {', '.join(sorted(valid_schemes))}"
        )

    # CS-specific parameters
    if bscheme == 'CS':
        cs_frac = config.get('cs_bond_fraction', 0.06)
        if not (0.0 < cs_frac < 0.3):
            raise ValueError(
                f"cs_bond_fraction should be in (0, 0.3), got {cs_frac}. "
                f"NAMD's default is 0.06."
        )
    elif 'cs_bond_fraction' in config:
        # CS parameter set but boundary scheme is not CS: display warning msg 
        import warnings
        warnings.warn(
            f"cs_bond_fraction value will be ignored since boundary_scheme is: '{bscheme}'."
        )

    # --- PBC compound grouping ---
    pbc_compound = config.get('pbc_compound', 'residue')
    if pbc_compound not in ('residue', 'fragment', 'atom'):
        raise ValueError(
            f"pbc_compound must be 'residue', 'fragment', or 'atom', "
            f"got '{pbc_compound}'."
        )

    # --- Parallelization ---
    n_workers = config.get('n_workers', 1)
    if not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError(f"n_workers must be a positive integer, got {n_workers}")

    # --- Frame range ---
    first = config.get('first_frame', 0)
    stride = config.get('stride', 1)

    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}")
    if first < 0:
        raise ValueError(f"first_frame must be >= 0, got {first}")

    last = config.get('last_frame', -1)
    if last == -1 or last >= n_frames:
        last = n_frames - 1
    if first > last:
        raise ValueError(
            f"first_frame ({first}) is after last_frame ({last}). "
            f"Trajectory has {n_frames} frames."
        )

    # --- Switching window ---
    mm_cutoff = config.get('mm_cutoff', 40.0)
    mm_switchdist = config.get('mm_switchdist')
    if mm_switchdist is not None and mm_switchdist >= mm_cutoff:
        raise ValueError(
            f"mm_switchdist ({mm_switchdist}) must be less than "
            f"mm_cutoff ({mm_cutoff}). The switching function "
            f"attenuates charges between switchdist and cutoff."
        )

    # --- Neutralization shell fraction ---
    neutralize_mm = config.get('neutralize_mm_charge', True)
    neutral_frac = config.get('neutralization_shell_fraction', 0.1)
    if neutralize_mm and not (0.0 < neutral_frac <= 1.0):
        raise ValueError(
            f"neutralization_shell_fraction must be in (0, 1], "
            f"got {neutral_frac}"
        )


def create_example_config():
    """Write a fully-commented example configuration file."""
    config = """# ezQMMM 2.0 Configuration
psf_file: system.psf
dcd_file: trajectory.dcd
qm_selection: "resid 100 and not backbone"
mm_cutoff: 40.0
mm_switchdist: 35.0   
first_frame: 0
last_frame: 100
stride: 10
method: B3LYP
basis: 6-31G*
charge: 0
multiplicity: 1
boundary_scheme: RCD
# CS virtual-charge displacement as a fraction of the MM1–MM2 bond length.
# Default 0.06 matches NAMD
cs_bond_fraction: 0.06

# Periodic boundary remapping of MM atoms.
# QM codes are not PBC-aware. They see point charges in plain Cartesian space. 
# pbc_compound: <value>,  remaps charge location at the correct minimum-image position w.r.t. QM, while avoiding creating stretched bonds at boundary 
# residue  : group by residue (default)
# fragment : group by covalent connectivity 
# atom     : per-atom minimum image 
pbc_compound: residue


# MM charge neutralization (default: true).
# Distributes any residual MM charge evenly across all point charges each
# frame to enforce target_mm_charge. Eliminates frame-to-frame fluctuations
# from ions drifting in/out of the cutoff. Set to false for benchmarking
# or to reproduce raw PSF charge behaviour.
neutralize_mm_charge: true
target_mm_charge: 0.0
neutralization_shell_fraction: 0.1  # outermost 10% of charges absorb the correction
output_dir: ./qmmm_calculations
output_prefix: qmmm
program: qchem

# Axes along which to tile periodic images (any combo of x/y/z or a/b/c).
# Leave empty or omit to disable. Examples:
#   supercell_axes: [x, y]       # membrane protein, normal along z
#   supercell_axes: [x, y, z]    # small solvated active site
#   supercell_axes: [z]          # elongated system (DNA, fibril)
supercell_axes: []

# Write a PDB per frame (shared PSF written once).
# Options: all, half, tenth, or any integer stride. null = disabled.
pdb_stride: null

# Parallel frame processing. Default 1 (serial).
# Each worker loads its own copy of the trajectory (~500MB per worker).
# Set to number of physical CPU cores for best performance.
n_workers: 1

qchem_keywords: |
  scf_convergence    8
  mem_total          8000

qchem_blocks: |
  $basis
  C 0
  cc-pVDZ
  ****
  H 0
  cc-pVDZ
  ****
  $end
"""
    with open('config_example.yaml', 'w') as f:
        f.write(config)
    print("Created: config_example.yaml")
