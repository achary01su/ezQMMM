"""
Minimum-image remapping and supercell image tiling.

Uses floor((x - x_ref)/L + 0.5), which is equivalent to round-half-up convention.
"""


import numpy as np
from MDAnalysis.analysis import distances

# Minimum image remapping
# ------------------------------------------------------------------

def remap_position(pos: np.ndarray, qm_center: np.ndarray,
                   box: np.ndarray) -> np.ndarray:
    """
    Remap a single (3,) position to minimum image relative to *qm_center*.

    It must be consitent with inlined remap in generator.extract_point_charges
    used for QM-input charge positions 
    """
    lx, ly, lz = box[0], box[1], box[2]
    return np.array([
        pos[0] - np.floor((pos[0] - qm_center[0]) / lx + 0.5) * lx,
        pos[1] - np.floor((pos[1] - qm_center[1]) / ly + 0.5) * ly,
        pos[2] - np.floor((pos[2] - qm_center[2]) / lz + 0.5) * lz,
    ])


# Could be used in generator.py
# I will do it later. For now, remap_positions_array is unused.
# I will keep this commented out. Use inline arithmetic as currently done.

#def remap_positions_array(pos: np.ndarray, qm_center: np.ndarray,
#                          box: np.ndarray) -> np.ndarray:
#    """
#    Vectorised remap of an (N, 3) position array to minimum image.
#    """
#    lx, ly, lz = box[0], box[1], box[2]
#    out = pos.copy()
#    out[:, 0] -= np.floor((pos[:, 0] - qm_center[0]) / lx + 0.5) * lx
#    out[:, 1] -= np.floor((pos[:, 1] - qm_center[1]) / ly + 0.5) * ly
#    out[:, 2] -= np.floor((pos[:, 2] - qm_center[2]) / lz + 0.5) * lz
#    return out

def remap_positions_by_compound(mm_ag, orig_pos: np.ndarray,
                                 qm_center: np.ndarray,
                                 box: np.ndarray,
                                 compound: str = 'residue') -> np.ndarray:
    """
    Remap MM atom positions to minimum image relative to QM centroid.

    The QM codes are not PBC-aware. They see point charges in plain Cartesian space. 
    This remap ensures each charge appears at the correct minimum-image position relative to QM, 
    while keeping bonded groups intact.

    For compound='residue' or 'fragment', two-step process:
      1. Unwrap all atoms to be near the reference atom.
      2. Shift the whole fragment/residue to minimum image of QM centroid.

    For compound='atom', each atom is independently wrapped; GROMACS style
    Likely to show stretched bonds across periodic boundaries).

    For compound='NONE', no remapping is performed, the original coordinates are returned.

    Parameters
    ----------
    mm_ag     : MDAnalysis AtomGroup of MM atoms within cutoff
    orig_pos  : (N, 3) original positions of mm_ag atoms
    qm_center : (3,) QM centroid
    box       : (6,) box dimensions (orthorhombic; only [0:3] used)
    compound  : 'residue'  - group by PSF residue (default, fast)
                'fragment' - group by covalent connectivity 
                             use when ligands/lipids/LPSs contains multiple residues)
                'atom'     - atoms-based. No grouping 
                'NONE'     - no remapping
    """
    lx, ly, lz = box[0], box[1], box[2]

    # No remapping
    if compound == 'NONE':
        return orig_pos
      
    # Per-atom remap 
    if compound == 'atom':
        out = orig_pos.copy()
        out[:, 0] -= np.floor((orig_pos[:, 0] - qm_center[0]) / lx + 0.5) * lx
        out[:, 1] -= np.floor((orig_pos[:, 1] - qm_center[1]) / ly + 0.5) * ly
        out[:, 2] -= np.floor((orig_pos[:, 2] - qm_center[2]) / lz + 0.5) * lz
        return out

    # Compound-aware remap
    if compound == 'residue':
        groups = mm_ag.residues
    elif compound == 'fragment':
        groups = mm_ag.fragments
    else:
        raise ValueError(
            f"compound must be 'residue', 'fragment', or 'atom', "
            f"got '{compound}'"
        )

    new_pos = orig_pos.copy()
    idx_to_row = {idx: i for i, idx in enumerate(mm_ag.indices)}

    for grp in groups:
        rows = [idx_to_row[a.index] for a in grp.atoms
                if a.index in idx_to_row]
        if not rows:
            continue

        ref = rows[0]
        ref_pos = new_pos[ref]

        # Step 1: unwrap all atoms to be near the reference atom
        for r in rows[1:]:
            new_pos[r, 0] -= np.floor((new_pos[r, 0] - ref_pos[0]) / lx + 0.5) * lx
            new_pos[r, 1] -= np.floor((new_pos[r, 1] - ref_pos[1]) / ly + 0.5) * ly
            new_pos[r, 2] -= np.floor((new_pos[r, 2] - ref_pos[2]) / lz + 0.5) * lz

        # Step 2: shift the whole unit to minimum image of QM center
        shift = np.array([
            -np.floor((ref_pos[0] - qm_center[0]) / lx + 0.5) * lx,
            -np.floor((ref_pos[1] - qm_center[1]) / ly + 0.5) * ly,
            -np.floor((ref_pos[2] - qm_center[2]) / lz + 0.5) * lz,
        ])
        for r in rows:
            new_pos[r] += shift

    return new_pos


# ------------------------------------------------------------------------
# Supercell image tiling
# ------------------------------------------------------------------------

def image_shells(cutoff: float, box: np.ndarray,
                 expand: tuple[bool, bool, bool]) -> tuple[int, int, int]:
    """
    Number of image shells per axis: ceil(cutoff / L) for active axes,
    0 for suppressed axes.
    """
    return tuple(
        int(np.ceil(cutoff / box[i])) if do_expand else 0
        for i, do_expand in enumerate(expand)
    )


def tile_images(charges: list, qm_pos: np.ndarray,
                cutoff: float, box: np.ndarray,
                expand: tuple[bool, bool, bool]
                ) -> tuple[list, tuple[int, int, int], int]:
    """
    Generate periodic images of primary charges along requested axes.
    Primary charges must already be remapped to minimum image positions
    relative to the QM centroid so the (0,0,0) shell skip correctly
    corresponds to what is already in the primary charge list.
    Returns (image_charges, shells, n_candidates).
    """
    if not charges:
        return [], (0, 0, 0), 0

    lx, ly, lz = box[0], box[1], box[2]
    nx, ny, nz = image_shells(cutoff, box, expand)
    image_charges = []
    n_candidates = 0

    rq = np.array([[x, y, z] for _, x, y, z in charges])
    rcharges = np.array([q for q, *_ in charges])

    for ix in range(-nx, nx + 1):
        for iy in range(-ny, ny + 1):
            for iz in range(-nz, nz + 1):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                shifted = rq + np.array([ix * lx, iy * ly, iz * lz])
                n_candidates += len(shifted)
                dists = distances.distance_array(
                    qm_pos, shifted, box=None
                ).min(axis=0)
                for idx in np.where(dists <= cutoff)[0]:
                    image_charges.append((
                        float(rcharges[idx]),
                        float(shifted[idx, 0]),
                        float(shifted[idx, 1]),
                        float(shifted[idx, 2]),
                    ))

    return image_charges, (nx, ny, nz), n_candidates
