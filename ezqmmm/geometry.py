"""
Minimum-image remapping and supercell image tiling.

All functions use the floor-based rounding convention
(round-half-up) rather than np.round (banker's rounding)
for consistency with standard MD codes.
"""


import numpy as np
from MDAnalysis.analysis import distances

# Minimum image remapping
# ------------------------------------------------------------------

def remap_position(pos: np.ndarray, qm_center: np.ndarray,
                   box: np.ndarray) -> np.ndarray:
    """
    Remap a single (3,) position to minimum image relative to *qm_center*.
    """
    lx, ly, lz = box[0], box[1], box[2]
    return np.array([
        pos[0] - np.floor((pos[0] - qm_center[0]) / lx + 0.5) * lx,
        pos[1] - np.floor((pos[1] - qm_center[1]) / ly + 0.5) * ly,
        pos[2] - np.floor((pos[2] - qm_center[2]) / lz + 0.5) * lz,
    ])


def remap_positions_array(pos: np.ndarray, qm_center: np.ndarray,
                          box: np.ndarray) -> np.ndarray:
    """
    Vectorised remap of an (N, 3) position array to minimum image.
    """
    lx, ly, lz = box[0], box[1], box[2]
    out = pos.copy()
    out[:, 0] -= np.floor((pos[:, 0] - qm_center[0]) / lx + 0.5) * lx
    out[:, 1] -= np.floor((pos[:, 1] - qm_center[1]) / ly + 0.5) * ly
    out[:, 2] -= np.floor((pos[:, 2] - qm_center[2]) / lz + 0.5) * lz
    return out


def remap_positions_by_residue(mm_ag, orig_pos: np.ndarray,
                                qm_center: np.ndarray,
                                box: np.ndarray) -> np.ndarray:
    """
    Remap MM atom positions to minimum image, preserving residue
    internal geometry.  Two-step process per residue:

    1. **Unwrap**: remap every atom to minimum image relative to the
       residue's reference atom (first atom).  This reassembles
       residues that straddle the periodic boundary.
    2. **Shift**: remap the reference atom to minimum image relative
       to the QM centroid, and apply the same displacement to every
       atom in the residue.

    Both steps use standard floor-based minimum image arithmetic,
    but step 1 keeps the residue whole and step 2 places it near QM.

    Parameters
    ----------
    mm_ag      : MDAnalysis AtomGroup of MM atoms within cutoff
    orig_pos   : (N, 3) original positions of mm_ag atoms
    qm_center  : (3,) QM centroid
    box        : (6,) box dimensions
    """
    lx, ly, lz = box[0], box[1], box[2]
    new_pos = orig_pos.copy()

    # Map universe atom index → position array row
    idx_to_row = {idx: i for i, idx in enumerate(mm_ag.indices)}

    for res in mm_ag.residues:
        rows = [idx_to_row[a.index] for a in res.atoms
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

        # Step 2: shift the whole residue to minimum image of QM center
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
