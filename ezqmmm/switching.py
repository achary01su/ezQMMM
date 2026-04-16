"""
NAMD-style quintic switching function.

S(r) = 1 - 10t³ + 15t⁴ - 6t⁵,   t = (r - sw) / (cut - sw)
S(r) = 1  for r ≤ sw
S(r) = 0  for r ≥ cut

Distance r = minimum distance from charge to any QM atom.
"""

from typing import Optional

import numpy as np
from MDAnalysis.analysis import distances

from ezqmmm.models import SwitchRecord


def apply_switching(charges: list, qm_pos: np.ndarray,
                    sw: float, cut: float, box,
                    frame: int,
                    n_primary: Optional[int] = None
                    ) -> tuple[list, list[SwitchRecord]]:
    """
    Vectorised quintic switching on a charge list.

    Parameters
    ----------
    charges   : list of (q, x, y, z)
    qm_pos    : (M, 3) QM atom positions
    sw        : switching start distance
    cut       : cutoff distance
    box       : MDAnalysis box array or None (None when image charges present)
    frame     : frame index (for SwitchRecord)
    n_primary : index boundary between primary and image charges

    Returns
    -------
    scaled : list of (q, x, y, z) with switching applied
    recs   : list of SwitchRecord for charges with scale < 1
    """
    if not charges:
        return [], []

    positions = np.array([[x, y, z] for _, x, y, z in charges])
    qs = np.array([q for q, *_ in charges])

    all_dists = distances.distance_array(
        qm_pos, positions, box=box
    ).min(axis=0)

    sw_range = cut - sw
    t = np.clip((all_dists - sw) / sw_range, 0.0, 1.0)
    scales = np.where(
        all_dists <= sw, 1.0,
        np.where(all_dists >= cut, 0.0,
                 1.0 - 10 * t**3 + 15 * t**4 - 6 * t**5)
    )

    scaled_qs = qs * scales
    scaled = [
        (float(scaled_qs[i]), float(positions[i, 0]),
         float(positions[i, 1]), float(positions[i, 2]))
        for i in range(len(charges))
    ]

    recs = []
    for i in np.where(scales < 1.0)[0]:
        is_img = (n_primary is not None) and (int(i) >= n_primary)
        recs.append(SwitchRecord(
            frame=frame,
            psf_charge=float(qs[i]),
            scaled_charge=float(scaled_qs[i]),
            scale=float(scales[i]),
            dist=float(all_dists[i]),
            position=positions[i],
            is_image=is_img,
        ))
    return scaled, recs
