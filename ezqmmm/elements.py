"""
Mass-to-element lookup for PSF-based topologies.

PSF files store CHARMM atom types, not element symbols.  This module
infers elements from atomic masses.  The table covers the elements most
common in biomolecular QM/MM: H, D, C, N, O, Na, Mg, P, S, Cl, K, Ca,
Fe, Cu, Zn.  Any mass not matched returns 'X'.

Note: hydrogen mass repartition (HMR) will break this lookup because
shifted hydrogen masses fall outside the standard range.
"""

MASS_TO_ELEMENT = {
    (0.9,  1.2):  'H',  (1.9,  2.2):  'D',  (11.9, 12.2): 'C',
    (13.9, 14.2): 'N',  (15.9, 16.2): 'O',  (22.9, 23.2): 'Na',
    (24.0, 24.6): 'Mg', (30.8, 31.5): 'P',  (31.9, 32.2): 'S',
    (35.2, 35.8): 'Cl', (38.9, 39.5): 'K',  (39.9, 40.5): 'Ca',
    (55.6, 56.1): 'Fe', (63.2, 63.8): 'Cu', (65.1, 65.7): 'Zn',
}


def get_element_from_mass(mass: float) -> str:
    """Return element symbol for *mass*, or 'X' if unrecognised."""
    for (lo, hi), elem in MASS_TO_ELEMENT.items():
        if lo <= mass <= hi:
            return elem
    return 'X'
