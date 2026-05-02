# ezQMMM 2.0 
Generates QM/MM single-point calculation input files from CHARMM/NAMD MD
trajectories. Supports **Q-Chem** and **ORCA**. 
**Psi4** support will come soon when adequatly tested.

---

## Installation

```bash
# Clone and install as a package (recommended)
git clone https://github.com/achary01su/ezQMMM.git
cd ezQMMM
pip install -e .
```

This installs the `ezqmmm` command and all dependencies (MDAnalysis, NumPy,
PyYAML).  The `-e` flag means editable so that code changes could be applied
without reinstalling.

Alternatively, run the standalone script without installing it as a package.
In that case, please make sure that the dependencies are installed already in 
the system. If they are not installed, one can use `pip` to install them as follows.

```bash
pip install MDAnalysis numpy pyyaml
python ezQMMM2.py config.yaml
```

---

## Quick Start

```bash
# Generate an example config
ezqmmm --example

# Rename the config file to config.yaml, edit the file, and then run
ezqmmm config.yaml
```

Or using the standalone script:
(note that the standalone code will not be updated beyond 2.0.1 version) 

```bash
python ezQMMM2.py --example
# Rename the config file to config.yaml, edit the file, and then run
python ezQMMM2.py config.yaml
```
---

## Configuration Reference

```yaml
# ── Trajectory inputs ────────────────────────────────────────────────────────
psf_file: system.psf          # CHARMM PSF topology file
dcd_file: trajectory.dcd      # DCD trajectory file

# ── QM region ────────────────────────────────────────────────────────────────
qm_selection: "resid 100"     # MDAnalysis atom selection string
charge: 0                     # Total QM charge
multiplicity: 1               # Spin multiplicity

# ── QM method ────────────────────────────────────────────────────────────────
method: B3LYP                 # Electronic structure method
basis: 6-31G*                 # Basis set (use 'gen' for custom, see *_blocks)
program: qchem                # Target program: orca | qchem 

# ── MM environment ───────────────────────────────────────────────────────────
mm_cutoff: 40.0               # Cutoff radius (Angstrom) for MM point charges
# mm_switchdist: 35.0         # Switching zone start (Angstrom); disabled by default
                              # It is recommended that user provide a mm_switchdist value
                              # lower than the mm_cutoff value

# ── Boundary scheme ──────────────────────────────────────────────────────────
boundary_scheme: RCD          # RCD | CS | Z1 | Z2 | Z3 | NONE

# ── MM charge neutralization ─────────────────────────────────────────────────
neutralize_mm_charge: true    # Adjust outermost charges to enforce target_mm_charge (default: true)
target_mm_charge: 0.0         # Target net MM charge in e (default: 0.0)
neutralization_shell_fraction: 0.1  # Fraction of outermost charges to adjust (default: 0.1)

# ── Frame selection ──────────────────────────────────────────────────────────
first_frame: 0                # First frame index (0-based)
last_frame: 100               # Last frame index (-1 = last frame)
stride: 10                    # Step between frames

# ── Output ───────────────────────────────────────────────────────────────────
output_dir: ./qmmm_calcs      # Output directory (created if absent)
output_prefix: qmmm           # Prefix for all output files

# ── Periodic images (supercell) ──────────────────────────────────────────────
supercell_axes: []            # Axes to tile: any combo of x/y/z or a/b/c
                              # e.g. [x, y] for membrane (z = normal)
                              #      [x, y, z] for fully periodic system
                              # Leave empty to disable

# ── Structure output ─────────────────────────────────────────────────────────
pdb_stride: null              # Write PDB every N frames (PSF written once)
                              # Options: all | half | tenth | integer | null

# ── Program-specific keywords ────────────────────────────────────────────────
qchem_keywords: |
  scf_convergence    8
  mem_total          8000

qchem_blocks: |
  $basis
  C 0
  cc-pVDZ
  ****
  $end
```

---

## Output Files

For each frame the following are generated under `output_dir`:

| File | Description |
|---|---|
| `<prefix>_frame<N>_orca.inp` | ORCA input file |
| `<prefix>_frame<N>_qchem.in` | Q-Chem input file |
| `<prefix>_frame<N>_charges.pc` | ORCA only — external point charge file |
| `<prefix>_struct.psf` | Single copy of original PSF — shared topology for all PDB frames |
| `<prefix>_frame<N>_struct.pdb` | Full-system PDB for frame N with beta-column labels |
| `<prefix>_boundary.log` | Per-frame boundary charge modification detail |
| `<prefix>_switching.log` | Per-frame switching-zone charge detail |
| `<prefix>_run.log` | Full run log: settings, per-frame QM/MM charges, summary |

### Beta column values in PDB

| Value | Region |
|---|---|
| 9 | QM atom |
| 8 | MM point charge (within cutoff) |
| 0 | Outside cutoff |

In VMD: load `<prefix>_struct.psf` first as the topology, then load each
`<prefix>_frame<N>_struct.pdb` as a coordinate frame. 
Visulize different beta values in VMD to validate that QM/MM partitions and set up

---

## MDAnalysis Selection Syntax

The `qm_selection` field uses MDAnalysis selection language, which is similar
to but **not identical** to VMD's Tcl-based atom selection.

### Common Selection Examples

```yaml
# Single residue by number
qm_selection: "resid 100"

# Multiple residues
qm_selection: "resid 100 101 102"

# Residue range (inclusive)
qm_selection: "resid 100:105"

# Specific segment and residue (CHARMM segid)
qm_selection: "segid PROA and resid 100"

# Residue by name (e.g. all HEM residues)
qm_selection: "resname HEM"

# Specific atoms within a residue
qm_selection: "resid 100 and name FE"

# Exclude backbone (N, CA, C, O) — keeps sidechain heavy atoms only
# Note: MDAnalysis backbone includes Cα, so 'not backbone' excludes it.
# To keep sidechain AND Cα use: "resid 100 and not name N C O"
qm_selection: "resid 100 and not backbone"

# Cofactor plus two specific coordinating residues (sidechain only)
qm_selection: "resname FAD or (resid 45 112 and not backbone)"

# Ligand plus all atoms within 3.5 Å of the ligand
# Note: 'around' excludes the reference group itself, so 'resname LIG'
# must be added back explicitly to include the ligand atoms too.
qm_selection: "resname LIG or (around 3.5 resname LIG)"

# Single metal ion plus its coordination shell
# Note: 'sphzone' uses the COG of the reference and INCLUDES reference
# atoms. For multiple metals use 'around' instead.
qm_selection: "sphzone 2.5 resname ZN"
```

### Key Differences from VMD

| Feature | MDAnalysis | VMD |
|---|---|---|
| Residue number | `resid 100` | `resid 100` |
| Residue range | `resid 100:105` | `resid 100 to 105` |
| Segment ID | `segid PROA` | `segname PROA` |
| Residue name | `resname HEM` | `resname HEM` |
| Atom name | `name CA` | `name CA` |
| Distance selection | `around 3.5 resname LIG` | `within 3.5 of resname LIG` |
| Sphere zone (includes reference) | `sphzone 2.5 resname ZN` | `within 2.5 of resname ZN` |
| Backbone atoms | `backbone` (N, CA, C, O only) | `backbone` (includes OT\* termini) |
| By index (0-based) | `index 0 1 2` | `index 0 1 2` |
| By serial (1-based) | *(not available)* | `serial 1 2 3` |
| Chain | `segid` (CHARMM PSF) | `segname` (PSF) / `chain` (PDB) |

**Key pitfalls:**
- **`within` → `around`**: `around` excludes the reference group; VMD's `within X of` includes it. Use `resname LIG or (around X resname LIG)` to replicate VMD behaviour.
- **`sphzone` includes reference atoms**: unlike `around`. For a single metal `sphzone 2.5 resname ZN` already contains the metal — no `or resname ZN` needed.
- **`backbone` difference**: MDAnalysis backbone is strictly {N, CA, C, O}; VMD additionally includes OT\* terminal oxygens. `not backbone` in MDAnalysis excludes Cα — use `not name N C O` to keep it.
- **Residue ranges**: VMD uses `resid 100 to 105`; MDAnalysis uses `resid 100:105`.

### Verifying Your Selection

```python
import MDAnalysis as mda
u = mda.Universe("system.psf", "trajectory.dcd")
sel = u.select_atoms("resid 100 and not backbone")
print(f"Selected {len(sel)} atoms")
for atom in sel:
    print(f"  {atom.segid} {atom.resid} {atom.resname} {atom.name} "
          f"mass={atom.mass:.3f} charge={atom.charge:.4f}")
```

Cross-check the atom count and identity against VMD before running ezQMMM 2.0.

---

## Method Details

### MM Point Charge Selection

MM atoms are selected by whole-residue inclusion: if any atom of a residue
falls within `mm_cutoff` of any QM atom, the entire residue is included.
Distance is measured as the minimum distance to any QM atom (not center of
mass), using the minimum image convention (PBC-aware).

After selection, boundary scheme corrections, image tiling, and switching
are applied.  If `neutralize_mm_charge: true` (default), any residual net
charge in the final MM point charge set is distributed evenly across the
outermost charges (controlled by `neutralization_shell_fraction`, default
10%) to enforce `target_mm_charge`. Charges near the QM region are left
untouched. This is the last step before writing to the input file.

### MM Charge Neutralization

As MD frames evolve, ions and charged residues drift in and out of the
cutoff sphere, causing the total MM charge to vary between frames. A single
monovalent ion entering or leaving the cutoff can shift the electrostatic
potential at QM by 0.2–0.5 eV — large enough to dominate the standard
deviation of vertical ionization energies or redox potentials sampled across
frames.

NAMD addresses this via the `qmPointChargeScheme` keyword (NAMD 2.14 User's
Guide), which offers `round` or `zero` options that adjust the most distant
point charges to enforce a target total charge. ezQMMM 2.0 follows the same
philosophy: only the outermost charges — those with the weakest electrostatic
influence on the QM wavefunction — absorb the correction. Charges near the
QM region are left completely untouched.

The outermost fraction is controlled by `neutralization_shell_fraction`
(default `0.1` = outermost 10% of charges by distance from QM). The residual
charge is distributed evenly across only those charges. For a typical system
with ~30,000 MM charges and a residual of ±1 e, the correction per atom in
the outer shell is ~0.0003 e — negligible at large distance but collectively
eliminating the frame-to-frame integer charge jumps.

Set `neutralize_mm_charge: false` to disable for benchmarking or to reproduce
raw PSF charge behaviour. The correction is applied after the boundary scheme,
image tiling, and switching function, operating on the final charge set
written to the input file.

### Switching Function

A NAMD-style quintic switching function smoothly attenuates charges between
`mm_switchdist` and `mm_cutoff`. Switching is **disabled by default** — set
`mm_switchdist` explicitly to enable it.

$$S(r) = 1 - 10t^3 + 15t^4 - 6t^5, \qquad t = \frac{r - r_\text{sw}}{r_\text{cut} - r_\text{sw}}$$

$$S(r) = 1 \quad \text{for } r \leq r_\text{sw}, \qquad S(r) = 0 \quad \text{for } r \geq r_\text{cut}$$

Distance $r$ is the minimum distance from the charge to any QM atom (not
center of mass). All switching events are recorded in
`<prefix>_switching.log`.

### Boundary Schemes

A link hydrogen atom is placed along each QM-MM cut bond at 1.09 Å from the
QM atom. MM1 is directly bonded to QM, MM2 is bonded to MM1, MM3 is bonded
to MM2 — consecutive atoms along the covalent bond chain.

| Scheme | Treatment | Purpose |
|---|---|---|
| `RCD` | MM1 removed; charge redistributed to midpoint virtual charges; MM2 adjusted | Preserves bond dipole and charge neutrality — recommended default |
| `CS` | MM1 removed; charge split into a dipole pair around MM2 | Preserves charges. Virtual point charges are placed very close to the MM2 atom position |
| `Z1` | MM1 charge zeroed | Simplest elimination; breaks charge neutrality |
| `Z2` | MM1 and MM2 charges zeroed | Eliminates two consecutive MM atoms |
| `Z3` | MM1, MM2, and MM3 charges zeroed | Eliminates three consecutive MM atoms|
| `NONE` | No modification | Use only if no QM/MM covalent bonds exist |

**RCD factor of 2**: the virtual charge at the MM1–MM2 midpoint is
$2q_\text{MM1}/n$. This preserves the bond dipole: $q \cdot d = 2q \cdot d/2$.
The MM2 charge is adjusted by $-q_\text{MM1}/n$ to maintain charge neutrality.
All modifications are recorded in `<prefix>_boundary.log`.

**Z2 and Z3**: Sometime these methods can yield a net zero charge (escpecially for CHARMM force fields) for proteins based on the selection of QM-MM boundary.

### Coordinate Remapping

`distance_array(..., box=dimensions)` selects MM atoms via PBC but keeps raw
trajectory coordinates. Before writing, all MM charge positions are remapped
to their minimum image equivalent relative to the QM centroid using the
floor-based minimum image convention:

$$x_\text{remapped} = x - \left\lfloor\frac{x - x_\text{QM}}{L_x} + 0.5\right\rfloor L_x$$

and equivalently for $y$ and $z$. This approach becomes critically important when the QM region 
is at the edge of protein with a smaller water padding than the half the `mm_cutoff` value.

### Supercell / Periodic Images

When `supercell_axes` is set, periodic images of the primary MM charges are
generated along the specified axes. The number of shells is derived
automatically as $n_x = \lceil r_\text{cut} / L_x \rceil$ per axis. Only
images within $r_\text{cut}$ of any QM atom are retained. Images are added
after the boundary scheme and before switching.

| System | Recommended setting |
|---|---|
| Large solvated protein | `supercell_axes: []` — images never within cutoff |
| Membrane protein (normal along $z$) | `supercell_axes: [x, y]` — periodic in bilayer plane only |
| Small solvated active site model | `supercell_axes: [x, y, z]` — full 3D tiling |
| Elongated system (e.g. DNA fiber, fibril) | `supercell_axes: [z]` — periodic along fiber axis only |

For large boxes (`img=0`): after remapping every primary charge is within
$r_\text{cut}$ of QM. The nearest possible image is at $L - r_\text{cut}$,
which for typical biomolecular boxes ($L \sim 140\ \mathring{\text{A}}$,
$r_\text{cut} = 60\ \mathring{\text{A}}$) gives $\sim 80\ \mathring{\text{A}}$ — well
outside the cutoff. Images only contribute when $L \lesssim 2\,r_\text{cut}$.

---

## Caveats and Limitations

### QM Region
- **Must not straddle a periodic boundary.** The remapping formula uses the mean QM position as reference. Wrap the QM region before use.
- **Element assignment is mass-based.** PSF files store CHARMM atom types, not element symbols. The lookup table covers H, D, C, N, O, Na, Mg, P, S, Cl, K, Ca, Fe, Cu, Zn. Unusual elements are assigned `X`. Check the `QM=` atom count in the console output. If you are in need for an new element that are not included in the list, please let us know. We can append that element.
- **Hydrogen mass repartition (HMR) is not supported.** If the MD simulation used HMR, mass-to-element conversion will fail because hydrogen masses are shifted above their standard values.

### MM Environment
- **Whole-residue inclusion** means a single atom near the cutoff edge pulls in its entire residue. For large residues (lipids, polymers) this can significantly increase the MM region size.
- **Charges are PSF partial charges.** No polarization is applied.
- **MM charge neutralization is on by default.** Only the outermost charges
  (default: outermost 10%, controlled by `neutralization_shell_fraction`)
  absorb the correction — charges near the QM region are untouched, since
  they have the strongest electrostatic influence on the wavefunction and
  should not be perturbed. A similar approach is used by
  NAMD through the `qmPointChargeScheme` keyword (NAMD 2.14 User's Guide), which
  also adjusts only the most distant point charges. Set
  `neutralize_mm_charge: false` for benchmarking purposes.
- **No MM geometry optimization** is performed. Input files are for single-point energy calculations only.

### Boundary
- **Bonds must be in the PSF.** `_find_boundary_bonds` relies on `atom.bonds` from MDAnalysis. If the PSF lacks explicit bonds, no link atoms or charge corrections will be applied.
- **Only C–C type cuts are well-tested.** Cutting across polar bonds (C–N, C–O) should be benchmarked.


### Supercell
- **Orthorhombic boxes only.** Triclinic boxes require lattice vector arithmetic and are not supported.
- **PBC is always active for primary MM selection.** The minimum image convention is used regardless of `supercell_axes`. Image copies supplement the primary selection and do not replace it.

### PDB / PSF Output
- **PDB remapping is per-residue.** MM atom positions in the PDB are shifted to minimum image near the QM centroid using a two-step process: 
(i) all atoms in a residue are unwrapped to minimum image relative to the residue's first atom (reassembling residues that straddle the periodic boundary), then 
(ii) the whole residue is shifted to minimum image relative to the QM centroid. This preserves internal bond lengths. Long bonds may still appear at the boundary between the MM cutoff region (beta=8) and the rest of the system (beta=0), since atoms outside the cutoff are not remapped. This does not affect the QM/MM input files.


- **The PSF is written once.** Identical for every frame, so ezQMMM 2.0 copies it once as `<prefix>_struct.psf`. Each frame produces only a PDB.
- **`pdb_stride` significantly slows down the run.** Writing a full-system PDB for large systems can take several seconds per frame. Use `pdb_stride: tenth` or a large integer for production; reserve `pdb_stride: all` for validation.
- **tempfactors are not in PSF files.** MDAnalysis initializes them to zero. The startup note is informational only.

### Program-Specific
- **ORCA**: point charges go in a separate `_charges.pc` file. Keep `.inp` and `.pc` in the same directory when running ORCA.
- **Q-Chem**: charges appear inline in `$external_charges` in `x y z q` order (Angstrom).

---

## References

- NAMD QM/MM: https://www.ks.uiuc.edu/Research/qmmm/
- Melo et al., *Nature Methods* **15**, 351–354 (2018)
- RCD and RC boundary schemes: Lin & Truhlar, *J. Phys. Chem. A* **109**, 3991–4004 (2005). DOI: 10.1021/jp0446332
- MDAnalysis: Michaud-Agrawal et al., *J. Comput. Chem.* **32**, 2319 (2011)
