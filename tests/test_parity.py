"""
Parity test — monolith vs package.

Verifies that both entry points produce identical QM/MM input files.
This is a migration safety net, not the definition of correctness.
Remove once the monolith is retired.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ezQMMM2 as mono
from ezqmmm.writers import write_orca, write_psi4, write_qchem


@pytest.fixture
def sample_data():
    coords = [('C', 1.0, 2.0, 3.0), ('H', 1.5, 2.5, 3.5)]
    charges = [(0.5, 10.0, 20.0, 30.0), (-0.3, 11.0, 21.0, 31.0)]
    return coords, charges


def _mono_writer(name):
    """Bind a monolith writer method to a throwaway object."""
    class Fake:
        pass
    gen = Fake()
    method = getattr(mono.QMMMGenerator, f'_write_{name}')
    return method.__get__(gen)


class TestWriterParity:
    @pytest.mark.parametrize("program,ext", [
        ('orca', '_orca.inp'),
        ('qchem', '_qchem.in'),
        ('psi4', '_psi4.dat'),
    ])
    def test_identical_output(self, tmp_path, sample_data, program, ext):
        coords, charges = sample_data
        pkg_fn = {'orca': write_orca, 'qchem': write_qchem, 'psi4': write_psi4}[program]
        mono_fn = _mono_writer(program)

        # Write to separate dirs so filenames match (ORCA embeds .pc name)
        mdir = tmp_path / 'mono'
        pdir = tmp_path / 'pkg'
        mdir.mkdir()
        pdir.mkdir()

        fm = str(mdir / f'test{ext}')
        fp = str(pdir / f'test{ext}')

        mono_fn(fm, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        pkg_fn(fp, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')

        assert Path(fm).read_text() == Path(fp).read_text(), \
            f"{program} output differs between monolith and package"

        # Check .pc file for ORCA
        if program == 'orca':
            pc_m = fm.replace('.inp', '_charges.pc')
            pc_p = fp.replace('.inp', '_charges.pc')
            assert Path(pc_m).read_text() == Path(pc_p).read_text(), \
                "ORCA .pc file differs"
