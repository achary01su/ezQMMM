"""Tests for ezqmmm.writers — QM/MM input files and log output."""

import numpy as np
import pytest
from pathlib import Path

from ezqmmm.writers import (
    write_orca, write_qchem, write_psi4,
    write_boundary_log, write_switching_log,
)
from ezqmmm.models import ChargeMod, SwitchRecord


@pytest.fixture
def sample_data():
    coords = [('C', 1.0, 2.0, 3.0), ('H', 1.5, 2.5, 3.5)]
    charges = [(0.5, 10.0, 20.0, 30.0), (-0.3, 11.0, 21.0, 31.0)]
    return coords, charges


# ===================================================================
# ORCA
# ===================================================================

class TestORCA:
    def test_header(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.inp')
        write_orca(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert '! B3LYP 6-31G*' in content
        assert '* xyz 0 1' in content
        assert content.strip().endswith('*')

    def test_point_charge_file(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.inp')
        write_orca(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        pc = Path(f.replace('.inp', '_charges.pc'))
        assert pc.exists()
        lines = pc.read_text().strip().split('\n')
        assert lines[0] == '2'  # charge count
        assert len(lines) == 3  # count + 2 charges

    def test_no_charges_no_pc_reference(self, tmp_path):
        f = str(tmp_path / 'test.inp')
        write_orca(f, [('C', 0, 0, 0)], [], 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert 'pointcharges' not in content

    def test_custom_keywords(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.inp')
        write_orca(f, coords, charges, 'B3LYP', '6-31G*', 0, 1,
                   'TightSCF\nPAL4', '')
        content = Path(f).read_text()
        assert '! TightSCF' in content
        assert '! PAL4' in content

    def test_custom_blocks(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.inp')
        write_orca(f, coords, charges, 'B3LYP', '6-31G*', 0, 1,
                   '', '%scf MaxIter 500\nend')
        content = Path(f).read_text()
        assert '%scf MaxIter 500' in content

    def test_charge_format_qxyz(self, tmp_path, sample_data):
        """ORCA .pc format is q x y z."""
        coords, charges = sample_data
        f = str(tmp_path / 'test.inp')
        write_orca(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        pc_line = Path(f.replace('.inp', '_charges.pc')).read_text().strip().split('\n')[1]
        parts = pc_line.split()
        assert len(parts) == 4
        assert float(parts[0]) == pytest.approx(0.5)  # q first


# ===================================================================
# Q-Chem
# ===================================================================

class TestQChem:
    def test_structure(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.in')
        write_qchem(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert '$molecule' in content
        assert '0 1' in content
        assert 'qm_mm                true' in content
        assert '$external_charges' in content

    def test_no_charges_no_qmmm(self, tmp_path):
        f = str(tmp_path / 'test.in')
        write_qchem(f, [('C', 0, 0, 0)], [], 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert 'qm_mm' not in content
        assert 'external_charges' not in content

    def test_charge_format_xyzq(self, tmp_path, sample_data):
        """Q-Chem external_charges format is x y z q."""
        coords, charges = sample_data
        f = str(tmp_path / 'test.in')
        write_qchem(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        # Find the first charge line after $external_charges
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '$external_charges' in line:
                charge_line = lines[i + 1]
                parts = charge_line.split()
                assert len(parts) == 4
                assert float(parts[3]) == pytest.approx(0.5)  # q last
                break


# ===================================================================
# Psi4
# ===================================================================

class TestPsi4:
    def test_structure(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.dat')
        write_psi4(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert 'molecule qmmm' in content
        assert 'no_com' in content
        assert 'no_reorient' in content
        assert 'Chrgfield = QMMM()' in content
        assert "energy('B3LYP')" in content

    def test_bohr_conversion(self, tmp_path, sample_data):
        coords, charges = sample_data
        f = str(tmp_path / 'test.dat')
        write_psi4(f, coords, charges, 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        # charge at x=10.0 Å → 18.897260 Bohr
        assert '18.897260' in content

    def test_no_charges_no_chrgfield(self, tmp_path):
        f = str(tmp_path / 'test.dat')
        write_psi4(f, [('C', 0, 0, 0)], [], 'B3LYP', '6-31G*', 0, 1, '', '')
        content = Path(f).read_text()
        assert 'Chrgfield' not in content
        assert 'no_com' not in content


# ===================================================================
# Log writers
# ===================================================================

class TestBoundaryLog:
    def test_contains_all_types(self, tmp_path):
        mods = [
            ChargeMod(frame=0, mod_type='removed', reason='MM1 removed (RCD)',
                       psf_charge=-0.3, applied_charge=0.0,
                       position=np.array([1, 2, 3.0]),
                       atom_index=5, segid='SEG1', resid=2,
                       resname='MM1', name='C2'),
            ChargeMod(frame=0, mod_type='virtual', reason='Virtual RCD midpoint',
                       psf_charge=0.0, applied_charge=-0.6,
                       position=np.array([4, 5, 6.0])),
            ChargeMod(frame=0, mod_type='modified', reason='MM2 adjusted (RCD)',
                       psf_charge=0.15, applied_charge=0.0,
                       position=np.array([7, 8, 9.0]),
                       atom_index=6, segid='SEG1', resid=2,
                       resname='MM1', name='H3'),
        ]
        f = tmp_path / 'boundary.log'
        with open(f, 'w') as fh:
            write_boundary_log(fh, mods)
        content = f.read_text()
        assert 'removed' in content
        assert 'virtual' in content
        assert 'modified' in content
        assert 'Total:' in content
        assert 'removed=1' in content
        assert 'modified=1' in content
        assert 'virtual=1' in content


class TestSwitchingLog:
    def test_format(self, tmp_path):
        recs = [
            SwitchRecord(frame=0, psf_charge=-0.5, scaled_charge=-0.25,
                          scale=0.5, dist=37.5,
                          position=np.array([10, 20, 30.0])),
        ]
        f = tmp_path / 'switching.log'
        with open(f, 'w') as fh:
            write_switching_log(fh, recs, 35.0, 40.0, (False, False, False))
        content = f.read_text()
        assert 'Switching-Function' in content
        assert '37.500' in content
        assert '0.50000' in content
        assert 'Total switching-zone charge events: 1' in content

    def test_disabled_switching(self, tmp_path):
        f = tmp_path / 'switching.log'
        with open(f, 'w') as fh:
            write_switching_log(fh, [], None, 40.0, (False, False, False))
        content = f.read_text()
        assert 'disabled' in content
