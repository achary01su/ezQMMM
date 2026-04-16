"""Tests for ezqmmm.models — ChargeMod and SwitchRecord."""

import numpy as np

from ezqmmm.models import ChargeMod, SwitchRecord


class TestChargeMod:
    def test_delta_computed(self):
        mod = ChargeMod(frame=0, mod_type='removed', reason='test',
                         psf_charge=0.5, applied_charge=0.0,
                         position=np.array([1.0, 2.0, 3.0]))
        assert abs(mod.delta - (-0.5)) < 1e-10

    def test_delta_positive(self):
        mod = ChargeMod(frame=0, mod_type='modified', reason='test',
                         psf_charge=0.1, applied_charge=0.4,
                         position=np.array([0, 0, 0.0]))
        assert abs(mod.delta - 0.3) < 1e-10

    def test_optional_fields_default(self):
        mod = ChargeMod(frame=1, mod_type='virtual', reason='midpoint',
                         psf_charge=0.0, applied_charge=0.3,
                         position=np.array([0, 0, 0.0]))
        assert mod.atom_index is None
        assert mod.segid == ''
        assert mod.resid == 0
        assert mod.resname == ''
        assert mod.name == ''

    def test_all_fields_set(self):
        mod = ChargeMod(frame=5, mod_type='removed', reason='Z1',
                         psf_charge=-0.3, applied_charge=0.0,
                         position=np.array([1, 2, 3.0]),
                         atom_index=42, segid='PROA', resid=100,
                         resname='ALA', name='CA')
        assert mod.atom_index == 42
        assert mod.segid == 'PROA'
        assert mod.resid == 100
        assert mod.resname == 'ALA'
        assert mod.name == 'CA'
        assert mod.frame == 5

    def test_position_is_array(self):
        mod = ChargeMod(frame=0, mod_type='virtual', reason='t',
                         psf_charge=0, applied_charge=0,
                         position=[1.0, 2.0, 3.0])
        assert isinstance(mod.position, np.ndarray)
        assert mod.position.shape == (3,)


class TestSwitchRecord:
    def test_creation(self):
        rec = SwitchRecord(frame=5, psf_charge=-0.5, scaled_charge=-0.25,
                            scale=0.5, dist=37.5,
                            position=np.array([10, 20, 30.0]))
        assert rec.is_image is False
        assert abs(rec.scale - 0.5) < 1e-10
        assert rec.frame == 5

    def test_image_flag(self):
        rec = SwitchRecord(frame=0, psf_charge=0.3, scaled_charge=0.15,
                            scale=0.5, dist=12.0,
                            position=np.array([0, 0, 0.0]), is_image=True)
        assert rec.is_image is True

    def test_zero_scale(self):
        rec = SwitchRecord(frame=0, psf_charge=-0.82, scaled_charge=0.0,
                            scale=0.0, dist=50.0,
                            position=np.array([0, 0, 0.0]))
        assert abs(rec.scaled_charge) < 1e-10
        assert abs(rec.scale) < 1e-10
