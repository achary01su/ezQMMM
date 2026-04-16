"""Tests for ezqmmm.config — parsing and validation."""

import os
import pytest
import yaml

from ezqmmm.config import parse_axes, parse_pdb_stride, validate_config, create_example_config


class TestParseAxes:
    @pytest.mark.parametrize("input,expected", [
        ([], (False, False, False)),
        (None, (False, False, False)),
        (['x', 'y'], (True, True, False)),
        ('x,y,z', (True, True, True)),
        (['a', 'c'], (True, False, True)),
        ('z', (False, False, True)),
        (['X', 'Y'], (True, True, False)),
        (['b'], (False, True, False)),
    ])
    def test_parse(self, input, expected):
        assert parse_axes(input) == expected


class TestParsePdbStride:
    @pytest.mark.parametrize("input,expected", [
        (None, None), ('all', 1), ('half', 2), ('tenth', 10),
        (5, 5), ('20', 20), (1, 1),
    ])
    def test_parse(self, input, expected):
        assert parse_pdb_stride(input) == expected

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            parse_pdb_stride('bogus')


class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config({
            'qm_selection': 'resid 1', 'program': 'orca',
            'stride': 10, 'first_frame': 0, 'last_frame': 50,
        }, 100)

    def test_missing_qm_selection(self):
        with pytest.raises(ValueError, match="qm_selection"):
            validate_config({'program': 'orca'}, 100)

    def test_missing_program(self):
        with pytest.raises(ValueError, match="program"):
            validate_config({'qm_selection': 'resid 1'}, 100)

    @pytest.mark.parametrize("program", ['gaussian', 'turbomole', ''])
    def test_invalid_program(self, program):
        with pytest.raises(ValueError, match="not recognised"):
            validate_config({'qm_selection': 'r', 'program': program}, 100)

    @pytest.mark.parametrize("scheme", ['FAKE', 'rcd2', ''])
    def test_invalid_scheme(self, scheme):
        with pytest.raises(ValueError):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'boundary_scheme': scheme}, 100)

    @pytest.mark.parametrize("stride", [0, -1, -100])
    def test_invalid_stride(self, stride):
        with pytest.raises(ValueError, match="stride"):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'stride': stride}, 100)

    def test_first_frame_negative(self):
        with pytest.raises(ValueError, match="first_frame"):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'first_frame': -1}, 100)

    def test_first_after_last(self):
        with pytest.raises(ValueError):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'first_frame': 50, 'last_frame': 10}, 100)

    @pytest.mark.parametrize("sw,cut", [(40, 40), (45, 40), (100, 50)])
    def test_switchdist_ge_cutoff(self, sw, cut):
        with pytest.raises(ValueError, match="must be less than"):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'mm_switchdist': sw, 'mm_cutoff': cut}, 100)

    @pytest.mark.parametrize("nf", [0.0, -0.1, 1.5, -1.0])
    def test_invalid_neutral_frac(self, nf):
        with pytest.raises(ValueError):
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'neutralize_mm_charge': True,
                             'neutralization_shell_fraction': nf}, 100)

    def test_neutral_frac_valid_at_boundaries(self):
        """0.01 and 1.0 should both pass."""
        for nf in [0.01, 0.5, 1.0]:
            validate_config({'qm_selection': 'r', 'program': 'orca',
                             'neutralize_mm_charge': True,
                             'neutralization_shell_fraction': nf}, 100)


class TestExampleConfig:
    def test_round_trip(self, tmp_path):
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            create_example_config()
            with open('config_example.yaml') as f:
                cfg = yaml.safe_load(f)
            assert cfg['psf_file'] == 'system.psf'
            assert cfg['program'] == 'qchem'
            assert cfg['boundary_scheme'] == 'RCD'
            assert cfg['neutralize_mm_charge'] is True
            assert cfg['neutralization_shell_fraction'] == 0.1
            assert cfg['mm_cutoff'] == 40.0
            assert cfg['charge'] == 0
            assert cfg['multiplicity'] == 1
        finally:
            os.chdir(orig)
