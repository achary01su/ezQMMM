"""Tests for ezqmmm.elements — mass-to-element lookup."""

import pytest

from ezqmmm.elements import MASS_TO_ELEMENT, get_element_from_mass


class TestGetElement:
    @pytest.mark.parametrize("mass,expected", [
        (1.008, 'H'), (2.014, 'D'), (12.011, 'C'), (14.007, 'N'),
        (15.999, 'O'), (22.99, 'Na'), (24.305, 'Mg'), (30.974, 'P'),
        (32.06, 'S'), (35.45, 'Cl'), (39.1, 'K'), (40.08, 'Ca'),
        (55.845, 'Fe'), (63.546, 'Cu'), (65.38, 'Zn'),
    ])
    def test_known_elements(self, mass, expected):
        assert get_element_from_mass(mass) == expected

    @pytest.mark.parametrize("mass", [0.5, 5.0, 99.0, 200.0, -1.0])
    def test_unknown_returns_x(self, mass):
        assert get_element_from_mass(mass) == 'X'

    def test_boundary_low_hydrogen(self):
        """Mass exactly at lower bound of H range should match."""
        assert get_element_from_mass(0.9) == 'H'

    def test_boundary_high_hydrogen(self):
        """Mass exactly at upper bound of H range should match."""
        assert get_element_from_mass(1.2) == 'H'

    def test_hmr_mass_unrecognised(self):
        """HMR shifts H mass to ~3-4. Should return X, not match anything."""
        assert get_element_from_mass(3.024) == 'X'

    def test_table_has_15_entries(self):
        """Sanity check: table covers 15 elements."""
        assert len(MASS_TO_ELEMENT) == 15
