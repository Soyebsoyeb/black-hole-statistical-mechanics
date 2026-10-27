import math

import pytest

from bhstatmech import constants as c
from bhstatmech import units as u
from bhstatmech.exceptions import DimensionError, PhysicalParameterError
from bhstatmech.thermodynamics import laws


def test_planck_quantities_codata_values():
    assert c.PLANCK_LENGTH == pytest.approx(1.616255e-35, rel=1e-5)
    assert c.PLANCK_MASS == pytest.approx(2.176434e-8, rel=1e-5)
    assert c.PLANCK_TIME == pytest.approx(5.391247e-44, rel=1e-5)


def test_planck_relations():
    assert c.PLANCK_LENGTH / c.C == pytest.approx(c.PLANCK_TIME, rel=1e-12)
    assert c.PLANCK_MASS * c.PLANCK_LENGTH * c.C == pytest.approx(c.HBAR, rel=1e-12)


def test_stefan_boltzmann_value():
    assert c.SIGMA_SB == pytest.approx(5.670374e-8, rel=1e-6)


def test_round_trip_conversions():
    assert u.kg_to_solar_masses(u.solar_masses_to_kg(3.7)) == pytest.approx(3.7)
    assert u.temperature_natural_to_si(u.temperature_si_to_natural(1.23)) == pytest.approx(1.23)
    assert u.entropy_dimensionless_to_si(u.entropy_si_to_dimensionless(4.5)) == pytest.approx(4.5)
    assert u.energy_to_omega(u.omega_to_energy(7.0)) == pytest.approx(7.0)
    assert u.mass_planck_to_kg(u.mass_kg_to_planck(5.0)) == pytest.approx(5.0)


def test_gravitational_radius_conversion():
    m = c.M_SUN
    assert u.gravitational_radii_to_meters(2.0, m) == pytest.approx(2.0 * c.G * m / c.C**2)
    assert u.meters_to_gravitational_radii(u.gravitational_radius(m), m) == pytest.approx(1.0)


def test_invalid_mass_raises_informative_error():
    with pytest.raises(PhysicalParameterError, match="positive"):
        u.gravitational_radius(-1.0)


def test_dimensional_consistency_of_black_hole_formulas():
    m = u.Quantity(c.M_SUN, u.KILOGRAM)
    T = laws.hawking_temperature_quantity(m)
    S = laws.entropy_quantity(m)
    A = laws.horizon_area_quantity(m)
    u.assert_dimension(T, u.KELVIN, "T_H")
    u.assert_dimension(S, u.JOULE_PER_KELVIN, "S_BH")
    u.assert_dimension(A, u.SQUARE_METER, "A")
    # numerical values agree with the plain-float implementations
    assert T.value == pytest.approx(float(laws.hawking_temperature(c.M_SUN)), rel=1e-12)
    assert S.value == pytest.approx(float(laws.bekenstein_hawking_entropy(c.M_SUN)), rel=1e-12)


def test_dimension_mismatch_raises():
    with pytest.raises(DimensionError):
        u.Quantity(1.0, u.METER) + u.Quantity(1.0, u.SECOND)
    with pytest.raises(DimensionError):
        u.assert_dimension(u.Quantity(1.0, u.METER), u.SECOND)


def test_planck_length_has_dimension_of_length():
    ell = (u.HBAR_Q * u.G_Q / u.C_Q**3) ** 0.5
    u.assert_dimension(ell, u.METER, "l_P")
    assert ell.value == pytest.approx(c.PLANCK_LENGTH, rel=1e-12)
