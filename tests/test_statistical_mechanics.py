import numpy as np
import pytest

from bhstatmech.exceptions import PhysicalParameterError
from bhstatmech.statistical_mechanics import (
    FiniteSpectrum, bh_canonical_exponent, bh_exponent_second_derivative, bh_saddle_energy,
    entropy_with_log_correction, log10_omega_eff, microcanonical_heat_capacity_bh,
    two_level_heat_capacity_exact, two_level_system,
)
from bhstatmech.thermodynamics import laws


def test_canonical_heat_capacity_equals_schottky():
    N = 40
    sp = two_level_system(N, 1.0)
    for beta in (0.1, 0.7, 2.0):
        assert sp.heat_capacity(beta) == pytest.approx(two_level_heat_capacity_exact(beta, N), rel=1e-10)


def test_log_Z_matches_product_formula():
    N, beta = 25, 0.9
    sp = two_level_system(N, 1.0)
    assert sp.log_Z(beta) == pytest.approx(N * np.log(1.0 + np.exp(-beta)), rel=1e-12)


def test_gibbs_entropy_and_free_energy_consistency():
    sp = two_level_system(30, 1.0)
    beta = 1.3
    F = sp.free_energy(beta)
    E = sp.mean_energy(beta)
    S = sp.entropy(beta)
    assert F == pytest.approx(E - S / beta, rel=1e-10)


def test_canonical_heat_capacity_is_nonnegative_for_finite_systems():
    sp = two_level_system(20, 1.0)
    assert all(sp.heat_capacity(b) >= 0 for b in np.linspace(0.05, 5, 30))


def test_degeneracy_sum_is_two_to_the_n():
    sp = two_level_system(12, 1.0)
    assert np.exp(sp.log_Z(0.0)) == pytest.approx(2**12, rel=1e-12)


def test_bh_canonical_exponent_has_a_minimum_not_a_maximum():
    beta = 8 * np.pi * 0.5  # saddle at E = 0.5
    E = bh_saddle_energy(beta)
    assert E == pytest.approx(0.5)
    f = lambda e: bh_canonical_exponent(e, beta)
    assert f(E) < f(0.9 * E) and f(E) < f(1.1 * E)  # minimum
    assert bh_exponent_second_derivative() > 0


def test_microcanonical_heat_capacity_negative_and_consistent_with_si():
    assert microcanonical_heat_capacity_bh(1.0) < 0
    # S = 4 pi E^2 => C = -2 S in Planck units; same relation as the SI result
    E = 0.7
    assert microcanonical_heat_capacity_bh(E) == pytest.approx(-2.0 * 4 * np.pi * E**2)
    assert laws.heat_capacity(1e30) == pytest.approx(-2.0 * laws.bekenstein_hawking_entropy(1e30))


def test_log10_omega_eff_for_solar_mass():
    s = laws.bekenstein_hawking_entropy(1.98847e30, dimensionless=True)
    assert log10_omega_eff(s) == pytest.approx(1.049e77 / np.log(10), rel=1e-3)


def test_log_correction_parameterisation():
    A = np.array([1e2, 1e4, 1e6])
    base = A / 4.0
    assert np.allclose(entropy_with_log_correction(A, 0.0), base)
    corr = entropy_with_log_correction(A, -1.5) - base
    assert np.allclose(corr, -1.5 * np.log(A))
    with pytest.raises(PhysicalParameterError):
        entropy_with_log_correction(-1.0, 0.0)


def test_spectrum_shape_mismatch():
    with pytest.raises(PhysicalParameterError):
        FiniteSpectrum(np.zeros(3), np.zeros(4))
