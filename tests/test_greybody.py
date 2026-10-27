import numpy as np
import pytest

from bhstatmech.exceptions import PhysicalParameterError
from bhstatmech.hawking_radiation import (
    absorption_cross_section, bose_occupation, dimensionless_frequency, fermi_occupation,
    greybody_factor, greybody_table, number_flux, planck_energy_flux_per_omega,
)
from bhstatmech import constants as c
from bhstatmech.thermodynamics import laws


def test_bose_fermi_limits_and_identities():
    x = np.array([0.01, 0.5, 3.0, 20.0])
    b, f = bose_occupation(x), fermi_occupation(x)
    assert np.allclose((b - f)[:3], 2.0 * bose_occupation(2 * x[:3]), rtol=1e-9)  # 1/(e^x-1) - 1/(e^x+1) = 2/(e^2x-1)
    assert np.allclose(1.0 / b + 1.0, np.exp(x))
    assert np.allclose(1.0 / f - 1.0, np.exp(x))
    assert b[0] == pytest.approx(1.0 / x[0], rel=1e-2)          # classical (Rayleigh-Jeans) limit
    assert f[0] == pytest.approx(0.5, abs=5e-3)                  # Pauli blocking
    assert np.isfinite(fermi_occupation(800.0)) and fermi_occupation(800.0) >= 0.0
    assert np.all(f < 1.0) and np.all(b > f)
    # relative accuracy at large x (where 1 - tanh would cancel)
    assert fermi_occupation(30.0) == pytest.approx(np.exp(-30.0) / (1 + np.exp(-30.0)), rel=1e-14)


def test_spectrum_invalid_inputs():
    with pytest.raises(PhysicalParameterError):
        bose_occupation(0.0)
    with pytest.raises(PhysicalParameterError):
        dimensionless_frequency(1.0, -3.0)


def test_dimensionless_frequency_scales_with_mass():
    omega = 1e3
    x1 = dimensionless_frequency(omega, float(laws.hawking_temperature(1e20)))
    x2 = dimensionless_frequency(omega, float(laws.hawking_temperature(2e20)))
    assert x2 / x1 == pytest.approx(2.0)  # T ~ 1/M => x ~ M


def test_planck_flux_integrates_to_stefan_boltzmann():
    from scipy.integrate import quad

    T = 300.0
    kT = c.K_B * T / c.HBAR
    val, _ = quad(lambda w: planck_energy_flux_per_omega(w, T), 1e-6 * kT, 60 * kT, limit=200)
    # two polarisations => the full Stefan-Boltzmann flux sigma T^4
    assert val == pytest.approx(c.SIGMA_SB * T**4, rel=1e-4)


def test_greybody_bounds_and_flux_conservation():
    for spin in (0, 1, 2):
        for ell in range(max(spin, 0), spin + 3):
            for y in (0.05, 0.2, 0.5, 1.0):
                r = greybody_factor(y, ell, spin)
                assert 0.0 <= r.gamma <= 1.0 + 1e-6
                assert r.flux_defect < 1e-6   # |a_in|^2 - |a_out|^2 = 1 (Wronskian)


def test_scalar_low_frequency_limit_gives_horizon_area():
    # Gamma_0 -> 4 y^2 (sigma -> A_horizon = 4 pi r_s^2); correction is linear in y
    ratios = [greybody_factor(y, 0).gamma / (4 * y * y) for y in (0.02, 0.01, 0.005)]
    assert ratios[0] > ratios[1] > ratios[2] > 1.0
    assert ratios[2] == pytest.approx(1.0, abs=0.03)
    # linear convergence: (ratio - 1) halves when y halves
    assert (ratios[1] - 1) / (ratios[2] - 1) == pytest.approx(2.0, rel=0.15)


def test_transmission_increases_with_frequency_and_decreases_with_ell():
    ys = [0.1, 0.3, 0.6, 1.0, 1.5]
    g0 = [greybody_factor(y, 0).gamma for y in ys]
    assert all(a < b for a, b in zip(g0, g0[1:])) or g0[-1] > 0.99
    g_ell = [greybody_factor(0.4, l).gamma for l in range(4)]
    assert all(a > b for a, b in zip(g_ell, g_ell[1:]))


def test_high_frequency_cross_section_oscillates_about_geometric_optics():
    y = np.array([0.8, 1.0, 1.2])
    G = greybody_table(y, 12)
    sigma = absorption_cross_section(y, G)
    geo = 27.0 * np.pi / 4.0
    assert np.all(np.abs(sigma / geo - 1.0) < 0.12)


def test_solver_tolerance_sensitivity_is_negligible():
    a = greybody_factor(0.3, 2, rtol=1e-8, atol=1e-10).gamma
    b = greybody_factor(0.3, 2, rtol=1e-11, atol=1e-13).gamma
    assert abs(a - b) < 1e-9


def test_matching_radius_sensitivity():
    a = greybody_factor(0.3, 1, x_match=300.0).gamma
    b = greybody_factor(0.3, 1, x_match=900.0).gamma
    assert abs(a - b) < 1e-5


def test_number_flux_shape_and_positivity():
    y = np.linspace(0.05, 1.0, 12)
    G = greybody_table(y, 3)
    flux = number_flux(y, G)
    assert flux.shape == y.shape and np.all(flux > 0)
    # at small y the s-wave dominates the sum over multipoles
    weights = (2 * np.arange(4) + 1)[:, None] * G * bose_occupation(4 * np.pi * y)[None, :]
    assert weights[1:, 0].sum() < 0.1 * weights[0, 0]
    assert flux[0] == pytest.approx(weights[:, 0].sum() / (2 * np.pi), rel=1e-12)


def test_invalid_greybody_inputs():
    with pytest.raises(PhysicalParameterError):
        greybody_factor(-1.0, 0)
    with pytest.raises(PhysicalParameterError):
        greybody_factor(0.5, 0, spin=1)


def test_numerical_solver_agrees_with_independent_wkb_near_barrier_top():
    from bhstatmech.hawking_radiation import wkb_transmission

    # (ell, frequencies spanning the barrier top sqrt(V0)); measured maxima 0.086, 0.063, 0.040
    cases = {2: ((0.95, 1.0, 1.05, 1.15, 1.3), 0.10), 3: ((1.3, 1.37, 1.45, 1.6), 0.07), 5: ((2.0, 2.13, 2.25), 0.05)}
    worst = {}
    for ell, (ys, tol) in cases.items():
        dev = max(abs(greybody_factor(y, ell).gamma - wkb_transmission(y, ell)) for y in ys)
        assert dev < tol, (ell, dev)
        worst[ell] = dev
    assert worst[2] > worst[3] > worst[5]          # WKB error shrinks with ell, as expected
    # far above the barrier both methods tend to 1
    assert greybody_factor(1.3, 2).gamma > 0.99 and wkb_transmission(1.3, 2) > 0.99
