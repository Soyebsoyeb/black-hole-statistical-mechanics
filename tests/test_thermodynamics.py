import numpy as np
import pytest

from bhstatmech import constants as c
from bhstatmech.evaporation import mass_loss as ml
from bhstatmech.exceptions import ConvergenceError, PhysicalParameterError
from bhstatmech.numerical_methods import (
    central_difference, derivative_error_scan, find_root, richardson_derivative,
)
from bhstatmech.spacetimes import Kerr, ReissnerNordstrom, Schwarzschild
from bhstatmech.thermodynamics import first_law, laws

MASSES = np.array([1e12, 1e20, 1e27, c.M_SUN, 1e32])


def test_temperature_solar_mass():
    assert laws.hawking_temperature(c.M_SUN) == pytest.approx(6.17e-8, rel=2e-3)


def test_entropy_scalings():
    S = laws.bekenstein_hawking_entropy(MASSES)
    A = laws.horizon_area(MASSES)
    assert np.allclose(np.diff(np.log(S)) / np.diff(np.log(MASSES)), 2.0, rtol=1e-12)
    assert np.allclose(S / A, S[0] / A[0], rtol=1e-12)  # S proportional to A
    assert np.allclose(S, laws.entropy_from_area(A), rtol=1e-12)


def test_entropy_area_over_four_planck_squared():
    A = laws.horizon_area(c.M_SUN)
    s = laws.entropy_from_area(A, dimensionless=True)
    assert s == pytest.approx(A / (4 * c.PLANCK_LENGTH**2), rel=1e-12)
    assert s == pytest.approx(1.049e77, rel=1e-3)


def test_class_and_functional_interfaces_agree():
    bh = Schwarzschild(1e30)
    assert bh.hawking_temperature == pytest.approx(float(laws.hawking_temperature(1e30)), rel=1e-12)
    assert bh.bekenstein_hawking_entropy == pytest.approx(float(laws.bekenstein_hawking_entropy(1e30)), rel=1e-12)
    assert bh.horizon_area == pytest.approx(float(laws.horizon_area(1e30)), rel=1e-12)


@pytest.mark.parametrize("m", MASSES)
def test_first_law_schwarzschild_finite_difference(m):
    assert laws.first_law_residual(float(m)) < 1e-8


def test_first_law_symbolic():
    assert laws.first_law_symbolic()


def test_first_law_error_scan_shows_truncation_then_roundoff():
    f = lambda m: float(laws.bekenstein_hawking_entropy(m))
    m0 = 1e30
    exact = 8 * np.pi * c.G * m0 / (c.HBAR * c.C) * c.K_B
    steps = m0 * np.array([1e-1, 1e-2, 1e-3, 1e-4])
    err = derivative_error_scan(f, exact, m0, steps)
    # truncation error O(h^2): factor ~100 per decade (exact for the quadratic: finite-difference exact)
    assert err.max() < 1e-12
    cubic = lambda x: x**3
    e2 = derivative_error_scan(cubic, 3.0, 1.0, [1e-1, 1e-2])
    assert e2[0] / e2[1] == pytest.approx(100.0, rel=1e-6)


def test_richardson_improves_central_difference():
    f, x = np.sin, 0.7
    h = 0.1
    e1 = abs(central_difference(f, x, h) - np.cos(x))
    e2 = abs(richardson_derivative(f, x, h) - np.cos(x))
    assert e2 < e1 / 50


def test_heat_capacity_negative_and_matches_numerical_derivative():
    for m in MASSES:
        assert laws.heat_capacity(m) < 0
    m0 = 1e30
    dMc2_dT = (m0 * (1 + 1e-6) - m0 * (1 - 1e-6)) * c.C**2 / (
        laws.hawking_temperature(m0 * (1 + 1e-6)) - laws.hawking_temperature(m0 * (1 - 1e-6)))
    assert dMc2_dT == pytest.approx(float(laws.heat_capacity(m0)), rel=1e-6)
    assert laws.heat_capacity(m0) == pytest.approx(-2.0 * laws.bekenstein_hawking_entropy(m0), rel=1e-12)


def test_rn_first_law():
    for rp, Q in [(2.0, 0.5), (1.5, 1.0), (3.0, 2.9)]:
        assert first_law.rn_first_law_residual(rp, Q) < 1e-6


def test_kerr_first_law():
    for a in (0.1, 0.5, 0.9):
        assert first_law.kerr_first_law_residual(1.0, a) < 1e-6


def test_ads_first_law():
    for rp in (0.3, 1.0, 4.0):
        assert first_law.ads_first_law_residual(rp, 1.0) < 1e-6


# ---- evaporation ----------------------------------------------------------------


def test_mass_loss_scaling_t_evap_proportional_to_m_cubed():
    t1 = ml.evaporation_time_analytic(1e11)
    t2 = ml.evaporation_time_analytic(2e11)
    assert t2 / t1 == pytest.approx(8.0, rel=1e-12)
    r = ml.mass_loss_rate(1e11) / ml.mass_loss_rate(2e11)
    assert r == pytest.approx(4.0, rel=1e-12)  # dM/dt ~ 1/M^2


def test_mass_loss_closed_form_coefficient():
    # photons only, blackbody at T_H with area 4 pi r_s^2: alpha = hbar c^4 / (15360 pi G^2)
    m = 1e12
    assert ml.mass_loss_rate(m) == pytest.approx(-ml.alpha_coefficient() / m**2, rel=1e-10)


def test_integrated_mass_loss_matches_analytic_solution():
    # M(t) = M0 (1 - t/t_evap)^(1/3) has an unbounded slope as t -> t_evap, so the
    # well-conditioned comparison is in the other direction: t/t_evap = 1 - (M/M0)^3.
    m0 = 1e12
    t, M = ml.integrate_mass_loss(m0, cutoff_planck_masses=1e-3 * m0 / c.PLANCK_MASS)  # stop at 1e-3 m0
    t_ev = ml.evaporation_time_analytic(m0)
    assert np.max(np.abs(t / t_ev - (1.0 - (M / m0) ** 3))) < 1e-9
    assert M[-1] == pytest.approx(1e-3 * m0, rel=1e-7)  # limited by the event root-finder accuracy
    assert np.all(np.diff(M) < 0)


def test_geometric_optics_cross_section_shortens_lifetime():
    assert ml.evaporation_time_analytic(1e12, "geometric_optics") < ml.evaporation_time_analytic(1e12)
    # emitting area of a sphere of radius b_c = 3 sqrt(3) GM/c^2 is 27/4 times the horizon area
    ratio = ml.effective_area(1e12, "geometric_optics") / ml.effective_area(1e12, "horizon")
    assert ratio == pytest.approx(27.0 / 4.0, rel=1e-12)
    bc = 3.0 * np.sqrt(3.0) * c.G * 1e12 / c.C**2
    assert ml.effective_area(1e12, "geometric_optics") == pytest.approx(4 * np.pi * bc**2, rel=1e-12)
    assert ml.evaporation_time_analytic(1e12) / ml.evaporation_time_analytic(1e12, "geometric_optics") == pytest.approx(27.0 / 4.0)
    with pytest.raises(PhysicalParameterError):
        ml.effective_area(1e12, "nonsense")


def test_adiabaticity_parameter_grows_as_mass_shrinks():
    e = [ml.adiabaticity_parameter(m) for m in (1e15, 1e12, 1e9)]
    assert e[0] < e[1] < e[2]
    assert e[1] / e[0] == pytest.approx(1e6, rel=1e-9)  # epsilon ~ M^-2
    m_half = ml.mass_at_adiabaticity(0.1)
    assert ml.adiabaticity_parameter(m_half) == pytest.approx(0.1, rel=1e-9)


def test_integration_below_cutoff_is_rejected():
    with pytest.raises(PhysicalParameterError):
        ml.integrate_mass_loss(c.PLANCK_MASS)


def test_find_root_requires_bracket():
    with pytest.raises(ConvergenceError, match="bracketed"):
        find_root(lambda x: x**2 + 1.0, -1.0, 1.0)
    assert find_root(lambda x: x**2 - 2.0, 0.0, 2.0).root == pytest.approx(np.sqrt(2.0), abs=1e-13)
