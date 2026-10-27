import numpy as np
import pytest

from bhstatmech import constants as c
from bhstatmech.exceptions import PhysicalParameterError
from bhstatmech.spacetimes import (
    Kerr, ReissnerNordstrom, Schwarzschild, SchwarzschildAdS, SchwarzschildDeSitter,
    kretschmann_symbolic,
)

SUN = Schwarzschild(c.M_SUN)


def test_solar_schwarzschild_radius():
    assert SUN.schwarzschild_radius == pytest.approx(2953.25, abs=0.01)  # 2 GM_sun/c^2 with the IAU nominal GM_sun


def test_metric_inverse_is_inverse():
    r = np.array([1.5, 3.0, 10.0]) * SUN.schwarzschild_radius
    g = SUN.metric_components(r)
    gi = SUN.inverse_metric_components(r)
    for a, b in zip(g, gi):
        assert np.allclose(a * b, 1.0)


def test_metric_singular_flag_at_horizon_is_nan_not_inf():
    g = SUN.metric_components(SUN.schwarzschild_radius)
    assert np.isnan(g[1])


def test_ingoing_eddington_finkelstein_is_regular_at_horizon():
    block, _, _ = SUN.ingoing_ef_metric(SUN.schwarzschild_radius)
    assert np.all(np.isfinite(block))
    assert abs(np.linalg.det(block)) > 0  # non-degenerate: coordinate singularity removed


def test_kretschmann_finite_at_horizon_diverges_at_origin():
    rs = SUN.schwarzschild_radius
    k_h = SUN.kretschmann(rs)
    assert np.isfinite(k_h)
    assert k_h == pytest.approx(12.0 / rs**4, rel=1e-12)
    assert SUN.kretschmann(1e-6 * rs) / k_h == pytest.approx(1e36, rel=1e-9)


def test_kretschmann_symbolic_matches_closed_form():
    import sympy as sp

    K, (M, r) = kretschmann_symbolic()
    assert sp.simplify(K - 48 * M**2 / r**6) == 0


def test_tortoise_log_divergence_near_horizon():
    rs = SUN.schwarzschild_radius
    eps = np.array([1e-3, 1e-4, 1e-5])
    rstar = SUN.tortoise(rs * (1 + eps))
    # r* ~ r_s ln(eps) + r_s: slope in ln(eps) is r_s
    slope = np.diff(rstar) / np.diff(np.log(eps))
    assert np.allclose(slope, rs, rtol=1e-3)


def test_tortoise_inverse_roundtrip():
    rs = SUN.schwarzschild_radius
    r = rs * np.array([1.0001, 1.5, 3.0, 50.0])
    assert np.allclose(SUN.radius_from_tortoise(SUN.tortoise(r)), r, rtol=1e-11)


def test_radial_null_rays_have_unit_coordinate_speed_in_tortoise_coordinate():
    r = SUN.schwarzschild_radius * np.array([2.0, 4.0])
    dt = SUN.radial_null_time(r[1], True) - SUN.radial_null_time(r[0], True)
    assert dt == pytest.approx((SUN.tortoise(r[1]) - SUN.tortoise(r[0])) / c.C)


def test_effective_potential_properties():
    rs = SUN.schwarzschild_radius
    r = rs * np.linspace(1.0001, 20, 400)
    for spin in (0, 1, 2):
        V = SUN.effective_potential(r, 2, spin)
        assert np.all(V >= 0)
        assert V[-1] < V.max() / 5  # decays at large r
    V0 = SUN.effective_potential(rs * (1 + 1e-7), 2, 0)
    assert V0 < 1e-6 * SUN.effective_potential(r, 2, 0).max()  # potential vanishes at the horizon


def test_invalid_inputs_raise():
    with pytest.raises(PhysicalParameterError):
        Schwarzschild(-1.0)
    with pytest.raises(PhysicalParameterError):
        Schwarzschild(float("nan"))
    with pytest.raises(PhysicalParameterError):
        SUN.f(-1.0)
    with pytest.raises(PhysicalParameterError):
        SUN.effective_potential(10.0, 0, 3)


# ---- Reissner-Nordstrom --------------------------------------------------------


def test_rn_reduces_to_schwarzschild_at_zero_charge():
    rn = ReissnerNordstrom(1.0, 0.0)
    assert rn.r_plus == pytest.approx(2.0)
    assert rn.temperature == pytest.approx(1.0 / (8 * np.pi))
    assert rn.entropy == pytest.approx(4 * np.pi)


def test_rn_extremal_limit_has_zero_temperature():
    for q in (0.9, 0.999, 0.999999, 1.0):
        T = ReissnerNordstrom(1.0, q).temperature
        assert T >= 0
    assert ReissnerNordstrom(1.0, 1.0).temperature == 0.0
    assert ReissnerNordstrom(1.0, 1.0).is_extremal
    assert ReissnerNordstrom(1.0, 1.0).entropy == pytest.approx(np.pi)


def test_rn_overextremal_is_rejected():
    with pytest.raises(PhysicalParameterError, match="naked"):
        ReissnerNordstrom(1.0, 1.01)


def test_rn_horizons_are_roots_of_f():
    rn = ReissnerNordstrom(1.0, 0.6)
    assert abs(rn.f(rn.r_plus)) < 1e-14 and abs(rn.f(rn.r_minus)) < 1e-14


def test_rn_heat_capacity_sign_change_at_sqrt3_Q():
    # M(r+, Q) at r+ = sqrt(3) Q has diverging C_Q; stable (C>0) for r+ < sqrt(3) Q
    Q = 0.5
    stable = ReissnerNordstrom(ReissnerNordstrom.mass_from(1.2 * Q, Q), Q)
    unstable = ReissnerNordstrom(ReissnerNordstrom.mass_from(4.0 * Q, Q), Q)
    assert stable.heat_capacity_fixed_Q > 0
    assert unstable.heat_capacity_fixed_Q < 0


def test_rn_smarr_relation():
    from bhstatmech.thermodynamics.first_law import rn_smarr_residual

    assert abs(rn_smarr_residual(ReissnerNordstrom(1.0, 0.7))) < 1e-13


# ---- Kerr ----------------------------------------------------------------------------


def test_kerr_limits():
    k0 = Kerr(1.0, 0.0)
    assert k0.r_plus == pytest.approx(2.0)
    assert k0.temperature == pytest.approx(1.0 / (8 * np.pi))
    ke = Kerr(1.0, 1.0)
    assert ke.temperature == 0.0
    assert ke.horizon_angular_velocity == pytest.approx(0.5)  # a/(2 M r+) = 1/2 at extremality
    assert ke.entropy == pytest.approx(2 * np.pi)


def test_kerr_smarr_relation():
    for a in (0.0, 0.3, 0.9, 0.9999):
        assert abs(Kerr(1.0, a).smarr_residual()) < 1e-13


def test_kerr_overspin_rejected():
    with pytest.raises(PhysicalParameterError):
        Kerr(1.0, 1.2)


# ---- AdS / dS ---------------------------------------------------------------------


def test_ads_hawking_page_values():
    L = 1.0
    bh = SchwarzschildAdS(L, L)  # r+ = L
    assert bh.free_energy == pytest.approx(0.0, abs=1e-15)
    assert bh.temperature == pytest.approx(SchwarzschildAdS.hawking_page_temperature(L))
    small = SchwarzschildAdS(0.3 * L, L)
    large = SchwarzschildAdS(3.0 * L, L)
    assert small.free_energy > 0 > large.free_energy
    assert small.heat_capacity < 0 < large.heat_capacity


def test_ads_minimum_temperature():
    L = 2.0
    rmin = L / np.sqrt(3.0)
    T = SchwarzschildAdS(rmin, L).temperature
    assert T == pytest.approx(SchwarzschildAdS.minimum_temperature(L))
    for rp in (0.5 * rmin, 0.9 * rmin, 1.1 * rmin, 3 * rmin):
        assert SchwarzschildAdS(rp, L).temperature > T


def test_ads_from_mass_roundtrip():
    bh = SchwarzschildAdS.from_mass(2.3, 1.7)
    assert bh.mass == pytest.approx(2.3, rel=1e-12)


def test_sds_two_horizons_two_temperatures():
    bh = SchwarzschildDeSitter(1.0, 20.0)
    assert bh.r_b < bh.r_c
    assert abs(bh.f(bh.r_b)) < 1e-12 and abs(bh.f(bh.r_c)) < 1e-12
    assert bh.temperature_b != pytest.approx(bh.temperature_c, rel=1e-2)
    # small M/L: r_b = 2M + 8 M^3 / L^2 + O(M^5/L^4)
    assert bh.r_b == pytest.approx(2.0 + 8.0 / 400.0, rel=1e-3)


def test_sds_nariai_bound():
    with pytest.raises(PhysicalParameterError, match="Nariai"):
        SchwarzschildDeSitter(1.0, 5.0)
    near = SchwarzschildDeSitter(1.0, np.sqrt(27.0) * (1 + 1e-6))
    assert near.temperature_b < 1e-2 and near.temperature_c < 1e-2
    assert near.r_c - near.r_b < 0.1 * near.r_b
