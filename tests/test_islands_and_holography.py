import numpy as np
import pytest

from bhstatmech.exceptions import PhysicalParameterError
from bhstatmech.generalized_entropy import IslandToyModel, page_time_no_correction
from bhstatmech.holography import (
    ads3_entropy, ads3_vacuum_length, btz_connected_length, btz_crossover_angle,
    btz_disconnected_length, btz_entropy, btz_geodesic_numeric, numeric_semicircle_length,
)

M = IslandToyModel()


def test_branch_structure_before_and_after_transition():
    tp = M.page_time()
    assert 0.0 < tp < 1.0
    early, late = tp * 0.5, 0.5 * (1 + tp)
    assert float(M.S_no_island(early)) < M.S_island_analytic(early)
    assert float(M.S_no_island(late)) > M.S_island_analytic(late)
    s = M.S_radiation([early, late])
    assert s[0] == pytest.approx(float(M.S_no_island(early)))
    assert s[1] == pytest.approx(M.S_island_analytic(late))


def test_page_transition_continuity_and_envelope_is_minimum():
    taus = np.linspace(0.0, 0.99, 400)
    env = M.S_radiation(taus)
    no = np.asarray(M.S_no_island(taus), dtype=float)
    assert np.all(env <= no + 1e-9)
    isl = np.array([M.S_island_analytic(float(t)) for t in taus])
    valid = ~np.isnan(isl)
    assert np.all(env[valid] <= isl[valid] + 1e-9)
    # rises then falls
    k = int(np.argmax(env))
    assert 0 < k < len(taus) - 1
    assert np.all(np.diff(env[: k + 1]) >= -1e-9) and np.all(np.diff(env[k:]) <= 1e-9)
    # at tau -> 1 the entropy goes to zero (late-time purification in the toy model)
    assert env[-1] < 0.2 * env[k]


def test_page_time_matches_root_condition():
    tp = M.page_time()
    assert float(M.S_no_island(tp)) == pytest.approx(M.S_island_analytic(tp), rel=1e-10)


def test_page_time_limit_kappa_to_zero():
    m = IslandToyModel(mu=1e-6, xi=1e-3)
    assert m.page_time() == pytest.approx(page_time_no_correction(m.r), abs=2e-3)


def test_analytic_and_numerical_extremisation_agree():
    for tau in (0.2, 0.5, 0.8, 0.95):
        u_num, val_num = M.u_star_numeric(tau)
        assert u_num == pytest.approx(M.u_star_analytic(tau), abs=1e-6)
        assert val_num == pytest.approx(M.S_island_analytic(tau), rel=1e-12)


def test_extremum_is_a_minimum_in_u():
    tau = 0.7
    u = M.u_star_analytic(tau)
    f = lambda x: float(M.S_gen(x, tau))
    assert f(u) < f(u * 0.9) and f(u) < f(u * 1.1)


def test_extremisation_stable_to_tolerance_and_bounds():
    base = M.u_star_numeric(0.6, xtol=1e-12)[0]
    for xtol in (1e-6, 1e-9, 1e-14):
        assert M.u_star_numeric(0.6, xtol=xtol)[0] == pytest.approx(base, abs=2e-6)
    assert M.u_star_numeric(0.6, u_max=50.0)[0] == pytest.approx(base, abs=1e-6)


def test_parameter_sensitivity_of_page_time():
    # larger r (more radiated entropy per lost black-hole entropy) => earlier Page time
    t = [IslandToyModel(r=r).page_time() for r in (1.0, 4 / 3, 2.0)]
    assert t[0] > t[1] > t[2]
    # larger kappa (costlier island) => later transition
    k = [IslandToyModel(mu=m, xi=0.05).page_time() for m in (0.5, 1.0, 3.0)]
    assert k[0] < k[1] < k[2]


def test_island_toy_model_invalid_params():
    with pytest.raises(PhysicalParameterError):
        IslandToyModel(S0=-1.0)


# ---- holography --------------------------------------------------------------------


def test_ads3_vacuum_entropy_central_charge():
    L, G = 1.0, 1.0
    c_cft = 3 * L / (2 * G)
    for l in (0.1, 1.0, 10.0):
        S = ads3_entropy(l, 1e-6, L, G)
        assert S == pytest.approx((c_cft / 3.0) * np.log(l / 1e-6), rel=1e-12)


def test_semicircle_numerical_integral_matches_closed_form():
    for l, eps in [(1.0, 1e-3), (10.0, 1e-2), (2.0, 1e-5)]:
        assert numeric_semicircle_length(l, eps) == pytest.approx(ads3_vacuum_length(l, eps), rel=1e-6)


def test_btz_numeric_geodesic_matches_closed_form():
    for dphi, rp in [(1.0, 1.0), (2.5, 1.0), (0.3, 2.0)]:
        a = btz_geodesic_numeric(dphi, rp, 1e-4)
        b = btz_connected_length(dphi, rp, 1e-4)
        assert a == pytest.approx(b, rel=1e-6)


def test_btz_small_interval_recovers_vacuum_result():
    # r_+ -> 0 is the massless BTZ (planar) limit: length -> 2 L ln(dphi / eps), i.e. the
    # Poincare-AdS3 result for an interval of coordinate width dphi (central charge c = 3L/2G)
    dphi, eps = 0.8, 1e-5
    assert btz_connected_length(dphi, 1e-6, eps) == pytest.approx(2 * np.log(dphi / eps), rel=1e-6)
    assert btz_connected_length(dphi, 1e-6, eps) == pytest.approx(ads3_vacuum_length(dphi, eps), rel=1e-6)


def test_btz_entanglement_plateau_and_branch_switch():
    rp, eps = 1.0, 1e-4
    x = btz_crossover_angle(rp)
    assert np.pi < x < 2 * np.pi
    assert btz_entropy(x - 0.05, rp, eps)[1] == "connected"
    assert btz_entropy(x + 0.05, rp, eps)[1] == "disconnected"
    # the minimum is continuous at the crossover
    sa = btz_entropy(x - 1e-9, rp, eps)[0]
    sb = btz_entropy(x + 1e-9, rp, eps)[0]
    assert sa == pytest.approx(sb, rel=1e-6)
    # S(A) is bounded above by the thermal (Bekenstein-Hawking-like) plateau
    s_full = btz_disconnected_length(2 * np.pi - 1e-3, rp, eps) / 4
    assert btz_entropy(2 * np.pi - 1e-3, rp, eps)[0] <= s_full


def test_btz_crossover_moves_toward_2pi_with_temperature():
    xs = [btz_crossover_angle(rp) for rp in (0.5, 1.0, 3.0)]
    assert xs[0] < xs[1] < xs[2] < 2 * np.pi


def test_btz_entropy_for_complement_differs_only_by_horizon_term():
    rp, eps = 2.0, 1e-4
    dphi = 1.0
    c, d = btz_connected_length(dphi, rp, eps), btz_disconnected_length(dphi, rp, eps)
    assert d - btz_connected_length(2 * np.pi - dphi, rp, eps) == pytest.approx(2 * np.pi * rp)
    assert np.isfinite(c) and np.isfinite(d)
