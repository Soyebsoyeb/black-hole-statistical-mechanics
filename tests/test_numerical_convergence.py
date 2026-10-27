import numpy as np
import pytest

from bhstatmech import constants as c
from bhstatmech.config import load_config
from bhstatmech.evaporation import mass_loss as ml
from bhstatmech.exceptions import ConvergenceError
from bhstatmech.numerical_methods import (
    central_difference, find_root, make_rng, minimize_bounded, solve_ode, spawn_rngs,
    tolerance_sensitivity,
)
from bhstatmech.spacetimes import SchwarzschildAdS


def test_central_difference_second_order_convergence():
    f, x = np.exp, 0.3
    errs = [abs(central_difference(f, x, h) - np.exp(x)) for h in (1e-1, 5e-2, 2.5e-2)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.allclose(orders, 2.0, atol=0.02)


def test_ode_solver_converges_with_tolerance():
    # y' = -y, exact e^{-t}
    errs = []
    for rtol in (1e-4, 1e-7, 1e-10):
        sol = solve_ode(lambda t, y: -y, (0.0, 5.0), [1.0], rtol=rtol, atol=rtol * 1e-2)
        errs.append(abs(sol.y[0, -1] - np.exp(-5.0)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-9


def test_ode_failure_raises_convergence_error():
    with pytest.raises(ConvergenceError):
        solve_ode(lambda t, y: [y[0] ** 2], (0.0, 5.0), [1.0], rtol=1e-8)  # finite-time blow-up at t = 1


def test_mass_loss_integration_insensitive_to_tolerance():
    t_ref = None
    vals = []
    for rtol in (1e-7, 1e-9, 1e-11):
        t, M = ml.integrate_mass_loss(1e12, cutoff_planck_masses=1e-2 * 1e12 / c.PLANCK_MASS, rtol=rtol)
        vals.append(t[-1])
    assert abs(vals[0] - vals[2]) / vals[2] < 1e-6
    assert abs(vals[1] - vals[2]) / vals[2] < 1e-8


def test_root_finder_tolerance_sensitivity_of_ads_inversion():
    L, M = 1.3, 2.0
    rp = [SchwarzschildAdS.from_mass(M, L).r_plus]
    assert SchwarzschildAdS(rp[0], L).mass == pytest.approx(M, rel=1e-13)
    # bracketing root-finders: result changes by <= requested tolerance
    f = lambda x: x**3 - 2
    for tol in (1e-6, 1e-10, 1e-14):
        assert find_root(f, 0.0, 2.0, xtol=tol, rtol=1e-15).root == pytest.approx(2 ** (1 / 3), abs=10 * tol)


def test_minimizer_tolerance_sensitivity():
    g = lambda x: (x - 0.3) ** 2 + 1.0
    out = tolerance_sensitivity(lambda t: minimize_bounded(g, -1.0, 2.0, xtol=t)[0], [1e-4, 1e-8, 1e-11])
    # the location of a smooth minimum cannot be resolved below ~sqrt(machine eps) = 1.5e-8,
    # whatever xtol is requested (the objective is flat to round-off there)
    assert np.all(np.abs(out - 0.3) < np.array([1e-3, 1e-7, 1e-7]))


def test_rng_reproducibility_and_independence():
    a = make_rng(5).standard_normal(5)
    assert np.array_equal(a, make_rng(5).standard_normal(5))
    gens = spawn_rngs(5, 3)
    draws = [g.standard_normal(100) for g in gens]
    assert abs(np.corrcoef(draws[0], draws[1])[0, 1]) < 0.35
    again = [g.standard_normal(100) for g in spawn_rngs(5, 3)]
    assert all(np.array_equal(x, y) for x, y in zip(draws, again))
    g = make_rng(7)
    assert make_rng(g) is g


def test_config_is_versioned_and_loadable():
    cfg = load_config()
    assert cfg["config_version"] == 1 and "master_seed" in cfg
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.toml")
