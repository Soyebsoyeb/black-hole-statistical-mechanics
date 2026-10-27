import numpy as np
import pytest

from bhstatmech.exceptions import PhysicalParameterError
from bhstatmech.lattice_models import (
    block_entropy, entropy_from_symplectic, ground_state_covariance, laplacian,
    stiffness_matrix, symplectic_eigenvalues, two_oscillator_entropy_exact,
)


def test_two_oscillator_matches_analytic_formula():
    m = 0.3
    X, P = ground_state_covariance(2, 1.0, m, "dirichlet")  # omega0^2 = 1 + m^2, k = 1
    exact = two_oscillator_entropy_exact(np.sqrt(1 + m**2), 1.0)
    assert block_entropy(X, P, [0]) == pytest.approx(exact, rel=1e-12)


def test_pure_global_state_has_unit_uncertainty_product():
    X, P = ground_state_covariance(20, 1.0, 0.2, "periodic")
    nu = symplectic_eigenvalues(X, P)  # whole system: pure state, all nu = 1/2
    assert np.allclose(nu, 0.5, atol=1e-9)
    assert entropy_from_symplectic(nu) == pytest.approx(0.0, abs=1e-9)


def test_complementary_blocks_have_equal_entropy():
    n = 24
    X, P = ground_state_covariance(n, 1.0, 0.1, "periodic")
    A = list(range(5, 13))
    B = [i for i in range(n) if i not in A]
    assert block_entropy(X, P, A) == pytest.approx(block_entropy(X, P, B), rel=1e-7)


def test_entropy_monotone_then_symmetric_in_block_size():
    n = 40
    X, P = ground_state_covariance(n, 1.0, 0.05, "periodic")
    s = [block_entropy(X, P, range(l)) for l in range(1, n)]
    assert np.argmax(s) in (n // 2 - 1, n // 2 - 2, n // 2)
    assert block_entropy(X, P, range(10)) == pytest.approx(block_entropy(X, P, range(n - 10)), rel=1e-7)


def test_logarithmic_scaling_with_central_charge_one():
    # periodic chain: S = (c/3) ln[(N/pi) sin(pi l/N)] + const with c = 1 (massless limit)
    n = 128
    X, P = ground_state_covariance(n, 1.0, 1e-3, "periodic")
    ls = np.array([4, 8, 16, 32])
    S = np.array([block_entropy(X, P, range(l)) for l in ls])
    chord = np.log(n / np.pi * np.sin(np.pi * ls / n))
    slope = np.polyfit(chord, S, 1)[0]
    assert slope == pytest.approx(1.0 / 3.0, rel=0.10)


def test_lattice_refinement_converges_at_fixed_physical_size():
    # fixed physical length: interval 1/4 of a ring of circumference 1; mass m = 1e-2 / length unit
    # entropy minus the ln(1/a) divergence converges as a -> 0.
    vals = []
    for N in (32, 64, 128):
        a = 1.0 / N
        X, P = ground_state_covariance(N, a, 1e-2, "periodic")
        S = block_entropy(X, P, range(N // 4))
        vals.append(S - (1.0 / 3.0) * np.log(1.0 / a))
    d1, d2 = abs(vals[1] - vals[0]), abs(vals[2] - vals[1])
    assert d2 < d1  # successive refinements get closer
    assert d2 < 0.02


def test_entropy_grows_with_refinement_like_log_of_cutoff():
    ss = []
    for N in (32, 64, 128):
        X, P = ground_state_covariance(N, 1.0 / N, 1e-2, "periodic")
        ss.append(block_entropy(X, P, range(N // 4)))
    increments = np.diff(ss)
    assert np.allclose(increments, np.log(2) / 3, rtol=0.1)  # UV divergence: S ~ (1/3) ln(1/a)


def test_boundary_conditions_differ_but_agree_far_from_boundary():
    n = 80
    res = {}
    for bc in ("periodic", "dirichlet", "neumann"):
        X, P = ground_state_covariance(n, 1.0, 0.05, bc)
        res[bc] = block_entropy(X, P, range(30, 40))  # interior block
    assert res["periodic"] == pytest.approx(res["dirichlet"], rel=0.05)
    # an end-anchored block has one entangling point only; its entropy is smaller
    X, P = ground_state_covariance(n, 1.0, 0.05, "dirichlet")
    assert block_entropy(X, P, range(10)) < block_entropy(X, P, range(30, 40))


def test_finite_size_correction_decreases_with_total_size():
    # entropy of fixed 10-site block approaches its infinite-chain value as N grows
    s = []
    for n in (40, 80, 160):
        X, P = ground_state_covariance(n, 1.0, 0.1, "periodic")
        s.append(block_entropy(X, P, range(10)))
    assert abs(s[2] - s[1]) < abs(s[1] - s[0])


def test_invalid_parameters():
    with pytest.raises(PhysicalParameterError):
        stiffness_matrix(10, 1.0, 0.0, "periodic")  # zero mode
    with pytest.raises(PhysicalParameterError):
        laplacian(10, "bogus")
    with pytest.raises(PhysicalParameterError):
        stiffness_matrix(10, -1.0, 0.1)
    with pytest.raises(PhysicalParameterError):
        laplacian(1, "dirichlet")
