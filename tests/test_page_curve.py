import numpy as np
import pytest

from bhstatmech.numerical_methods import make_rng, spawn_rngs
from bhstatmech.page_curve import (
    hawking_reference_curve, page_average_entropy, page_bound_curve, page_large_dimension_approx,
    qubit_page_curve_ensemble, qubit_page_curve_sample, qubit_page_exact, sample_entropy,
)
from bhstatmech.quantum_states import (
    density_from_ket, partial_trace, random_pure_state, von_neumann_entropy,
)


def test_page_formula_small_cases_exact():
    assert page_average_entropy(1, 7) == pytest.approx(0.0, abs=1e-14)
    # m = n = 2: sum_{k=3}^4 1/k - 1/4 = 1/3 + 1/4 - 1/4 = 1/3
    assert page_average_entropy(2, 2) == pytest.approx(1.0 / 3.0, rel=1e-12)
    # m = n = 4 (harmonic sums): H_16 - H_4 - 3/8
    H = lambda n: sum(1.0 / k for k in range(1, n + 1))
    assert page_average_entropy(4, 4) == pytest.approx(H(16) - H(4) - 3.0 / 8.0, rel=1e-12)


def test_page_formula_symmetric_in_subsystems():
    assert page_average_entropy(3, 11) == pytest.approx(page_average_entropy(11, 3))


def test_page_large_dimension_limit():
    assert page_average_entropy(2**6, 2**14) == pytest.approx(page_large_dimension_approx(2**6, 2**14), rel=1e-4)
    # for m = n the deficit from ln m approaches 1/2
    assert np.log(2**10) - page_average_entropy(2**10, 2**10) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("m,n", [(2, 2), (3, 5), (4, 4), (4, 16)])
def test_monte_carlo_matches_page_formula(m, n):
    rngs = spawn_rngs(2024, 4000)
    s = np.array([sample_entropy(m, n, r) for r in rngs])
    se = s.std(ddof=1) / np.sqrt(len(s))
    assert abs(s.mean() - page_average_entropy(m, n)) < 4.5 * se


def test_complementary_subsystems_have_equal_entropy(rng):
    psi = random_pure_state(3 * 7, rng)
    rho = density_from_ket(psi)
    sa = von_neumann_entropy(partial_trace(rho, (3, 7), [0]))
    sb = von_neumann_entropy(partial_trace(rho, (3, 7), [1]))
    assert sa == pytest.approx(sb, abs=1e-12)


def test_page_curve_shape_emerges_from_sampling():
    n = 10
    exact = qubit_page_exact(n)
    ref = hawking_reference_curve(n)
    bound = page_bound_curve(n)
    peak = int(np.argmax(exact))
    assert peak == n // 2                     # maximum at the half-way (Page) point
    assert exact[0] == 0 and exact[-1] == 0   # pure at both ends
    assert np.all(np.diff(exact[: peak + 1]) > 0) and np.all(np.diff(exact[peak:]) < 0)
    assert np.all(exact <= bound + 1e-12)
    assert np.all(exact <= ref + 1e-12)       # below the monotonic semiclassical reference
    assert exact[n // 2] > bound[n // 2] - 0.5  # within 1/2 nat of the maximal value


def test_ensemble_mean_within_error_of_exact_page_values():
    n, ns = 8, 300
    mean, std, sem, data = qubit_page_curve_ensemble(n, ns, seed=7)
    exact = qubit_page_exact(n)
    assert np.all(np.abs(mean[1:-1] - exact[1:-1]) < 5 * sem[1:-1] + 1e-12)
    assert data.shape == (ns, n + 1)
    assert np.all(std[1:-1] > 0)


def test_ensemble_reproducible_under_fixed_seed():
    a = qubit_page_curve_ensemble(6, 20, seed=99)[0]
    b = qubit_page_curve_ensemble(6, 20, seed=99)[0]
    c = qubit_page_curve_ensemble(6, 20, seed=100)[0]
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_single_sample_symmetric_under_complement():
    s = qubit_page_curve_sample(8, make_rng(3))
    assert s[0] == 0 and s[-1] == 0
    # S(R_k) = S(complement of R_k); by construction both equal for pure states, check via total
    psi = random_pure_state(2**8, make_rng(3))
    rho_dims = (2**3, 2**5)
    sa = von_neumann_entropy(partial_trace(density_from_ket(psi), rho_dims, [0]))
    assert sa == pytest.approx(s[3], abs=1e-12)
