import numpy as np
import pytest

from bhstatmech.quantum_states import (
    random_density_matrix, renyi_entropy, spectrum, von_neumann_entropy,
)
from bhstatmech.replica_methods import (
    extrapolate_from_integers, log_moment, renyi_moment, replica_entropy_error_scan,
    replica_entropy_fd, replica_entropy_richardson,
)


@pytest.fixture
def rho():
    return random_density_matrix(6, seed=1)


def test_moment_at_n_one_is_trace(rho):
    p = spectrum(rho)
    assert renyi_moment(p, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert log_moment(p, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert renyi_moment(p, 2.0) == pytest.approx(np.real(np.trace(rho @ rho)))


def test_replica_derivative_matches_von_neumann(rho):
    p, ex = spectrum(rho), von_neumann_entropy(rho)
    assert replica_entropy_fd(p, 1e-4) == pytest.approx(ex, abs=1e-8)
    assert replica_entropy_richardson(p, 1e-2) == pytest.approx(ex, abs=1e-9)


def test_replica_error_scales_quadratically_with_step_then_floors(rho):
    errs = replica_entropy_error_scan(rho, [1e-1, 1e-2, 1e-3])
    assert errs[0] / errs[1] == pytest.approx(100.0, rel=0.1)
    assert errs[1] / errs[2] == pytest.approx(100.0, rel=0.1)
    tiny = replica_entropy_error_scan(rho, [1e-9])[0]
    assert tiny < 1e-5


def test_replica_entropy_of_maximally_mixed_state():
    p = np.ones(8) / 8
    assert replica_entropy_fd(p, 1e-4) == pytest.approx(np.log(8), abs=1e-9)


def test_replica_entropy_pure_state_is_zero():
    p = np.array([1.0, 0.0, 0.0])
    assert replica_entropy_fd(p, 1e-4) == pytest.approx(0.0, abs=1e-12)


def test_renyi_entropy_consistent_with_moments(rho):
    p = spectrum(rho)
    for n in (0.5, 2.0, 3.0, 7.0):
        assert renyi_entropy(rho, n) == pytest.approx(log_moment(p, n) / (1 - n), rel=1e-12)


def test_integer_only_continuation_is_much_worse_than_exact_spectrum(rho):
    p, ex = spectrum(rho), von_neumann_entropy(rho)
    direct = abs(replica_entropy_fd(p, 1e-4) - ex)
    for n_max in (3, 5, 8):
        assert abs(extrapolate_from_integers(p, n_max) - ex) > 1e3 * direct
    # but the error shrinks as more integer replicas are used
    e = [abs(extrapolate_from_integers(p, n) - ex) for n in (3, 5, 8)]
    assert e[0] > e[1] > e[2]
