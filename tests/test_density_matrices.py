import numpy as np
import pytest

from bhstatmech.exceptions import QuantumStateError
from bhstatmech.quantum_states import (
    K0, K1, bell_state, check_density_matrix, check_ket, classically_correlated_state,
    density_from_ket, entropy_from_probabilities, entropy_from_schmidt, entropy_matrix_log,
    ghz_state, is_hermitian, is_positive_semidefinite, normalize, partial_trace,
    partially_entangled_state, product_state, purity, random_density_matrix, random_hermitian,
    random_pure_state, random_unitary, reduced_density_from_ket, renyi_entropy, separable_mixture,
    spectrum, tensor, thermal_state, von_neumann_entropy, werner_state, SZ,
)
from bhstatmech import constants as c

LN2 = np.log(2.0)


@pytest.mark.parametrize("which", ["phi+", "phi-", "psi+", "psi-"])
def test_bell_states_global_pure_local_maximally_mixed(which):
    psi = bell_state(which)
    check_ket(psi)
    rho = density_from_ket(psi)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-12)
    for keep in ([0], [1]):
        r = partial_trace(rho, (2, 2), keep)
        assert von_neumann_entropy(r) == pytest.approx(LN2, abs=1e-12)
        assert np.allclose(r, np.eye(2) / 2)


def test_physical_units_option():
    r = partial_trace(density_from_ket(bell_state()), (2, 2), [0])
    assert von_neumann_entropy(r, physical=True) == pytest.approx(c.K_B * LN2, rel=1e-12)


def test_partial_trace_preserves_trace_and_positivity(rng):
    rho = random_density_matrix(12, seed=rng)
    for dims, keep in [((3, 4), [0]), ((3, 4), [1]), ((2, 2, 3), [0, 2]), ((2, 2, 3), [1])]:
        r = partial_trace(rho, dims, keep)
        assert np.trace(r).real == pytest.approx(1.0, abs=1e-12)
        check_density_matrix(r)


def test_partial_trace_matches_pure_state_shortcut(rng):
    psi = random_pure_state(2 * 3 * 4, rng)
    dims = (2, 3, 4)
    for keep in ([0], [1], [2], [0, 2], [1, 2]):
        a = partial_trace(density_from_ket(psi), dims, keep)
        b = reduced_density_from_ket(psi, dims, keep)
        assert np.allclose(a, b, atol=1e-12)


def test_partial_trace_of_product_state_returns_factor():
    a, b = random_density_matrix(2, seed=1), random_density_matrix(3, seed=2)
    rho = np.kron(a, b)
    assert np.allclose(partial_trace(rho, (2, 3), [0]), a)
    assert np.allclose(partial_trace(rho, (2, 3), [1]), b)


def test_zero_entropy_for_pure_states(rng):
    for d in (2, 5, 16):
        rho = density_from_ket(random_pure_state(d, rng))
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-10)
        assert purity(rho) == pytest.approx(1.0, abs=1e-12)


def test_product_state_entropies():
    psi = product_state(K0, (K0 + K1) / np.sqrt(2))
    rho = density_from_ket(psi)
    for keep in ([0], [1]):
        assert von_neumann_entropy(partial_trace(rho, (2, 2), keep)) == pytest.approx(0.0, abs=1e-12)


def test_partially_entangled_entropy_formula():
    theta = 0.4
    p = np.cos(theta) ** 2
    expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    psi = partially_entangled_state(theta)
    r = reduced_density_from_ket(psi, (2, 2), [0])
    assert von_neumann_entropy(r) == pytest.approx(expected, rel=1e-12)


def test_ghz_reduced_state_is_classical_mixture():
    psi = ghz_state(4)
    r = reduced_density_from_ket(psi, (2,) * 4, [0])
    assert np.allclose(r, np.eye(2) / 2)
    assert von_neumann_entropy(reduced_density_from_ket(psi, (2,) * 4, [0, 1])) == pytest.approx(LN2)


def test_independent_entropy_implementations_agree(rng):
    # eigenvalue-based, Schmidt/SVD-based, and matrix-logarithm-based
    # use the smaller factor (5) so the reduced state is full rank (matrix log is defined)
    psi = random_pure_state(5 * 6, rng)
    r = reduced_density_from_ket(psi, (5, 6), [0])
    s_eig = von_neumann_entropy(r)
    assert s_eig == pytest.approx(entropy_from_schmidt(psi, 5, 6), rel=1e-12)
    assert s_eig == pytest.approx(entropy_matrix_log(r), rel=1e-9)
    rho = random_density_matrix(7, seed=3)
    assert von_neumann_entropy(rho) == pytest.approx(entropy_matrix_log(rho), rel=1e-9)


def test_entropy_bounds(rng):
    for d in (2, 4, 9):
        rho = random_density_matrix(d, seed=rng)
        s = von_neumann_entropy(rho)
        assert 0.0 <= s <= np.log(d) + 1e-12
    assert von_neumann_entropy(np.eye(5) / 5) == pytest.approx(np.log(5))


def test_renyi_limits_and_ordering():
    rho = random_density_matrix(6, seed=11)
    s1 = von_neumann_entropy(rho)
    assert renyi_entropy(rho, 1.0) == pytest.approx(s1)
    assert renyi_entropy(rho, 1.0 + 1e-7) == pytest.approx(s1, rel=1e-5)
    assert renyi_entropy(rho, 0.0) == pytest.approx(np.log(np.linalg.matrix_rank(rho)))
    assert renyi_entropy(rho, 2.0) == pytest.approx(-np.log(purity(rho)), rel=1e-12)
    ns = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
    vals = [renyi_entropy(rho, n) for n in ns]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))  # non-increasing in n
    assert vals[-1] == pytest.approx(-np.log(spectrum(rho)[0]))


def test_zero_eigenvalues_handled():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    assert entropy_from_probabilities(p) == pytest.approx(LN2)
    rho = np.diag(p).astype(complex)
    assert von_neumann_entropy(rho) == pytest.approx(LN2)
    with pytest.raises(QuantumStateError):
        entropy_matrix_log(rho)


def test_random_ensembles_valid(rng):
    for ens in ("hilbert-schmidt", "bures"):
        rho = random_density_matrix(8, ensemble=ens, seed=rng)
        check_density_matrix(rho)
    low_rank = random_density_matrix(8, rank=3, seed=rng)
    assert np.linalg.matrix_rank(low_rank, tol=1e-10) == 3
    H = random_hermitian(6, rng)
    assert is_hermitian(H)
    with pytest.raises(ValueError):
        random_density_matrix(4, ensemble="nope")


def test_random_unitary_is_unitary_and_haar_like(rng):
    U = random_unitary(16, rng)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-12)
    # |U_00|^2 has mean 1/d for Haar unitaries
    vals = [abs(random_unitary(8, rng)[0, 0]) ** 2 for _ in range(600)]
    assert np.mean(vals) == pytest.approx(1 / 8, rel=0.12)


def test_thermal_state_matches_boltzmann():
    H = np.diag([0.0, 1.0, 2.0]).astype(complex)
    rho = thermal_state(H, 0.7)
    p = np.exp(-0.7 * np.array([0, 1, 2]))
    assert np.allclose(np.diag(rho).real, p / p.sum())
    # very large beta*E must not overflow
    assert np.isfinite(thermal_state(H * 1e4, 1.0)).all()
    assert von_neumann_entropy(thermal_state(H, 0.0)) == pytest.approx(np.log(3))


def test_validation_errors_are_informative():
    with pytest.raises(QuantumStateError, match="not normalised"):
        check_ket(np.array([1.0, 1.0], dtype=complex))
    with pytest.raises(QuantumStateError, match="trace"):
        check_density_matrix(np.eye(2, dtype=complex))
    with pytest.raises(QuantumStateError, match="Hermitian"):
        check_density_matrix(np.array([[0.5, 1j], [0.5j, 0.5]]))
    with pytest.raises(QuantumStateError, match="negative"):
        check_density_matrix(np.array([[1.5, 0], [0, -0.5]], dtype=complex))
    with pytest.raises(QuantumStateError):
        normalize(np.zeros(3))
    with pytest.raises(QuantumStateError):
        partial_trace(np.eye(4) / 4, (2, 2), [2])
    with pytest.raises(QuantumStateError):
        spectrum(np.array([[1.5, 0], [0, -0.5]], dtype=complex))


def test_hermiticity_and_positivity_helpers():
    assert is_positive_semidefinite(np.eye(3))
    assert not is_positive_semidefinite(-np.eye(3))
    assert is_hermitian(SZ) and not is_hermitian(np.array([[0, 1], [0, 0]]))
