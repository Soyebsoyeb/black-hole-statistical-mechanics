import numpy as np
import pytest

from bhstatmech.evaporation import simulate_nonunitary_evaporation, simulate_unitary_evaporation
from bhstatmech.exceptions import PhysicalParameterError, QuantumStateError
from bhstatmech.quantum_states import (
    density_from_ket, evolve_density, evolve_ket, is_unitary, partial_trace, random_hermitian,
    random_pure_state, random_unitary, unitary_from_hamiltonian, von_neumann_entropy,
)


def test_hamiltonian_evolution_is_unitary_and_norm_preserving(rng):
    H = random_hermitian(8, rng)
    for t in (0.1, 1.0, 7.3):
        U = unitary_from_hamiltonian(H, t)
        assert is_unitary(U)
        psi = random_pure_state(8, rng)
        assert np.linalg.norm(evolve_ket(psi, U)) == pytest.approx(1.0, abs=1e-12)
    # group property
    U1, U2 = unitary_from_hamiltonian(H, 0.4), unitary_from_hamiltonian(H, 0.9)
    assert np.allclose(U1 @ U2, unitary_from_hamiltonian(H, 1.3), atol=1e-12)


def test_evolution_preserves_global_entropy_but_changes_local(rng):
    H = random_hermitian(16, rng)
    U = unitary_from_hamiltonian(H, 2.0)
    rho = density_from_ket(np.eye(16)[0].astype(complex))
    rho_t = evolve_density(rho, U)
    assert von_neumann_entropy(rho_t) == pytest.approx(0.0, abs=1e-10)
    assert von_neumann_entropy(partial_trace(rho_t, (4, 4), [0])) > 0.1


def test_nonhermitian_generator_rejected():
    with pytest.raises(QuantumStateError):
        unitary_from_hamiltonian(np.array([[0, 1], [0, 0]], dtype=complex), 1.0)


N = 8


@pytest.fixture(scope="module")
def uni():
    return simulate_unitary_evaporation(N, seed=5, keep_spectra=True)


def test_unitary_evaporation_conserves_norm_and_purity(uni):
    assert uni.norm_deviation < 1e-12
    assert np.all(uni.s_global == 0.0)
    assert np.all(uni.purity_global == 1.0)


def test_unitary_evaporation_complementary_entropies_equal(uni):
    assert np.array_equal(uni.s_radiation, uni.s_black_hole)
    assert uni.s_radiation[0] == 0.0 and uni.s_radiation[-1] == 0.0
    assert np.allclose(uni.mutual_information, 2 * uni.s_radiation)


def test_unitary_evaporation_obeys_dimension_bound(uni):
    k = uni.steps
    bound = np.minimum(k, N - k) * np.log(2.0)
    assert np.all(uni.s_radiation <= bound + 1e-9)
    assert np.all(uni.s_radiation >= 0)


def test_unitary_evaporation_spectra_normalised(uni):
    assert len(uni.spectra) == N
    assert all(abs(s.sum() - 1.0) < 1e-10 for s in uni.spectra)


def test_unitary_evaporation_reproducible_by_seed():
    a = simulate_unitary_evaporation(7, seed=11)
    b = simulate_unitary_evaporation(7, seed=11)
    c = simulate_unitary_evaporation(7, seed=12)
    assert np.array_equal(a.s_radiation, b.s_radiation)
    assert not np.allclose(a.s_radiation, c.s_radiation)


def test_unitary_average_over_seeds_rises_then_falls():
    runs = np.array([simulate_unitary_evaporation(8, seed=s).s_radiation for s in range(40)])
    mean = runs.mean(axis=0)
    assert np.argmax(mean) == 4
    assert mean[4] > mean[1] and mean[4] > mean[7]
    assert mean[-1] == 0.0


def test_nonunitary_reference_properties():
    res = simulate_nonunitary_evaporation(6, seed=3)
    k = res.steps
    assert np.allclose(res.s_radiation, k * np.log(2.0))   # monotonic growth, no turnover
    assert np.all(np.diff(res.s_radiation) > 0)
    assert np.allclose(res.mutual_information, 0.0, atol=1e-12)  # radiation uncorrelated with the remnant
    assert res.purity_global[-1] == pytest.approx(2.0 ** (-6))   # purity lost
    assert res.s_global[-1] == pytest.approx(6 * np.log(2.0), rel=1e-9)  # final state: R maximally mixed


def test_nonunitary_vs_unitary_final_state():
    uni = simulate_unitary_evaporation(6, seed=3)
    non = simulate_nonunitary_evaporation(6, seed=3)
    assert uni.s_radiation[-1] == 0.0 < non.s_radiation[-1]


def test_invalid_sizes():
    with pytest.raises(PhysicalParameterError):
        simulate_unitary_evaporation(1, seed=0)
    with pytest.raises(PhysicalParameterError):
        simulate_nonunitary_evaporation(1, seed=0)


def test_random_unitary_reproducible():
    assert np.array_equal(random_unitary(5, 3), random_unitary(5, 3))
