import numpy as np
import pytest

from bhstatmech.entanglement import (
    concurrence_two_qubit, entanglement_energies, entanglement_spectrum, log_negativity,
    mutual_information, negativity, partial_transpose, subsystem_entropy,
)
from bhstatmech.entanglement import horizon
from bhstatmech.quantum_states import (
    K0, K1, bell_state, classically_correlated_state, density_from_ket, partial_trace,
    product_state, random_density_matrix, random_pure_state, separable_mixture, tensor,
    von_neumann_entropy, werner_state,
)

LN2 = np.log(2.0)


def test_mutual_information_zero_for_product_states(rng):
    a, b = random_density_matrix(3, seed=rng), random_density_matrix(4, seed=rng)
    assert mutual_information(np.kron(a, b), (3, 4)) == pytest.approx(0.0, abs=1e-10)
    pure = density_from_ket(product_state(K0, (K0 + K1) / np.sqrt(2)))
    assert mutual_information(pure, (2, 2)) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_bell_is_twice_ln2():
    assert mutual_information(density_from_ket(bell_state()), (2, 2)) == pytest.approx(2 * LN2)


def test_pure_state_mutual_information_is_twice_entropy(rng):
    psi = random_pure_state(12, rng)
    rho = density_from_ket(psi)
    s = von_neumann_entropy(partial_trace(rho, (3, 4), [0]))
    assert mutual_information(rho, (3, 4)) == pytest.approx(2 * s, rel=1e-9)


def test_correlation_without_entanglement():
    rho = classically_correlated_state()
    assert mutual_information(rho, (2, 2)) == pytest.approx(LN2)
    assert negativity(rho, (2, 2)) == pytest.approx(0.0, abs=1e-12)
    assert concurrence_two_qubit(rho) == pytest.approx(0.0, abs=1e-9)


def test_negativity_bell_and_werner_threshold():
    assert negativity(density_from_ket(bell_state()), (2, 2)) == pytest.approx(0.5)
    assert log_negativity(density_from_ket(bell_state()), (2, 2)) == pytest.approx(1.0)
    assert negativity(werner_state(0.5), (2, 2)) > 0
    assert negativity(werner_state(0.3), (2, 2)) == pytest.approx(0.0, abs=1e-12)
    # exact threshold p = 1/3
    assert negativity(werner_state(1 / 3 + 1e-3), (2, 2)) > 0
    assert negativity(werner_state(1 / 3 - 1e-3), (2, 2)) == pytest.approx(0.0, abs=1e-12)


def test_negativity_zero_for_separable_mixtures():
    for p in (0.0, 0.3, 1.0):
        assert negativity(separable_mixture(p), (2, 2)) == pytest.approx(0.0, abs=1e-12)


def test_concurrence_bell_and_werner():
    assert concurrence_two_qubit(density_from_ket(bell_state())) == pytest.approx(1.0, abs=1e-9)
    # Werner: C = max(0, (3p - 1)/2)
    for p in (0.2, 0.6, 0.9):
        assert concurrence_two_qubit(werner_state(p)) == pytest.approx(max(0.0, (3 * p - 1) / 2), abs=1e-9)


def test_partial_transpose_involution_and_trace(rng):
    rho = random_density_matrix(6, seed=rng)
    pt = partial_transpose(rho, (2, 3), 1)
    assert np.trace(pt).real == pytest.approx(1.0)
    assert np.allclose(partial_transpose(pt, (2, 3), 1), rho)
    # transposing A or B gives matrices related by a full transpose
    assert np.allclose(partial_transpose(rho, (2, 3), 0), partial_transpose(rho, (2, 3), 1).T)


def test_entanglement_spectrum_normalised_and_sorted(rng):
    psi = random_pure_state(4 * 8, rng)
    sp = entanglement_spectrum(psi, (4, 8), [0])
    assert sp.sum() == pytest.approx(1.0)
    assert np.all(np.diff(sp) <= 1e-15)
    assert len(sp) == 4
    en = entanglement_energies(np.array([0.5, 0.5, 0.0]))
    assert np.allclose(en, LN2)
    assert subsystem_entropy(psi, (4, 8), [0]) == pytest.approx(subsystem_entropy(psi, (4, 8), [1]), rel=1e-9)


# ---- horizon models ---------------------------------------------------------------


def test_horizon_pair_state_reduced_and_global_entropies():
    psi = horizon.horizon_pair_state()
    rho = density_from_ket(psi)
    r_in = partial_trace(rho, (2, 2), [0])
    r_out = partial_trace(rho, (2, 2), [1])
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-12)
    assert von_neumann_entropy(r_in) == pytest.approx(LN2)
    assert von_neumann_entropy(r_out) == pytest.approx(LN2)
    assert mutual_information(rho, (2, 2)) == pytest.approx(2 * LN2)
    assert np.allclose(r_out, np.eye(2) / 2)  # tracing out the interior leaves a mixed state


def test_thermofield_pair_reproduces_bose_occupation_and_entropy():
    x = 1.5
    d = 80
    psi = horizon.thermofield_pair_state(x, d)
    rho_out = partial_trace(density_from_ket(psi), (d, d), [1])
    nbar = np.real(np.sum(np.arange(d) * np.diag(rho_out)))
    assert nbar == pytest.approx(1.0 / np.expm1(x), rel=1e-9)
    assert von_neumann_entropy(rho_out) == pytest.approx(horizon.bose_mode_entropy(x), rel=1e-9)
    # truncation convergence: the entropy error shrinks monotonically with the cutoff d
    errs = []
    for dd in (4, 8, 16, 32):
        rho_d = partial_trace(density_from_ket(horizon.thermofield_pair_state(x, dd)), (dd, dd), [1])
        errs.append(abs(von_neumann_entropy(rho_d) - horizon.bose_mode_entropy(x)))
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-10


def test_pair_emission_entropies_follow_expected_structure():
    n = 5
    k, s_rad, s_int, mi = horizon.pair_model_entropies(n)
    assert np.allclose(s_rad, k * LN2)         # each emitted mode is maximally mixed
    assert np.allclose(s_int, n * LN2)         # interior is n maximally mixed partners
    assert np.allclose(mi, 2 * k * LN2)        # pairwise correlations accumulate
    # unequal entanglement: smaller entropy per pair
    _, s_rad2, _, _ = horizon.pair_model_entropies(4, theta=0.3)
    assert s_rad2[-1] < 4 * LN2


def test_pair_model_complement_shortcut_matches_direct_evaluation():
    # S(R_k u interior) from the complement must equal the direct 2^(k+N)-dimensional evaluation (small N)
    from bhstatmech.quantum_states import reduced_density_from_ket

    n = 3
    psi = horizon.pair_emission_state(n, 0.5)
    dims = (2,) * (2 * n)
    k, s_r, s_i, mi = horizon.pair_model_entropies(n, 0.5)
    for kk in range(n + 1):
        direct = von_neumann_entropy(reduced_density_from_ket(psi, dims, list(range(kk)) + list(range(n, 2 * n))))
        assert s_r[kk] + s_i[kk] - mi[kk] == pytest.approx(direct, abs=1e-12)
