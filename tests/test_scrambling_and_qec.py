import numpy as np
import pytest

from bhstatmech.error_correction import (
    STABILIZERS_5, encode_five_qubit, encode_repetition, erasure_recoverability,
    five_qubit_codewords, five_qubit_recover, logical_fidelity_five, pauli_string,
    repetition_decode, repetition_recover, repetition_syndrome,
)
from bhstatmech.page_curve import page_average_entropy
from bhstatmech.quantum_states import is_unitary
from bhstatmech.scrambling import (
    circuit_unitary, entanglement_growth, otoc, otoc_vs_depth, pauli_z_on,
)

A, B = 0.6, 0.8j


# ---- repetition code -----------------------------------------------------------------


@pytest.mark.parametrize("err", ["III", "XII", "IXI", "IIX"])
def test_repetition_code_corrects_single_bit_flips(err):
    psi = encode_repetition(A, B)
    corrupted = pauli_string(err) @ psi
    fixed = repetition_recover(corrupted)
    assert np.allclose(repetition_decode(fixed), (A, B))


def test_repetition_syndromes_distinguish_locations():
    psi = encode_repetition(A, B)
    syn = {q: repetition_syndrome(pauli_string("".join("X" if i == q else "I" for i in range(3))) @ psi) for q in range(3)}
    assert len(set(syn.values())) == 3
    assert repetition_syndrome(psi) == (1, 1)


def test_repetition_code_fails_on_phase_errors():
    psi = encode_repetition(A, B)
    out = repetition_recover(pauli_string("ZII") @ psi)
    assert not np.allclose(repetition_decode(out), (A, B))  # bit-flip code does not protect phase


def test_repetition_code_fails_on_two_flips():
    psi = encode_repetition(A, B)
    out = repetition_recover(pauli_string("XXI") @ psi)
    assert not np.allclose(repetition_decode(out), (A, B))


# ---- five-qubit code -----------------------------------------------------------------


def test_five_qubit_codewords_orthonormal_and_stabilised():
    z, o = five_qubit_codewords()
    assert np.linalg.norm(z) == pytest.approx(1.0) and np.linalg.norm(o) == pytest.approx(1.0)
    assert abs(np.vdot(z, o)) < 1e-12
    for s in STABILIZERS_5:
        S = pauli_string(s)
        assert np.allclose(S @ z, z) and np.allclose(S @ o, o)


@pytest.mark.parametrize("qubit", range(5))
@pytest.mark.parametrize("pauli", "XYZ")
def test_five_qubit_code_corrects_every_single_qubit_pauli(qubit, pauli):
    psi = encode_five_qubit(A, B)
    err = ["I"] * 5
    err[qubit] = pauli
    recovered = five_qubit_recover(pauli_string("".join(err)) @ psi)
    assert logical_fidelity_five((A, B), recovered) == pytest.approx(1.0, abs=1e-10)
    assert abs(np.vdot(psi, recovered)) == pytest.approx(1.0, abs=1e-10)


def test_five_qubit_code_fails_on_weight_two_errors():
    psi = encode_five_qubit(A, B)
    bad = pauli_string("XXIII") @ psi
    try:
        recovered = five_qubit_recover(bad)
    except ValueError:
        return
    assert abs(np.vdot(psi, recovered)) < 1 - 1e-6


def test_erasure_recoverability_of_five_qubit_code():
    z, o = five_qubit_codewords()
    cw = np.stack([z, o], axis=1)
    full = 2 * np.log(2.0)
    assert erasure_recoverability(cw, 5, []) == pytest.approx(full, abs=1e-9)
    # any two erased qubits are recoverable ([[5,1,3]] corrects two erasures)
    for pair in ([0, 1], [1, 3], [2, 4]):
        assert erasure_recoverability(cw, 5, pair) == pytest.approx(full, abs=1e-9)
    # three erased qubits destroy part of the information
    assert erasure_recoverability(cw, 5, [0, 1, 2]) < full - 0.1


# ---- scrambling ---------------------------------------------------------------------


def test_circuit_unitary_is_unitary_and_reproducible():
    U = circuit_unitary(5, 4, seed=3)
    assert is_unitary(U)
    assert np.array_equal(U, circuit_unitary(5, 4, seed=3))


def test_otoc_identity_and_commuting_operators():
    n = 4
    U = np.eye(2**n, dtype=complex)
    assert otoc(U, pauli_z_on(0, n), pauli_z_on(3, n)) == pytest.approx(1.0)
    # Z operators commute with each other, so F = 1 under any diagonal unitary
    D = np.diag(np.exp(1j * np.random.default_rng(0).uniform(0, 6, 2**n)))
    assert otoc(D, pauli_z_on(0, n), pauli_z_on(3, n)) == pytest.approx(1.0)


def test_otoc_light_cone_and_decay():
    n, dist = 6, 4
    mean, sem = otoc_vs_depth(n, dist, depth_max=8, n_realizations=24, seed=17)
    assert mean[0] == pytest.approx(1.0)
    # no signal outside the light cone: depth d < dist cannot connect sites 0 and dist
    assert np.all(mean[: dist - 1] > 1 - 1e-9)
    assert mean[-1] < 0.35
    assert np.all(sem >= 0)


def test_entanglement_growth_saturates_near_page_value():
    n = 8
    # open-chain brickwork needs depth well beyond n to saturate (depth 10 is still ~0.24 nats short)
    mean, sem, page = entanglement_growth(n, depth_max=24, n_realizations=20, seed=5)
    assert mean[0] == 0.0
    assert np.all(np.diff(mean[:5]) > 0)  # initial growth
    assert mean[10] < page - 0.1          # not yet saturated at depth 10
    assert abs(mean[-1] - page) < max(5 * sem[-1], 0.03)
    assert page == pytest.approx(page_average_entropy(16, 16))
    assert np.all(mean <= 4 * np.log(2) + 1e-9)
