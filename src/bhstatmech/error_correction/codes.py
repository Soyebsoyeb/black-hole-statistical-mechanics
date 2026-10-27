"""Three-qubit bit-flip repetition code and the five-qubit perfect code.

Three-qubit code: ``|0_L> = |000>``, ``|1_L> = |111>``; corrects any single
X (bit-flip) error.  Syndromes ``(Z0 Z1, Z1 Z2)`` identify the flipped qubit.

Five-qubit code (Laflamme-Miquel-Paz-Zurek; Bennett et al.): stabilisers are
cyclic shifts of ``XZZXI``; it corrects an arbitrary single-qubit error.
Implemented here as a stabiliser projector construction with explicit
syndrome measurement by projection and recovery by the Pauli lookup table.

Relation to holography (conceptual only): in AdS/CFT the bulk operator
algebra of a region can be reconstructed from several different boundary
subregions (Almheiri-Dong-Harlow).  Redundant encoding of a logical qubit
into physical qubits, where erasure of a small subset leaves the logical
information recoverable, is a toy analogue of that statement.  It does not
derive holography.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..quantum_states.canonical import I2, SX, SY, SZ
from ..quantum_states.core import tensor

PAULI = {"I": I2, "X": SX, "Y": SY, "Z": SZ}


@lru_cache(maxsize=None)
def _pauli_string_cached(s: str) -> np.ndarray:
    m = tensor(*[PAULI[c] for c in s])
    m.setflags(write=False)
    return m


def pauli_string(s: str) -> np.ndarray:
    """Tensor product of single-qubit Paulis, e.g. ``'XZZXI'`` (cached, read-only array)."""
    return _pauli_string_cached(s)


# ---- three-qubit repetition code -------------------------------------------------


def encode_repetition(alpha: complex, beta: complex) -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    psi[0], psi[7] = alpha, beta
    return psi


_REP_SYNDROME = {(+1, +1): None, (-1, +1): 0, (-1, -1): 1, (+1, -1): 2}


def repetition_syndrome(psi: np.ndarray) -> tuple[int, int]:
    """Deterministic syndrome for a state within a single error subspace."""
    s = []
    for stab in ("ZZI", "IZZ"):
        op = pauli_string(stab)
        e = float(np.real(psi.conj() @ op @ psi))
        s.append(int(round(e)))
    return tuple(s)  # type: ignore[return-value]


def repetition_recover(psi: np.ndarray) -> np.ndarray:
    q = _REP_SYNDROME[repetition_syndrome(psi)]
    if q is None:
        return psi
    err = ["I"] * 3
    err[q] = "X"
    return pauli_string("".join(err)) @ psi


def repetition_decode(psi: np.ndarray) -> tuple[complex, complex]:
    return psi[0], psi[7]


# ---- five-qubit code ----------------------------------------------------------------

STABILIZERS_5 = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


def _five_qubit_projector() -> np.ndarray:
    P = np.eye(32, dtype=complex)
    for s in STABILIZERS_5:
        P = P @ (np.eye(32) + pauli_string(s)) / 2.0
    return P


_X_L = pauli_string("XXXXX")
_Z_L = pauli_string("ZZZZZ")


def five_qubit_codewords() -> tuple[np.ndarray, np.ndarray]:
    P = _five_qubit_projector()
    v = np.zeros(32, dtype=complex)
    v[0] = 1.0
    zero = P @ v
    zero /= np.linalg.norm(zero)
    one = _X_L @ zero
    return zero, one


def encode_five_qubit(alpha: complex, beta: complex) -> np.ndarray:
    z, o = five_qubit_codewords()
    return alpha * z + beta * o


@lru_cache(maxsize=None)
def _syndrome_table_5() -> dict:
    table = {}
    for q in range(5):
        for p in "XYZ":
            e = ["I"] * 5
            e[q] = p
            E = pauli_string("".join(e))
            syn = tuple(+1 if np.allclose(pauli_string(s) @ E, E @ pauli_string(s)) else -1 for s in STABILIZERS_5)
            table["".join(e)] = syn
    return table


def five_qubit_recover(psi: np.ndarray) -> np.ndarray:
    """Measure the stabilisers (deterministic after a single-qubit error) and undo the error."""
    syn = tuple(int(round(float(np.real(psi.conj() @ pauli_string(s) @ psi)))) for s in STABILIZERS_5)
    if syn == (1, 1, 1, 1):
        return psi
    for err, s in _syndrome_table_5().items():
        if s == syn:
            return pauli_string(err) @ psi
    raise ValueError(f"unrecognised syndrome {syn}")


def logical_fidelity_five(psi_in: tuple[complex, complex], recovered: np.ndarray) -> float:
    z, o = five_qubit_codewords()
    a = z.conj() @ recovered
    b = o.conj() @ recovered
    target = np.array(psi_in, dtype=complex)
    return float(abs(np.vdot(target, np.array([a, b]))) ** 2)


# ---- erasure / mutual-information diagnostics -----------------------------------------


def erasure_recoverability(codeword_state: np.ndarray, n_qubits: int, erased: list[int], ref_dim_state: np.ndarray | None = None) -> float:
    """Coherent information-style diagnostic: ``I(ref : kept)`` for a maximally entangled
    reference of the logical qubit, with ``erased`` physical qubits discarded.

    ``codeword_state`` must be a 2-column matrix of the logical basis ``[|0_L>, |1_L>]``
    (shape ``(2**n, 2)``); the reference qubit is maximally entangled with the logical qubit.
    A value of ``2 ln 2`` (nats) means the logical qubit is fully recoverable from the kept qubits.
    """
    from ..quantum_states.core import reduced_density_from_ket
    from ..quantum_states.entropy import von_neumann_entropy

    cw = codeword_state
    psi = (np.kron(np.array([1, 0]), cw[:, 0]) + np.kron(np.array([0, 1]), cw[:, 1])) / np.sqrt(2.0)
    dims = (2,) + (2,) * n_qubits
    kept = [i + 1 for i in range(n_qubits) if i not in erased]
    s_ref_kept = von_neumann_entropy(reduced_density_from_ket(psi, dims, [0] + kept))
    s_ref = von_neumann_entropy(reduced_density_from_ket(psi, dims, [0]))
    s_kept = von_neumann_entropy(reduced_density_from_ket(psi, dims, kept))
    return s_ref + s_kept - s_ref_kept
