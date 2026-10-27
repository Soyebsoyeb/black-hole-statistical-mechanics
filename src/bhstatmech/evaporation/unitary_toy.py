"""Finite-dimensional evaporation toy models (unitary and non-unitary).

Setting
-------
``N`` qubits in total.  After ``j`` emission steps the radiation ``R_j`` is
qubits ``0..j-1`` and the remaining black hole ``B_j`` is qubits ``j..N-1``.

Unitary model
~~~~~~~~~~~~~
Start from the pure product state ``|0...0>``.  At step ``j`` apply a
Haar-random unitary ``U_j`` to the *current black-hole register* (qubits
``j-1..N-1``, which contains the qubit about to be emitted) and then declare
qubit ``j-1`` to be radiation.  Every operation is unitary on the global
Hilbert space, so the global state stays pure and ``S(R_j) = S(B_j)``.
The rise-then-fall of ``S(R_j)`` is not imposed; it emerges from the
scrambling dynamics.

Non-unitary reference (Hawking's information-loss picture)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Same black-hole scrambling, but the emitted qubit is *replaced by a maximally
mixed qubit uncorrelated with everything else* and the original is traced out
(completely positive, trace preserving, not unitary):

    rho_{R B} -> rho_R (x) I/2 (x) Tr_q[ U rho_B U^dagger ].

Then ``S(R_j) = j ln 2`` grows monotonically, ``I(R:B) = 0`` and the global
purity decays as ``2^-j Tr(rho_B^2)``.

Complexity
~~~~~~~~~~
State vector: O(2^N) memory; Haar unitary on the black-hole register:
O(4^(N-j)) memory and O(8^(N-j)) time for the QR factorisation.  The
non-unitary model tracks only the black-hole density matrix, also O(4^(N-j)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import PhysicalParameterError
from ..numerical_methods.rng import make_rng
from ..quantum_states.core import partial_trace
from ..quantum_states.entropy import entropy_from_probabilities, von_neumann_entropy
from ..quantum_states.random_states import random_unitary


@dataclass
class EvaporationResult:
    steps: np.ndarray
    s_radiation: np.ndarray
    s_black_hole: np.ndarray
    s_global: np.ndarray
    purity_global: np.ndarray
    mutual_information: np.ndarray
    norm_deviation: float = 0.0
    spectra: list = field(default_factory=list)  # entanglement spectrum of R_j per step (unitary model)


def _apply_unitary_on_tail(psi: np.ndarray, n_qubits: int, first: int, U: np.ndarray) -> np.ndarray:
    """Apply ``U`` to qubits ``first..N-1`` of the state vector."""
    d_head = 2**first
    return (psi.reshape(d_head, -1) @ U.T).reshape(-1)


def simulate_unitary_evaporation(n_qubits: int, seed: int, scramble: bool = True, keep_spectra: bool = False) -> EvaporationResult:
    if n_qubits < 2:
        raise PhysicalParameterError("need at least two qubits")
    rng = make_rng(seed)
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0
    steps = np.arange(n_qubits + 1)
    s_r = np.zeros(n_qubits + 1)
    s_b = np.zeros(n_qubits + 1)
    max_norm_dev = 0.0
    spectra = []
    for j in range(1, n_qubits + 1):
        first = j - 1
        if scramble:
            U = random_unitary(2 ** (n_qubits - first), rng)
            psi = _apply_unitary_on_tail(psi, n_qubits, first, U)
        max_norm_dev = max(max_norm_dev, abs(np.linalg.norm(psi) - 1.0))
        if j < n_qubits:
            sv = np.linalg.svd(psi.reshape(2**j, 2 ** (n_qubits - j)), compute_uv=False)
            p = sv**2
            s_r[j] = s_b[j] = entropy_from_probabilities(p)
            if keep_spectra:
                spectra.append(np.sort(p)[::-1])
        else:
            s_r[j] = s_b[j] = 0.0
            if keep_spectra:
                spectra.append(np.array([1.0]))
    s_global = np.zeros(n_qubits + 1)  # pure at every step
    purity = np.ones(n_qubits + 1)
    mi = s_r + s_b - s_global
    return EvaporationResult(steps, s_r, s_b, s_global, purity, mi, max_norm_dev, spectra)


def simulate_nonunitary_evaporation(n_qubits: int, seed: int) -> EvaporationResult:
    """Information-loss reference; tracks only the black-hole density matrix."""
    if n_qubits < 2:
        raise PhysicalParameterError("need at least two qubits")
    rng = make_rng(seed)
    rho_b = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    rho_b[0, 0] = 1.0
    steps = np.arange(n_qubits + 1)
    s_r = np.zeros(n_qubits + 1)
    s_b = np.zeros(n_qubits + 1)
    s_g = np.zeros(n_qubits + 1)
    pur = np.ones(n_qubits + 1)
    for j in range(1, n_qubits + 1):
        n_b = n_qubits - j + 1  # qubits in the register before emission
        U = random_unitary(2**n_b, rng)
        rho_b = U @ rho_b @ U.conj().T
        # emit (trace out) the first qubit of the register
        dims = (2, 2 ** (n_b - 1))
        rho_b = partial_trace(rho_b, dims, [1])
        s_r[j] = j * np.log(2.0)
        s_b[j] = von_neumann_entropy(rho_b)
        s_g[j] = s_r[j] + s_b[j]  # R is uncorrelated with B
        pur[j] = 2.0 ** (-j) * float(np.real(np.trace(rho_b @ rho_b)))
    mi = s_r + s_b - s_g  # = 0 by construction (product state)
    return EvaporationResult(steps, s_r, s_b, s_g, pur, mi)
