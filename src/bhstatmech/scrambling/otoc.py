"""Out-of-time-order correlators and entanglement growth in random circuits.

Conventions
-----------
For a unitary ``U`` and Pauli operators ``W``, ``V`` the infinite-temperature
OTOC is::

    F(U) = (1/d) Tr[ W(t)^dagger V^dagger W(t) V ],   W(t) = U^dagger W U,   d = 2^N,

with ``F = 1`` for commuting operators and ``F -> 0`` for a fully scrambled
(Haar-random) unitary acting on a large system (``<F> = O(1/d^2)``).

Circuit model: brickwork of Haar-random two-qubit gates on an open chain.
One *layer* applies gates on pairs (0,1),(2,3),... then on (1,2),(3,4),....
The OTOC between a single-site ``Z`` at site 0 and a ``Z`` at distance ``x`` shows
the light-cone: ``F = 1`` until the front reaches ``x`` (roughly ``x`` layers) and
then decays.  This is a small-system numerical diagnostic; it does **not** establish a
universal black-hole scrambling time (``t* ~ (1/2 pi T) ln S``).
"""

from __future__ import annotations

import numpy as np

from ..exceptions import PhysicalParameterError
from ..numerical_methods.rng import make_rng
from ..quantum_states.canonical import I2, SZ
from ..quantum_states.entropy import entropy_from_probabilities
from ..quantum_states.random_states import random_unitary


def _apply_two_qubit_gate(state: np.ndarray, gate: np.ndarray, i: int, n: int) -> np.ndarray:
    """Apply a 4x4 gate to qubits (i, i+1) of an ``n``-qubit tensor (any trailing axes kept)."""
    sh = state.shape
    s = state.reshape(2**i, 4, -1)
    s = np.einsum("ab,ibj->iaj", gate, s)
    return s.reshape(sh)


def brickwork_layer(state: np.ndarray, n: int, parity: int, rng) -> np.ndarray:
    for i in range(parity, n - 1, 2):
        state = _apply_two_qubit_gate(state, random_unitary(4, rng), i, n)
    return state


def circuit_unitary(n: int, depth: int, seed) -> np.ndarray:
    """Dense unitary of ``depth`` brickwork layers (alternating parity)."""
    if n < 2:
        raise PhysicalParameterError("need at least 2 qubits")
    rng = make_rng(seed)
    U = np.eye(2**n, dtype=complex)
    for layer in range(depth):
        U = brickwork_layer(U, n, layer % 2, rng)
    return U


def pauli_z_on(site: int, n: int) -> np.ndarray:
    ops = [I2] * n
    ops[site] = SZ
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def otoc(U: np.ndarray, W: np.ndarray, V: np.ndarray) -> float:
    """F = Re (1/d) Tr[ W(t)^dag V^dag W(t) V ] with W(t) = U^dag W U."""
    d = U.shape[0]
    Wt = U.conj().T @ W @ U
    return float(np.real(np.trace(Wt.conj().T @ V.conj().T @ Wt @ V)) / d)


def otoc_vs_depth(n: int, distance: int, depth_max: int, n_realizations: int, seed: int):
    """Average OTOC between Z_0 and Z_distance versus circuit depth (same circuit prefix)."""
    from ..numerical_methods.rng import spawn_rngs

    W, V = pauli_z_on(0, n), pauli_z_on(distance, n)
    out = np.zeros((n_realizations, depth_max + 1))
    for r, rng in enumerate(spawn_rngs(seed, n_realizations)):
        U = np.eye(2**n, dtype=complex)
        out[r, 0] = otoc(U, W, V)
        for layer in range(depth_max):
            U = brickwork_layer(U, n, layer % 2, rng)
            out[r, layer + 1] = otoc(U, W, V)
    return out.mean(axis=0), out.std(axis=0, ddof=1) / np.sqrt(n_realizations)


def entanglement_growth(n: int, depth_max: int, n_realizations: int, seed: int, cut: int | None = None):
    """Half-chain entanglement entropy versus depth for a product initial state |0...0>.

    Returns mean and standard error (nats), plus the Page value for the cut.
    """
    from ..numerical_methods.rng import spawn_rngs
    from ..page_curve.page import page_average_entropy

    cut = n // 2 if cut is None else cut
    out = np.zeros((n_realizations, depth_max + 1))
    for r, rng in enumerate(spawn_rngs(seed, n_realizations)):
        psi = np.zeros((2,) * n, dtype=complex)
        psi[(0,) * n] = 1.0
        for layer in range(depth_max):
            psi = brickwork_layer(psi.reshape(2**n), n, layer % 2, rng).reshape((2,) * n)
            sv = np.linalg.svd(psi.reshape(2**cut, -1), compute_uv=False)
            out[r, layer + 1] = entropy_from_probabilities(sv**2)
    page = page_average_entropy(2**cut, 2 ** (n - cut))
    return out.mean(axis=0), out.std(axis=0, ddof=1) / np.sqrt(n_realizations), page
