"""Canonical test states."""

from __future__ import annotations

import numpy as np

from ..exceptions import QuantumStateError
from .core import basis_ket, density_from_ket, tensor

K0 = basis_ket(2, 0)
K1 = basis_ket(2, 1)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def product_state(*kets: np.ndarray) -> np.ndarray:
    return tensor(*kets)


def bell_state(which: str = "phi+") -> np.ndarray:
    s = 1.0 / np.sqrt(2.0)
    table = {
        "phi+": s * (tensor(K0, K0) + tensor(K1, K1)),
        "phi-": s * (tensor(K0, K0) - tensor(K1, K1)),
        "psi+": s * (tensor(K0, K1) + tensor(K1, K0)),
        "psi-": s * (tensor(K0, K1) - tensor(K1, K0)),
    }
    if which not in table:
        raise QuantumStateError(f"unknown Bell state {which!r}")
    return table[which]


def partially_entangled_state(theta: float) -> np.ndarray:
    """cos(theta)|00> + sin(theta)|11>; maximally entangled at theta = pi/4."""
    return np.cos(theta) * tensor(K0, K0) + np.sin(theta) * tensor(K1, K1)


def ghz_state(n: int) -> np.ndarray:
    if n < 2:
        raise QuantumStateError("GHZ state needs at least 2 qubits")
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def classically_correlated_state() -> np.ndarray:
    """(|00><00| + |11><11|)/2: perfectly correlated, separable (no entanglement)."""
    return 0.5 * (density_from_ket(tensor(K0, K0)) + density_from_ket(tensor(K1, K1)))


def separable_mixture(p: float) -> np.ndarray:
    """p |00><00| + (1-p) |+,+><+,+| (a separable mixed state)."""
    if not 0.0 <= p <= 1.0:
        raise QuantumStateError("p must lie in [0, 1]")
    plus = (K0 + K1) / np.sqrt(2.0)
    return p * density_from_ket(tensor(K0, K0)) + (1 - p) * density_from_ket(tensor(plus, plus))


def werner_state(p: float) -> np.ndarray:
    """p |phi+><phi+| + (1-p) I/4; entangled iff p > 1/3."""
    if not 0.0 <= p <= 1.0:
        raise QuantumStateError("p must lie in [0, 1]")
    return p * density_from_ket(bell_state("phi+")) + (1 - p) * np.eye(4) / 4.0


def thermal_state(H: np.ndarray, beta: float) -> np.ndarray:
    """exp(-beta H)/Z via eigendecomposition with log-sum-exp stabilisation."""
    w, V = np.linalg.eigh((H + H.conj().T) / 2)
    a = -beta * w
    a -= a.max()
    p = np.exp(a)
    p /= p.sum()
    return (V * p) @ V.conj().T
