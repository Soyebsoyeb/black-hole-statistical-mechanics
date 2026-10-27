"""Entanglement and correlation measures for finite-dimensional bipartite states.

Limitations (see docs/quantum_entanglement.md)
----------------------------------------------
* Von Neumann entropy of a reduced state quantifies entanglement only for
  *pure* global states.
* Mutual information counts all (classical + quantum) correlations; the
  classically correlated state has I = ln 2 and zero entanglement.
* Negativity is a necessary-but-not-sufficient entanglement criterion in
  general (it misses bound entanglement); it is exact for 2x2 and 2x3 systems.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..exceptions import QuantumStateError
from ..quantum_states.core import partial_trace
from ..quantum_states.entropy import (
    entropy_from_probabilities, mutual_information, spectrum, von_neumann_entropy,
)

__all__ = [
    "mutual_information", "partial_transpose", "negativity", "log_negativity",
    "concurrence_two_qubit", "entanglement_spectrum", "entanglement_energies", "subsystem_entropy",
]


def partial_transpose(rho: np.ndarray, dims: Sequence[int], sys: int) -> np.ndarray:
    """Transpose subsystem ``sys`` of ``rho``."""
    dims = tuple(dims)
    n = len(dims)
    if not 0 <= sys < n:
        raise QuantumStateError("invalid subsystem index")
    t = rho.reshape(dims + dims)
    t = np.swapaxes(t, sys, sys + n)
    d = int(np.prod(dims))
    return t.reshape(d, d)


def negativity(rho: np.ndarray, dims: Sequence[int], sys: int = 1) -> float:
    """N = (||rho^{T_B}||_1 - 1)/2 = sum of |negative eigenvalues| of the partial transpose."""
    w = np.linalg.eigvalsh(partial_transpose(rho, dims, sys))
    return float(-w[w < 0].sum())


def log_negativity(rho: np.ndarray, dims: Sequence[int], sys: int = 1) -> float:
    return float(np.log2(2.0 * negativity(rho, dims, sys) + 1.0))


def concurrence_two_qubit(rho: np.ndarray) -> float:
    """Wootters concurrence of a two-qubit state."""
    if rho.shape != (4, 4):
        raise QuantumStateError("concurrence is implemented for two qubits only")
    sy = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(sy, sy)
    rho_tilde = YY @ rho.conj() @ YY
    ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(rho @ rho_tilde)))[::-1], 0.0, None))
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def entanglement_spectrum(psi: np.ndarray, dims: Sequence[int], keep: Sequence[int]) -> np.ndarray:
    """Eigenvalues (descending, summing to 1) of the reduced density matrix of a pure state."""
    from ..quantum_states.core import reduced_density_from_ket

    return spectrum(reduced_density_from_ket(psi, dims, keep))


def entanglement_energies(spec: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    """-ln(lambda_i) for the non-zero entanglement-spectrum eigenvalues."""
    s = np.asarray(spec, dtype=float)
    return -np.log(s[s > floor])


def subsystem_entropy(psi: np.ndarray, dims: Sequence[int], keep: Sequence[int]) -> float:
    from ..quantum_states.core import reduced_density_from_ket

    return von_neumann_entropy(reduced_density_from_ket(psi, dims, keep))
