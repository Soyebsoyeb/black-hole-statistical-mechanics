"""State vectors, density matrices, tensor products, partial traces, evolution.

Conventions
-----------
* Subsystems are ordered left to right; subsystem 0 is the most significant
  index of the computational basis (``kron`` ordering).
* ``dims`` is the tuple of local Hilbert-space dimensions.
* All invariants (normalisation, Hermiticity, positivity) are checked with an
  explicit tolerance and violations raise
  :class:`~bhstatmech.exceptions.QuantumStateError`; nothing is silently
  corrected.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np

from ..exceptions import QuantumStateError

TOL = 1e-10


# ---- construction ------------------------------------------------------------


def basis_ket(dim: int, index: int) -> np.ndarray:
    if not 0 <= index < dim:
        raise QuantumStateError(f"basis index {index} out of range for dimension {dim}")
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def normalize(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n == 0:
        raise QuantumStateError("cannot normalise the zero vector")
    return psi / n


def tensor(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of vectors or matrices."""
    return reduce(np.kron, ops)


def density_from_ket(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


# ---- validation ----------------------------------------------------------------


def check_ket(psi: np.ndarray, tol: float = TOL) -> None:
    if psi.ndim != 1:
        raise QuantumStateError("state vector must be one-dimensional")
    if abs(np.linalg.norm(psi) - 1.0) > tol:
        raise QuantumStateError(f"state vector not normalised: norm = {np.linalg.norm(psi)!r}")


def is_hermitian(rho: np.ndarray, tol: float = TOL) -> bool:
    return bool(np.allclose(rho, rho.conj().T, atol=tol))


def is_positive_semidefinite(rho: np.ndarray, tol: float = TOL) -> bool:
    return bool(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() >= -tol)


def check_density_matrix(rho: np.ndarray, tol: float = TOL) -> None:
    """Raise unless ``rho`` is square, Hermitian, positive semidefinite, unit trace."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise QuantumStateError("density matrix must be square")
    if not is_hermitian(rho, tol):
        raise QuantumStateError("density matrix is not Hermitian")
    if abs(np.trace(rho) - 1.0) > tol:
        raise QuantumStateError(f"density matrix trace is {np.trace(rho)!r}, expected 1")
    if not is_positive_semidefinite(rho, tol):
        raise QuantumStateError("density matrix has negative eigenvalues")


def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol))


# ---- partial traces ---------------------------------------------------------------


def _check_dims(dims: Sequence[int], keep: Sequence[int]) -> tuple[tuple[int, ...], list[int]]:
    dims = tuple(int(d) for d in dims)
    keep = sorted(set(int(k) for k in keep))
    if any(k < 0 or k >= len(dims) for k in keep):
        raise QuantumStateError(f"subsystem indices {keep} invalid for {len(dims)} subsystems")
    return dims, keep


def partial_trace(rho: np.ndarray, dims: Sequence[int], keep: Sequence[int]) -> np.ndarray:
    """Reduced density matrix on subsystems ``keep`` (all others are traced out)."""
    dims, keep = _check_dims(dims, keep)
    n = len(dims)
    if rho.shape != (int(np.prod(dims)),) * 2:
        raise QuantumStateError("density-matrix shape does not match dims")
    t = rho.reshape(dims + dims)
    cur = n
    for idx in sorted((i for i in range(n) if i not in keep), reverse=True):
        t = np.trace(t, axis1=idx, axis2=idx + cur)
        cur -= 1
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return t.reshape(d_keep, d_keep)


def reduced_density_from_ket(psi: np.ndarray, dims: Sequence[int], keep: Sequence[int]) -> np.ndarray:
    """Reduced density matrix of a pure state without forming the full ``|psi><psi|``."""
    dims, keep = _check_dims(dims, keep)
    if psi.size != int(np.prod(dims)):
        raise QuantumStateError("state-vector length does not match dims")
    tr = [i for i in range(len(dims)) if i not in keep]
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    M = np.transpose(psi.reshape(dims), keep + tr).reshape(d_keep, -1)
    return M @ M.conj().T


def schmidt_coefficients(psi: np.ndarray, dim_a: int, dim_b: int) -> np.ndarray:
    """Singular values of the coefficient matrix (Schmidt coefficients, descending)."""
    return np.linalg.svd(psi.reshape(dim_a, dim_b), compute_uv=False)


# ---- evolution -----------------------------------------------------------------------


def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:
    """exp(-i H t) via eigendecomposition (stable for Hermitian H)."""
    if not is_hermitian(H):
        raise QuantumStateError("Hamiltonian must be Hermitian")
    w, V = np.linalg.eigh((H + H.conj().T) / 2)
    return (V * np.exp(-1j * w * t)) @ V.conj().T


def evolve_ket(psi: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ psi


def evolve_density(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T
