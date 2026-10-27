"""Entanglement entropy of a free scalar field on a 1D lattice (coupled harmonic chain).

Hamiltonian (hbar = 1)::

    H = sum_i [ p_i^2 / 2 ] + (1/2) x^T K x,       K = m^2 I + L / a^2,

where ``x_i = sqrt(a) phi_i`` and ``p_i = sqrt(a) pi_i`` are canonically
normalised lattice-field variables (``[x_i, p_j] = i delta_ij``), ``a`` is the
lattice spacing, ``m`` the mass (an infrared regulator), and ``L`` is the graph
Laplacian of the chain with boundary condition:

* ``'periodic'`` : ring of N sites,
* ``'dirichlet'``: x_0 = x_{N+1} = 0 (fixed ends),
* ``'neumann'``  : free ends (zero mode lifted by ``m > 0``).

Ground state: Gaussian with covariance ``<x x> = (1/2) K^{-1/2}``,
``<p p> = (1/2) K^{1/2}``, ``<x p + p x> = 0``.

Entanglement entropy of a block A: ``S = sum_k s(nu_k)`` with
``s(nu) = (nu + 1/2) ln(nu + 1/2) - (nu - 1/2) ln(nu - 1/2)`` and
``nu_k^2`` the eigenvalues of ``X_A P_A`` (symplectic eigenvalues, ``nu >= 1/2``;
``nu = 1/2`` contributes zero).

UV divergence: in the continuum, S diverges like ``(c/3) ln(l/a)`` (c = 1) for an
interval with two endpoints in 1+1 dimensions; the lattice spacing ``a`` is the
regulator.  The lattice result must not be identified with black-hole entropy
without a derivation.

Complexity: building K^{+-1/2} costs O(N^3) time, O(N^2) memory.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import PhysicalParameterError


def laplacian(n: int, boundary: str) -> np.ndarray:
    if n < 2:
        raise PhysicalParameterError("need at least 2 sites")
    L = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    if boundary == "dirichlet":
        return L
    if boundary == "neumann":
        L[0, 0] = L[-1, -1] = 1.0
        return L
    if boundary == "periodic":
        L[0, -1] = L[-1, 0] = -1.0
        return L
    raise PhysicalParameterError(f"unknown boundary condition {boundary!r}")


def stiffness_matrix(n: int, a: float, m: float, boundary: str = "periodic") -> np.ndarray:
    if a <= 0 or m < 0:
        raise PhysicalParameterError("lattice spacing must be positive and mass non-negative")
    K = m**2 * np.eye(n) + laplacian(n, boundary) / a**2
    if boundary in ("periodic", "neumann") and m == 0:
        raise PhysicalParameterError("zero mode: periodic/neumann chains need m > 0")
    return K


def ground_state_covariance(n: int, a: float = 1.0, m: float = 1e-3, boundary: str = "periodic"):
    """Return ``(X, P)``: ground-state position and momentum covariance matrices."""
    K = stiffness_matrix(n, a, m, boundary)
    w2, V = np.linalg.eigh(K)
    if w2.min() <= 0:
        raise PhysicalParameterError("stiffness matrix is not positive definite")
    w = np.sqrt(w2)
    X = 0.5 * (V / w) @ V.T
    P = 0.5 * (V * w) @ V.T
    return X, P


def symplectic_eigenvalues(X_A: np.ndarray, P_A: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvals(X_A @ P_A)
    nu = np.sqrt(np.clip(ev.real, 0.25, None))  # nu >= 1/2 for physical states
    return np.sort(nu)


def entropy_from_symplectic(nu: np.ndarray) -> float:
    nu = np.asarray(nu, dtype=float)
    hi = (nu + 0.5) * np.log(nu + 0.5)
    d = nu - 0.5
    lo = np.where(d > 1e-14, d * np.log(np.where(d > 1e-14, d, 1.0)), 0.0)
    return float(np.sum(hi - lo))


def block_entropy(X: np.ndarray, P: np.ndarray, sites) -> float:
    idx = np.asarray(sites)
    return entropy_from_symplectic(symplectic_eigenvalues(X[np.ix_(idx, idx)], P[np.ix_(idx, idx)]))


def entropy_vs_block_size(n: int, a: float, m: float, boundary: str, sizes, start: int = 0) -> np.ndarray:
    X, P = ground_state_covariance(n, a, m, boundary)
    return np.array([block_entropy(X, P, range(start, start + int(l))) for l in sizes])


def two_oscillator_entropy_exact(omega0: float, k: float) -> float:
    """Analytic entropy of one site of two coupled oscillators.

    H = (p1^2 + p2^2)/2 + omega0^2 (x1^2 + x2^2)/2 + k (x1 - x2)^2 / 2.
    Normal modes: omega_+ = omega0, omega_- = sqrt(omega0^2 + 2k); then
    <x1^2> = (1/ omega_+ + 1/omega_-)/4, <p1^2> = (omega_+ + omega_-)/4.
    """
    wp, wm = omega0, np.sqrt(omega0**2 + 2.0 * k)
    x2 = 0.25 * (1.0 / wp + 1.0 / wm)
    p2 = 0.25 * (wp + wm)
    return entropy_from_symplectic(np.array([np.sqrt(x2 * p2)]))
