"""Replica identity and Renyi entropies for finite-dimensional states.

``S(rho) = -d/dn ln Tr(rho^n) |_{n=1}``.  For a finite-dimensional density
matrix ``Tr(rho^n) = sum_i lambda_i^n`` is an entire function of ``n``, so the
analytic continuation from integer to real ``n`` is trivial *given the
spectrum*.  In gravity one only has the integer-n replica path integrals and
must continue the (unknown) function from integers; that step is non-trivial
and is *not* captured by this module.  ``extrapolate_from_integers`` shows how
fragile continuation from integer data alone is, even in finite dimension.
"""

from __future__ import annotations

import numpy as np

from ..quantum_states.entropy import spectrum, von_neumann_entropy


def renyi_moment(p: np.ndarray, n: float) -> float:
    """Tr(rho^n) from eigenvalues ``p``."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(np.sum(np.exp(n * np.log(p))))


def log_moment(p: np.ndarray, n: float) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(np.log(np.sum(np.exp(n * np.log(p)))))


def replica_entropy_fd(p: np.ndarray, h: float = 1e-4) -> float:
    """-d/dn ln Tr rho^n at n=1 by central finite difference of step ``h``."""
    return -(log_moment(p, 1.0 + h) - log_moment(p, 1.0 - h)) / (2.0 * h)


def replica_entropy_richardson(p: np.ndarray, h: float = 1e-2) -> float:
    d1 = replica_entropy_fd(p, h)
    d2 = replica_entropy_fd(p, h / 2.0)
    return (4.0 * d2 - d1) / 3.0


def replica_entropy_error_scan(rho: np.ndarray, steps) -> np.ndarray:
    """Absolute error of the replica finite-difference entropy against the direct value."""
    p = spectrum(rho)
    exact = von_neumann_entropy(rho)
    return np.array([abs(replica_entropy_fd(p, h) - exact) for h in steps])


def extrapolate_from_integers(p: np.ndarray, n_max: int) -> float:
    """Estimate S from integer replica data alone.

    Fit the polynomial of degree ``n_max - 1`` through ``(n, ln Tr rho^n)`` for
    ``n = 1..n_max`` (``n = 1`` gives 0) and differentiate at ``n = 1``.
    """
    ns = np.arange(1, n_max + 1, dtype=float)
    ys = np.array([log_moment(p, n) for n in ns])
    coeffs = np.polynomial.polynomial.polyfit(ns - 1.0, ys, n_max - 1)
    return -float(coeffs[1])
