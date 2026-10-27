"""Random states, unitaries and density matrices with documented ensembles.

* ``random_pure_state``: Haar-random, a normalised complex Gaussian vector.
* ``random_unitary``: Haar measure on U(d) via QR of a complex Ginibre matrix
  with the phase correction of Mezzadri (2007).
* ``random_density_matrix``:
    - ``'hilbert-schmidt'``: ``rho = G G^dagger / Tr(G G^dagger)`` with ``G`` a
      ``d x k`` complex Ginibre matrix.  ``k = d`` gives the Hilbert-Schmidt
      measure; ``k < d`` gives rank-``k`` states induced by partial trace of a
      Haar-random pure state on ``C^d (x) C^k``.
    - ``'bures'``: ``rho ~ (1 + U) G G^dagger (1 + U^dagger)``, normalised, with
      ``U`` Haar-random (Bures measure).
* ``random_hermitian`` draws from the Gaussian unitary ensemble (GUE).

All functions take a ``seed`` or a ``numpy.random.Generator``.
"""

from __future__ import annotations

import numpy as np

from ..numerical_methods.rng import make_rng


def _ginibre(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    return (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0)


def random_pure_state(dim: int, seed=None) -> np.ndarray:
    rng = make_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def random_unitary(dim: int, seed=None) -> np.ndarray:
    rng = make_rng(seed)
    Q, R = np.linalg.qr(_ginibre(rng, dim, dim))
    d = np.diag(R)
    return Q * (d / np.abs(d))


def random_hermitian(dim: int, seed=None) -> np.ndarray:
    rng = make_rng(seed)
    A = _ginibre(rng, dim, dim)
    return (A + A.conj().T) / 2.0


def random_density_matrix(dim: int, rank: int | None = None, ensemble: str = "hilbert-schmidt", seed=None) -> np.ndarray:
    rng = make_rng(seed)
    if ensemble == "hilbert-schmidt":
        k = dim if rank is None else rank
        G = _ginibre(rng, dim, k)
        rho = G @ G.conj().T
    elif ensemble == "bures":
        U = random_unitary(dim, rng)
        G = _ginibre(rng, dim, dim)
        A = (np.eye(dim) + U) @ G
        rho = A @ A.conj().T
    else:
        raise ValueError(f"unknown ensemble {ensemble!r}")
    return rho / np.trace(rho).real
