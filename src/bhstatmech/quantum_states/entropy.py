"""Von Neumann and Renyi entropies (numerically stable, eigenvalue based).

``S(rho) = -k_B Tr(rho ln rho)``; zero eigenvalues contribute their limiting
value ``lim_{p->0} p ln p = 0``.  Entropies are returned in units of k_B
(dimensionless, natural log) unless ``physical=True``, which multiplies by the
SI Boltzmann constant.

Independent implementations are provided for cross-checking:

* :func:`von_neumann_entropy` : eigenvalues of rho,
* :func:`entropy_from_schmidt` : singular values of a pure bipartite state,
* :func:`entropy_matrix_log` : ``-Tr(rho logm(rho))`` (full-rank states only).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm

from .. import constants as const
from ..exceptions import QuantumStateError
from .core import TOL

_EIG_TOL = 1e-10


def spectrum(rho: np.ndarray, tol: float = _EIG_TOL) -> np.ndarray:
    """Eigenvalues of ``rho`` (descending), checked for positivity and unit trace."""
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    if w.min() < -tol:
        raise QuantumStateError(f"negative eigenvalue {w.min():.3e}: not a valid density matrix")
    if abs(w.sum() - 1.0) > 1e-8:
        raise QuantumStateError(f"trace {w.sum():.12f} != 1")
    return np.clip(w, 0.0, None)[::-1]


def entropy_from_probabilities(p: np.ndarray, physical: bool = False) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    s = float(-np.sum(p * np.log(p)))
    return s * const.K_B if physical else s


def von_neumann_entropy(rho: np.ndarray, physical: bool = False) -> float:
    return entropy_from_probabilities(spectrum(rho), physical)


def entropy_from_schmidt(psi: np.ndarray, dim_a: int, dim_b: int, physical: bool = False) -> float:
    s = np.linalg.svd(psi.reshape(dim_a, dim_b), compute_uv=False)
    return entropy_from_probabilities(s**2 / np.sum(s**2), physical)


def entropy_matrix_log(rho: np.ndarray) -> float:
    """-Tr(rho logm(rho)); valid only for strictly positive rho (independent check)."""
    w = spectrum(rho)
    if w.min() <= 1e-12:
        raise QuantumStateError("matrix-log entropy requires a full-rank state")
    return float(-np.real(np.trace(rho @ logm(rho))))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def renyi_entropy(rho: np.ndarray, n: float, physical: bool = False) -> float:
    """S_n = ln Tr(rho^n) / (1 - n); n -> 1 gives von Neumann, n = inf gives -ln lambda_max."""
    p = spectrum(rho)
    return renyi_from_probabilities(p, n, physical)


def renyi_from_probabilities(p: np.ndarray, n: float, physical: bool = False) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if n < 0:
        raise ValueError("Renyi index must be non-negative")
    if abs(n - 1.0) < 1e-14:
        return entropy_from_probabilities(p, physical)
    if np.isinf(n):
        s = -np.log(p.max())
    elif n == 0:
        s = np.log(p.size)
    else:
        # log-sum-exp for stability: ln sum p^n
        s = float(np.log(np.sum(np.exp(n * np.log(p)))) / (1.0 - n))
    return s * const.K_B if physical else float(s)


def mutual_information(rho_ab: np.ndarray, dims: tuple[int, int]) -> float:
    """I(A:B) = S(A) + S(B) - S(AB)."""
    from .core import partial_trace

    ra = partial_trace(rho_ab, dims, [0])
    rb = partial_trace(rho_ab, dims, [1])
    return von_neumann_entropy(ra) + von_neumann_entropy(rb) - von_neumann_entropy(rho_ab)
