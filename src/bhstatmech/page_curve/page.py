"""Average entanglement entropy of random pure states (Page curve).

For a Haar-random pure state on ``C^m (x) C^n`` with ``m <= n`` the average
entropy of the ``m``-dimensional factor is (Page 1993; Foong-Kanno 1994;
Sen 1996)::

    <S_{m,n}> = sum_{k=n+1}^{mn} 1/k - (m - 1)/(2 n)
              = psi(mn + 1) - psi(n + 1) - (m - 1)/(2n),

with psi the digamma function; for ``1 << m <= n`` this is ``ln m - m/(2n)``.
The expression is symmetric under ``m <-> n`` (complementary subsystems of a
pure state have equal entropy), implemented by ordering the arguments.

A *Page curve* is obtained by fixing the total dimension ``N = m n`` and
moving degrees of freedom from the complement into the subsystem.  Nothing in
this module imposes a rise-then-fall shape: it follows from sampling.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma

from ..exceptions import PhysicalParameterError
from ..numerical_methods.rng import make_rng, spawn_rngs
from ..quantum_states.entropy import entropy_from_probabilities
from ..quantum_states.random_states import random_pure_state


def page_average_entropy(m: int, n: int) -> float:
    """Exact Haar-average entropy (nats) of one factor of an ``m x n`` bipartite pure state."""
    if m < 1 or n < 1:
        raise PhysicalParameterError("dimensions must be positive")
    if m > n:
        m, n = n, m
    return float(digamma(m * n + 1.0) - digamma(n + 1.0) - (m - 1.0) / (2.0 * n))


def page_large_dimension_approx(m: int, n: int) -> float:
    """ln m - m/(2n) for m <= n."""
    if m > n:
        m, n = n, m
    return float(np.log(m) - m / (2.0 * n))


def sample_entropy(m: int, n: int, rng) -> float:
    psi = random_pure_state(m * n, rng)
    s = np.linalg.svd(psi.reshape(m, n), compute_uv=False)
    return entropy_from_probabilities(s**2)


def qubit_page_curve_sample(n_qubits: int, rng) -> np.ndarray:
    """``S(R_k)`` for k = 0..N for one Haar-random pure state of N qubits (R_k = first k qubits)."""
    psi = random_pure_state(2**n_qubits, rng)
    out = np.zeros(n_qubits + 1)
    for k in range(1, n_qubits):
        s = np.linalg.svd(psi.reshape(2**k, 2 ** (n_qubits - k)), compute_uv=False)
        out[k] = entropy_from_probabilities(s**2)
    return out


def qubit_page_curve_ensemble(n_qubits: int, n_samples: int, seed: int):
    """Mean, sample standard deviation and standard error of the mean over ``n_samples`` states."""
    rngs = spawn_rngs(seed, n_samples)
    data = np.array([qubit_page_curve_sample(n_qubits, r) for r in rngs])
    return data.mean(axis=0), data.std(axis=0, ddof=1), data.std(axis=0, ddof=1) / np.sqrt(n_samples), data


def qubit_page_exact(n_qubits: int) -> np.ndarray:
    """Exact Haar average of S(R_k) for k = 0..N using Page's formula."""
    return np.array([0.0] + [page_average_entropy(2**k, 2 ** (n_qubits - k)) for k in range(1, n_qubits)] + [0.0])


def hawking_reference_curve(n_qubits: int) -> np.ndarray:
    """Semiclassical reference ``S_rad(k) = k ln 2`` (monotonic, no information return)."""
    return np.arange(n_qubits + 1) * np.log(2.0)


def page_bound_curve(n_qubits: int) -> np.ndarray:
    """``min(k, N-k) ln 2``: the maximal entropy allowed by subsystem dimensions."""
    k = np.arange(n_qubits + 1)
    return np.minimum(k, n_qubits - k) * np.log(2.0)
