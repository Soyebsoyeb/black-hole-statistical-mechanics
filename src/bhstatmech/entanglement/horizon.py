"""Toy models of entanglement across a horizon.

These are *finite-dimensional models*, not quantum-field calculations near a
horizon.

1. ``horizon_pair_state``: |Psi> = (|0>_in|0>_out + |1>_in|1>_out)/sqrt(2)
   generalised to ``cos(theta)|00> + sin(theta)|11>``.
2. ``thermofield_pair_state``: truncated two-mode squeezed state

       |Psi> = sqrt(1 - x^2) sum_{n=0}^{d-1} x^n |n>_in |n>_out,
       x^2 = exp(-hbar omega / k_B T_H),

   which is the structure produced by a Bogoliubov transformation between
   infalling and outgoing modes: tracing out the interior gives a thermal
   (Bose) distribution for the outgoing mode with ``<n> = 1/(exp(hbar omega/kT)-1)``.
   The state is normalised after truncation; convergence in ``d`` is checked
   in tests.
3. ``pair_emission_state``: ``N`` independent pairs; radiation = first ``k``
   outgoing modes, interior = all interior partners.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import PhysicalParameterError
from ..quantum_states.canonical import partially_entangled_state
from ..quantum_states.core import reduced_density_from_ket, tensor
from ..quantum_states.entropy import von_neumann_entropy


def horizon_pair_state(theta: float = np.pi / 4) -> np.ndarray:
    """Two-qubit pair state; subsystem order (in, out)."""
    return partially_entangled_state(theta)


def thermofield_pair_state(x_hw_over_kT: float, d: int) -> np.ndarray:
    """Truncated two-mode squeezed state with ``x = hbar omega / (k_B T_H)``; order (in, out)."""
    if x_hw_over_kT <= 0:
        raise PhysicalParameterError("hbar*omega/(k_B T) must be positive")
    n = np.arange(d)
    amp = np.exp(-0.5 * x_hw_over_kT * n)  # x^n with x^2 = exp(-hw/kT)
    amp /= np.linalg.norm(amp)
    psi = np.zeros(d * d, dtype=complex)
    psi[n * d + n] = amp
    return psi


def bose_mode_entropy(x_hw_over_kT: float) -> float:
    """Entropy (nats) of a thermal bosonic mode: (n+1) ln(n+1) - n ln n, n = 1/(e^x - 1)."""
    nbar = 1.0 / np.expm1(x_hw_over_kT)
    return float((nbar + 1.0) * np.log1p(nbar) - nbar * np.log(nbar))


def pair_emission_state(n_pairs: int, theta: float = np.pi / 4) -> np.ndarray:
    """N pair states; qubit ordering ``out_1..out_N, in_1..in_N``."""
    pair = partially_entangled_state(theta).reshape(2, 2)  # [in, out]
    # build amplitude tensor with index order (out_1..out_N, in_1..in_N)
    psi = np.ones((), dtype=complex)
    for _ in range(n_pairs):
        psi = np.multiply.outer(psi, pair)  # axes ... (in_i, out_i)
    # axes currently (in_1, out_1, in_2, out_2, ...)
    outs = [2 * i + 1 for i in range(n_pairs)]
    ins = [2 * i for i in range(n_pairs)]
    return np.transpose(psi, outs + ins).reshape(-1)


def pair_model_entropies(n_pairs: int, theta: float = np.pi / 4):
    """Return ``k, S_rad(k), S_interior_partners(k), I(R_k : interior)`` for k = 0..N.

    Radiation R_k is the first k outgoing modes.  Interior I is all N interior
    partners.  The global state is pure, so I(R:I) = S(R) + S(I) - S(RI), and
    S(RI) equals the entropy of the complement (the N-k not-yet-emitted outgoing
    modes), which is evaluated instead of the huge ``2^(N+k)``-dimensional
    reduced state.
    """
    psi = pair_emission_state(n_pairs, theta)
    dims = (2,) * (2 * n_pairs)
    ks, s_r, s_in, mi = [], [], [], []
    interior = list(range(n_pairs, 2 * n_pairs))
    for k in range(n_pairs + 1):
        rad = list(range(k))
        s_rad = von_neumann_entropy(reduced_density_from_ket(psi, dims, rad)) if k else 0.0
        s_int = von_neumann_entropy(reduced_density_from_ket(psi, dims, interior))
        rest = list(range(k, n_pairs))  # complement of R_k and the interior
        s_both = von_neumann_entropy(reduced_density_from_ket(psi, dims, rest)) if rest else 0.0
        ks.append(k)
        s_r.append(s_rad)
        s_in.append(s_int)
        mi.append(s_rad + s_int - s_both)
    return np.array(ks), np.array(s_r), np.array(s_in), np.array(mi)
