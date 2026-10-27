"""First-law and Smarr checks for charged, rotating and AdS black holes (Planck units).

Each function returns the relative residual of the first law evaluated with
central finite differences, i.e. numerical differentiation of ``M(S, Q)`` or
``M(S, J)`` against the analytically known conjugate potentials.
"""

from __future__ import annotations

import numpy as np

from ..spacetimes.natural import Kerr, ReissnerNordstrom, SchwarzschildAdS


def rn_first_law_residual(r_plus: float, Q: float, h: float = 1e-6) -> float:
    """Check ``dM = T dS + Phi dQ`` along a random direction in (r+, Q) space."""
    d_rp, d_Q = 0.6, 0.8  # unit direction
    M_of = lambda rp, q: ReissnerNordstrom.mass_from(rp, q)

    def bh(rp, q):
        return ReissnerNordstrom(M_of(rp, q), q)

    plus, minus = bh(r_plus + h * d_rp, Q + h * d_Q), bh(r_plus - h * d_rp, Q - h * d_Q)
    dM = plus.M - minus.M
    dS = plus.entropy - minus.entropy
    dQ = plus.Q - minus.Q
    ref = bh(r_plus, Q)
    rhs = ref.temperature * dS + ref.electric_potential * dQ
    return abs(dM - rhs) / abs(dM)


def kerr_first_law_residual(M: float, a: float, h: float = 1e-6) -> float:
    """Check ``dM = T dS + Omega dJ`` for a displacement in (M, J)."""
    J = M * a
    dM_, dJ_ = 0.6 * h, 0.8 * h

    def bh(m, j):
        return Kerr(m, j / m)

    plus, minus = bh(M + dM_, J + dJ_), bh(M - dM_, J - dJ_)
    dM = plus.M - minus.M
    dS = plus.entropy - minus.entropy
    dJ = plus.J - minus.J
    ref = bh(M, J)
    rhs = ref.temperature * dS + ref.horizon_angular_velocity * dJ
    return abs(dM - rhs) / abs(dM)


def ads_first_law_residual(r_plus: float, L: float, h: float = 1e-6) -> float:
    """Check ``dM = T dS`` at fixed L."""
    plus, minus = SchwarzschildAdS(r_plus + h, L), SchwarzschildAdS(r_plus - h, L)
    dM = plus.mass - minus.mass
    dS = plus.entropy - minus.entropy
    return abs(dM - SchwarzschildAdS(r_plus, L).temperature * dS) / abs(dM)


def rn_smarr_residual(bh: ReissnerNordstrom) -> float:
    """M - 2 T S - Phi Q (Smarr relation for RN)."""
    return bh.M - 2.0 * bh.temperature * bh.entropy - bh.electric_potential * bh.Q
