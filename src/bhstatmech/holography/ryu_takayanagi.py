"""Ryu-Takayanagi entropy in controlled AdS geometries (units hbar = k_B = 1).

Static classical prescription::

    S_A = Area(gamma_A) / (4 G_N),

with gamma_A the minimal bulk surface homologous to the boundary region A.
This is the *classical* formula; covariant (HRT) and quantum-extremal-surface
corrections are not implemented here and are not equivalent to it.

Geometries
----------
AdS3 vacuum, Poincare slice (``ds^2 = L^2 (dz^2 + dx^2)/z^2``): geodesics are
semicircles.  For an interval of width ``l`` with UV cutoff ``eps``::

    Length = 2 L ln(l / eps),     S = (c/3) ln(l/eps),   c = 3 L / (2 G_N)  (Brown-Henneaux).

BTZ black hole at horizon radius ``r_+`` (temperature ``T = r_+ / (2 pi L^2)``),
constant-time slice.  Two candidate geodesics for an interval of boundary
angular width ``Delta phi`` on the ring of circumference ``2 pi``:

* connected: ``Length_c = 2 L ln[(2 r_inf / r_+) sinh(r_+ Delta phi / (2 L))]``
* disconnected (wraps the horizon): ``Length_d = Length_c(2 pi - Delta phi) + 2 pi r_+``.

The entropy is ``min(Length_c, Length_d)/(4 G_N)``.  The crossover (entanglement
plateau) occurs where the lengths are equal.  The numerical minimiser below
finds geodesics by direct minimisation, independent of the closed forms.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ..exceptions import PhysicalParameterError


# ---- AdS3 Poincare ---------------------------------------------------------


def ads3_vacuum_length(l: float, eps: float, L: float = 1.0) -> float:
    if not (l > 0 and eps > 0 and l > 2 * eps):
        raise PhysicalParameterError("require l > 2*eps > 0")
    return 2.0 * L * np.log(l / eps)


def ads3_entropy(l: float, eps: float, L: float = 1.0, G_N: float = 1.0) -> float:
    return ads3_vacuum_length(l, eps, L) / (4.0 * G_N)


def numeric_semicircle_length(l: float, eps: float, L: float = 1.0) -> float:
    """Direct integration of the length of the semicircle of radius l/2, cut at z = eps.

    Parametrise ``x = (l/2) cos(t), z = (l/2) sin(t)``; ``dl = L dt / sin t``.
    """
    t0 = np.arcsin(2.0 * eps / l)
    val, _ = quad(lambda t: L / np.sin(t), t0, np.pi - t0, epsabs=1e-12, epsrel=1e-12)
    return float(val)


# ---- BTZ -----------------------------------------------------------------------


def btz_connected_length(dphi: float, r_plus: float, eps_r: float, L: float = 1.0) -> float:
    """Geodesic length with radial cutoff ``r_inf = 1/eps_r`` (``eps_r`` is the UV cutoff)."""
    r_inf = 1.0 / eps_r
    return 2.0 * L * np.log((2.0 * r_inf / r_plus) * np.sinh(r_plus * dphi / (2.0 * L)))


def btz_disconnected_length(dphi: float, r_plus: float, eps_r: float, L: float = 1.0) -> float:
    return btz_connected_length(2.0 * np.pi - dphi, r_plus, eps_r, L) + 2.0 * np.pi * r_plus


def btz_entropy(dphi: float, r_plus: float, eps_r: float, L: float = 1.0, G_N: float = 1.0):
    """Minimum over the two candidate surfaces; returns (S, 'connected' | 'disconnected')."""
    c = btz_connected_length(dphi, r_plus, eps_r, L)
    d = btz_disconnected_length(dphi, r_plus, eps_r, L)
    return (min(c, d) / (4.0 * G_N), "connected" if c <= d else "disconnected")


def btz_crossover_angle(r_plus: float, L: float = 1.0) -> float:
    """Angle in (pi, 2 pi) at which the connected and horizon-wrapping candidates have equal length (the cutoff cancels)."""
    from scipy.optimize import brentq

    g = lambda dphi: (btz_connected_length(dphi, r_plus, 1e-6, L)
                      - btz_disconnected_length(dphi, r_plus, 1e-6, L))
    return float(brentq(g, np.pi, 2.0 * np.pi - 1e-6))


def btz_geodesic_numeric(dphi: float, r_plus: float, eps_r: float, L: float = 1.0) -> float:
    """Independent check: minimise the length over the turning radius using the exact integral.

    On a constant-time slice the metric is ``L^2 dr^2/(r^2 - r_+^2) + r^2 dphi^2``.
    A geodesic with turning radius ``r_t`` subtends
    ``Delta phi(r_t) = 2 int_{r_t}^{inf} L r_t dr / (r sqrt((r^2 - r_+^2)(r^2 - r_t^2)))``;
    we invert this numerically for ``r_t`` and then integrate the proper length
    ``2 int_{r_t}^{r_inf} L r dr / sqrt((r^2 - r_+^2)(r^2 - r_t^2)) * sqrt(r^2-r_t^2)/... `` as coded below.
    """
    from scipy.optimize import brentq

    r_inf = 1.0 / eps_r

    def dphi_of(rt):
        # phi change on one branch: int L r_t dr / (r sqrt((r^2 - r+^2)(r^2 - rt^2)))
        # substitute r = r_t cosh(s) to remove the endpoint singularity
        g = lambda s: L * rt / (rt * np.cosh(s) * np.sqrt(rt**2 * np.cosh(s) ** 2 - r_plus**2))
        val, _ = quad(g, 0.0, np.arccosh(r_inf / rt), epsabs=1e-10, epsrel=1e-10, limit=200)
        return 2.0 * val

    rt = brentq(lambda x: dphi_of(x) - dphi, r_plus * (1 + 1e-7), r_inf * 0.5, xtol=1e-13)
    g = lambda s: L * rt * np.cosh(s) / np.sqrt(rt**2 * np.cosh(s) ** 2 - r_plus**2)
    val, _ = quad(g, 0.0, np.arccosh(r_inf / rt), epsabs=1e-10, epsrel=1e-10, limit=200)
    return float(2.0 * val)
