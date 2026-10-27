"""Schwarzschild spacetime in SI units.

Metric (signature -+++)::

    ds^2 = -f(r) c^2 dt^2 + dr^2 / f(r) + r^2 dOmega^2,   f = 1 - r_s/r,
    r_s = 2GM/c^2.

Curvature invariant (Kretschmann scalar)::

    K = 48 G^2 M^2 / (c^4 r^6) = 12 r_s^2 / r^6   [m^-4]

which is finite at r = r_s (coordinate singularity) and diverges only at r = 0.

The coordinate expressions g_rr = 1/f diverge at the horizon.  This module
returns ``nan`` there rather than ``inf`` and provides the ingoing
Eddington-Finkelstein form, which is regular on the future horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as const
from ..exceptions import PhysicalParameterError


@dataclass(frozen=True)
class Schwarzschild:
    """Non-rotating, uncharged black hole of mass ``mass_kg``."""

    mass_kg: float

    def __post_init__(self) -> None:
        m = self.mass_kg
        if not (np.isfinite(m) and m > 0.0):
            raise PhysicalParameterError(f"mass must be positive and finite, got {m!r}")

    # ---- horizon quantities -------------------------------------------------
    @property
    def schwarzschild_radius(self) -> float:
        """r_s = 2GM/c^2 [m]."""
        return 2.0 * const.G * self.mass_kg / const.C**2

    @property
    def horizon_radius(self) -> float:
        """Event-horizon location (equals r_s) [m]."""
        return self.schwarzschild_radius

    @property
    def surface_gravity(self) -> float:
        """kappa = c^4 / (4GM) [m s^-2] (acceleration measured with Killing time t)."""
        return const.C**4 / (4.0 * const.G * self.mass_kg)

    @property
    def hawking_temperature(self) -> float:
        """T_H = hbar kappa / (2 pi c k_B) = hbar c^3 / (8 pi G M k_B) [K]."""
        return const.HBAR * self.surface_gravity / (2.0 * np.pi * const.C * const.K_B)

    @property
    def horizon_area(self) -> float:
        """A = 4 pi r_s^2 [m^2]."""
        return 4.0 * np.pi * self.schwarzschild_radius**2

    @property
    def bekenstein_hawking_entropy(self) -> float:
        """S_BH = k_B c^3 A / (4 G hbar) [J/K]."""
        return const.K_B * const.C**3 * self.horizon_area / (4.0 * const.G * const.HBAR)

    # ---- metric --------------------------------------------------------------
    def f(self, r):
        """Metric function 1 - r_s/r (valid for r > 0)."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise PhysicalParameterError("radius must be positive")
        return 1.0 - self.schwarzschild_radius / r

    def metric_components(self, r, theta=np.pi / 2):
        """Diagonal components (g_tt, g_rr, g_thth, g_phph) in Schwarzschild coordinates.

        ``g_rr`` is ``nan`` at the coordinate singularity r = r_s.
        """
        r = np.asarray(r, dtype=float)
        f = self.f(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_rr = np.where(f == 0.0, np.nan, 1.0 / f)
        return (-f * const.C**2, g_rr, r**2, r**2 * np.sin(theta) ** 2)

    def inverse_metric_components(self, r, theta=np.pi / 2):
        r = np.asarray(r, dtype=float)
        f = self.f(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_tt_inv = np.where(f == 0.0, np.nan, -1.0 / (f * const.C**2))
        return (g_tt_inv, f, 1.0 / r**2, 1.0 / (r**2 * np.sin(theta) ** 2))

    def ingoing_ef_metric(self, r, theta=np.pi / 2):
        """Ingoing Eddington-Finkelstein metric in (v, r, theta, phi), regular at r_s.

        ds^2 = -f c^2 dv^2 + 2 c dv dr + r^2 dOmega^2 with v = t + r*/c.
        Returns the 2x2 (v, r) block ``[[g_vv, g_vr], [g_vr, 0]]`` as an array
        of shape ``(..., 2, 2)`` plus the angular components.
        """
        r = np.asarray(r, dtype=float)
        f = self.f(r)
        block = np.zeros(r.shape + (2, 2))
        block[..., 0, 0] = -f * const.C**2
        block[..., 0, 1] = const.C
        block[..., 1, 0] = const.C
        return block, r**2, r**2 * np.sin(theta) ** 2

    # ---- curvature ------------------------------------------------------------
    def kretschmann(self, r):
        """K = 48 G^2 M^2 / (c^4 r^6) [m^-4]."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise PhysicalParameterError("radius must be positive")
        return 48.0 * const.G**2 * self.mass_kg**2 / (const.C**4 * r**6)

    # ---- tortoise coordinate and null rays -----------------------------------
    def tortoise(self, r):
        """r* = r + r_s ln|r/r_s - 1|, defined for r != r_s.

        Evaluated as ``r + r_s * log(|r - r_s| / r_s)`` to avoid forming the
        divergent quantity 1/f.  It tends to -infinity logarithmically at the
        horizon.
        """
        r = np.asarray(r, dtype=float)
        rs = self.schwarzschild_radius
        with np.errstate(divide="ignore"):
            return r + rs * np.log(np.abs(r - rs) / rs)

    def radius_from_tortoise(self, rstar):
        """Invert r*(r) on the exterior r > r_s (Newton iteration in ln(r/r_s - 1))."""
        rs = self.schwarzschild_radius
        y = np.atleast_1d(np.asarray(rstar, dtype=float)) / rs - 1.0  # solve x + ln x = y, x = r/rs - 1
        s = np.where(y > 1.0, np.log(np.maximum(y, 1.0)), y)  # s = ln x, start on the convex right side
        for _ in range(100):
            h = np.exp(s) + s - y
            s_new = s - h / (np.exp(s) + 1.0)
            if np.all(np.abs(s_new - s) < 1e-15 * (1 + np.abs(s))):
                s = s_new
                break
            s = s_new
        out = rs * (1.0 + np.exp(s))
        return out if np.ndim(rstar) else float(out[0])

    def radial_null_time(self, r, outgoing: bool = True, t0: float = 0.0):
        """Coordinate time along a radial null ray: t = t0 +/- r*/c."""
        sign = 1.0 if outgoing else -1.0
        return t0 + sign * self.tortoise(r) / const.C

    # ---- perturbation potentials -----------------------------------------------
    def effective_potential(self, r, ell: int, spin: int = 0):
        """Regge-Wheeler-type potential V(r) [m^-2] for massless spin-s fields.

        The radial equation is ``d^2 psi/dr*^2 + (omega^2/c^2 - V) psi = 0`` with

            V = f [ ell(ell+1)/r^2 + (1 - s^2) r_s / r^3 ],   s = 0, 1, 2

        (s = 0 scalar, s = 1 electromagnetic, s = 2 axial gravitational).
        """
        if spin not in (0, 1, 2):
            raise PhysicalParameterError("spin must be 0, 1 or 2")
        if ell < max(spin, 0):
            raise PhysicalParameterError(f"ell must be >= spin ({spin}), got {ell}")
        r = np.asarray(r, dtype=float)
        rs = self.schwarzschild_radius
        return self.f(r) * (ell * (ell + 1) / r**2 + (1 - spin**2) * rs / r**3)


# ---------------------------------------------------------------------------
# Symbolic verification
# ---------------------------------------------------------------------------


def kretschmann_symbolic():
    """Derive K for the Schwarzschild metric with SymPy (geometrized units G=c=1).

    Returns the simplified expression, expected to equal ``48 M**2 / r**6``.
    """
    import sympy as sp

    t, r, th, ph = sp.symbols("t r theta phi", positive=True)
    M = sp.symbols("M", positive=True)
    x = [t, r, th, ph]
    f = 1 - 2 * M / r
    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)
    ginv = g.inv()
    n = 4
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                Gam[a][b][c] = sp.simplify(
                    sum(
                        ginv[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d]))
                        for d in range(n)
                    )
                    / 2
                )

    def riemann_up(a, b, c, d):
        expr = sp.diff(Gam[a][b][d], x[c]) - sp.diff(Gam[a][b][c], x[d])
        expr += sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n))
        return sp.simplify(expr)

    K = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    Rup = riemann_up(a, b, c, d)
                    if Rup == 0:
                        continue
                    Rdown = g[a, a] * Rup  # metric is diagonal
                    # R^{abcd} R_{abcd} = R_{abcd}^2 g^aa g^bb g^cc g^dd
                    K += Rdown**2 * ginv[a, a] * ginv[b, b] * ginv[c, c] * ginv[d, d]
    return sp.simplify(K), (M, r)
