"""Reissner-Nordstrom, Kerr, Schwarzschild-AdS and Schwarzschild-dS geometries.

All quantities in this module are in Planck units, G = hbar = c = k_B = 1
(Gaussian charge units for Q).  Convert with :mod:`bhstatmech.units`.  In
these units the Bekenstein-Hawking entropy is S = A/4 and T = kappa / (2 pi).

Thermodynamic ensemble conventions
----------------------------------
* RN: heat capacity at fixed charge Q, first law dM = T dS + Phi dQ.
* Kerr: heat capacity at fixed angular momentum J, dM = T dS + Omega dJ.
* SAdS (4D, fixed AdS radius L, no pressure term): dM = T dS, with the mass
  parameter M interpreted as the energy (the usual AdS/CFT boundary energy in
  4D with the standard normalisation of the timelike Killing vector).
* SdS: two horizons with different temperatures; no single global T.

Numerical conditioning
----------------------
Near extremality M^2 - Q^2 (or M^2 - a^2) suffers catastrophic cancellation,
so the square roots are evaluated as ``sqrt((M - Q) (M + Q))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..exceptions import PhysicalParameterError


def _check_positive(x: float, name: str) -> None:
    if not (np.isfinite(x) and x > 0):
        raise PhysicalParameterError(f"{name} must be positive and finite, got {x!r}")


# ---------------------------------------------------------------------------
# Reissner-Nordstrom
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReissnerNordstrom:
    """Charged black hole, ``f(r) = 1 - 2M/r + Q^2/r^2``; requires ``|Q| <= M``."""

    M: float
    Q: float

    def __post_init__(self) -> None:
        _check_positive(self.M, "M")
        if not np.isfinite(self.Q):
            raise PhysicalParameterError("Q must be finite")
        if abs(self.Q) > self.M:
            raise PhysicalParameterError(
                f"|Q|={abs(self.Q)} exceeds M={self.M}: naked singularity (no horizon)"
            )

    @property
    def _delta(self) -> float:
        return float(np.sqrt((self.M - abs(self.Q)) * (self.M + abs(self.Q))))

    @property
    def r_plus(self) -> float:
        return self.M + self._delta

    @property
    def r_minus(self) -> float:
        return self.Q**2 / self.r_plus  # r+ r- = Q^2: avoids cancellation near extremality

    @property
    def is_extremal(self) -> bool:
        return self._delta == 0.0

    def f(self, r):
        r = np.asarray(r, dtype=float)
        return 1.0 - 2.0 * self.M / r + self.Q**2 / r**2

    @property
    def surface_gravity(self) -> float:
        return self._delta / self.r_plus**2  # (r+ - r-) / (2 r+^2) with r+ - r- = 2 delta

    @property
    def temperature(self) -> float:
        return self.surface_gravity / (2.0 * np.pi)

    @property
    def area(self) -> float:
        return 4.0 * np.pi * self.r_plus**2

    @property
    def entropy(self) -> float:
        return self.area / 4.0

    @property
    def electric_potential(self) -> float:
        return self.Q / self.r_plus

    @property
    def heat_capacity_fixed_Q(self) -> float:
        """C_Q = T (dS/dr+) / (dT/dr+) at fixed Q; diverges at r+ = sqrt(3) |Q|."""
        rp, Q = self.r_plus, self.Q
        dS = 2.0 * np.pi * rp
        dT = (3.0 * Q**2 - rp**2) / (4.0 * np.pi * rp**4)
        if dT == 0.0:
            return float("inf")
        return self.temperature * dS / dT

    def kretschmann(self, r):
        """K = 8 (6 M^2 r^2 - 12 M Q^2 r + 7 Q^4) / r^8 (G = c = 1)."""
        r = np.asarray(r, dtype=float)
        M, Q = self.M, self.Q
        return 8.0 * (6.0 * M**2 * r**2 - 12.0 * M * Q**2 * r + 7.0 * Q**4) / r**8

    @staticmethod
    def mass_from(r_plus: float, Q: float) -> float:
        """M(r+, Q) = (r+^2 + Q^2) / (2 r+)."""
        return (r_plus**2 + Q**2) / (2.0 * r_plus)


# ---------------------------------------------------------------------------
# Kerr
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Kerr:
    """Rotating black hole with spin parameter ``a = J/M``; requires ``|a| <= M``."""

    M: float
    a: float

    def __post_init__(self) -> None:
        _check_positive(self.M, "M")
        if not np.isfinite(self.a):
            raise PhysicalParameterError("a must be finite")
        if abs(self.a) > self.M:
            raise PhysicalParameterError(f"|a|={abs(self.a)} exceeds M={self.M}: naked singularity")

    @property
    def _delta(self) -> float:
        return float(np.sqrt((self.M - abs(self.a)) * (self.M + abs(self.a))))

    @property
    def J(self) -> float:
        return self.M * self.a

    @property
    def r_plus(self) -> float:
        return self.M + self._delta

    @property
    def r_minus(self) -> float:
        return self.a**2 / self.r_plus  # r+ r- = a^2, avoids cancellation

    @property
    def surface_gravity(self) -> float:
        return self._delta / (2.0 * self.M * self.r_plus)

    @property
    def temperature(self) -> float:
        return self.surface_gravity / (2.0 * np.pi)

    @property
    def area(self) -> float:
        return 8.0 * np.pi * self.M * self.r_plus

    @property
    def entropy(self) -> float:
        return self.area / 4.0

    @property
    def horizon_angular_velocity(self) -> float:
        return self.a / (2.0 * self.M * self.r_plus)

    @property
    def is_extremal(self) -> bool:
        return self._delta == 0.0

    def smarr_residual(self) -> float:
        """M - 2 T S - 2 Omega J, which vanishes identically."""
        return self.M - 2.0 * self.temperature * self.entropy - 2.0 * self.horizon_angular_velocity * self.J


# ---------------------------------------------------------------------------
# Schwarzschild - anti de Sitter (4D)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchwarzschildAdS:
    """``f(r) = 1 - 2M/r + r^2/L^2`` parametrised by the horizon radius."""

    r_plus: float
    L: float

    def __post_init__(self) -> None:
        _check_positive(self.r_plus, "r_plus")
        _check_positive(self.L, "L")

    @classmethod
    def from_mass(cls, M: float, L: float) -> "SchwarzschildAdS":
        _check_positive(M, "M")
        _check_positive(L, "L")
        g = lambda rp: 0.5 * rp * (1.0 + rp**2 / L**2) - M  # monotonic increasing
        hi = max(2.0 * M, 1.0)
        while g(hi) < 0:
            hi *= 2.0
        return cls(brentq(g, 0.0, hi, xtol=1e-15, rtol=1e-15), L)

    @property
    def mass(self) -> float:
        return 0.5 * self.r_plus * (1.0 + self.r_plus**2 / self.L**2)

    @property
    def temperature(self) -> float:
        return (1.0 + 3.0 * self.r_plus**2 / self.L**2) / (4.0 * np.pi * self.r_plus)

    @property
    def entropy(self) -> float:
        return np.pi * self.r_plus**2

    @property
    def free_energy(self) -> float:
        """F = M - T S = (r+/4)(1 - r+^2/L^2); thermal AdS has F = 0 (Casimir energy dropped)."""
        return 0.25 * self.r_plus * (1.0 - self.r_plus**2 / self.L**2)

    @property
    def heat_capacity(self) -> float:
        """C = dM/dT = 2 pi r+^2 (1 + 3y) / (3y - 1), y = r+^2/L^2."""
        y = self.r_plus**2 / self.L**2
        if abs(3.0 * y - 1.0) < 1e-300:
            return float("inf")
        return 2.0 * np.pi * self.r_plus**2 * (1.0 + 3.0 * y) / (3.0 * y - 1.0)

    @staticmethod
    def minimum_temperature(L: float) -> float:
        """T_min = sqrt(3) / (2 pi L), attained at r+ = L / sqrt(3)."""
        return np.sqrt(3.0) / (2.0 * np.pi * L)

    @staticmethod
    def hawking_page_temperature(L: float) -> float:
        """T_HP = 1 / (pi L), where F = 0 at r+ = L."""
        return 1.0 / (np.pi * L)


# ---------------------------------------------------------------------------
# Schwarzschild - de Sitter (4D)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchwarzschildDeSitter:
    """``f(r) = 1 - 2M/r - r^2/L^2`` with a black-hole and a cosmological horizon.

    The two horizons have different surface gravities, hence different
    temperatures, except in the Nariai limit ``27 M^2 -> L^2`` where both
    vanish.  No global thermal equilibrium is assumed.
    """

    M: float
    L: float

    def __post_init__(self) -> None:
        _check_positive(self.M, "M")
        _check_positive(self.L, "L")
        if 27.0 * self.M**2 >= self.L**2:
            raise PhysicalParameterError(
                "27 M^2 >= L^2: no distinct black-hole and cosmological horizons (Nariai bound)"
            )

    def _roots(self) -> tuple[float, float]:
        # h(r) = r f(r) = r - 2M - r^3/L^2 has its maximum at r_n = L/sqrt(3), where
        # h(r_n) > 0 iff 27 M^2 < L^2 (checked in __post_init__).  h(0) = -2M < 0
        # and h(L) = -2M < 0, so one root lies on each side of r_n.
        r_n = self.L / np.sqrt(3.0)
        h = lambda r: r - 2.0 * self.M - r**3 / self.L**2
        rb = brentq(h, 0.0, r_n, xtol=1e-15, rtol=1e-15)
        rc = brentq(h, r_n, self.L, xtol=1e-15, rtol=1e-15)
        return float(rb), float(rc)

    @property
    def r_b(self) -> float:
        return self._roots()[0]

    @property
    def r_c(self) -> float:
        return self._roots()[1]

    def f(self, r):
        r = np.asarray(r, dtype=float)
        return 1.0 - 2.0 * self.M / r - r**2 / self.L**2

    def _fprime(self, r: float) -> float:
        return 2.0 * self.M / r**2 - 2.0 * r / self.L**2

    @property
    def kappa_b(self) -> float:
        return 0.5 * self._fprime(self.r_b)

    @property
    def kappa_c(self) -> float:
        return -0.5 * self._fprime(self.r_c)

    @property
    def temperature_b(self) -> float:
        return self.kappa_b / (2.0 * np.pi)

    @property
    def temperature_c(self) -> float:
        return self.kappa_c / (2.0 * np.pi)

    @property
    def entropy_b(self) -> float:
        return np.pi * self.r_b**2

    @property
    def entropy_c(self) -> float:
        return np.pi * self.r_c**2
