"""Unit conversions and a minimal dimensional-analysis system.

The :class:`Quantity` class tracks exponents of (mass, length, time,
temperature) so that physical formulas can be checked for dimensional
consistency in tests.  It is deliberately small: it is not a general units
library.

Dimensionless variables used in this package
--------------------------------------------
``x = hbar*omega / (k_B*T_H)``  : frequency in units of the Hawking temperature.
``y = omega * r_s / c``          : frequency in units of the light-crossing time of r_s.
                                   For Schwarzschild, x = 4*pi*y.
``S / k_B``                      : entropy in units of Boltzmann's constant.
``r / r_s``                      : radius in units of the Schwarzschild radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from . import constants as const
from .exceptions import DimensionError, PhysicalParameterError

# ---------------------------------------------------------------------------
# Dimensional analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimension:
    """Exponents of (mass, length, time, temperature)."""

    mass: Fraction = Fraction(0)
    length: Fraction = Fraction(0)
    time: Fraction = Fraction(0)
    temperature: Fraction = Fraction(0)

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.temperature + other.temperature,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.temperature - other.temperature,
        )

    def __pow__(self, p) -> "Dimension":
        p = Fraction(p)
        return Dimension(self.mass * p, self.length * p, self.time * p, self.temperature * p)


def _d(m=0, l=0, t=0, k=0) -> Dimension:
    return Dimension(Fraction(m), Fraction(l), Fraction(t), Fraction(k))


DIMENSIONLESS = _d()
KILOGRAM = _d(m=1)
METER = _d(l=1)
SECOND = _d(t=1)
KELVIN = _d(k=1)
JOULE = _d(m=1, l=2, t=-2)
HERTZ = _d(t=-1)
JOULE_PER_KELVIN = _d(m=1, l=2, t=-2, k=-1)
WATT = _d(m=1, l=2, t=-3)
SQUARE_METER = _d(l=2)


@dataclass(frozen=True)
class Quantity:
    """A float with a :class:`Dimension`.  Supports ``* / **`` and ``+ -``."""

    value: float
    dim: Dimension = DIMENSIONLESS

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.dim * other.dim)
        return Quantity(self.value * float(other), self.dim)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.dim / other.dim)
        return Quantity(self.value / float(other), self.dim)

    def __rtruediv__(self, other):
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dim)

    def __pow__(self, p):
        return Quantity(self.value ** float(p), self.dim ** p)

    def __add__(self, other):
        self._check_same(other, "add")
        return Quantity(self.value + other.value, self.dim)

    def __sub__(self, other):
        self._check_same(other, "subtract")
        return Quantity(self.value - other.value, self.dim)

    def _check_same(self, other, verb):
        if not isinstance(other, Quantity) or other.dim != self.dim:
            raise DimensionError(f"cannot {verb} quantities with different dimensions")


def assert_dimension(q: Quantity, expected: Dimension, name: str = "quantity") -> None:
    """Raise :class:`DimensionError` unless ``q.dim == expected``."""
    if q.dim != expected:
        raise DimensionError(f"{name} has dimension {q.dim}, expected {expected}")


# Constants as dimensionful quantities
G_Q = Quantity(const.G, _d(m=-1, l=3, t=-2))
C_Q = Quantity(const.C, _d(l=1, t=-1))
HBAR_Q = Quantity(const.HBAR, _d(m=1, l=2, t=-1))
KB_Q = Quantity(const.K_B, JOULE_PER_KELVIN)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _positive(x: float, name: str) -> float:
    if not (x > 0.0) or not math.isfinite(x):
        raise PhysicalParameterError(f"{name} must be positive and finite, got {x!r}")
    return x


def solar_masses_to_kg(m_sun: float) -> float:
    return m_sun * const.M_SUN


def kg_to_solar_masses(m_kg: float) -> float:
    return m_kg / const.M_SUN


def gravitational_radius(mass_kg: float) -> float:
    """GM/c^2 in metres."""
    return const.G * _positive(mass_kg, "mass") / const.C**2


def meters_to_gravitational_radii(r_m: float, mass_kg: float) -> float:
    return r_m / gravitational_radius(mass_kg)


def gravitational_radii_to_meters(r_over_rg: float, mass_kg: float) -> float:
    return r_over_rg * gravitational_radius(mass_kg)


def temperature_si_to_natural(T_kelvin: float) -> float:
    """Temperature in units of the Planck temperature (k_B T / E_P)."""
    return T_kelvin / const.PLANCK_TEMPERATURE


def temperature_natural_to_si(T_planck: float) -> float:
    return T_planck * const.PLANCK_TEMPERATURE


def entropy_si_to_dimensionless(S_si: float) -> float:
    """S [J/K] -> S / k_B."""
    return S_si / const.K_B


def entropy_dimensionless_to_si(S_over_kb: float) -> float:
    return S_over_kb * const.K_B


def omega_to_energy(omega: float) -> float:
    """hbar*omega in joules for an angular frequency in rad/s."""
    return const.HBAR * omega


def energy_to_omega(E_joule: float) -> float:
    return E_joule / const.HBAR


def mass_kg_to_planck(mass_kg: float) -> float:
    return mass_kg / const.PLANCK_MASS


def mass_planck_to_kg(mass_planck: float) -> float:
    return mass_planck * const.PLANCK_MASS


def length_m_to_planck(L: float) -> float:
    return L / const.PLANCK_LENGTH


def time_s_to_planck(t: float) -> float:
    return t / const.PLANCK_TIME
