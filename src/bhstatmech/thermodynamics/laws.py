"""Black-hole thermodynamic laws (SI) with identity checks.

Bekenstein-Hawking::

    T_H  = hbar c^3 / (8 pi G M k_B)
    A    = 4 pi r_s^2 = 16 pi G^2 M^2 / c^4
    S_BH = k_B c^3 A / (4 G hbar) = 4 pi k_B G M^2 / (hbar c)

First law (Schwarzschild): ``d(M c^2) = T_H dS_BH``.
Heat capacity: ``C = d(M c^2)/dT_H = -8 pi G M^2 k_B / (hbar c) = -2 S_BH``.
"""

from __future__ import annotations

import math

import numpy as np

from .. import constants as const
from ..exceptions import PhysicalParameterError
from ..numerical_methods.finite_difference import central_difference
from ..units import (
    JOULE_PER_KELVIN, KELVIN, SQUARE_METER, METER, C_Q, G_Q, HBAR_Q, KB_Q, Quantity,
)


def _check_mass(m):
    m = np.asarray(m, dtype=float)
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise PhysicalParameterError("mass must be positive and finite")
    return m


def schwarzschild_radius(mass_kg):
    return 2.0 * const.G * _check_mass(mass_kg) / const.C**2


def hawking_temperature(mass_kg):
    return const.HBAR * const.C**3 / (8.0 * np.pi * const.G * _check_mass(mass_kg) * const.K_B)


def horizon_area(mass_kg):
    return 4.0 * np.pi * schwarzschild_radius(mass_kg) ** 2


def entropy_from_area(area_m2, dimensionless: bool = False):
    A = np.asarray(area_m2, dtype=float)
    s = const.C**3 * A / (4.0 * const.G * const.HBAR)  # S / k_B
    return s if dimensionless else const.K_B * s


def bekenstein_hawking_entropy(mass_kg, dimensionless: bool = False):
    """S_BH in J/K, or S_BH/k_B if ``dimensionless``."""
    M = _check_mass(mass_kg)
    s = 4.0 * np.pi * const.G * M**2 / (const.HBAR * const.C)
    return s if dimensionless else const.K_B * s


def heat_capacity(mass_kg):
    """C = d(Mc^2)/dT_H [J/K] for Schwarzschild; always negative."""
    M = _check_mass(mass_kg)
    return -8.0 * np.pi * const.G * M**2 * const.K_B / (const.HBAR * const.C)


def first_law_residual(mass_kg: float, h_rel: float = 1e-5) -> float:
    """Relative residual of ``dE/dM = T dS/dM`` with E = M c^2, using a central difference.

    Returns ``|c^2 - T dS/dM| / c^2``.
    """
    M = float(_check_mass(mass_kg))
    dSdM = central_difference(lambda m: float(bekenstein_hawking_entropy(m)), M, h_rel * M)
    lhs = const.C**2
    rhs = float(hawking_temperature(M)) * dSdM
    return abs(lhs - rhs) / lhs


def first_law_symbolic() -> bool:
    """Verify dM = T dS exactly in natural units with SymPy (T = 1/(8 pi M), S = 4 pi M^2)."""
    import sympy as sp

    M = sp.symbols("M", positive=True)
    T = 1 / (8 * sp.pi * M)
    S = 4 * sp.pi * M**2
    return sp.simplify(T * sp.diff(S, M) - 1) == 0


# ---- dimensional-analysis versions (used in tests) ---------------------------


def hawking_temperature_quantity(mass: Quantity) -> Quantity:
    return HBAR_Q * C_Q**3 / (8.0 * math.pi * G_Q * mass * KB_Q)


def entropy_quantity(mass: Quantity) -> Quantity:
    return 4.0 * math.pi * KB_Q * G_Q * mass**2 / (HBAR_Q * C_Q)


def horizon_area_quantity(mass: Quantity) -> Quantity:
    return 16.0 * math.pi * G_Q**2 * mass**2 / C_Q**4


__all__ = [
    "schwarzschild_radius", "hawking_temperature", "horizon_area", "entropy_from_area",
    "bekenstein_hawking_entropy", "heat_capacity", "first_law_residual", "first_law_symbolic",
    "hawking_temperature_quantity", "entropy_quantity", "horizon_area_quantity",
    "KELVIN", "JOULE_PER_KELVIN", "SQUARE_METER", "METER",
]
