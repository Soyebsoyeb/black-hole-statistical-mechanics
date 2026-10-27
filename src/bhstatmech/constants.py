"""Physical constants (SI) and derived Planck quantities.

Values are the CODATA 2018 recommended values (exact where the SI defines
them).  ``G`` carries a relative standard uncertainty of about 2e-5, which
propagates linearly into every quantity below that depends on it.  The solar
mass follows from the IAU 2015 nominal solar mass parameter GM_sun = 1.3271244e20 m^3 s^-2
(M_sun = GM_sun / G), so r_s(M_sun) = 2 GM_sun / c^2 = 2953.25 m independent of G.

Conventions
-----------
* SI units are used for astrophysical quantities.
* Geometrized units (G = c = 1) are used inside relativistic derivations where
  stated.
* Planck/natural units (G = hbar = c = k_B = 1) are used for the additional
  geometries (``bhstatmech.spacetimes.natural``) and holographic expressions.

Never mix formulas from different conventions without restoring constants
explicitly.  Conversion helpers live in :mod:`bhstatmech.units`.
"""

from __future__ import annotations

import math

# ---- fundamental constants (SI) -------------------------------------------
G: float = 6.67430e-11  # m^3 kg^-1 s^-2
C: float = 299_792_458.0  # m s^-1 (exact)
HBAR: float = 1.054571817e-34  # J s
K_B: float = 1.380649e-23  # J K^-1 (exact)
EV: float = 1.602176634e-19  # J (exact)

# ---- derived constants -----------------------------------------------------
SIGMA_SB: float = math.pi**2 * K_B**4 / (60.0 * HBAR**3 * C**2)  # W m^-2 K^-4

PLANCK_LENGTH: float = math.sqrt(HBAR * G / C**3)  # m
PLANCK_MASS: float = math.sqrt(HBAR * C / G)  # kg
PLANCK_TIME: float = math.sqrt(HBAR * G / C**5)  # s
PLANCK_ENERGY: float = PLANCK_MASS * C**2  # J
PLANCK_TEMPERATURE: float = PLANCK_ENERGY / K_B  # K

# ---- astrophysical --------------------------------------------------------
GM_SUN: float = 1.3271244e20  # m^3 s^-2, IAU 2015 nominal solar mass parameter (exact by definition)
M_SUN: float = GM_SUN / G  # kg, ~1.98841e30 (follows from the IAU nominal GM_sun and the G above)
YEAR: float = 365.25 * 86400.0  # s (Julian year)
AGE_OF_UNIVERSE: float = 13.8e9 * YEAR  # s, approximate

__all__ = [
    "G", "C", "HBAR", "K_B", "EV", "SIGMA_SB",
    "PLANCK_LENGTH", "PLANCK_MASS", "PLANCK_TIME", "PLANCK_ENERGY",
    "PLANCK_TEMPERATURE", "GM_SUN", "M_SUN", "YEAR", "AGE_OF_UNIVERSE",
]
