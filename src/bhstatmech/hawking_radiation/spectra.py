"""Thermal occupation numbers and emission rates.

A thermal occupation curve illustrates the spectrum of Hawking quanta; it is
not a derivation of Hawking radiation (which follows from the Bogoliubov
mixing of modes defined on the collapsing background, see
``docs/hawking_radiation.md`` and ``entanglement.horizon``).

Number emission rate per mode sum (one bosonic species, scalar field)::

    d^2N / (dt d omega) = (1/2 pi) sum_l (2 l + 1) Gamma_l(omega) / (exp(hbar omega / k_B T_H) - 1)

Assumptions: spherically symmetric, non-rotating hole; each (l, m) mode is one
independent channel; Gamma_l is the transmission probability of the
Regge-Wheeler potential barrier; the emission is into vacuum at infinity.
Rotating holes or several species require additional sums over m and species
(and spin-dependent potentials).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from scipy.special import expit

from .. import constants as const
from ..exceptions import PhysicalParameterError


def dimensionless_frequency(omega, T_kelvin: float):
    """x = hbar omega / (k_B T)."""
    if T_kelvin <= 0:
        raise PhysicalParameterError("temperature must be positive")
    return const.HBAR * np.asarray(omega, dtype=float) / (const.K_B * T_kelvin)


def _check_x(x):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise PhysicalParameterError("hbar*omega/(k_B T) must be positive")
    return x


def bose_occupation(x):
    """1 / (e^x - 1) with x = hbar omega / (k_B T_H)."""
    return 1.0 / np.expm1(_check_x(x))


def fermi_occupation(x):
    """1 / (e^x + 1), evaluated as the logistic function expit(-x) (accurate and overflow-free)."""
    return expit(-_check_x(x))


def number_flux(
    omega_over_c_rs, gammas: np.ndarray,
    ell_values: Sequence[int] | None = None,
):
    """Dimensionless number-rate density ``(1/2 pi) sum_l (2l+1) Gamma_l n(x)``.

    ``omega_over_c_rs = y = omega r_s / c``; for Schwarzschild ``x = 4 pi y``.
    ``gammas`` is an array of shape ``(n_ell, n_y)`` of transmission
    probabilities for ``ell = 0..n_ell-1`` (or ``ell_values``).
    The physical rate is ``d^2N/dt d omega = (this) `` since the formula is
    already per unit omega; multiply by nothing further.
    """
    y = np.asarray(omega_over_c_rs, dtype=float)
    G = np.asarray(gammas, dtype=float)
    ells = np.arange(G.shape[0]) if ell_values is None else np.asarray(ell_values)
    n = bose_occupation(4.0 * np.pi * y)
    return (1.0 / (2.0 * np.pi)) * np.sum((2 * ells + 1)[:, None] * G * n[None, :], axis=0)


def planck_energy_flux_per_omega(omega, T_kelvin: float):
    """Blackbody energy flux per unit area, time and angular frequency, summed over two polarisations.

    Integrates to ``sigma_SB T^4``.
    """
    x = dimensionless_frequency(omega, T_kelvin)
    return const.HBAR * np.asarray(omega) ** 3 / (4.0 * np.pi**2 * const.C**2) / np.expm1(x)
