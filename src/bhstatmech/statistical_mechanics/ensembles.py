"""Finite-dimensional statistical mechanics and black-hole state counting.

Units: k_B = 1 unless ``kB`` is passed.  A spectrum is stored as energy levels
with log-degeneracies ``ln g`` so that very large degeneracies do not overflow.

Gravitating systems
-------------------
For a Schwarzschild black hole the microcanonical entropy ``S(E) = 4 pi E^2``
(Planck units) is convex, so ``C = dE/dT < 0``.  The canonical integrand
``exp[S(E) - beta E]`` then has a *minimum* (not a maximum) at the saddle
``E* = beta / (8 pi)``: the canonical partition function does not exist as a
convergent integral and canonical and microcanonical descriptions disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from ..exceptions import PhysicalParameterError


@dataclass(frozen=True)
class FiniteSpectrum:
    energies: np.ndarray
    log_degeneracy: np.ndarray

    def __post_init__(self):
        if self.energies.shape != self.log_degeneracy.shape:
            raise PhysicalParameterError("energies and log_degeneracy must have equal shape")

    # canonical quantities ---------------------------------------------------
    def log_Z(self, beta: float) -> float:
        return float(logsumexp(self.log_degeneracy - beta * self.energies))

    def weights(self, beta: float) -> np.ndarray:
        a = self.log_degeneracy - beta * self.energies
        return np.exp(a - logsumexp(a))

    def mean_energy(self, beta: float) -> float:
        return float(np.sum(self.weights(beta) * self.energies))

    def energy_variance(self, beta: float) -> float:
        w = self.weights(beta)
        return float(np.sum(w * self.energies**2) - np.sum(w * self.energies) ** 2)

    def free_energy(self, beta: float, kB: float = 1.0) -> float:
        return -kB * self.log_Z(beta) / (kB * beta)  # F = -(1/beta) ln Z (T = 1/(kB beta))

    def entropy(self, beta: float) -> float:
        """Canonical (Gibbs) entropy / k_B = beta <E> + ln Z."""
        return beta * self.mean_energy(beta) + self.log_Z(beta)

    def heat_capacity(self, beta: float) -> float:
        """C / k_B = beta^2 Var(E) >= 0 (canonical ensemble)."""
        return beta**2 * self.energy_variance(beta)

    # microcanonical -----------------------------------------------------------
    def microcanonical_entropy(self) -> np.ndarray:
        """S(E)/k_B = ln g(E) at each level."""
        return self.log_degeneracy


def two_level_system(n_atoms: int, epsilon: float = 1.0) -> FiniteSpectrum:
    """N independent two-level atoms (E = k epsilon, degeneracy C(N, k))."""
    k = np.arange(n_atoms + 1)
    lng = gammaln(n_atoms + 1) - gammaln(k + 1) - gammaln(n_atoms - k + 1)
    return FiniteSpectrum(k * epsilon, lng)


def two_level_heat_capacity_exact(beta: float, n_atoms: int, epsilon: float = 1.0) -> float:
    """Schottky: C/k_B = N (beta eps/2)^2 / cosh^2(beta eps / 2)."""
    return n_atoms * (beta * epsilon / 2.0) ** 2 / np.cosh(beta * epsilon / 2.0) ** 2


# ---- black-hole-like density of states (Planck units) --------------------------


def bh_microcanonical_entropy(E):
    return 4.0 * np.pi * np.asarray(E, dtype=float) ** 2


def bh_canonical_exponent(E, beta: float):
    """S(E) - beta E, the exponent of the canonical integrand."""
    E = np.asarray(E, dtype=float)
    return bh_microcanonical_entropy(E) - beta * E


def bh_saddle_energy(beta: float) -> float:
    """Stationary point E* = beta / (8 pi) of S(E) - beta E; a *minimum*."""
    return beta / (8.0 * np.pi)


def bh_exponent_second_derivative() -> float:
    """d^2/dE^2 [S - beta E] = 8 pi > 0: the saddle is a minimum."""
    return 8.0 * np.pi


def microcanonical_heat_capacity_bh(E):
    """C = dE/dT with T = 1/(8 pi E): C = -8 pi E^2 (Planck units)."""
    return -8.0 * np.pi * np.asarray(E, dtype=float) ** 2


# ---- effective state counting ----------------------------------------------------


def log_omega_eff(entropy_over_kB):
    """ln Omega_eff = S_BH / k_B (Omega_eff itself overflows double precision)."""
    return np.asarray(entropy_over_kB, dtype=float)


def log10_omega_eff(entropy_over_kB):
    return np.asarray(entropy_over_kB, dtype=float) / np.log(10.0)


def entropy_with_log_correction(area_over_lp2, alpha: float, constant: float = 0.0):
    """S/k_B = A/(4 l_P^2) + alpha ln(A/l_P^2) + constant.

    ``alpha`` is a model-dependent coefficient; different approaches in the
    literature yield different values and no universal value is assumed.
    """
    A = np.asarray(area_over_lp2, dtype=float)
    if np.any(A <= 0):
        raise PhysicalParameterError("area must be positive")
    return A / 4.0 + alpha * np.log(A) + constant
