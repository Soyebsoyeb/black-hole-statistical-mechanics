"""Generalized-entropy / island toy model.

THIS IS A TRANSPARENT TOY, NOT A DERIVATION FROM EINSTEIN'S EQUATIONS.

Ingredients (dimensionless entropies in units of k_B, with tau = t / t_evap)
------------------------------------------------------------------------
* Black-hole entropy during quasi-static evaporation of a 4D Schwarzschild
  hole, ``M(t) = M0 (1 - tau)^{1/3}`` and ``S_BH ~ M^2``::

      S_H(tau) = S0 (1 - tau)^{2/3}.

* Entropy of the Hawking radiation without an island, from entropy
  conservation in emission with ratio ``r`` of radiated entropy to lost
  black-hole entropy (``r = 4/3`` for an ideal massless 3+1 gas, no greybody
  factors; a stated assumption)::

      S_rad(tau) = r [S0 - S_H(tau)].

* Island branch.  ``u >= 0`` is the displacement of the island boundary
  (candidate extremal surface) in units of the horizon scale.  The area-like
  term is linear in ``u`` and the bulk entropy of ``R union I`` falls
  exponentially as the island absorbs partners of the radiation::

      S_gen(u; tau) = S_H(1 + mu u) + S_rad exp(-u / xi).

  Parameters: ``mu`` (area growth rate per unit displacement) and ``xi``
  (decorrelation length).  Only ``kappa = mu xi`` and ``r`` matter for the
  transition time.

Prescription (schematic island formula)::

    S(R) = min{ S_no-island, ext_u S_gen(u) }.

The stationary point ``u* = xi ln( S_rad / (kappa S_H) )`` exists (and is a
minimum in u) only when ``S_rad > kappa S_H``; otherwise only the trivial
boundary point ``u = 0`` remains, with ``S_gen(0) = S_H + S_rad >= S_rad``, so
the no-island branch dominates automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import PhysicalParameterError
from ..numerical_methods.solvers import find_root, minimize_bounded


@dataclass(frozen=True)
class IslandToyModel:
    S0: float = 1000.0
    r: float = 4.0 / 3.0
    mu: float = 1.0
    xi: float = 0.05

    def __post_init__(self):
        if min(self.S0, self.r, self.mu, self.xi) <= 0:
            raise PhysicalParameterError("all toy-model parameters must be positive")

    @property
    def kappa(self) -> float:
        return self.mu * self.xi

    def S_H(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self.S0 * np.clip(1.0 - tau, 0.0, None) ** (2.0 / 3.0)

    def S_rad(self, tau):
        return self.r * (self.S0 - self.S_H(tau))

    def S_no_island(self, tau):
        return self.S_rad(tau)

    def S_gen(self, u, tau):
        """Generalized entropy of the island candidate as a function of the boundary position."""
        return self.S_H(tau) * (1.0 + self.mu * np.asarray(u)) + self.S_rad(tau) * np.exp(-np.asarray(u) / self.xi)

    # analytic stationary point -------------------------------------------------
    def island_exists(self, tau: float) -> bool:
        return float(self.S_rad(tau)) > self.kappa * float(self.S_H(tau))

    def u_star_analytic(self, tau: float) -> float:
        if not self.island_exists(tau):
            return 0.0
        return self.xi * float(np.log(self.S_rad(tau) / (self.kappa * self.S_H(tau))))

    def S_island_analytic(self, tau: float) -> float:
        """Value at the nontrivial extremum; ``nan`` if it does not exist."""
        if not self.island_exists(tau):
            return float("nan")
        SH, Sr = float(self.S_H(tau)), float(self.S_rad(tau))
        return SH * (1.0 + self.kappa * (1.0 + np.log(Sr / (self.kappa * SH))))

    # numerical extremisation ----------------------------------------------------
    def u_star_numeric(self, tau: float, xtol: float = 1e-12, u_max: float = 10.0):
        """Minimise S_gen over u in [0, u_max] with a bounded scalar minimiser."""
        u, val = minimize_bounded(lambda x: float(self.S_gen(x, tau)), 0.0, u_max, xtol=xtol)
        return u, val

    def S_island_numeric(self, tau: float, **kw) -> float:
        if not self.island_exists(tau):
            return float("nan")
        return self.u_star_numeric(tau, **kw)[1]

    # full prescription ----------------------------------------------------------
    def S_radiation(self, tau):
        """min over branches (the island-formula entropy of the radiation)."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty_like(tau)
        for i, t in enumerate(tau):
            isl = self.S_island_analytic(float(t))
            no = float(self.S_no_island(t))
            out[i] = no if np.isnan(isl) else min(no, isl)
        return out

    def page_time(self) -> float:
        """Transition time tau_P solving S_no-island = S_island (root finding)."""
        f = lambda t: float(self.S_no_island(t)) - self.S_island_analytic(t)
        # island exists once S_rad > kappa S_H; start the bracket just after that
        lo = 0.0
        for t in np.linspace(0.0, 0.999, 2000):
            if self.island_exists(float(t)):
                lo = float(t)
                break
        hi = 0.999
        if f(lo) > 0:
            return lo  # island dominates as soon as it exists
        return find_root(f, lo, hi).root


def page_time_no_correction(r: float) -> float:
    """Closed form for kappa -> 0: S_rad = S_H gives (1-tau)^{2/3} = r/(1+r)."""
    return float(1.0 - (r / (1.0 + r)) ** 1.5)
