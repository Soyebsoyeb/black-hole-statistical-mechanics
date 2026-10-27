"""Quasi-static mass-loss model and backreaction estimates.

Luminosity prescription (documented approximation)
--------------------------------------------------
The black hole radiates as a blackbody of effective area ``A_eff`` at the
Hawking temperature, photons only (two polarisations), no greybody factors,
no other species::

    L = sigma_SB A_eff T_H^4,   dM/dt = -L / c^2.

With ``A_eff = A_horizon`` this gives ``dM/dt = -alpha/M^2`` with
``alpha = hbar c^4 / (15360 pi G^2)`` and the lifetime
``t_evap = M0^3 / (3 alpha) = 5120 pi G^2 M0^3 / (hbar c^4)``.
The ``geometric_optics`` option replaces the horizon by the sphere of the
critical impact parameter ``b_c = 3 sqrt(3) GM/c^2 = (3 sqrt(3)/2) r_s``: a
black-body of that radius has emitting area ``4 pi b_c^2 = 27 pi r_s^2``, i.e.
``27/4`` times the horizon area (equivalently, the capture cross-section
``sigma = (27/4) pi r_s^2 = pi b_c^2`` is one quarter of the emitting area).
This is the high-frequency limit of the emission; it is verified numerically in
``experiments/05_greybody_factors.py``.  Both lifetimes are *approximations*: real
lifetimes depend on the particle content and greybody factors.

The model is not extended through the Planck-scale endpoint; integrations
stop at a user-chosen cutoff mass (default 10 Planck masses).

Backreaction / adiabaticity estimate
------------------------------------
The quasi-static, semiclassical picture requires the black hole to change
slowly on the timescale of one thermal quantum, ``tau_q = hbar / (k_B T_H)``.
Define ``epsilon = tau_q * |dM/dt| / M``.  ``epsilon << 1`` is necessary (not
sufficient) for validity; ``epsilon ~ 1`` signals that the approximation has
broken down.  This is a heuristic criterion, not a prediction of the endpoint.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .. import constants as const
from ..exceptions import PhysicalParameterError
from ..numerical_methods.solvers import solve_ode
from ..thermodynamics.laws import hawking_temperature, horizon_area

CrossSection = Literal["horizon", "geometric_optics"]


def effective_area(mass_kg: float, cross_section: CrossSection = "horizon") -> float:
    A = float(horizon_area(mass_kg))
    if cross_section == "horizon":
        return A
    if cross_section == "geometric_optics":
        return A * 27.0 / 4.0  # 4 pi b_c^2 = 27 pi r_s^2 over the horizon area 4 pi r_s^2
    raise PhysicalParameterError(f"unknown cross_section {cross_section!r}")


def luminosity(mass_kg: float, cross_section: CrossSection = "horizon") -> float:
    """L = sigma_SB A_eff T_H^4 [W]."""
    return const.SIGMA_SB * effective_area(mass_kg, cross_section) * float(hawking_temperature(mass_kg)) ** 4


def mass_loss_rate(mass_kg: float, cross_section: CrossSection = "horizon") -> float:
    """dM/dt [kg/s] (negative)."""
    return -luminosity(mass_kg, cross_section) / const.C**2


def alpha_coefficient(cross_section: CrossSection = "horizon") -> float:
    """alpha in dM/dt = -alpha/M^2  [kg^3/s]."""
    factor = 1.0 if cross_section == "horizon" else 27.0 / 4.0
    return factor * const.HBAR * const.C**4 / (15360.0 * np.pi * const.G**2)


def evaporation_time_analytic(mass_kg: float, cross_section: CrossSection = "horizon") -> float:
    """t_evap = M^3 / (3 alpha) [s] (to M = 0; an idealisation, see module docstring)."""
    return mass_kg**3 / (3.0 * alpha_coefficient(cross_section))


def integrate_mass_loss(
    m0_kg: float,
    cross_section: CrossSection = "horizon",
    cutoff_planck_masses: float = 10.0,
    rtol: float = 1e-10,
    atol: float = 0.0,
):
    """Integrate dM/dt with an adaptive solver in the variable u = M^3.

    ``d(M^3)/dt = 3 M^2 dM/dt = -3 alpha`` is constant, so the ODE is solved in
    the dimensionless variable ``y = (M/M0)^3`` and time in units of t_evap
    to avoid dynamic-range problems; the solver result is compared to the
    analytic solution in tests and experiments.

    Returns ``(t, M)`` arrays in seconds and kilograms, terminated at the cutoff mass.
    """
    t_ev = evaporation_time_analytic(m0_kg, cross_section)
    m_cut = cutoff_planck_masses * const.PLANCK_MASS
    if m0_kg <= m_cut:
        raise PhysicalParameterError("initial mass must exceed the cutoff mass")
    # Use the physical ODE dm/dtau with m = M/M0, tau = t/t_ev:  dm/dtau = -1/(3 m^2)
    event = lambda tau, y: y[0] - m_cut / m0_kg
    event.terminal = True
    event.direction = -1
    sol = solve_ode(lambda tau, y: [-1.0 / (3.0 * y[0] ** 2)], (0.0, 1.0), [1.0],
                    rtol=rtol, atol=atol if atol else 1e-14, events=event, dense_output=False)
    return sol.t * t_ev, sol.y[0] * m0_kg


def adiabaticity_parameter(mass_kg: float, cross_section: CrossSection = "horizon") -> float:
    """epsilon = (hbar / k_B T_H) |dM/dt| / M (dimensionless)."""
    T = float(hawking_temperature(mass_kg))
    tau_q = const.HBAR / (const.K_B * T)
    return tau_q * abs(mass_loss_rate(mass_kg, cross_section)) / mass_kg


def mass_at_adiabaticity(epsilon: float = 0.1, cross_section: CrossSection = "horizon") -> float:
    """Mass [kg] at which the adiabaticity parameter equals ``epsilon``.

    epsilon scales as M^-2, so M = M_P * sqrt(c0 / epsilon) with c0 = epsilon(M_P) / ... .
    """
    ref = adiabaticity_parameter(const.PLANCK_MASS, cross_section)
    return const.PLANCK_MASS * np.sqrt(ref / epsilon)
