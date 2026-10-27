"""Greybody factors for massless fields on Schwarzschild.

Radial equation (units r_s = 1, c = 1, frequency ``y = omega r_s / c``)::

    d^2 psi / dr*^2 + (y^2 - V_l(r)) psi = 0,
    V_l = f [ l(l+1)/r^2 + (1 - s^2)/r^3 ],   f = 1 - 1/r,   dr*/dr = 1/f.

Method
------
Integrate from near the horizon (pure ingoing wave, ``psi = e^{-i y r*}``)
outward with an adaptive DOP853 integrator, in the logarithmic variable
``s = ln(r - 1)`` which is regular across the whole domain (``dr*/ds = r``).
At a large matching radius the solution is decomposed into Riccati-Hankel
functions of ``y r*``::

    psi = a_in h_l^-(y r*) + a_out h_l^+(y r*),   h_l^+/-(x) ~ exp(+/- i (x - l pi/2)),

and ``Gamma_l = 1 / |a_in|^2`` (unit transmitted amplitude at the horizon).

Diagnostics
-----------
* ``flux_defect = | |a_in|^2 - |a_out|^2 - 1 | / |a_in|^2``: probability-conservation
  (Wronskian) violation relative to the incident flux.  It is normalised by
  ``|a_in|^2`` because the difference of two numbers of size ``|a_in|^2`` can
  only be resolved to ``eps |a_in|^2`` (when ``Gamma ~ 1e-11`` the absolute
  defect is dominated by round-off).
* Bounds ``0 <= Gamma <= 1`` are checked and violations (beyond the matching
  error) raise :class:`ConvergenceError`.

Validation targets (see ``tests/`` and ``docs/validation.md``):
low-frequency ``Gamma_0 -> 4 y^2`` (absorption cross-section -> horizon area,
Das-Gibbons-Mathur) and the high-frequency geometric-optics cross-section
``(27/4) pi r_s^2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn

from ..exceptions import ConvergenceError, PhysicalParameterError


@dataclass(frozen=True)
class GreybodyResult:
    gamma: float
    a_in: complex
    a_out: complex
    flux_defect: float


def potential(r, ell: int, spin: int = 0):
    """Dimensionless Regge-Wheeler potential (r in units of r_s)."""
    r = np.asarray(r, dtype=float)
    return (1.0 - 1.0 / r) * (ell * (ell + 1) / r**2 + (1 - spin**2) / r**3)


def _riccati_hankel(ell: int, x: float):
    """Return (h_minus, h_minus', h_plus, h_plus') of the Riccati-Hankel functions.

    h^+(x) = x (j_l + i y_l)... normalised so that h^+ ~ -i exp(i(x - l pi/2));
    only moduli of the expansion coefficients are used, so the constant phase
    is immaterial.
    """
    j, jp = x * spherical_jn(ell, x), spherical_jn(ell, x) + x * spherical_jn(ell, x, derivative=True)
    y, yp = x * spherical_yn(ell, x), spherical_yn(ell, x) + x * spherical_yn(ell, x, derivative=True)
    hp, hpp = j + 1j * y, jp + 1j * yp
    hm, hmp = j - 1j * y, jp - 1j * yp
    return hm, hmp, hp, hpp


def greybody_factor(
    y: float,
    ell: int,
    spin: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    x_match: float = 400.0,
    s_min: float = -22.0,
    r_star_min_cap: float | None = None,
) -> GreybodyResult:
    """Transmission probability ``Gamma_l(y)`` for the spin-``spin`` potential."""
    if y <= 0:
        raise PhysicalParameterError("frequency y = omega r_s / c must be positive")
    if ell < spin:
        raise PhysicalParameterError("ell must be >= spin")
    r_star_max = max(x_match / y, 60.0 * (ell + 1))
    # r* = r + ln(r - 1) with r in units of r_s; invert for the final r by Newton
    def rstar(r):
        return r + np.log(r - 1.0)

    r_end = r_star_max
    for _ in range(100):
        r_end = r_end - (rstar(r_end) - r_star_max) / (1.0 + 1.0 / (r_end - 1.0))
    s_max = np.log(r_end - 1.0)

    def rhs(s, u):
        x = np.exp(s)
        r = 1.0 + x
        v = (x / r) * (ell * (ell + 1) / r**2 + (1 - spin**2) / r**3)
        psi, pi_ = u[0], u[1]
        # d/ds = (dr*/ds) d/dr* with dr*/ds = r;  psi' = pi, pi' = (V - y^2) psi
        return [r * pi_, r * (v - y * y) * psi]

    r0 = 1.0 + np.exp(s_min)
    rs0 = rstar(r0)
    psi0 = np.exp(-1j * y * rs0)
    u0 = np.array([psi0, -1j * y * psi0], dtype=complex)
    sol = solve_ivp(rhs, (s_min, s_max), u0, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise ConvergenceError(f"greybody integration failed for y={y}, ell={ell}: {sol.message}")
    psi, dpsi = sol.y[0, -1], sol.y[1, -1]  # psi and dpsi/dr*
    xr = y * rstar(r_end)
    hm, hmp, hp, hpp = _riccati_hankel(ell, xr)
    # psi = a_in hm + a_out hp ;  dpsi/dr* = y (a_in hm' + a_out hp')
    A = np.array([[hm, hp], [y * hmp, y * hpp]], dtype=complex)
    a_in, a_out = np.linalg.solve(A, np.array([psi, dpsi]))
    gamma = 1.0 / abs(a_in) ** 2
    defect = abs(abs(a_in) ** 2 - abs(a_out) ** 2 - 1.0) / abs(a_in) ** 2
    if not (-1e-6 <= gamma <= 1.0 + 1e-3):
        raise ConvergenceError(f"transmission probability {gamma} outside [0, 1] for y={y}, ell={ell}")
    return GreybodyResult(float(gamma), complex(a_in), complex(a_out), float(defect))


def wkb_transmission(y: float, ell: int, spin: int = 0) -> float:
    """First-order WKB (barrier-top) transmission ``1 / (1 + exp[2 pi (V0 - y^2) / sqrt(-2 V0'')])``.

    ``V0`` is the maximum of the potential and ``V0''`` its second derivative with respect to the
    tortoise coordinate at the maximum.  This is an *independent approximation* (it does not integrate
    the wave equation) used to cross-check :func:`greybody_factor`: it is accurate near and above the
    barrier top for ``ell >= 2`` with an error that decreases as ``ell`` grows, and it is unreliable deep
    below the barrier (tunnelling regime).
    """
    from scipy.optimize import minimize_scalar

    V = lambda r: float(potential(r, ell, spin))
    res = minimize_scalar(lambda r: -V(r), bounds=(1.05, 6.0 + ell), method="bounded", options={"xatol": 1e-12})
    r0, V0 = res.x, V(res.x)
    f = lambda r: 1.0 - 1.0 / r
    dV = lambda r: (V(r + 1e-6) - V(r - 1e-6)) / 2e-6
    g = lambda r: f(r) * dV(r)
    d2 = f(r0) * (g(r0 + 1e-4) - g(r0 - 1e-4)) / 2e-4  # d^2 V / d r*^2 = f d/dr (f dV/dr)
    return float(1.0 / (1.0 + np.exp(2.0 * np.pi * (V0 - y * y) / np.sqrt(-2.0 * d2))))


def greybody_table(y_values, ell_max: int, spin: int = 0, **kw) -> np.ndarray:
    """``Gamma[ell, i]`` for ``ell = spin..ell_max`` (rows start at ``ell = spin``; lower rows are 0)."""
    y_values = np.asarray(y_values, dtype=float)
    G = np.zeros((ell_max + 1, y_values.size))
    for ell in range(spin, ell_max + 1):
        for i, y in enumerate(y_values):
            G[ell, i] = greybody_factor(float(y), ell, spin, **kw).gamma
    return G


def absorption_cross_section(y_values, gamma_table: np.ndarray) -> np.ndarray:
    """sigma / r_s^2 = (pi / y^2) sum_l (2 l + 1) Gamma_l(y)  (units r_s = 1)."""
    y = np.asarray(y_values, dtype=float)
    ells = np.arange(gamma_table.shape[0])
    return (np.pi / y**2) * np.sum((2 * ells + 1)[:, None] * gamma_table, axis=0)
