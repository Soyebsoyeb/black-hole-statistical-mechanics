"""Checked wrappers around SciPy root finding, minimisation and ODE solvers.

Every wrapper raises :class:`~bhstatmech.exceptions.ConvergenceError` rather
than silently returning an unconverged answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp

from ..exceptions import ConvergenceError


@dataclass(frozen=True)
class RootResult:
    root: float
    iterations: int
    converged: bool


def find_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-14,
    rtol: float = 4 * np.finfo(float).eps,
    maxiter: int = 200,
) -> RootResult:
    """Brent root finding on a sign-changing bracket ``[a, b]``."""
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb) and fa != 0 and fb != 0:
        raise ConvergenceError(f"root not bracketed on [{a}, {b}]: f(a)={fa:.3e}, f(b)={fb:.3e}")
    root, info = optimize.brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter, full_output=True)
    if not info.converged:
        raise ConvergenceError(f"brentq did not converge: {info.flag}")
    return RootResult(float(root), int(info.iterations), True)


def minimize_bounded(
    f: Callable[[float], float], a: float, b: float, xtol: float = 1e-10
) -> tuple[float, float]:
    """Bounded scalar minimisation; returns ``(x_min, f_min)``."""
    res = optimize.minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": xtol})
    if not res.success:
        raise ConvergenceError(f"minimize_scalar failed: {res.message}")
    return float(res.x), float(res.fun)


def solve_ode(fun, t_span, y0, rtol: float = 1e-10, atol: float = 1e-12, method: str = "DOP853", **kw):
    """``solve_ivp`` that raises on failure."""
    sol = solve_ivp(fun, t_span, y0, rtol=rtol, atol=atol, method=method, **kw)
    if not sol.success:
        raise ConvergenceError(f"ODE integration failed: {sol.message}")
    return sol


def tolerance_sensitivity(
    func: Callable[[float], float], tolerances: Sequence[float]
) -> np.ndarray:
    """Evaluate ``func(tol)`` for each tolerance (sensitivity study helper)."""
    return np.asarray([func(t) for t in tolerances])
