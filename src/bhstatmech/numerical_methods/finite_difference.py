"""Finite-difference derivatives and error scans.

A central difference has truncation error O(h^2) and round-off error
O(eps/h); the optimal step is of order eps**(1/3) * |x|.  ``derivative_error_scan``
exposes both regimes explicitly.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """Second-order central difference ``(f(x+h) - f(x-h)) / (2h)``."""
    if h <= 0:
        raise ValueError("step h must be positive")
    return (f(x + h) - f(x - h)) / (2.0 * h)


def richardson_derivative(f: Callable[[float], float], x: float, h: float) -> float:
    """Fourth-order derivative from Richardson extrapolation of two central differences."""
    d1 = central_difference(f, x, h)
    d2 = central_difference(f, x, h / 2.0)
    return (4.0 * d2 - d1) / 3.0


def derivative_error_scan(
    f: Callable[[float], float],
    df_exact: float,
    x: float,
    steps: Sequence[float],
) -> np.ndarray:
    """Relative error of the central difference for each step in ``steps``."""
    errs = []
    for h in steps:
        approx = central_difference(f, x, h)
        errs.append(abs(approx - df_exact) / abs(df_exact))
    return np.asarray(errs)
