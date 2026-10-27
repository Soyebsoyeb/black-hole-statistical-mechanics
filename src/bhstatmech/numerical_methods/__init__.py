"""Numerical utilities: finite differences, checked solvers, reproducible RNG."""

from .finite_difference import (
    central_difference,
    derivative_error_scan,
    richardson_derivative,
)
from .rng import make_rng, spawn_rngs
from .solvers import RootResult, find_root, minimize_bounded, solve_ode, tolerance_sensitivity

__all__ = [
    "central_difference", "richardson_derivative", "derivative_error_scan",
    "make_rng", "spawn_rngs",
    "RootResult", "find_root", "minimize_bounded", "solve_ode", "tolerance_sensitivity",
]
