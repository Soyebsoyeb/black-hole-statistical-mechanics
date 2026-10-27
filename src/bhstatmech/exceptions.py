"""Exception hierarchy used throughout the package."""

from __future__ import annotations


class BHStatMechError(Exception):
    """Base class for all package-specific errors."""


class PhysicalParameterError(BHStatMechError, ValueError):
    """A physical parameter is outside its allowed range (e.g. negative mass)."""


class DimensionError(BHStatMechError, TypeError):
    """Dimensional analysis check failed."""


class ConvergenceError(BHStatMechError, RuntimeError):
    """A numerical method failed to converge within its stated tolerance."""


class QuantumStateError(BHStatMechError, ValueError):
    """A state or operator violates a quantum-mechanical invariant."""
