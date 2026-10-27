"""Gaussian lattice models of quantum-field entanglement."""

from .harmonic_chain import (
    block_entropy, entropy_from_symplectic, entropy_vs_block_size, ground_state_covariance,
    laplacian, stiffness_matrix, symplectic_eigenvalues, two_oscillator_entropy_exact,
)

__all__ = [n for n in dir() if not n.startswith("_")]
