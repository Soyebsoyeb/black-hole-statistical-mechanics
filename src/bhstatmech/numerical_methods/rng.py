"""Reproducible random number generation (no hidden global state)."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a ``numpy.random.Generator``; an existing generator is passed through."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Create ``n`` statistically independent generators from one master seed."""
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]
