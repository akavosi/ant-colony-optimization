"""Centralized random-number-generator control for reproducibility.

Every stochastic component in this project (population initialization,
selection, crossover, mutation) draws from a `numpy.random.Generator`
instance created here, seeded explicitly. Given the same seed, a run
is bit-for-bit reproducible, so every animation in the README and blog
post can be regenerated exactly from the published seed.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a fresh, independent NumPy random generator.

    Args:
        seed: Integer seed. The same seed always reproduces the same
            sequence of draws, and therefore the same animation.

    Returns:
        A `numpy.random.Generator` (PCG64-backed) seeded deterministically.
    """
    return np.random.default_rng(seed)
