"""Probabilistic tour construction.

Each ant builds a complete tour one city at a time. At each step, the
probability of moving to an unvisited city j from the current city i is
proportional to \\(\\tau_{ij}^\\alpha \\cdot \\eta_{ij}^\\beta\\), where
\\(\\tau\\) is pheromone and \\(\\eta = 1/d\\) is a "visibility" heuristic
that favors nearby cities independent of pheromone. This is the
random-proportional rule from Ant System (Dorigo, Maniezzo & Colorni).
"""

from __future__ import annotations

import numpy as np


def _visibility(distances: np.ndarray) -> np.ndarray:
    """Heuristic desirability 1/d, with the (unused) zero diagonal
    safely mapped to 0 rather than producing a division warning."""
    with np.errstate(divide="ignore"):
        eta = np.where(distances > 0, 1.0 / distances, 0.0)
    return eta


def construct_tour(
    pheromone: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    start_city: int = 0,
) -> np.ndarray:
    """Build one ant's tour via the random-proportional rule.

    Args:
        pheromone: Shape (n, n) current pheromone levels.
        distances: Shape (n, n) distance matrix.
        rng: Seeded random generator.
        alpha: Exponent controlling how strongly pheromone influences
            the choice. alpha=0 ignores pheromone entirely.
        beta: Exponent controlling how strongly the nearness heuristic
            influences the choice. beta=0 ignores distance entirely
            (pure pheromone-following).
        start_city: Which city every ant starts from. Fixed at 0 by
            convention; since a tour's length doesn't depend on its
            starting point, this doesn't bias the search.

    Returns:
        Shape (n,) permutation of city indices, in visit order.
    """
    n = pheromone.shape[0]
    eta = _visibility(distances)
    desirability = (pheromone**alpha) * (eta**beta)

    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    tour[0] = start_city
    visited[start_city] = True
    current = start_city

    for step in range(1, n):
        weights = desirability[current].copy()
        weights[visited] = 0.0
        total = weights.sum()
        if total <= 0.0:
            # Degenerate case (e.g. alpha=beta=0, or numerical underflow):
            # fall back to uniform choice among unvisited cities so
            # construction never gets stuck.
            candidates = np.flatnonzero(~visited)
            next_city = int(rng.choice(candidates))
        else:
            probabilities = weights / total
            next_city = int(rng.choice(n, p=probabilities))

        tour[step] = next_city
        visited[next_city] = True
        current = next_city

    return tour


def construct_colony(
    pheromone: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
    num_ants: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Construct a full colony's worth of tours for one iteration.

    Args:
        pheromone, distances, rng, alpha, beta: See `construct_tour`.
        num_ants: How many tours to construct.

    Returns:
        Shape (num_ants, n) array of tours.
    """
    n = pheromone.shape[0]
    tours = np.empty((num_ants, n), dtype=int)
    for i in range(num_ants):
        tours[i] = construct_tour(pheromone, distances, rng, alpha, beta)
    return tours
