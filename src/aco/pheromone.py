"""Pheromone matrix operations: the shared "memory" ants read and write.

The pheromone matrix is symmetric (this is an undirected TSP instance —
an edge (i, j) carries one pheromone level regardless of which
direction it's traversed) and is the single most important object in
this project to visualize well: it's the direct analogue of the
fitness landscape in the genetic algorithm project.
"""

from __future__ import annotations

import numpy as np


def init_pheromone(n: int, initial_level: float = 1.0) -> np.ndarray:
    """Uniform initial pheromone on every edge (including the diagonal,
    which is simply never read since ants never travel from a city to
    itself — left nonzero rather than masked out, to keep this function
    trivial; `probabilities` in `construction.py` excludes visited
    cities directly).

    Args:
        n: Number of cities.
        initial_level: Starting pheromone level on every edge.

    Returns:
        Shape (n, n) symmetric array.
    """
    return np.full((n, n), initial_level, dtype=float)


def evaporate(pheromone: np.ndarray, rho: float) -> np.ndarray:
    """Apply evaporation: `pheromone *= (1 - rho)` on every edge.

    Args:
        pheromone: Shape (n, n) current pheromone levels.
        rho: Evaporation rate in [0, 1]. Higher means faster forgetting.

    Returns:
        A new array (does not mutate the input), so callers can compare
        pre/post-evaporation states if needed for diagnostics.
    """
    return pheromone * (1.0 - rho)


def deposit(
    pheromone: np.ndarray,
    tours: np.ndarray,
    tour_lengths: np.ndarray,
    deposit_strength: float = 1.0,
) -> np.ndarray:
    """Deposit pheromone along each ant's tour, inversely weighted by length.

    Every edge on a given ant's tour receives `deposit_strength /
    tour_length` additional pheromone. Shorter tours deposit more per
    edge, which is the entire mechanism by which the colony learns —
    edges that tend to appear on short tours accumulate pheromone
    faster than edges that only appear on long ones.

    Args:
        pheromone: Shape (n, n) current pheromone levels.
        tours: Shape (num_ants, n) array of city-index permutations.
        tour_lengths: Shape (num_ants,) total length of each tour.
        deposit_strength: Scales the total amount deposited (denoted Q
            in the ACO literature).

    Returns:
        A new pheromone array with deposits added (input not mutated).
    """
    updated = pheromone.copy()
    for tour, length in zip(tours, tour_lengths):
        amount = deposit_strength / length
        a, b = tour, np.roll(tour, -1)
        updated[a, b] += amount
        updated[b, a] += amount
    return updated


def pheromone_entropy(pheromone: np.ndarray) -> float:
    """Average, normalized entropy of each city's outgoing edge distribution.

    For each city, normalize its row of the pheromone matrix (excluding
    the self-edge) into a probability distribution and compute its
    Shannon entropy; average over all cities and normalize by the
    maximum possible entropy (log(n-1), i.e. a uniform distribution) to
    land in [0, 1]. This is the ACO analogue of the genetic algorithm
    project's population-diversity metric: 1.0 means pheromone is
    spread uniformly (the colony has no preference yet, maximally
    "diverse"); values near 0 mean the colony has committed heavily to
    a small number of edges per city.

    Args:
        pheromone: Shape (n, n) pheromone levels.

    Returns:
        A scalar in [0, 1].
    """
    n = pheromone.shape[0]
    mask = ~np.eye(n, dtype=bool)
    row_entropies = np.empty(n)
    max_entropy = np.log(n - 1)
    for i in range(n):
        row = pheromone[i][mask[i]]
        total = row.sum()
        p = row / total if total > 0 else np.full_like(row, 1.0 / len(row))
        p_safe = np.where(p > 0, p, 1.0)  # avoid log(0); those terms contribute 0 anyway
        row_entropies[i] = -np.sum(p * np.log(p_safe))
    return float(np.mean(row_entropies) / max_entropy) if max_entropy > 0 else 0.0


def deposit_elite_bonus(
    pheromone: np.ndarray,
    best_tour: np.ndarray,
    best_length: float,
    num_elite: float,
    deposit_strength: float = 1.0,
) -> np.ndarray:
    """Extra deposit along the best-known tour (Elitist Ant System).

    This is the direct analogue of elitism in the genetic algorithm
    project: it doesn't just let the best solution survive unchanged,
    it actively reinforces the paths that produced it, which is what
    lets a single excellent tour keep influencing the colony's search
    long after it was found.

    Args:
        pheromone: Shape (n, n) current pheromone levels.
        best_tour: Shape (n,) the best tour found so far.
        best_length: That tour's length.
        num_elite: Weight of the elite deposit, in the same units as
            "number of extra ants depositing on this tour" (the
            standard Elitist Ant System parameterization).
        deposit_strength: Same Q as in `deposit`.

    Returns:
        A new pheromone array with the elite bonus added.
    """
    updated = pheromone.copy()
    amount = num_elite * deposit_strength / best_length
    a, b = best_tour, np.roll(best_tour, -1)
    updated[a, b] += amount
    updated[b, a] += amount
    return updated
