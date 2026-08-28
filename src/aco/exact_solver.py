"""Exact TSP solver via Held-Karp dynamic programming.

Used to compute a *provable* optimal tour length for the convergence
chart — the same role the known analytic global optimum played for the
genetic algorithm project, but TSP has no closed-form optimum, so it's
computed exactly instead. O(2^n * n^2) time and O(2^n * n) space, which
is trivial for the n=17 instance used throughout this project (~37M
basic operations, well under a second) but becomes impractical well
before n=25 — this is an exact solver for small instances, not a
general-purpose TSP algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class ExactSolution:
    """Result of an exact (or brute-force) TSP solve."""

    tour: np.ndarray  # shape (n,), permutation of city indices
    length: float


def solve_held_karp(distances: np.ndarray) -> ExactSolution:
    """Solve TSP exactly via Held-Karp dynamic programming.

    Fixes city 0 as the tour's start (valid since a tour's length is
    invariant to its starting point) and builds up optimal partial
    tours over increasing subsets of the remaining cities, represented
    as bitmasks.

    Args:
        distances: Shape (n, n) symmetric distance matrix.

    Returns:
        An `ExactSolution` with the optimal tour and its length.
    """
    n = distances.shape[0]
    if n <= 1:
        return ExactSolution(tour=np.arange(n), length=0.0)
    if n == 2:
        return ExactSolution(tour=np.array([0, 1]), length=2 * distances[0, 1])

    # dp[mask][j]: min cost of a path starting at city 0, visiting
    # exactly the cities in `mask` (which always includes city 0 and
    # j), and ending at city j. Cities are 0-indexed; bit i of `mask`
    # (for i >= 1) represents city i being visited; city 0 is implicit.
    other_cities = list(range(1, n))
    num_others = n - 1
    size = 1 << num_others

    NEG = float("inf")
    dp = np.full((size, num_others), NEG)
    parent = np.full((size, num_others), -1, dtype=int)

    # Base case: path 0 -> city (bit index i), visiting only that city.
    for i in range(num_others):
        dp[1 << i, i] = distances[0, other_cities[i]]

    for mask in range(1, size):
        for j in range(num_others):
            if not (mask & (1 << j)):
                continue
            cost_j = dp[mask, j]
            if cost_j == NEG:
                continue
            city_j = other_cities[j]
            remaining = ((size - 1) ^ mask)
            k = 0
            rem = remaining
            while rem:
                if rem & 1:
                    new_mask = mask | (1 << k)
                    candidate = cost_j + distances[city_j, other_cities[k]]
                    if candidate < dp[new_mask, k]:
                        dp[new_mask, k] = candidate
                        parent[new_mask, k] = j
                rem >>= 1
                k += 1

    full_mask = size - 1
    best_cost = NEG
    best_j = -1
    for j in range(num_others):
        cost = dp[full_mask, j] + distances[other_cities[j], 0]
        if cost < best_cost:
            best_cost = cost
            best_j = j

    # Reconstruct the tour by walking `parent` back from (full_mask, best_j).
    tour_rest = []
    mask, j = full_mask, best_j
    while j != -1:
        tour_rest.append(other_cities[j])
        prev_j = parent[mask, j]
        mask ^= (1 << j)
        j = prev_j
    tour_rest.reverse()

    tour = np.array([0] + tour_rest)
    return ExactSolution(tour=tour, length=float(best_cost))


def solve_brute_force(distances: np.ndarray) -> ExactSolution:
    """Solve TSP exactly by trying every permutation. Reference-quality,
    exponential-in-the-worst-way; only used to independently verify
    `solve_held_karp` on small instances in tests, never in production
    code paths.
    """
    n = distances.shape[0]
    best_tour = None
    best_length = float("inf")
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = sum(distances[tour[i], tour[(i + 1) % n]] for i in range(n))
        if length < best_length:
            best_length = length
            best_tour = tour
    return ExactSolution(tour=np.array(best_tour), length=float(best_length))
