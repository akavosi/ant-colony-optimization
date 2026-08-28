"""Tests for `src.aco.exact_solver`.

The Held-Karp implementation is checked against independent brute-force
enumeration on small instances — this ground truth feeds directly into
the convergence chart's "distance to proven optimum" story, so it needs
to be verified against something other than itself.
"""

from __future__ import annotations

import numpy as np

from src.aco.exact_solver import solve_brute_force, solve_held_karp
from src.aco.graph import random_city_layout, tour_length


def test_held_karp_matches_brute_force_on_random_instances():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = rng.integers(4, 9)  # brute force is only feasible this small
        layout = random_city_layout(int(n), rng, min_separation=3.0)

        hk = solve_held_karp(layout.distances)
        bf = solve_brute_force(layout.distances)

        assert np.isclose(hk.length, bf.length), f"seed={seed} n={n}: {hk.length} vs {bf.length}"


def test_held_karp_tour_is_a_valid_permutation():
    rng = np.random.default_rng(0)
    layout = random_city_layout(10, rng, min_separation=5.0)
    result = solve_held_karp(layout.distances)

    assert sorted(result.tour.tolist()) == list(range(10))


def test_held_karp_reported_length_matches_recomputed_tour_length():
    rng = np.random.default_rng(1)
    layout = random_city_layout(9, rng, min_separation=5.0)
    result = solve_held_karp(layout.distances)

    recomputed = tour_length(result.tour, layout.distances)
    assert np.isclose(result.length, recomputed)


def test_held_karp_trivial_cases():
    # Two cities: the only tour is back-and-forth.
    distances = np.array([[0.0, 5.0], [5.0, 0.0]])
    result = solve_held_karp(distances)
    assert np.isclose(result.length, 10.0)


def test_held_karp_beats_or_matches_a_greedy_nearest_neighbor_heuristic():
    """Sanity check with a real, independent baseline: a simple greedy
    nearest-neighbor construction can never do better than the true
    optimum, so Held-Karp's answer must be <= it on every instance.
    """
    rng = np.random.default_rng(7)
    layout = random_city_layout(12, rng, min_separation=5.0)

    # Greedy nearest-neighbor tour starting from city 0.
    unvisited = set(range(1, 12))
    tour = [0]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: layout.distances[last, c])
        tour.append(nxt)
        unvisited.remove(nxt)
    greedy_length = tour_length(np.array(tour), layout.distances)

    exact = solve_held_karp(layout.distances)
    assert exact.length <= greedy_length + 1e-9
