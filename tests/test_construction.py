"""Tests for `src.aco.construction`."""

from __future__ import annotations

import numpy as np

from src.aco.construction import construct_colony, construct_tour
from src.aco.graph import random_city_layout
from src.aco.pheromone import init_pheromone


def _layout(n=8, seed=0):
    return random_city_layout(n, np.random.default_rng(seed), min_separation=3.0)


def test_construct_tour_visits_every_city_exactly_once():
    layout = _layout()
    pheromone = init_pheromone(layout.n)
    rng = np.random.default_rng(1)

    tour = construct_tour(pheromone, layout.distances, rng, alpha=1.0, beta=2.0)

    assert sorted(tour.tolist()) == list(range(layout.n))


def test_construct_tour_starts_at_the_requested_city():
    layout = _layout()
    pheromone = init_pheromone(layout.n)
    rng = np.random.default_rng(1)

    tour = construct_tour(pheromone, layout.distances, rng, alpha=1.0, beta=2.0, start_city=3)
    assert tour[0] == 3


def test_construct_tour_is_reproducible_given_a_seed():
    layout = _layout()
    pheromone = init_pheromone(layout.n)

    a = construct_tour(pheromone, layout.distances, np.random.default_rng(5), alpha=1.0, beta=2.0)
    b = construct_tour(pheromone, layout.distances, np.random.default_rng(5), alpha=1.0, beta=2.0)
    np.testing.assert_array_equal(a, b)


def test_zero_alpha_beta_falls_back_to_valid_random_tour():
    """alpha=beta=0 makes every unvisited city equally desirable (weights
    are all zero -> the uniform-choice fallback path), which must still
    produce a valid permutation rather than crashing or looping."""
    layout = _layout()
    pheromone = init_pheromone(layout.n)
    rng = np.random.default_rng(2)

    tour = construct_tour(pheromone, layout.distances, rng, alpha=0.0, beta=0.0)
    assert sorted(tour.tolist()) == list(range(layout.n))


def test_strong_beta_prefers_nearest_neighbor_like_tours():
    """With alpha=0 (pheromone ignored) and very high beta, construction
    should behave close to greedy nearest-neighbor: each step should
    usually pick a genuinely close city, not a far one. We check this
    statistically rather than exactly, since it's still probabilistic."""
    layout = _layout(n=10, seed=3)
    pheromone = init_pheromone(layout.n)
    rng = np.random.default_rng(4)

    tour = construct_tour(pheromone, layout.distances, rng, alpha=0.0, beta=20.0)
    step_lengths = [layout.distances[tour[i], tour[i + 1]] for i in range(len(tour) - 1)]

    # Compare against a random-tour baseline: a near-greedy tour's
    # average step should be well below the average of all pairwise
    # distances in the instance.
    all_dists = layout.distances[~np.eye(layout.n, dtype=bool)]
    assert np.mean(step_lengths) < np.mean(all_dists)


def test_higher_pheromone_edge_is_preferred_when_beta_is_zero():
    """With beta=0 (distance ignored) and alpha>0, an edge with much
    higher pheromone should be chosen far more often than chance."""
    n = 5
    distances = np.ones((n, n)) - np.eye(n)  # all distances equal -> beta has no effect anyway
    pheromone = init_pheromone(n, initial_level=1.0)
    pheromone[0, 1] = pheromone[1, 0] = 100.0  # heavily favor city 1 from city 0

    rng = np.random.default_rng(0)
    choices = [construct_tour(pheromone, distances, rng, alpha=2.0, beta=0.0)[1] for _ in range(200)]
    frac_city_1 = np.mean(np.array(choices) == 1)

    assert frac_city_1 > 0.9  # should be picked nearly every time


def test_construct_colony_shape():
    layout = _layout()
    pheromone = init_pheromone(layout.n)
    rng = np.random.default_rng(0)

    tours = construct_colony(pheromone, layout.distances, rng, num_ants=15, alpha=1.0, beta=2.0)
    assert tours.shape == (15, layout.n)
    for tour in tours:
        assert sorted(tour.tolist()) == list(range(layout.n))
