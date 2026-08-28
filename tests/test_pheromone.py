"""Tests for `src.aco.pheromone`."""

from __future__ import annotations

import numpy as np

from src.aco.pheromone import deposit, deposit_elite_bonus, evaporate, init_pheromone


def test_init_pheromone_is_uniform():
    pheromone = init_pheromone(5, initial_level=2.5)
    assert pheromone.shape == (5, 5)
    assert np.all(pheromone == 2.5)


def test_evaporate_scales_correctly_and_does_not_mutate_input():
    pheromone = np.full((4, 4), 10.0)
    original = pheromone.copy()

    result = evaporate(pheromone, rho=0.3)

    np.testing.assert_array_almost_equal(result, np.full((4, 4), 7.0))
    np.testing.assert_array_equal(pheromone, original)  # input untouched


def test_evaporate_zero_rate_is_a_no_op():
    pheromone = np.full((4, 4), 5.0)
    result = evaporate(pheromone, rho=0.0)
    np.testing.assert_array_almost_equal(result, pheromone)


def test_evaporate_full_rate_zeros_everything():
    pheromone = np.full((4, 4), 5.0)
    result = evaporate(pheromone, rho=1.0)
    np.testing.assert_array_almost_equal(result, np.zeros((4, 4)))


def test_deposit_adds_more_pheromone_for_shorter_tours():
    pheromone = init_pheromone(4, initial_level=0.0)
    tours = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    # Same tour, but we'll compare against a longer-length version to
    # confirm the *shorter* one deposits more per unit.
    short = deposit(pheromone, tours[:1], np.array([10.0]))
    long_ = deposit(pheromone, tours[:1], np.array([100.0]))

    assert short[0, 1] > long_[0, 1]


def test_deposit_is_symmetric():
    pheromone = init_pheromone(4, initial_level=0.0)
    tours = np.array([[0, 2, 1, 3]])
    result = deposit(pheromone, tours, np.array([10.0]))
    np.testing.assert_array_almost_equal(result, result.T)


def test_deposit_only_touches_edges_on_the_tour():
    pheromone = init_pheromone(5, initial_level=0.0)
    tour = np.array([0, 1, 2, 3, 4])  # edges: 0-1,1-2,2-3,3-4,4-0
    result = deposit(pheromone, tour[None, :], np.array([10.0]))

    # Edge 0-2 is not on this tour and should remain untouched.
    assert result[0, 2] == 0.0
    assert result[0, 1] > 0.0  # this edge IS on the tour


def test_deposit_does_not_mutate_input():
    pheromone = init_pheromone(4, initial_level=1.0)
    original = pheromone.copy()
    deposit(pheromone, np.array([[0, 1, 2, 3]]), np.array([10.0]))
    np.testing.assert_array_equal(pheromone, original)


def test_elite_bonus_increases_with_num_elite():
    pheromone = init_pheromone(4, initial_level=0.0)
    tour = np.array([0, 1, 2, 3])

    small_bonus = deposit_elite_bonus(pheromone, tour, best_length=10.0, num_elite=1.0)
    large_bonus = deposit_elite_bonus(pheromone, tour, best_length=10.0, num_elite=5.0)

    assert large_bonus[0, 1] > small_bonus[0, 1]
    assert np.isclose(large_bonus[0, 1], 5 * small_bonus[0, 1])
