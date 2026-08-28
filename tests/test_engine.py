"""Tests for `src.aco.engine`."""

from __future__ import annotations

import numpy as np

from src.aco.engine import ACOConfig, run_aco
from src.aco.exact_solver import solve_held_karp
from src.aco.graph import random_city_layout


def _layout(n=10, seed=0):
    return random_city_layout(n, np.random.default_rng(seed), min_separation=5.0)


def test_run_is_deterministic_given_a_seed():
    layout = _layout()
    config = ACOConfig(num_iterations=15, seed=123)

    a = run_aco(layout, config)
    b = run_aco(layout, config)

    assert len(a.generations) == len(b.generations)
    for state_a, state_b in zip(a.generations, b.generations):
        np.testing.assert_array_equal(state_a.tours, state_b.tours)
        assert state_a.best_ever_length == state_b.best_ever_length


def test_best_ever_length_never_increases():
    layout = _layout()
    history = run_aco(layout, ACOConfig(num_iterations=40, seed=3))

    lengths = [s.best_ever_length for s in history.generations]
    assert all(b <= a for a, b in zip(lengths, lengths[1:]))


def test_every_tour_is_a_valid_permutation():
    layout = _layout(n=8)
    history = run_aco(layout, ACOConfig(num_iterations=10, num_ants=12, seed=1))

    for state in history.generations:
        for tour in state.tours:
            assert sorted(tour.tolist()) == list(range(8))


def test_diversity_is_bounded_and_trends_downward():
    layout = _layout()
    history = run_aco(layout, ACOConfig(num_iterations=50, seed=2))

    diversities = [s.diversity for s in history.generations]
    assert all(0.0 <= d <= 1.0 + 1e-9 for d in diversities)
    early_avg = np.mean(diversities[:5])
    late_avg = np.mean(diversities[-5:])
    assert late_avg < early_avg


def test_pheromone_stays_finite_and_nonnegative():
    """Guards against evaporation/deposit interactions producing NaN or
    negative pheromone over a long run — an easy failure mode to
    introduce accidentally when reordering evaporate/deposit steps."""
    layout = _layout()
    history = run_aco(layout, ACOConfig(num_iterations=100, seed=4))

    for state in history.generations:
        assert np.all(np.isfinite(state.pheromone))
        assert np.all(state.pheromone >= 0.0)


def test_default_config_finds_the_provable_optimum_on_a_small_instance():
    """Regression/sanity test: on a modest instance, Elitist Ant System
    with reasonable settings should reliably find the true optimum
    within a generous iteration budget. This is the same kind of check
    as the genetic algorithm project's convergence regression test."""
    layout = random_city_layout(10, np.random.default_rng(0), min_separation=8.0)
    exact = solve_held_karp(layout.distances)

    history = run_aco(layout, ACOConfig(num_iterations=100, seed=0))

    assert np.isclose(history.final.best_ever_length, exact.length, rtol=1e-6)


def test_flagship_configuration_converges_to_the_true_optimum():
    """Pins down the actual 17-city instance and tuned defaults used
    throughout every rendered visualization in this project (see
    `ACOConfig`'s docstring for why these specific values were chosen).
    Guards against an accidental change to the defaults silently
    breaking the config the whole project's assets were generated from.
    """
    layout = random_city_layout(17, np.random.default_rng(0), min_separation=8.0)
    exact = solve_held_karp(layout.distances)

    history = run_aco(layout, ACOConfig())  # tuned defaults, seed=18

    assert np.isclose(history.final.best_ever_length, exact.length, rtol=1e-6)


def test_elite_tour_length_matches_recomputed_length():
    layout = _layout()
    history = run_aco(layout, ACOConfig(num_iterations=20, seed=6))

    from src.aco.graph import tour_length as recompute

    final = history.final
    assert np.isclose(
        recompute(final.best_ever_tour, layout.distances), final.best_ever_length
    )
