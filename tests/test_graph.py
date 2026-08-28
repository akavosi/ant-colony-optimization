"""Tests for `src.aco.graph`."""

from __future__ import annotations

import numpy as np
import pytest

from src.aco.graph import random_city_layout, tour_length


def test_layout_has_correct_shape_and_bounds():
    rng = np.random.default_rng(0)
    bounds = ((0.0, 100.0), (0.0, 100.0))
    layout = random_city_layout(17, rng, bounds=bounds, min_separation=8.0)

    assert layout.coords.shape == (17, 2)
    assert layout.distances.shape == (17, 17)
    (x_min, x_max), (y_min, y_max) = bounds
    assert np.all(layout.coords[:, 0] >= x_min) and np.all(layout.coords[:, 0] <= x_max)
    assert np.all(layout.coords[:, 1] >= y_min) and np.all(layout.coords[:, 1] <= y_max)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    layout = random_city_layout(10, rng, min_separation=5.0)

    np.testing.assert_array_almost_equal(layout.distances, layout.distances.T)
    np.testing.assert_array_almost_equal(np.diag(layout.distances), np.zeros(10))


def test_distance_matrix_matches_euclidean_distance():
    rng = np.random.default_rng(2)
    layout = random_city_layout(8, rng, min_separation=5.0)

    i, j = 2, 5
    expected = np.linalg.norm(layout.coords[i] - layout.coords[j])
    assert np.isclose(layout.distances[i, j], expected)


def test_min_separation_is_respected():
    rng = np.random.default_rng(3)
    layout = random_city_layout(12, rng, min_separation=8.0)

    off_diagonal = layout.distances[~np.eye(12, dtype=bool)]
    assert off_diagonal.min() >= 8.0


def test_layout_is_reproducible_given_a_seed():
    a = random_city_layout(10, np.random.default_rng(42), min_separation=5.0)
    b = random_city_layout(10, np.random.default_rng(42), min_separation=5.0)
    np.testing.assert_array_equal(a.coords, b.coords)


def test_impossible_layout_raises_rather_than_looping_forever():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        # 50 cities with min_separation=50 cannot fit in a 10x10 box.
        random_city_layout(50, rng, bounds=((0.0, 10.0), (0.0, 10.0)), min_separation=50.0)


def test_tour_length_is_a_closed_loop():
    # A unit square: length should be exactly 4 regardless of starting city.
    distances = np.array([
        [0.0, 1.0, np.sqrt(2), 1.0],
        [1.0, 0.0, 1.0, np.sqrt(2)],
        [np.sqrt(2), 1.0, 0.0, 1.0],
        [1.0, np.sqrt(2), 1.0, 0.0],
    ])
    tour = np.array([0, 1, 2, 3])
    assert np.isclose(tour_length(tour, distances), 4.0)


def test_tour_length_is_invariant_to_rotation_and_direction():
    distances = np.array([
        [0.0, 1.0, np.sqrt(2), 1.0],
        [1.0, 0.0, 1.0, np.sqrt(2)],
        [np.sqrt(2), 1.0, 0.0, 1.0],
        [1.0, np.sqrt(2), 1.0, 0.0],
    ])
    original = tour_length(np.array([0, 1, 2, 3]), distances)
    rotated = tour_length(np.array([2, 3, 0, 1]), distances)
    reversed_tour = tour_length(np.array([0, 3, 2, 1]), distances)

    assert np.isclose(original, rotated)
    assert np.isclose(original, reversed_tour)
