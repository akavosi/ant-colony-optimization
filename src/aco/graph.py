"""City layout and distance computation for the TSP instance ACO searches.

Kept deliberately simple: cities are points in the plane, distances are
Euclidean. The interesting complexity in this project is in the search
algorithm and the visualization, not the problem instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CityLayout:
    """A fixed set of city coordinates and their pairwise distances.

    Attributes:
        coords: Shape (n, 2) array of (x, y) positions.
        distances: Shape (n, n) symmetric matrix of Euclidean distances,
            with zero diagonal.
        bounds: ((x_min, x_max), (y_min, y_max)), for consistent
            rendering across every visualization in this project.
    """

    coords: np.ndarray
    distances: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]

    @property
    def n(self) -> int:
        return len(self.coords)


def random_city_layout(
    n: int,
    rng: np.random.Generator,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 100.0), (0.0, 100.0)),
    min_separation: float = 8.0,
) -> CityLayout:
    """Sample `n` city positions, rejecting layouts with near-duplicate cities.

    Args:
        n: Number of cities.
        rng: Seeded random generator.
        bounds: ((x_min, x_max), (y_min, y_max)) placement domain.
        min_separation: Minimum allowed distance between any two cities.
            Purely a visualization concern — two cities placed on top of
            each other would make the tour and pheromone edges illegible
            regardless of how good the underlying algorithm is.

    Returns:
        A `CityLayout`.

    Raises:
        RuntimeError: if a well-separated layout isn't found within a
            generous number of attempts (indicates `n` is too large for
            `bounds` at the requested `min_separation`).
    """
    (x_min, x_max), (y_min, y_max) = bounds

    for _ in range(500):
        coords = np.stack(
            [rng.uniform(x_min, x_max, n), rng.uniform(y_min, y_max, n)], axis=1
        )
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        if dists.min() >= min_separation:
            np.fill_diagonal(dists, 0.0)
            return CityLayout(coords=coords, distances=dists, bounds=bounds)

    raise RuntimeError(
        f"Could not place {n} cities with min_separation={min_separation} "
        f"in bounds={bounds} after 500 attempts. Reduce n, reduce "
        f"min_separation, or enlarge bounds."
    )


def tour_length(tour: np.ndarray, distances: np.ndarray) -> float:
    """Total length of a closed tour (returns to its starting city).

    Args:
        tour: Shape (n,) permutation of city indices.
        distances: Shape (n, n) distance matrix.

    Returns:
        Sum of edge lengths, including the closing edge from the last
        city back to the first.
    """
    return float(distances[tour, np.roll(tour, -1)].sum())
