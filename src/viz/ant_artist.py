"""Animates a legible subset of ants walking their constructed tours.

Showing all `num_ants` (24 by default) moving at once would be visual
noise without adding information — the pheromone map already
summarizes the colony's collective experience. A small, fixed subset
(`NUM_ANTS_SHOWN`) is enough to read as "a colony of ants walking,"
while the pheromone edges carry the actual signal.

Motion uses arc-length parameterization along each ant's own tour
polyline, so a single parameter t in [0, 1] moves every shown ant at a
roughly constant visual speed regardless of how uneven that ant's
particular edge lengths are — a t-fraction of the way through the
animation corresponds to a t-fraction of the *distance* walked, not
naively a t-fraction of the *cities* visited.
"""

from __future__ import annotations

import numpy as np

from src.viz import theme

NUM_ANTS_SHOWN = 9


def select_shown_ants(num_ants: int, rng: np.random.Generator) -> np.ndarray:
    """Pick a fixed-size, fixed-identity subset of ant indices to animate.

    Args:
        num_ants: Total colony size.
        rng: Seeded random generator (called once; the same subset is
            reused for the whole run, so a viewer can track individual
            ants across iterations rather than seeing a new random
            subset flash up each time).

    Returns:
        Shape (min(NUM_ANTS_SHOWN, num_ants),) array of ant indices.
    """
    k = min(NUM_ANTS_SHOWN, num_ants)
    return rng.choice(num_ants, size=k, replace=False)


class TourWalker:
    """Precomputes arc-length parameterization for one ant's tour, so
    repeated position queries at different t are O(1) after setup."""

    def __init__(self, coords: np.ndarray, tour: np.ndarray):
        loop = np.append(tour, tour[0])
        points = coords[loop]
        seg_lengths = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
        self._cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._total = self._cum[-1]
        self._points = points

    def position_at(self, t: float) -> np.ndarray:
        """Position at fraction `t` (0=start city, 1=back at start city)."""
        if self._total <= 0:
            return self._points[0]
        target = t * self._total
        seg = np.searchsorted(self._cum, target, side="right") - 1
        seg = np.clip(seg, 0, len(self._points) - 2)
        seg_start_dist = self._cum[seg]
        seg_len = self._cum[seg + 1] - seg_start_dist
        local_t = 0.0 if seg_len <= 0 else (target - seg_start_dist) / seg_len
        return self._points[seg] + local_t * (self._points[seg + 1] - self._points[seg])


def draw_ants(ax, coords: np.ndarray, tours: np.ndarray, shown_indices: np.ndarray, t: float) -> None:
    """Draw the shown ants at walk-fraction `t` along their own tours.

    Args:
        ax: Matplotlib axes to draw on.
        coords: Shape (n, 2) city positions.
        tours: Shape (num_ants, n) this iteration's constructed tours.
        shown_indices: Which ant indices to actually render (see
            `select_shown_ants`).
        t: Fraction in [0, 1] of the way along each shown ant's tour.
    """
    positions = np.array(
        [TourWalker(coords, tours[i]).position_at(t) for i in shown_indices]
    )
    # Slight, fixed per-ant alpha variation so shown ants read as
    # individuals rather than identical clones, per the project brief.
    alphas = 0.55 + 0.35 * (np.arange(len(shown_indices)) % 3) / 2.0
    for pos, a in zip(positions, alphas):
        ax.scatter(
            [pos[0]], [pos[1]], s=30, color=theme.COLOR_NEW,
            alpha=a, edgecolors="none", zorder=4,
        )
