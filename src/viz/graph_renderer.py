"""Renders the city layout and pheromone-weighted edges.

The pheromone matrix is the direct analogue of the genetic algorithm
project's fitness landscape, and it has the same latent risk of visual
noise: with n=17 cities there are up to 136 possible edges, and drawing
all of them — even at low opacity — reads as visual clutter rather than
information (the exact failure mode the Rastrigin landscape's first
draft hit). Two deliberate choices avoid it here:

1. Edge prominence is computed *relative to the current spread* of
   pheromone values, not their absolute magnitude. When pheromone is
   still near-uniform (early in a run), nothing stands out and the
   canvas reads as calm; prominence emerges only as the colony's
   preferences actually differentiate.
2. Regardless of the distribution's shape, at most `max_edges_drawn`
   edges are ever rendered — the strongest candidates only, not a
   faint wash across every possible connection.
"""

from __future__ import annotations

import numpy as np

from src.viz import theme


def _edge_prominence(
    pheromone: np.ndarray, max_edges_drawn: int, initial_level: float
) -> list[tuple[int, int, float]]:
    """Return (i, j, weight) for the most pheromone-reinforced edges.

    Weight measures how far an edge's pheromone has risen *above the
    initial uniform baseline*, relative to the current global maximum —
    not a min-max normalization over just the drawn subset. That
    distinction matters at the very start of a run: with a naive
    min-max-of-the-drawn-edges normalization, a perfectly uniform
    pheromone matrix (iteration 0, before any ant has walked) has zero
    spread, which degenerates into "every candidate edge gets maximum
    weight" — the opposite of the intended behavior, and exactly the
    kind of busy, uninformative first impression this design is meant
    to avoid. Measuring against the fixed initial baseline instead
    means an undifferentiated matrix correctly produces no prominent
    edges at all (nothing has been learned yet, so nothing is drawn).
    """
    n = pheromone.shape[0]
    iu = np.triu_indices(n, k=1)
    values = pheromone[iu]

    global_max = values.max()
    denom = global_max - initial_level
    if denom <= 1e-9:
        return []  # nothing has differentiated from the baseline yet

    k = min(max_edges_drawn, len(values))
    top_idx = np.argpartition(-values, k - 1)[:k]

    edges = []
    for idx in top_idx:
        weight = float(np.clip((values[idx] - initial_level) / denom, 0.0, 1.0))
        if weight <= 0.02:
            continue  # hasn't meaningfully risen above baseline; skip rather than draw a near-invisible line
        i, j = iu[0][idx], iu[1][idx]
        edges.append((int(i), int(j), weight))
    return edges


def draw_graph(
    ax,
    coords: np.ndarray,
    pheromone: np.ndarray,
    initial_level: float,
    max_edges_drawn: int = 45,
    min_linewidth: float = 0.4,
    max_linewidth: float = 3.2,
    min_alpha: float = 0.05,
    max_alpha: float = 0.85,
) -> None:
    """Draw cities and their most-prominent pheromone-weighted edges.

    Args:
        ax: Matplotlib axes to draw on.
        coords: Shape (n, 2) city positions.
        pheromone: Shape (n, n) current pheromone levels.
        initial_level: The pheromone level every edge started at (see
            `_edge_prominence` — prominence is measured relative to
            this baseline, not the drawn subset's own min/max).
        max_edges_drawn: Hard cap on edges rendered, regardless of the
            pheromone distribution's shape.
        min_linewidth, max_linewidth: Edge width range.
        min_alpha, max_alpha: Edge opacity range.
    """
    for i, j, w in _edge_prominence(pheromone, max_edges_drawn, initial_level):
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=theme.ACCENT,
            linewidth=min_linewidth + w * (max_linewidth - min_linewidth),
            alpha=min_alpha + w * (max_alpha - min_alpha),
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=46, color=theme.TEXT_PRIMARY, edgecolors=theme.BG_ELEVATED,
        linewidths=1.0, zorder=3,
    )


def draw_best_tour(ax, coords: np.ndarray, tour: np.ndarray, alpha: float = 0.95) -> None:
    """Overlay the best-known tour as a closed gold polyline.

    Args:
        ax: Matplotlib axes to draw on.
        coords: Shape (n, 2) city positions.
        tour: Shape (n,) permutation of city indices.
        alpha: Line opacity.
    """
    loop = np.append(tour, tour[0])
    ax.plot(
        coords[loop, 0], coords[loop, 1],
        color=theme.COLOR_ELITE, linewidth=2.2, alpha=alpha,
        solid_capstyle="round", zorder=2,
    )
