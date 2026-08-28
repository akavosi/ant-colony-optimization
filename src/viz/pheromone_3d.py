"""Renders the pheromone matrix itself as a 3D bar chart.

This is the ACO project's answer to "what's the natural 3D visual
here" — there's no continuous landscape the way there was for the
genetic algorithm's fitness function (TSP is discrete), but the
pheromone matrix is exactly the kind of mathematical object a 3D bar
chart is made for: two city indices as the base plane, pheromone level
as height. Watching a nearly-flat field of 289 cells collapse into a
sparse handful of dramatic spikes — precisely the edges of the optimal
tour — makes the "the colony's memory becomes the answer" idea
concrete in a way the 2D edge map only implies.
"""

from __future__ import annotations

import numpy as np

from src.viz import theme

DEFAULT_ELEV = 32
DEFAULT_AZIM = -50


def draw_pheromone_bars(
    ax,
    pheromone: np.ndarray,
    best_tour: np.ndarray,
    bar_width: float = 0.75,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
) -> None:
    """Draw every off-diagonal pheromone cell as a 3D bar.

    Args:
        ax: A 3D axes from `theme.setup_3d_axes`.
        pheromone: Shape (n, n) pheromone levels.
        best_tour: Shape (n,) best-known tour, used only to color bars
            on the tour's edges gold, distinguishing "the answer" from
            background pheromone at a glance.
        bar_width: Footprint of each bar in the (i, j) grid.
        elev, azim: Camera angles (degrees).
    """
    n = pheromone.shape[0]
    tour_edges = set()
    for a, b in zip(best_tour, np.roll(best_tour, -1)):
        tour_edges.add((int(a), int(b)))
        tour_edges.add((int(b), int(a)))

    xs, ys, zs, dzs, colors, zorders = [], [], [], [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xs.append(i - bar_width / 2)
            ys.append(j - bar_width / 2)
            zs.append(0.0)
            dzs.append(float(pheromone[i, j]))
            is_tour_edge = (i, j) in tour_edges
            colors.append(theme.COLOR_ELITE if is_tour_edge else theme.RULE_STRONG)
            zorders.append(5.0 if is_tour_edge else 1.0 + float(pheromone[i, j]))

    # Draw background (non-tour) bars first, tour-edge bars last, so the
    # gold "answer" bars always render on top regardless of camera angle
    # — with computed_zorder disabled (see theme.setup_3d_axes), draw
    # order plus an explicit zorder is what actually controls this.
    order = np.argsort(zorders)
    xs, ys, zs, dzs = (np.array(a)[order] for a in (xs, ys, zs, dzs))
    colors = [colors[k] for k in order]

    ax.bar3d(
        xs, ys, zs, bar_width, bar_width, dzs,
        color=colors, shade=True, zorder=2, edgecolor="none",
    )

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_zlim(0, max(dzs.max(), 1.0) * 1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.7))
