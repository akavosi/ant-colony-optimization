"""Small-multiples "filmstrip" of the pheromone map at chosen iterations.

Structurally identical to the genetic algorithm project's filmstrip —
same rationale (a reader skimming rather than watching gets the whole
story in one image), same grid layout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.aco.engine import ACOHistory
from src.viz import theme
from src.viz.graph_renderer import draw_best_tour, draw_graph


def render_filmstrip(
    history: ACOHistory,
    iterations: list[int],
    cols: int = 3,
    panel_size: float = 3.0,
    dpi: int = 170,
):
    """Render a grid of pheromone-map snapshots at the given iterations.

    Args:
        history: Output of `run_aco`.
        iterations: Iteration indices to show, in display order.
        cols: Panels per row.
        panel_size: Size of each square panel, in inches.
        dpi: Render resolution.

    Returns:
        (H, W, 4) uint8 RGBA array.
    """
    by_iter = {s.iteration: s for s in history.generations}
    rows = -(-len(iterations) // cols)
    coords = history.layout.coords
    (x_min, x_max), (y_min, y_max) = history.layout.bounds
    pad = 0.06 * (x_max - x_min)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(panel_size * cols, panel_size * rows + 0.35 * rows),
        dpi=dpi,
    )
    fig.patch.set_facecolor(theme.BG_BASE)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(iterations):
            ax.axis("off")
            continue
        it = iterations[i]
        state = by_iter.get(it, history.final)

        theme.apply_base_style(fig, ax)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        draw_graph(ax, coords, state.pheromone, history.config.initial_pheromone)
        draw_best_tour(ax, coords, state.best_ever_tour)

        ax.set_title(
            f"iter {it}",
            fontsize=11, color=theme.TEXT_PRIMARY,
            fontfamily=theme.FONT_MONO_STACK, pad=6,
        )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.28)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf
