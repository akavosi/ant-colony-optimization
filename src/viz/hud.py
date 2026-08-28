"""Corner-anchored HUD: iteration counter plus tour-length/diversity sparklines.

Deliberately small and fixed-position so it acts like a scoreboard
rather than competing with the graph for attention. Structurally
identical to the genetic algorithm project's HUD (same corner-plate
design, same two-sparkline layout) — only the labels change, since
"best fitness" becomes "best tour length" and the x-axis is iterations
rather than generations.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.viz import theme


class Hud:
    """Owns the HUD's matplotlib artists so they can be updated per-frame
    without being recreated (which would be slow and would flicker)."""

    def __init__(self, fig, graph_ax):
        # Text block, top-left, monospace to match the blog's numeric style.
        # Placed in FIGURE coordinates (fig.text), not axes coordinates,
        # deliberately: unlike the genetic algorithm project's static
        # landscape, this project's graph axes gets `ax.clear()`'d and
        # redrawn every frame (pheromone edges genuinely change), which
        # would silently destroy an axes-level text artist after the
        # first frame. A soft background plate keeps this legible
        # regardless of what part of the graph/edges sit underneath it.
        left, bottom, width, height = graph_ax.get_position().bounds
        text_x = left + 0.03 * width
        text_y = bottom + 0.965 * height
        self.text = fig.text(
            text_x,
            text_y,
            "",
            transform=fig.transFigure,
            fontfamily=theme.FONT_MONO_STACK,
            fontsize=11,
            color=theme.TEXT_PRIMARY,
            va="top",
            ha="left",
            linespacing=1.6,
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=theme.BG_ELEVATED,
                edgecolor=theme.RULE,
                linewidth=0.7,
                alpha=0.9,
            ),
        )

        # Sparkline axes, bottom-anchored inset, thin and unobtrusive.
        # Backed by the same soft plate as the HUD text for legibility
        # against any part of the landscape (this corner is the darkest
        # region of the colormap, same issue as the text block above).
        plate = FancyBboxPatch(
            (0.645, 0.048),
            0.32,
            0.175,
            transform=fig.transFigure,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor=theme.BG_ELEVATED,
            edgecolor=theme.RULE,
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
        )
        fig.add_artist(plate)

        self.spark_ax = fig.add_axes([0.66, 0.065, 0.28, 0.14])
        self.spark_ax.set_facecolor("none")
        self.spark_ax.set_zorder(4)
        for spine in self.spark_ax.spines.values():
            spine.set_visible(False)
        self.spark_ax.set_xticks([])
        self.spark_ax.set_yticks([])

        (self.fitness_line,) = self.spark_ax.plot(
            [], [], color=theme.COLOR_ELITE, linewidth=1.4
        )
        (self.diversity_line,) = self.spark_ax.plot(
            [], [], color=theme.TEXT_FAINT, linewidth=1.1
        )
        self.spark_ax.set_xlim(0, 1)
        self.spark_ax.set_ylim(0, 1)

        self.label = self.spark_ax.text(
            0.0,
            1.18,
            "best length · diversity",
            transform=self.spark_ax.transAxes,
            fontfamily=theme.FONT_MONO_STACK,
            fontsize=7.5,
            color=theme.TEXT_MUTED,
            va="bottom",
            ha="left",
        )

    def update(
        self,
        iteration: int,
        best_length: float,
        length_history: list[float],
        diversity_history: list[float],
    ) -> None:
        self.text.set_text(
            f"iter {iteration:>3d}\nbest {best_length:7.2f}"
        )

        n = len(length_history)
        if n >= 2:
            xs = np.linspace(0, 1, n)
            f = np.array(length_history)
            f_norm = 1.0 - (f - f.min()) / (np.ptp(f) + 1e-9)  # invert: shorter=better=higher line
            d = np.array(diversity_history)
            d_norm = (d - d.min()) / (np.ptp(d) + 1e-9)
            self.fitness_line.set_data(xs, f_norm)
            self.diversity_line.set_data(xs, d_norm)
