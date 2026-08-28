"""Static 2D chart: gap to the proven optimum (log scale) and pheromone
diversity over iterations. Mirrors the genetic algorithm project's
convergence chart exactly in structure and rationale — see that
project's `convergence_chart.py` for why the log scale matters (a
linear plot goes flat once the gap is small, hiding exactly the
fine-tuning behavior worth showing).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.aco.engine import ACOHistory
from src.aco.exact_solver import solve_held_karp
from src.viz import theme


def render_convergence_chart(
    history: ACOHistory, dpi: int = 180, figsize: tuple[float, float] = (9.0, 5.4)
):
    """Render the annotated convergence chart as an RGBA array.

    Args:
        history: Output of `run_aco`.
        dpi: Render resolution.
        figsize: Figure size in inches.

    Returns:
        (H, W, 4) uint8 RGBA array.
    """
    exact = solve_held_karp(history.layout.distances)

    iters = [s.iteration for s in history.generations]
    gap = [max(s.best_ever_length - exact.length, 1e-9) for s in history.generations]
    diversity = [s.diversity for s in history.generations]

    converged_iter = next(
        (s.iteration for s in history.generations
         if np.isclose(s.best_ever_length, exact.length, rtol=1e-6)),
        None,
    )
    jumps = [
        (history.generations[i].iteration,
         history.generations[i - 1].best_ever_length - history.generations[i].best_ever_length)
        for i in range(1, len(history.generations))
    ]
    commit_iter = max(jumps, key=lambda t: t[1])[0] if jumps else None

    fig, (ax_gap, ax_div) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(theme.BG_BASE)

    for ax in (ax_gap, ax_div):
        ax.set_facecolor(theme.BG_BASE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme.RULE_STRONG)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=theme.TEXT_MUTED, labelsize=9.5, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(theme.FONT_MONO_STACK)
        ax.grid(axis="y", color=theme.RULE, linewidth=0.6, alpha=0.6)

    ax_gap.plot(iters, gap, color=theme.ACCENT, linewidth=1.8, zorder=3)
    ax_gap.set_yscale("log")
    ax_gap.set_ylabel(
        "gap to proven optimum\n(tour length, log)",
        fontsize=10, color=theme.TEXT_BODY, fontfamily=theme.FONT_BODY_STACK,
    )
    ax_gap.set_title(
        "Convergence: how far the best tour is from the provable optimum",
        fontsize=14, color=theme.TEXT_PRIMARY, fontfamily=theme.FONT_DISPLAY_STACK,
        loc="left", pad=12,
    )

    ax_div.plot(iters, diversity, color=theme.TEXT_FAINT, linewidth=1.6, zorder=3)
    ax_div.set_ylabel(
        "pheromone\ndiversity", fontsize=10, color=theme.TEXT_BODY,
        fontfamily=theme.FONT_BODY_STACK,
    )
    ax_div.set_xlabel(
        "iteration", fontsize=10, color=theme.TEXT_BODY,
        fontfamily=theme.FONT_BODY_STACK,
    )

    def annotate(ax, it, text, y_frac, color):
        if it is None:
            return
        ax.axvline(it, color=color, linewidth=1.0, linestyle=(0, (3, 2)), alpha=0.75, zorder=2)
        y_min, y_max = ax.get_ylim()
        if ax.get_yscale() == "log":
            y = 10 ** (np.log10(y_min) + y_frac * (np.log10(y_max) - np.log10(y_min)))
        else:
            y = y_min + y_frac * (y_max - y_min)
        ax.annotate(
            text, xy=(it, y), xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=color, fontfamily=theme.FONT_MONO_STACK, va="center",
        )

    annotate(ax_gap, commit_iter, f"iter {commit_iter}: biggest single jump", 0.88, theme.ACCENT_BRIGHT)
    annotate(ax_gap, converged_iter, f"iter {converged_iter}: proven optimum found", 0.68, theme.GOLD)
    annotate(ax_div, commit_iter, "", 0.5, theme.ACCENT_BRIGHT)
    annotate(ax_div, converged_iter, "", 0.5, theme.GOLD)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf
