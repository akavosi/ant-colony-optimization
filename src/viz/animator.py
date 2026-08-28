"""Turns an `ACOHistory` into a sequence of rendered RGBA frames.

Structurally mirrors the genetic algorithm project's animator (same
easing, same progress-dependent timing shrink, same extra hold on
improvement), adapted for ACO's iteration structure: each iteration
gets a short "ants walking their tours" phase (using the *pre-update*
pheromone, since that's what informed those tours) followed by a
brief hold on the *post-update* pheromone map and best-tour overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.aco.engine import ACOHistory
from src.viz import theme
from src.viz.ant_artist import draw_ants, select_shown_ants
from src.viz.graph_renderer import draw_best_tour, draw_graph
from src.viz.hud import Hud

FPS = 24


def _ease_in_out(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


@dataclass
class TimingPlan:
    walk_frames: int
    hold_frames: int


def _timing_for_iteration(iter_index: int, total_iters: int, is_new_best: bool) -> TimingPlan:
    """Frames shrink from a slow, legible start to a brisk finish —
    same philosophy and shape as the genetic algorithm project's
    `_timing_for_generation`, retuned for ACO's iteration count."""
    progress = iter_index / max(total_iters - 1, 1)
    walk = int(round(7 - 4 * progress))
    hold = int(round(7 - 5 * progress))
    if is_new_best:
        hold += 5
    return TimingPlan(walk_frames=max(walk, 2), hold_frames=max(hold, 2))


def frame_count_plan(history: ACOHistory) -> list[TimingPlan]:
    """Compute each iteration's `TimingPlan` without rendering anything
    (see the GA project's identically-purposed function)."""
    total = len(history.generations)
    return [
        _timing_for_iteration(i, total, state.is_new_best)
        for i, state in enumerate(history.generations)
    ]


def render_frames(history: ACOHistory, dpi: int = 140, size_in: float = 6.4):
    """Stream the full animation as RGBA uint8 frames, one at a time.

    See the GA project's `render_frames` for why this is a generator
    rather than a list — the same memory lesson applies here.
    """
    layout = history.layout
    coords = layout.coords
    total_iters = len(history.generations)

    fig = plt.figure(figsize=(size_in, size_in), dpi=dpi)
    ax = fig.add_axes([0.045, 0.045, 0.91, 0.91])
    theme.apply_base_style(fig, ax)

    (x_min, x_max), (y_min, y_max) = layout.bounds
    pad = 0.06 * (x_max - x_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    hud = Hud(fig, ax)

    ants_rng = np.random.default_rng(history.config.seed + 1_000_000)
    shown_ants = select_shown_ants(history.config.num_ants, ants_rng)

    length_hist: list[float] = []
    diversity_hist: list[float] = []

    def render_current() -> np.ndarray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()

    prev_pheromone = np.full((layout.n, layout.n), history.config.initial_pheromone)

    try:
        for i, state in enumerate(history.generations):
            timing = _timing_for_iteration(i, total_iters, state.is_new_best)
            length_hist.append(state.best_ever_length)
            diversity_hist.append(state.diversity)

            # --- Walk phase: ants move along their tours over the
            # pheromone state that informed those tours (pre-update). ---
            for step in range(timing.walk_frames):
                t = _ease_in_out(np.array((step + 1) / timing.walk_frames))
                ax.clear()
                theme.apply_base_style(fig, ax)
                ax.set_xlim(x_min - pad, x_max + pad)
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])

                draw_graph(ax, coords, prev_pheromone, history.config.initial_pheromone)
                draw_best_tour(ax, coords, state.best_ever_tour)
                draw_ants(ax, coords, state.tours, shown_ants, float(t))

                hud.update(state.iteration, state.best_ever_length, length_hist, diversity_hist)
                yield render_current()

            # --- Hold phase: settled, post-update pheromone + best tour. ---
            ax.clear()
            theme.apply_base_style(fig, ax)
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            draw_graph(ax, coords, state.pheromone, history.config.initial_pheromone)
            draw_best_tour(ax, coords, state.best_ever_tour)
            hud.update(state.iteration, state.best_ever_length, length_hist, diversity_hist)

            for _ in range(timing.hold_frames):
                yield render_current()

            prev_pheromone = state.pheromone
    finally:
        plt.close(fig)
