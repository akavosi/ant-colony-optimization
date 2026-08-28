"""Animates the pheromone matrix's 3D bars growing and shrinking over
iterations, with a slow camera drift.

Bar heights are interpolated between consecutive iterations' pheromone
states for smooth growth/decay rather than discrete jumps. Unlike the
genetic algorithm project's 3D trajectory animation, no keyframe
subsampling is needed here — a single frame of ~270 simple bar
polygons renders roughly 6x faster than that project's shaded terrain
surface, cheap enough to afford every iteration directly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.aco.engine import ACOHistory
from src.viz.pheromone_3d import DEFAULT_AZIM, DEFAULT_ELEV, draw_pheromone_bars
from src.viz.theme import setup_3d_axes

FPS = 18
INTERP_FRAMES = 3
HOLD_FRAMES = 2
NEW_BEST_BONUS_FRAMES = 3
AZIM_DRIFT_DEG = 22.0


def _ease_in_out(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def total_frame_count(history: ACOHistory) -> int:
    """Frame count for `render_pheromone_bars_3d`, computed without rendering."""
    total = 0
    for state in history.generations:
        total += INTERP_FRAMES + HOLD_FRAMES
        if state.is_new_best:
            total += NEW_BEST_BONUS_FRAMES
    return total


def render_pheromone_bars_3d(history: ACOHistory, dpi: int = 128, size_in: float = 6.0):
    """Stream the animated 3D pheromone bar chart as RGBA frames.

    Args:
        history: Output of `run_aco`.
        dpi, size_in: Figure resolution / size (defaults chosen so
            size_in*dpi is divisible by 16, avoiding silent encoder
            padding — see the genetic algorithm project's identical fix).

    Yields:
        (H, W, 4) uint8 RGBA frames.
    """
    n = history.layout.n
    total_frames = total_frame_count(history)
    frame_idx = 0

    fig = plt.figure(figsize=(size_in, size_in), dpi=dpi)

    prev_pheromone = np.full((n, n), history.config.initial_pheromone)

    def render_at(pheromone: np.ndarray, best_tour: np.ndarray, azim: float) -> np.ndarray:
        fig.clear()
        ax = setup_3d_axes(fig)
        draw_pheromone_bars(ax, pheromone, best_tour, elev=DEFAULT_ELEV, azim=azim)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()

    try:
        for state in history.generations:
            n_interp = INTERP_FRAMES + (NEW_BEST_BONUS_FRAMES if state.is_new_best else 0)
            for step in range(n_interp):
                t = _ease_in_out(np.array((step + 1) / n_interp))
                blended = prev_pheromone + (state.pheromone - prev_pheromone) * t
                azim = DEFAULT_AZIM + AZIM_DRIFT_DEG * (frame_idx / max(total_frames - 1, 1))
                yield render_at(blended, state.best_ever_tour, azim)
                frame_idx += 1

            for _ in range(HOLD_FRAMES):
                azim = DEFAULT_AZIM + AZIM_DRIFT_DEG * (frame_idx / max(total_frames - 1, 1))
                yield render_at(state.pheromone, state.best_ever_tour, azim)
                frame_idx += 1

            prev_pheromone = state.pheromone
    finally:
        plt.close(fig)
