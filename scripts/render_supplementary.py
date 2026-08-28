"""Renders every supplementary visualization from a single ACO run:
the 3D pheromone-bars hero image, the animated 3D pheromone-bars
video, the annotated 2D convergence chart, and the small-multiples
filmstrip.

Usage:
    python scripts/render_supplementary.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

from src.aco.engine import ACOConfig, run_aco
from src.aco.graph import random_city_layout
from src.export.render import GifVariant, export_stream
from src.viz.animator_pheromone_3d import FPS as FPS_3D
from src.viz.animator_pheromone_3d import render_pheromone_bars_3d, total_frame_count
from src.viz.convergence_chart import render_convergence_chart
from src.viz.filmstrip import render_filmstrip
from src.viz.pheromone_3d import draw_pheromone_bars
from src.viz.theme import setup_3d_axes

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "generated"
LAYOUT_SEED = 0
NUM_CITIES = 17

# Iterations shown in the filmstrip, chosen to trace the specific story
# of the default seed: rapid early progress (0, 3, 9), the approach to
# a long near-optimal plateau (18), the plateau itself (27), and the
# eventual exact resolution (54). If you change `ACOConfig.seed`,
# re-inspect `state.is_new_best` events and update this list.
FILMSTRIP_ITERATIONS = [0, 3, 9, 18, 27, 54]


def render_3d_hero(history) -> Path:
    fig = plt.figure(figsize=(8, 8), dpi=170)
    ax = setup_3d_axes(fig)
    draw_pheromone_bars(ax, history.final.pheromone, history.final.best_ever_tour)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    path = OUT_DIR / "aco_pheromone_3d_hero.png"
    iio.imwrite(path, buf)
    plt.close(fig)
    return path


def render_3d_animation(history) -> dict:
    total = total_frame_count(history)
    return export_stream(
        render_pheromone_bars_3d(history),
        OUT_DIR,
        "aco_pheromone_3d",
        total_frames=total,
        fps=FPS_3D,
        gif_variants=(
            GifVariant(name="repo", stride=2, width=440, max_colors=110),
            GifVariant(name="blog", stride=3, width=300, max_colors=64),
        ),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layout = random_city_layout(NUM_CITIES, np.random.default_rng(LAYOUT_SEED), min_separation=8.0)
    history = run_aco(layout, ACOConfig())
    print(
        f"ACO run complete: {len(history.generations)} iterations, "
        f"best length {history.final.best_ever_length:.2f}"
    )

    t0 = time.time()
    path = render_3d_hero(history)
    print(f"3D hero image -> {path.name} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    buf = render_convergence_chart(history)
    chart_path = OUT_DIR / "aco_convergence_chart.png"
    iio.imwrite(chart_path, buf)
    print(f"Convergence chart -> {chart_path.name} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    buf = render_filmstrip(history, iterations=FILMSTRIP_ITERATIONS, cols=3)
    filmstrip_path = OUT_DIR / "aco_filmstrip.png"
    iio.imwrite(filmstrip_path, buf)
    print(f"Filmstrip -> {filmstrip_path.name} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    paths = render_3d_animation(history)
    print(f"3D pheromone animation ({time.time() - t0:.1f}s):")
    for kind, p in paths.items():
        print(f"  {kind:12s} {p.name}")


if __name__ == "__main__":
    main()
