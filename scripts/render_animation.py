"""Runs Elitist Ant System on the project's 17-city instance, renders
the flagship animation, and exports MP4 + GIF variants + PNG snapshots.

Usage:
    python scripts/render_animation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.aco.engine import ACOConfig, run_aco
from src.aco.exact_solver import solve_held_karp
from src.aco.graph import random_city_layout
from src.export.render import export_stream
from src.viz.animator import FPS, frame_count_plan, render_frames

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "generated"

# The project's fixed problem instance: 17 cities is deliberately small
# enough that Held-Karp gives a *provable* optimum (see exact_solver.py),
# the same role the known analytic optimum played in the genetic
# algorithm project.
LAYOUT_SEED = 0
NUM_CITIES = 17


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layout = random_city_layout(NUM_CITIES, np.random.default_rng(LAYOUT_SEED), min_separation=8.0)
    exact = solve_held_karp(layout.distances)
    print(f"Instance: {NUM_CITIES} cities, provable optimum = {exact.length:.2f}")

    config = ACOConfig()  # tuned defaults, seed=18
    t0 = time.time()
    history = run_aco(layout, config)
    print(
        f"ACO run complete: {len(history.generations)} iterations in {time.time() - t0:.2f}s, "
        f"best length {history.final.best_ever_length:.2f} "
        f"(gap to optimum: {history.final.best_ever_length - exact.length:.4f})"
    )

    plan = frame_count_plan(history)
    total_frames = sum(p.walk_frames + p.hold_frames for p in plan)

    t0 = time.time()
    paths = export_stream(
        render_frames(history), OUT_DIR, "aco_evolution", total_frames=total_frames, fps=FPS
    )
    print(f"Exported animation + snapshots in {time.time() - t0:.2f}s")
    for kind, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {kind:12s} {path.name:32s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
