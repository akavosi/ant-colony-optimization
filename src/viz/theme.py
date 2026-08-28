"""Shared visual language for the whole OR visualization series.

This module is intentionally self-contained (no imports from the rest
of `src`) so it can be copied verbatim into the Ant Colony Optimization
and Simulated Annealing projects, keeping the three animations
visually consistent as one body of work.

Palette is derived from the akavosi.github.io blog's own design tokens
(warm parchment background, navy accent, muted gold) so the animations
read as belonging to the site rather than as a generic stock demo.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

from src.viz import fonts as _fonts  # noqa: F401  (registers project fonts on import)

# --- Core palette (matches blog CSS custom properties) ---
BG_BASE = "#efe9db"
BG_ELEVATED = "#faf7ee"
BG_SOFT = "#f2ecdb"
TEXT_PRIMARY = "#221c13"
TEXT_BODY = "#3c3323"
TEXT_MUTED = "#7c6f57"
TEXT_FAINT = "#a89b80"
ACCENT = "#2f3d63"
ACCENT_BRIGHT = "#4d5f96"
RULE = "#ddd1af"
RULE_STRONG = "#c7b788"
GOLD = "#a9781f"
GOLD_BRIGHT = "#c9a24b"
AMBER = "#b5792a"

# --- Semantic roles, reused across GA / ACO / SA for a shared visual grammar ---
COLOR_STANDARD = TEXT_FAINT      # ordinary candidate / individual / ant
COLOR_ELITE = GOLD_BRIGHT        # current best / elite
COLOR_NEW = ACCENT_BRIGHT        # freshly generated / newly accepted
COLOR_REJECTED = RULE_STRONG     # discarded / rejected candidate
COLOR_ACCEPTED_WORSE = AMBER     # SA: accepted despite being worse
COLOR_TRAIL = ACCENT             # trajectory of the running-best solution

FONT_DISPLAY = "Fraunces"
FONT_BODY = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

# Fallback font stacks in case the display fonts aren't installed in the
# rendering environment (matplotlib silently falls back otherwise, which
# would break the intended look without warning).
FONT_DISPLAY_STACK = [FONT_DISPLAY, "Georgia", "serif"]
FONT_BODY_STACK = [FONT_BODY, "DejaVu Sans", "sans-serif"]
FONT_MONO_STACK = [FONT_MONO, "DejaVu Sans Mono", "monospace"]


def landscape_colormap() -> LinearSegmentedColormap:
    """Two-tone sequential colormap from soft parchment to deep navy.

    Used for fitness-landscape contours: low objective value (good) is
    dark navy, high objective value (poor) fades toward the page
    background, so the map reads as "figure emerging from the page"
    rather than a generic viridis/heat palette.
    """
    return LinearSegmentedColormap.from_list(
        "parchment_navy", [BG_ELEVATED, "#c9c3ae", "#7d84a0", ACCENT, "#1a2138"]
    )


def apply_base_style(fig, ax) -> None:
    """Apply consistent figure/axes chrome shared by every frame."""
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_ELEVATED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def setup_3d_axes(fig, rect=(0.0, 0.0, 1.0, 1.0)):
    """Create and style a 3D axes matching the series' visual language.

    Shared across every project in this series that needs a 3D view
    (fitness landscapes, pheromone matrices, energy surfaces, ...) —
    transparent panes matching the page background, no grid lines, and
    `computed_zorder = False`.

    That last flag matters more than it looks: mplot3d's default
    automatic z-ordering (distance-from-camera based) frequently hides
    scatter/line/bar artists behind a surface or behind each other even
    when their own coordinates are clearly in front — a well-known
    mplot3d limitation. With `computed_zorder = False`, matplotlib
    respects each artist's explicit `zorder` instead, which is what
    actually works. Every artist drawn on an axes from this function
    must set `zorder` deliberately as a result — there is no free
    automatic depth sorting to fall back on.
    """
    ax = fig.add_axes(rect, projection="3d")
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_BASE)
    ax.computed_zorder = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(BG_BASE)
        axis.pane.set_alpha(1.0)
        axis.line.set_color(RULE)
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine_getter in (ax.xaxis, ax.yaxis, ax.zaxis):
        spine_getter.line.set_linewidth(0)
    return ax
