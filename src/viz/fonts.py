"""Registers the project's display fonts with matplotlib's font manager.

matplotlib only auto-discovers fonts installed system-wide in locations
its cache already knows about. Since this project ships/installs its
own copies of Fraunces and IBM Plex (to match the blog's typography
exactly rather than falling back to generic sans/serif), we register
them explicitly on import. Safe to call multiple times.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.font_manager as fm

_FONT_DIRS = [Path.home() / ".fonts"]
_registered = False


def register_project_fonts() -> None:
    global _registered
    if _registered:
        return
    for font_dir in _FONT_DIRS:
        for path in glob.glob(str(font_dir / "*.ttf")):
            fm.fontManager.addfont(path)
    _registered = True


register_project_fonts()
