"""Shared helpers for the experiment scripts (not an experiment itself)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bhstatmech import constants as c
from bhstatmech.config import load_config
from bhstatmech.visualization import CYCLE, LINESTYLES, PALETTE, apply_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CFG = load_config()
SEED = int(CFG["master_seed"])

apply_style()
RESULTS.mkdir(parents=True, exist_ok=True)

__all__ = ["np", "plt", "c", "CFG", "SEED", "CYCLE", "LINESTYLES", "PALETTE", "ROOT", "RESULTS",
           "figure_saver", "save_results", "label_line"]


def figure_saver(script_file: str):
    """Return ``save(fig, name, caption, params, seed=None)`` bound to the calling script."""
    script = f"experiments/{Path(script_file).name}"

    def save(fig, name, caption, params, seed=None):
        save_figure(fig, name, script=script, caption=caption, parameters=params, seed=seed)
        print(f"  saved figure {name}")

    return save


def save_results(name: str, payload: dict) -> None:
    """Write a JSON record of the numbers an experiment produced (quoted by the docs)."""
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"  saved results {name}.json")


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, **kw):
    """Direct label placed at data coordinates (x, y) with an offset in axis fractions."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", color=color,
                fontsize=9, **kw)
