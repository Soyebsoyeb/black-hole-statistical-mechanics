"""Plot style shared by every figure, plus the figure-provenance convention.

Colour roles use a small categorical palette that stays distinguishable under
the common colour-vision deficiencies (Okabe-Ito).  Series are always
identified by legend/direct labels, not by colour alone, and line styles vary.

Provenance convention
---------------------
Every figure ``figures/<name>.png`` has a sidecar ``figures/<name>.json``
with: generating script, parameters, seed, package version, and the caption.
``save_figure`` writes both and appends the entry to ``figures/manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import __version__  # noqa: E402

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "grey": "#7F7F7F",
}
CYCLE = [PALETTE[k] for k in ("blue", "vermilion", "green", "orange", "purple", "sky")]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.2),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "mathtext.default": "regular",
    })


def repo_figures_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "figures"


def save_figure(fig, name: str, script: str, caption: str, parameters: dict, seed=None, directory: Path | None = None) -> None:
    """Save PNG + PDF + provenance JSON and update ``manifest.json``."""
    d = Path(directory) if directory else repo_figures_dir()
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png")
    fig.savefig(d / f"{name}.pdf")
    plt.close(fig)
    meta = {
        "name": name,
        "png": f"{name}.png",
        "pdf": f"{name}.pdf",
        "generating_script": script,
        "package_version": __version__,
        "seed": seed,
        "parameters": parameters,
        "caption": caption,
    }
    (d / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[name] = meta
    mpath.write_text(json.dumps(dict(sorted(manifest.items())), indent=2, default=str) + "\n")
