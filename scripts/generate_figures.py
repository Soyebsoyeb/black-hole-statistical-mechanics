"""Regenerate all figures (the experiments are the only figure source)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    runpy.run_path(str(Path(__file__).with_name("run_all_experiments.py")), run_name="__main__")
