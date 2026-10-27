"""Run one experiment by number or name: ``python scripts/run_experiment.py 09``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments"


def find(key: str) -> Path:
    matches = sorted(p for p in EXP.glob("[0-9][0-9]_*.py") if p.name.startswith(key) or key in p.stem)
    if len(matches) != 1:
        raise SystemExit(f"experiment {key!r} matched {[m.name for m in matches]}; available: {[p.name for p in sorted(EXP.glob('[0-9][0-9]_*.py'))]}")
    return matches[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = find(sys.argv[1])
    raise SystemExit(subprocess.call([sys.executable, str(path)], cwd=EXP))
