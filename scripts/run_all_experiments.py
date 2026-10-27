"""Run every experiment in order (``--clean`` first removes figures/ and data/results/)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

if __name__ == "__main__":
    if "--clean" in sys.argv:
        for d in ("figures", "data/results"):
            for f in (ROOT / d).glob("*"):
                if f.name != "README.md":
                    shutil.rmtree(f) if f.is_dir() else f.unlink()
    failed = []
    for path in sorted(EXP.glob("[0-9][0-9]_*.py")):
        t0 = time.time()
        print(f"=== {path.name}", flush=True)
        rc = subprocess.call([sys.executable, str(path)], cwd=EXP)
        print(f"    exit {rc}, {time.time() - t0:.0f} s", flush=True)
        if rc:
            failed.append(path.name)
    if failed:
        raise SystemExit(f"failed: {failed}")
