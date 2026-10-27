"""Build black-hole-statistical-mechanics.zip next to the repository folder and verify it."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = ROOT.name
OUT = ROOT.parent / f"{NAME}.zip"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", "build", "dist", ".ipynb_checkpoints", ".venv"}
EXCLUDE_SUFFIX = (".pyc", ".zip")


def keep(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    return p.is_file() and not (set(rel.parts) & EXCLUDE_DIRS) and not any(part.endswith(".egg-info") for part in rel.parts) and p.suffix not in EXCLUDE_SUFFIX


if __name__ == "__main__":
    files = sorted(p for p in ROOT.rglob("*") if keep(p) and p.name != "MANIFEST.json")
    manifest = {"archive": OUT.name, "n_files": len(files),
                "files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    files.append(ROOT / "MANIFEST.json")
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"{NAME}/{p.relative_to(ROOT)}")
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        names = z.namelist()
        assert bad is None, bad
        for p in files:
            assert hashlib.sha256(z.read(f"{NAME}/{p.relative_to(ROOT)}")).hexdigest() == hashlib.sha256(p.read_bytes()).hexdigest(), p
    print(f"{OUT}: {len(names)} entries, {OUT.stat().st_size / 1e6:.2f} MB, verified (CRC and SHA-256)")
    sys.exit(0)
