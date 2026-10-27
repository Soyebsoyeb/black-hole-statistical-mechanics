"""Versioned scientific configuration (TOML, read with the standard library)."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

SUPPORTED_CONFIG_VERSION = 1


def repo_root() -> Path:
    """Repository root (works from a source checkout / editable install)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    """Load ``configs/default.toml`` (or ``path``) and check its version."""
    p = Path(path) if path else repo_root() / "configs" / "default.toml"
    with open(p, "rb") as fh:
        cfg = tomllib.load(fh)
    ver = cfg.get("config_version")
    if ver != SUPPORTED_CONFIG_VERSION:
        raise ValueError(f"unsupported config_version {ver!r} in {p}; expected {SUPPORTED_CONFIG_VERSION}")
    return cfg
