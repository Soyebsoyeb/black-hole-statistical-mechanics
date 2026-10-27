import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documented_quickstart_runs_and_prints_expected_values():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "quickstart.py")],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    assert "r_s            = 2953.25 m" in out
    assert "S(AB)          = 0.000 nats" in out
    assert "S(A)           = 0.693 nats" in out
    assert "peak at k = 5" in out
