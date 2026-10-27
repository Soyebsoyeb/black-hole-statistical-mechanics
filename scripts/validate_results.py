"""Check generated outputs: figure files and sidecars exist, results are finite, key numbers agree with theory."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "data" / "results"
errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        errors.append(msg)


def finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(finite(v) for v in obj)
    return True


stems = sorted(p.stem for p in FIG.glob("fig*.png"))
check(len(stems) >= 20, f"{len(stems)} figures present (>= 20 required)")
for s in stems:
    for ext in ("pdf", "json"):
        check((FIG / f"{s}.{ext}").exists(), f"{s}.{ext}")
    meta = json.loads((FIG / f"{s}.json").read_text())
    check(bool(meta.get("caption")) and bool(meta.get("generating_script")), f"{s}: caption and script recorded")
    check((ROOT / meta["generating_script"]).exists(), f"{s}: script {meta['generating_script']} exists")
manifest = json.loads((FIG / "manifest.json").read_text())
check(set(manifest) >= set(stems) if isinstance(manifest, dict) else True, "manifest covers all figures")

for p in sorted(RES.glob("*.json")):
    check(finite(json.loads(p.read_text())), f"{p.name}: all numbers finite")

def get(name):
    return json.loads((RES / name).read_text())

r = get("11_replica.json"); check(r["min_richardson_error"] < 1e-8, "replica derivative reproduces S_vN (Richardson error < 1e-8)")
r = get("09_page_curve.json"); check(abs(r["page_deficit_N40"] - 0.5) < 1e-6, "Page deficit at N=40 equals 1/2 nat")
check(all(v["max_abs_z_mean_vs_exact"] < 4 for v in r["ensemble_stats"].values()), "Monte Carlo Page curves within 4 s.e.m. of exact")
r = get("13_island_model.json"); check(r["max_u_error"] < 1e-6, "island extremum analytic vs numeric < 1e-6")
r = get("14_holography.json"); check(r["max_numeric_vs_closed_form"] < 1e-5, "BTZ geodesic numeric vs closed form < 1e-5")
r = get("10_unitary_evaporation.json"); check(r["max_norm_deviation"] < 1e-12 and r["unitary_final_S_R"] < 1e-10, "unitary toy stays pure and returns S(R) to 0")
r = get("15_scrambling.json"); check(abs(r["final_entropy"] - r["page_value"]) < 0.05, "scrambling circuit entropy saturates near Page value")
r = get("08_lattice.json"); check(abs(r["refinement_slope"] - 1 / 3) < 5e-3, "lattice UV slope within 0.005 of 1/3")
r = get("16_qec.json"); check(abs(r["erasure_five_worst_over_ln2"][2] - 2) < 1e-8, "5-qubit code recovers from any 2 erasures")
sys.exit(1 if errors else 0)
