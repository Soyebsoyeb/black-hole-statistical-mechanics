# Data

This repository contains no external datasets. Every number is produced by code in `src/bhstatmech` and `experiments/`.

| Path | Content | Produced by |
| --- | --- | --- |
| `data/results/*.json` | Numbers quoted in the documentation, one file per experiment | `experiments/NN_*.py` through `save_results` |
| `data/results/greybody_scalar.csv` | Scalar greybody-factor table (omega, ell, Gamma) | `experiments/05_greybody_factors.py` |
| `data/metadata/provenance.md` | How outputs relate to code, seeds and configuration | hand-written |

Physical constants come from `src/bhstatmech/constants.py` (CODATA-style values entered by hand, `GM_sun` from the IAU nominal value). They were not downloaded or checked against an online source in this build.
All random numbers derive from `master_seed` in `configs/default.toml` through `numpy.random.SeedSequence`.
