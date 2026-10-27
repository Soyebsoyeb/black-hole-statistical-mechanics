# Provenance

- Configuration: `configs/default.toml` (versioned by `config_version`), loaded by `bhstatmech.config.load_config`.
- Randomness: `bhstatmech.numerical_methods.rng.make_rng` / `spawn_rngs` derive independent streams from `master_seed` with `SeedSequence`.
- Figures: `figures/figNN_name.{png,pdf}` plus `figNN_name.json` with caption, generating script, parameters, seed and package version; `figures/manifest.json` indexes them.
- Results: `data/results/NN_name.json`, written by the same experiment that draws the figure.
- No figure or number is typed in by hand; a figure whose script has not been rerun after a code change is stale.
- Erratum recorded during development: the geometric-optics emission area was first coded as 27/16 of the horizon area; the correct factor is 27/4 (4 pi b_c^2 = 27 pi r_s^2). Code, test and Fig 19 were corrected.
