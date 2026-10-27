# Documentation Index

This directory documents the physics, numerical methods, validation and limitations of the `bhstatmech` package. Equations use GitHub-rendered LaTeX (`$...$`, `$$...$$`).

| Document | Contents |
| --- | --- |
| [theory.md](theory.md) | Overview of the physical picture and how the modules relate |
| [mathematical_derivations.md](mathematical_derivations.md) | Derivations of the formulas used in code |
| [black_hole_thermodynamics.md](black_hole_thermodynamics.md) | Geometries, laws, heat capacity, Hawking-Page, evaporation |
| [statistical_mechanics.md](statistical_mechanics.md) | Ensembles, state counting, log corrections |
| [hawking_radiation.md](hawking_radiation.md) | Thermal spectra, greybody factors, emission |
| [quantum_entanglement.md](quantum_entanglement.md) | Entanglement measures and horizon pair models |
| [quantum_information.md](quantum_information.md) | States, entropies, random ensembles, error correction, scrambling |
| [lattice_field_theory.md](lattice_field_theory.md) | Gaussian harmonic-chain entanglement |
| [information_paradox.md](information_paradox.md) | Statement of the problem and the toy models used |
| [page_curve.md](page_curve.md) | Page's formula and the evaporation toy |
| [generalized_entropy.md](generalized_entropy.md) | Generalized entropy and the island toy model |
| [islands_and_replica_methods.md](islands_and_replica_methods.md) | Replica trick and its numerical analogue |
| [holography.md](holography.md) | Ryu-Takayanagi in AdS3 and BTZ |
| [numerical_methods.md](numerical_methods.md) | Solvers, convergence, reproducibility |
| [validation.md](validation.md) | Tests and quantitative checks with actual results |
| [limitations.md](limitations.md) | What is and is not claimed |
| [references.md](references.md) | Bibliography (not independently verified) |

The research-style report is at [../reports/black_hole_statistical_mechanics_report.md](../reports/black_hole_statistical_mechanics_report.md).

Scope statement. This repository computes standard results (thermodynamics, thermal spectra, Page averages) exactly or numerically, and illustrates modern ideas (islands, replicas, holography) with explicitly labelled toy models. It does not resolve the black-hole information paradox, and no component derives Hawking radiation from quantum field theory in curved spacetime.
