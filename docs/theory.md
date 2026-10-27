# Theory Overview

## The physical picture

A classical black hole of mass $M$ has a horizon of area $A$. Four ingredients tie it to statistical mechanics:

1. **Thermodynamics.** Temperature $T_H = \hbar\kappa/(2\pi c k_B)$ with surface gravity $\kappa$, entropy $S_{BH} = k_B c^3 A/(4G\hbar)$, and the first law $dM = T\,dS + \Omega\,dJ + \Phi\,dQ$.
2. **Radiation.** Quantum fields near the horizon produce an approximately thermal flux at $T_H$, filtered by a potential barrier (greybody factor).
3. **Entanglement.** The thermal state of the radiation is the reduced state of a pure state entangled with partner modes behind the horizon, so the radiation entropy grows as the black hole evaporates.
4. **Unitarity.** If the global state is pure, the radiation entropy must fall back to zero. Page's argument says it must peak near half-way, at the Page time.

## Module map

| Question | Module | Document |
| --- | --- | --- |
| What are $T$, $S$, $C$? | `spacetimes`, `thermodynamics` | [black_hole_thermodynamics.md](black_hole_thermodynamics.md) |
| How long does evaporation take? | `evaporation.mass_loss` | same |
| How do ensembles behave? | `statistical_mechanics` | [statistical_mechanics.md](statistical_mechanics.md) |
| What does the radiation look like? | `hawking_radiation` | [hawking_radiation.md](hawking_radiation.md) |
| How is entanglement quantified? | `quantum_states`, `entanglement` | [quantum_entanglement.md](quantum_entanglement.md) |
| Field-theory entanglement scaling? | `lattice_models` | [lattice_field_theory.md](lattice_field_theory.md) |
| Page curve, unitary vs non-unitary | `page_curve`, `evaporation.unitary_toy` | [page_curve.md](page_curve.md) |
| Islands, replicas | `generalized_entropy`, `replica_methods` | [generalized_entropy.md](generalized_entropy.md) |
| Holographic entropy | `holography` | [holography.md](holography.md) |
| Scrambling, error correction | `scrambling`, `error_correction` | [quantum_information.md](quantum_information.md) |

## Units

Astrophysical quantities use SI. The charged, rotating and (A)dS geometries and all holography and island code use Planck units $G=\hbar=c=k_B=1$, stated in each module docstring. A small `Quantity` class checks dimensions at the SI boundary.
