# The Information Paradox and the Toy Models Used Here

## Statement

A black hole formed from a pure state evaporates into radiation that, in the semiclassical calculation, is exactly thermal. If evaporation ends and nothing else remains, a pure state has evolved into a mixed one, which violates unitary evolution. Equivalently, the semiclassical entropy of the radiation grows monotonically, while for a globally pure state it cannot exceed the entropy of the complement, so it must come back down.

## What this repository does

This repository does **not** resolve the paradox and does not derive any resolution from gravity. It contrasts three finite-dimensional models:

| Model | Radiation entropy | Global state | Module |
| --- | --- | --- | --- |
| Hawking reference | $k\ln2$, monotonic | mixed, no correlations | `page_curve.hawking_reference_curve` |
| Unitary random-scrambling toy | Page curve | pure | `evaporation.unitary_toy` |
| Non-unitary (information-loss) reference | $k\ln2$ | mixed | `evaporation.unitary_toy` |

Results: Figures 29, 32, 33. In the unitary toy the mean peak entropy over 24 seeds is 2.960 nats versus the exact Page average 2.966 for 10 qubits, the global state stays pure to $9\times10^{-16}$, and the final radiation entropy is zero. In the non-unitary reference the final radiation entropy is $10\ln2=6.93$.

## Interpretation limits

The unitary toy *assumes* unitarity (a Haar-random unitary on the black-hole register at each step). It shows what a unitary evaporation looks like, not why gravity would produce it. The non-unitary reference is a caricature of information loss, not a proposal.
