# Validation

All numbers below were produced by this repository's code in the build recorded in the README (test run: **210 passed**; `scripts/validate_results.py`: all checks pass). Quantities marked "analytic" are compared with a closed form; none is compared with external data.

## Automated tests (`python -m pytest`)

210 tests across 15 test files cover constants and units, geometry, thermodynamics, ensembles, density matrices, entanglement, lattice models, Page curve, unitarity, replica methods, greybody factors, islands and holography, scrambling and QEC, numerical convergence, and the quick-start script.

## Quantitative checks

| Check | Result |
| --- | --- |
| Kretschmann $K r_s^4$ at the horizon (symbolic vs numeric) | 12 to $10^{-15}$ |
| Ingoing EF determinant $/c^2$ | $-1.0000000000000044$ |
| Entropy $\propto M^2$ log-log slope | 2.0, area-law deviation $4\times10^{-16}$ |
| First-law finite-difference residual (Schwarzschild) | $\sim10^{-12}$, scales as $h^2$ |
| Evaporation ODE vs analytic | time error $4.8\times10^{-12}$ |
| Schottky heat capacity (two-level, $N=40$) | max deviation $3\times10^{-15}$ |
| Log-correction slope recovered | $-1.4999999999$ for input $-3/2$ |
| Bell state | $S_A=\ln2$, $I=2\ln2$ |
| Greybody flux conservation | relative defect $\le1.1\times10^{-9}$ |
| Greybody low-frequency $\Gamma_0/4y^2$ at $y=0.02,0.01,0.005$ | 1.071, 1.034, 1.016 (tends to 1) |
| Greybody high-frequency $\sigma/(27\pi r_s^2/4)$ at largest $y$ | 0.967 (oscillates between 0.934 and 1.097) |
| Greybody tolerance sensitivity | $\le9\times10^{-11}$ relative |
| Scalar emitted power over horizon-blackbody power | 7.19, versus geometric optics 6.76 (i.e. $27/4=6.75$) |
| Page deficit $N=40$ | 0.49999999999947 nat |
| Monte Carlo Page vs exact | max 1.4 to 2.3 standard errors |
| Unitary toy: norm deviation, final $S(R)$ | $8.9\times10^{-16}$, 0 |
| Lattice UV slope $(N=32\ldots512)$ | 0.33334 vs $1/3$ |
| Lattice CFT-window slope ($N=256$, $ma=10^{-3}$) | 0.297, **11% below** $1/3$ (infrared effect) |
| Island numeric vs analytic | $4.5\times10^{-9}$ in $u^*$; $1.1\times10^{-13}$ in $S$ |
| BTZ numeric vs closed form | $1.2\times10^{-7}$ |
| Replica derivative (Richardson best) | $1.1\times10^{-13}$ |
| 5-qubit code | corrects all 15 single-qubit Paulis; any 2 erasures recoverable, any 3 not |
| Circuit entanglement, 12 qubits, 30 layers | $3.6496\pm0.0016$ vs Page value $3.6590$ |

## Honest notes

- The circuit entropy is about 0.009 nat (roughly 6 standard errors) below the Page average: a brickwork circuit with open boundaries is not a Haar-random state, and depth may not be fully saturated. We report the difference rather than adjusting the claim.
- The lattice CFT-window slope does not reach $1/3$ for the default mass, for the infrared reason given in [lattice_field_theory.md](lattice_field_theory.md).
- The first geometric-optics emission factor was coded as 27/16 and corrected to 27/4 during development; the tests and Fig 19 use the corrected value.
- No result is compared with experiment or independent third-party code.
