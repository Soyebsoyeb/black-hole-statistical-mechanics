# Black-Hole Thermodynamics

Implemented in `bhstatmech.spacetimes`, `bhstatmech.thermodynamics` and `bhstatmech.evaporation.mass_loss`. Derivations are in [mathematical_derivations.md](mathematical_derivations.md).

## Quantities and checks

| Quantity | Result of this repository | Figure |
| --- | --- | --- |
| Schwarzschild radius, solar mass | 2953.25 m | Fig 01 |
| Kretschmann at the horizon, $K r_s^4$ | 12 (symbolic and numerical) | Fig 02 |
| Tortoise log-slope fit | 1.0000026 | Fig 03 |
| $T_H$, solar mass | $6.17\times10^{-8}$ K | Fig 05 |
| $S_{BH}/k_B$, solar mass | $1.05\times10^{77}$ | Fig 06 |
| First-law finite-difference residual | $\sim10^{-12}$ relative, scaling as $h^2$ | Fig 07 |
| Entropy log-log slope against $M$ | 2.0 | Fig 06 |
| $T_H$ equals the CMB temperature at | $4.5\times10^{22}$ kg | Fig 05 |
| Hawking-Page (AdS, $L=1$) | $T_{min}=0.27566$, $T_{HP}=0.31831$ | Fig 09 |

## Heat capacity and stability

Schwarzschild has $C=-2S<0$, so it cannot equilibrate with an infinite bath. In AdS the large-black-hole branch has $C>0$ and is stable above $T_{min}$; the free energy crosses zero at $T_{HP}$ (Fig 08, Fig 09). Reissner-Nordstrom and Kerr reach zero temperature at extremality; limits are computed with $\sqrt{(M-Q)(M+Q)}$ to avoid cancellation (Fig 10).

## Evaporation

The photon-only model gives $t_{evap}=8.4\times10^{19}$ s for $10^{12}$ kg. The mass whose photon-only lifetime equals the age of the universe is $1.7\times10^{11}$ kg. These lifetimes are model-dependent estimates: the real lifetime depends on the particle content and greybody factors.

The adiabaticity parameter $\varepsilon$ equals $5.2\times10^{-4}$ at the Planck mass, so this particular criterion does not flag a breakdown of the quasi-static picture before the Planck scale; it is a necessary condition only (Fig 11). The model is not continued through the endpoint.
