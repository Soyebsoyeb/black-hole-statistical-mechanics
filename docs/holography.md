# Holographic Entanglement Entropy

## Ryu-Takayanagi

For a boundary region $A$ in a state dual to a static bulk, $S_A=\mathrm{Area}(\gamma_A)/4G_N$ with $\gamma_A$ the minimal bulk surface anchored on $\partial A$. In AdS$_3$ the surface is a geodesic and the central charge is $c=3L/2G_N$ (Brown-Henneaux).

## Implementation (`bhstatmech.holography`)

- **Poincare AdS$_3$**: length $2L\ln(\ell/\epsilon)$ so $S=\tfrac c3\ln(\ell/\epsilon)$; a numerical quadrature of the semicircle matches (Fig 36a).
- **BTZ**: connected geodesic $2L\ln[(2r_\infty/r_+)\sinh(r_+\Delta\phi/2L)]$ and the horizon-wrapping candidate (complement geodesic plus $2\pi r_+$). They cross at $\Delta\phi=5.5919$ for $r_+=L=1$ (cutoff independent). An independent turning-radius integral reproduces the closed form to $1.2\times10^{-7}$ (Fig 36b,c).

## Relation to quantum error correction

Erasure-recoverability of the five-qubit code (Fig 38) is a finite analogy to subregion reconstruction in AdS/CFT (Almheiri-Dong-Harlow). It does not derive or test holography.

## Limits

Only static, two-dimensional-boundary geometries are treated; no time dependence, no quantum corrections, no higher dimensions, and no connection between this module and the Hawking-radiation module beyond conceptual discussion.
