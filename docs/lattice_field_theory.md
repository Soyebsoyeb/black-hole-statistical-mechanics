# Lattice Field Theory: Entanglement of a Harmonic Chain

A free scalar field discretised on a one-dimensional chain is a Gaussian system, so its entanglement entropy follows exactly from the covariance matrices (derivation in [mathematical_derivations.md](mathematical_derivations.md)). This is *not* a black-hole calculation; it demonstrates the field-theory features that horizon-entropy arguments rely on.

## Results (Fig 26)

- **UV divergence.** At fixed physical length the entropy grows as $\tfrac13\ln(1/a)$: the fitted slope is 0.33334 for $N=32,\dots,512$ (small fixed physical mass, see `experiments/08_lattice_entanglement.py`). The entropy minus $\tfrac13\ln N$ is constant to $10^{-5}$: the divergence is a boundary (area-like, here logarithmic) effect.
- **CFT scaling.** For a critical chain $S=\tfrac c3\ln[(N/\pi)\sin(\pi\ell/N)]+$ const with $c=1$ (Calabrese-Cardy). With $N=256$ and $ma=10^{-3}$ the fitted slope is 0.297, 11% below $1/3$. The deficit is an infrared mass effect: the correlation length $1/(ma)$ is only a few times the chain length. Larger masses reduce the slope further and the slope approaches $1/3$ as $mN\to0$.
- **Boundaries.** A block at an end of an open chain has one entangling point and half the entropy of an interior block (about 0.5 versus 1.0 nats at $\ell=40$, $ma=0.05$).
- **Exactness test.** Two coupled oscillators reproduce the analytic symplectic-eigenvalue formula (`tests/test_lattice_models.py`).

## Interpretation

The area law in $d=1$ is logarithmic at criticality; in higher dimensions the entropy is proportional to the boundary area times a cutoff-dependent coefficient. That the black-hole entropy takes the finite value $A/4G$ suggests a gravitational regularisation of this divergence, which is the motivation of entanglement-entropy proposals; no such calculation is carried out here.
