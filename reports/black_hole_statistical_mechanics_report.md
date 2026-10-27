# Black Hole Statistical Mechanics: Computational Study of Horizon Thermodynamics, Entanglement and Information Recovery in Toy Models

Technical report accompanying the `bhstatmech` package, version 0.1.0. All numbers are from `data/results/*.json` produced by the code in this repository. Figure numbers refer to the README gallery.

## Abstract

We assemble a reproducible numerical laboratory that (i) evaluates black-hole thermodynamics for Schwarzschild, Reissner-Nordstrom, Kerr and (anti-)de Sitter geometries and verifies the first law and the Hawking-Page transition, (ii) computes scalar greybody factors for Schwarzschild by direct integration of the radial wave equation, (iii) implements standard quantum-information measures, (iv) compares the Page curve for Haar-random states and a random-unitary evaporation toy with a non-unitary reference, (v) studies a phenomenological island model, a finite-dimensional replica identity, Ryu-Takayanagi lengths in AdS$_3$/BTZ, OTOC scrambling, and two quantum error-correcting codes. All components are checked against closed forms or independent numerical methods; 210 automated tests pass. The work reproduces known results and illustrates modern proposals with explicitly labelled toy models. It does not resolve the information paradox and makes no new physical claim.

## 1. Introduction

Black holes have temperature and entropy, radiate approximately thermally, and appear to turn pure states into mixed ones unless information leaves with the radiation. The Page curve states what a unitary evaporation must look like. Recent semiclassical proposals reproduce it using extremal surfaces with islands. The aim here is a computational companion that makes each ingredient concrete and testable at small scale.

## 2. Methods

Conventions: SI for astrophysical numbers, Planck units for charged, rotating, (A)dS, holographic and island code. Randomness is seeded from one master seed. Details: [docs/numerical_methods.md](../docs/numerical_methods.md), derivations in [docs/mathematical_derivations.md](../docs/mathematical_derivations.md).

Greybody factors: the scalar radial equation is integrated from the horizon with an ingoing wave and matched to outgoing/incoming asymptotics; $\Gamma=1/|a_{in}|^2$.

Page curve: exact average $\langle S_{m,n}\rangle=\sum_{k=n+1}^{mn}1/k-(m-1)/2n$ and Monte Carlo over Haar-random states. Evaporation toy: at each step a Haar-random unitary acts on the remaining black-hole qubits, then one qubit is emitted. Non-unitary reference: the emitted qubit is traced out of a mixed register and treated as uncorrelated.

Island toy: $S_{gen}(u;\tau)=S_H(1+\mu u)+S_{rad}e^{-u/\xi}$ with $S_H=S_0(1-\tau)^{2/3}$, $S_{rad}=r(S_0-S_H)$.

## 3. Results

### 3.1 Thermodynamics (Figs 1-14)

For one solar mass: $r_s=2953.25$ m, $T_H=6.17\times10^{-8}$ K, $S/k_B=1.05\times10^{77}$. The first-law residual is $\sim10^{-12}$ and converges as $h^2$. Schwarzschild has $C=-2S$; in AdS the Hawking-Page transition is at $T_{HP}=1/\pi L=0.31831$ with $T_{min}=0.27566$. A photon-only evaporation model gives $t_{evap}=8.4\times10^{19}$ s for $10^{12}$ kg; the numerical ODE matches the analytic solution to $5\times10^{-12}$. The adiabaticity parameter reaches only $5\times10^{-4}$ at the Planck mass, so this heuristic criterion does not mark a breakdown earlier.

### 3.2 Hawking emission (Figs 15-19)

Peak photon energy of the thermal spectrum scales as $1/M$ (299, 100, 30 MeV for $10^{11}$, $3\times10^{11}$, $10^{12}$ kg). The greybody transmission obeys flux conservation to $10^{-9}$, tends to $4y^2$ at low frequency (ratio 1.016 at $y=0.005$), and the absorption cross-section oscillates about the geometric-optics value $27\pi r_s^2/4$ (ratio between 0.934 and 1.097). Integrated scalar emission power exceeds the horizon-area blackbody by a factor 7.19, against 6.76 for geometric optics ($27/4=6.75$).

### 3.3 Entanglement and horizon pair models (Figs 20-26)

Standard states give the expected entropies ($\ln2$ for a Bell pair; negativity 1/2; classically correlated state has zero negativity). The truncated two-mode squeezed state reproduces the Bose entropy as the truncation grows. In the $N$-pair model $S(R_k)=k\ln2$ for maximally entangled pairs, the monotonic semiclassical behaviour. For a harmonic chain, the entropy at fixed physical size grows with slope 0.33334 in $\ln(1/a)$, matching $c/3$ with $c=1$; the finite-mass periodic-chain window slope at $N=256$, $ma=10^{-3}$ is 0.297, 11% lower, attributable to the finite correlation length.

### 3.4 Page curve, evaporation, replicas (Figs 27-33)

Page's formula gives a deficit of 0.5000 nat at the symmetric point for $N=40$; Monte Carlo means agree with the exact average to within 1.4-2.3 standard errors. The unitary toy reproduces the Page curve (peak 2.960 vs 2.966 nats, 10 qubits) with purity maintained to $9\times10^{-16}$; the non-unitary reference gives final radiation entropy $10\ln2$. The replica identity recovers $S=1.503146$ to $10^{-13}$ by Richardson extrapolation; polynomial continuation from integer replicas converges slowly (error 0.16 at $n_{max}=2$, 0.012 at 12).

### 3.5 Islands, holography, scrambling, error correction (Figs 34-38)

The island toy has an extremum at $u^*$ that agrees between analytic and numerical evaluation to $4.5\times10^{-9}$, and a Page time $\tau_P=0.6203$ for the chosen parameters. In BTZ the connected and horizon-wrapping geodesics cross at $\Delta\phi=5.5919$ for $r_+=L=1$; a turning-radius quadrature agrees with the closed form to $1.2\times10^{-7}$. The OTOC front arrives at layers 1, 3, 3, 7, 9 for separations 1, 2, 3, 5, 7. Half-chain entanglement of a 12-qubit brickwork circuit reaches $3.6496\pm0.0016$, slightly below the Haar-average Page value 3.6590. The five-qubit code corrects all 15 single-qubit Paulis and recovers the logical qubit after erasure of any two qubits, failing at three, consistent with distance 3.

## 4. Discussion

The numerical results reproduce the known qualitative hierarchy: thermodynamics fixes $T$ and $S$; thermal radiation with greybody filtering has an emission area near $27/4$ times the horizon at high frequency; a unitary evaporation toy follows the Page curve while a non-unitary one does not; islands in a toy model provide an alternative route to a turnover. The toy island model and pair model are constructed to show these features, so they are consistent with, but not evidence for, the physical proposals.

## 5. Limitations

See [docs/limitations.md](../docs/limitations.md). In particular: scalar field on Schwarzschild only; photon-only evaporation; free parameters (log-correction coefficient, island model constants) are illustrative; no gravitational replica geometry; static AdS$_3$ geodesics only; sizes limited to about 14 qubits; the bibliography is unverified.

## 6. Development note

During development the geometric-optics emission area was coded as 27/16 of the horizon area, which was wrong; the correct value is 27/4 and all dependent code, tests and the Fig 19 caption were corrected and regenerated.

## 7. References

See [docs/references.md](../docs/references.md). The list was compiled without access to bibliographic databases and has not been verified.
