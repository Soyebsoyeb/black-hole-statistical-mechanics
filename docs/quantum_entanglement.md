# Quantum Entanglement Measures and Horizon Pair Models

## Measures (`bhstatmech.entanglement`)

For a bipartite state $\rho_{AB}$: von Neumann entropy $S(\rho)=-\mathrm{Tr}\rho\ln\rho$, mutual information $I(A\!:\!B)=S_A+S_B-S_{AB}$, negativity $\mathcal N=(\lVert\rho^{T_B}\rVert_1-1)/2$, logarithmic negativity, and two-qubit concurrence. Checks (Fig 20, 23): product state $S=0$; Bell state $S_A=\ln2$, $I=2\ln2$, $\mathcal N=1/2$; classically correlated mixture $I=\ln2$, $\mathcal N=0$ (correlated but not entangled); Werner state ($p=0.5$) has $\mathcal N=0.125$, and the negativity threshold on the sampled grid is $p\approx0.335$ (analytic $1/3$).

## Horizon pair models (`bhstatmech.entanglement.horizon`)

These are finite-dimensional *models* of the structure produced by a Bogoliubov transformation, not field-theory calculations.

1. **Two-qubit pair** $\cos\theta|00\rangle+\sin\theta|11\rangle$ (inside, outside).
2. **Truncated two-mode squeezed (thermofield) state** $|\Psi\rangle=\sqrt{1-x^2}\sum_{n<d}x^n|n\rangle_{in}|n\rangle_{out}$ with $x^2=e^{-\hbar\omega/k_BT_H}$. Tracing out the interior gives a Bose distribution with $\langle n\rangle=1/(e^{\hbar\omega/kT}-1)$ and entropy $(\bar n+1)\ln(\bar n+1)-\bar n\ln\bar n$. The truncation error in the entropy decays with $d$ (Fig 24, tabulated in `data/results/07_horizon_entanglement.json`).
3. **$N$-pair emission** (Fig 25). With radiation $R_k$ the first $k$ emitted modes and interior $I$ all partners, $S(R_k)=k\,h(\theta)$ and $I(R_k\!:\!I)=2k\,h(\theta)$ ($h$ the pair entropy). For $\theta=\pi/4$ this gives $S(R_k)=k\ln2$ and mutual information $2k\ln2$, exactly the monotonic Hawking-like entropy.

The model deliberately has no mechanism that returns information: the emitted modes stay entangled with the interior partners forever. It reproduces the *semiclassical* entropy growth that the Page curve is contrasted with.
