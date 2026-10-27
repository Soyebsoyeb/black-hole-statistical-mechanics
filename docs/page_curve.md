# The Page Curve

## Formula

For a Haar-random pure state on $\mathcal H_m\otimes\mathcal H_n$ ($m\le n$) the average subsystem entropy is $\langle S\rangle=\sum_{k=n+1}^{mn}1/k-(m-1)/(2n)$ ([mathematical_derivations.md](mathematical_derivations.md)). For $N$ qubits with $k$ in the radiation, $m=2^{\min(k,N-k)}$, $n=2^{\max(k,N-k)}$, giving a curve that rises as $k\ln2$ up to the Page point $k=N/2$ and falls symmetrically.

## Results

- Fig 27: curves for $N=8,\dots,48$; normalised curves approach $\min(k,N-k)/N$. The deficit from the maximal value at the symmetric point is $0.5000$ nat at $N=40$ (Fig 27b).
- Fig 28: Monte Carlo ensembles (400/300/150/60 states for $N=8/10/12/14$) agree with the exact average: the maximum deviation over interior points is 1.4 to 2.3 standard errors (as expected for the maximum of many correlated points). The state-to-state standard deviation is at most 0.030 nat at $N=8$ and 0.0045 at $N=14$ and shrinks with $N$.
- Fig 29: the semiclassical $k\ln2$ violates the bound $S\le(N-k)\ln2$ for $k>N/2$.

## Caveats

A Haar-random global state is a statement about typical states, not about the dynamics of any specific black hole. The Page time $t_P$ of a real evaporating black hole depends on entropy accounting (for instance on greybody factors and on which species radiate) and is not computed here.
