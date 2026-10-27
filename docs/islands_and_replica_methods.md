# Replica Methods and Their Numerical Analogue

## Identity

$$S(\rho)=-\lim_{n\to1}\partial_n\ln\mathrm{Tr}\rho^n.$$

In gravity, $\mathrm{Tr}\rho^n$ is computed at integer $n$ by a replica path integral, and the continuation to $n\to1$ yields the quantum extremal surface prescription (Lewkowycz-Maldacena for the classical case; Penington et al. and Almheiri et al. for replica wormholes). That calculation is **not** implemented here.

## What is implemented (`bhstatmech.replica_methods`)

For finite-dimensional density matrices the moments $\mathrm{Tr}\rho^n$ are known for all real $n$ from the spectrum, so the identity can be tested numerically.

| Test (random $d=8$ state, $S=1.503146$) | Result |
| --- | --- |
| Central difference, best step | error $6.3\times10^{-12}$ |
| Richardson extrapolation, best step | error $1.1\times10^{-13}$ |
| Polynomial continuation from integers $n\le 2,\dots,12$ | error 0.159 at $n_{max}=2$ down to 0.012 at $n_{max}=12$ |

(Figs 30 and 31.) The last row illustrates why analytic continuation from integer replicas is the difficult step: convergence is slow and not guaranteed to be to the right function. This is a finite-dimensional analogy only.
