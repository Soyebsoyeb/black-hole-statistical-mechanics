# Numerical Methods

- **Reproducibility.** One `master_seed` in `configs/default.toml`; every stream is derived with `numpy.random.SeedSequence`. Rerunning an experiment reproduces its figure and JSON.
- **Derivatives.** Central differences ($O(h^2)$) and Richardson extrapolation ($O(h^4)$) with error scans that show the truncation/round-off trade-off (Fig 07, 31).
- **Root finding.** Bracketed Brent method with explicit bracket checks (`find_root`), raising on failure.
- **ODEs.** `solve_ode` wraps `scipy.integrate.solve_ivp` (DOP853 for the greybody solver) and raises `ConvergenceError` on failure. The greybody results change by at most $9\times10^{-11}$ relative when the tolerances are tightened, and the flux (Wronskian) defect is at most $1.1\times10^{-9}$ relative.
- **Evaporation ODE.** Solved in scaled variables $m=M/M_0$, $\tau=t/t_{evap}$; time error against the analytic solution is $4.8\times10^{-12}$.
- **Cancellation.** Extremal-limit expressions use $\sqrt{(M-Q)(M+Q)}$; Fermi occupations use `scipy.special.expit`.
- **Memory.** Reduced states of pure vectors use the Schmidt decomposition; purity-based quantities of $2^{N}\times2^N$ matrices are avoided when a complement shortcut exists.
- **Statistical error.** Monte Carlo figures show standard errors of the mean separately from sample standard deviations.
