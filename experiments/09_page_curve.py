"""Experiment 09: Page curves from exact averages and Monte Carlo ensembles of random pure states."""

from _common import *  # noqa: F401,F403
from bhstatmech.page_curve import (
    hawking_reference_curve, page_average_entropy, page_bound_curve, qubit_page_curve_ensemble, qubit_page_exact,
)

save = figure_saver(__file__)
pcfg = CFG["page_curve"]
print("Experiment 09: Page curves")

# ---- Fig 27: multiple Hilbert-space dimensions (exact Haar averages) -----------------------------------------------------
Ns = [8, 12, 16, 24, 48]
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for Nq, col, ls in zip(Ns, CYCLE + [PALETTE["grey"]], LINESTYLES):
    curve = qubit_page_exact(Nq)
    a.plot(np.arange(Nq + 1) / Nq, curve / (Nq * np.log(2)), color=col, ls=ls, label=f"N = {Nq} qubits")
a.plot([0, 0.5, 1], [0, 0.5, 0], color=PALETTE["black"], lw=0.8, ls=":", label=r"$\min(k, N-k)/N$ bound")
a.set_xlabel("fraction of qubits in the radiation, $k/N$")
a.set_ylabel(r"$\langle S(R_k)\rangle / (N\ln 2)$")
a.set_title("(a) Average radiation entropy (exact Page formula)")
a.legend(fontsize=8.5)
Ngrid = np.arange(2, 41, 2)
deficit = np.array([(Nq / 2) * np.log(2) - page_average_entropy(2 ** (Nq // 2), 2 ** (Nq // 2)) for Nq in Ngrid])
b.plot(Ngrid, deficit, "o-", color=PALETTE["blue"], label=r"$\frac{N}{2}\ln 2 - \langle S\rangle_{k=N/2}$")
b.axhline(0.5, color=PALETTE["vermilion"], ls="--", label="large-N limit 1/2 nat")
b.set_xlabel("total qubits N")
b.set_ylabel("deficit from the maximal entropy [nats]")
b.set_title("(b) Finite-size correction at the Page point")
b.legend()
save(fig, "fig27_page_curves_dimensions",
     "Haar-average entropy of the first k of N qubits from Page's exact formula <S_{m,n}> = psi(mn+1) - psi(n+1) - (m-1)/(2n). The normalised curves approach the "
     "piecewise-linear bound min(k, N-k)/N as N grows (the rise-then-fall shape comes from sampling pure states of a fixed total dimension, not from an imposed curve). "
     "At the symmetric point the average falls short of the maximal value (N/2) ln 2 by 1/2 nat in the large-N limit (b).",
     {"N": Ns})

# ---- Fig 28: ensemble averages with uncertainty bands -----------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=False)
stats = {}
for ax_, Nq in zip(axs.ravel(), pcfg["qubit_counts"]):
    ns = int(pcfg["n_samples"][str(Nq)])
    mean, std, sem, data = qubit_page_curve_ensemble(Nq, ns, SEED + Nq)
    k = np.arange(Nq + 1)
    exact = qubit_page_exact(Nq)
    ax_.fill_between(k, mean - std, mean + std, color=PALETTE["sky"], alpha=0.45, label=r"$\pm 1$ sample s.d. (state-to-state spread)")
    ax_.fill_between(k, mean - 2 * sem, mean + 2 * sem, color=PALETTE["blue"], alpha=0.6, label=r"$\pm 2$ s.e.m. (statistical error of the mean)")
    ax_.plot(k, mean, color=PALETTE["blue"], lw=1.6, label=f"Monte Carlo mean ({ns} states)")
    ax_.plot(k, exact, color=PALETTE["vermilion"], ls="--", label="exact Haar average")
    ax_.plot(k, page_bound_curve(Nq), color=PALETTE["black"], ls=":", lw=0.9, label=r"$\min(k,N-k)\ln2$")
    ax_.set_title(f"N = {Nq} qubits (dim {2**Nq})")
    ax_.set_xlabel("qubits in radiation $k$")
    ax_.set_ylabel(r"$S(R_k)$ [nats]")
    z = np.abs(mean[1:-1] - exact[1:-1]) / sem[1:-1]
    stats[str(Nq)] = {"n_samples": ns, "max_abs_z_mean_vs_exact": float(z.max()), "max_std": float(std.max()), "peak_mean": float(mean.max()), "peak_exact": float(exact.max())}
axs[0, 0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
save(fig, "fig28_page_curve_ensembles",
     "Ensemble of Haar-random pure states of N qubits; S(R_k) is the entropy of the first k qubits. Dark band: +/- 2 standard errors of the mean (this is the Monte Carlo error); "
     "light band: +/- 1 sample standard deviation (the genuine state-to-state fluctuation, which shrinks as N grows). The Monte Carlo mean agrees with the exact average within the "
     "statistical error. Results are reproducible for the seeds recorded in the figure sidecar.",
     {"qubit_counts": pcfg["qubit_counts"], "n_samples": pcfg["n_samples"]}, seed=SEED)

# ---- Fig 29: semiclassical reference versus unitary --------------------------------------------------------------------------
Nq = 12
k = np.arange(Nq + 1)
fig, ax = plt.subplots()
ax.plot(k, hawking_reference_curve(Nq), color=PALETTE["vermilion"], ls="--", label=r"semiclassical reference $k\ln2$ (radiation never purifies)")
ax.plot(k, qubit_page_exact(Nq), color=PALETTE["blue"], label="Page average for a globally pure state")
ax.plot(k, page_bound_curve(Nq), color=PALETTE["black"], ls=":", lw=1, label=r"dimension bound $\min(k,N-k)\ln2$")
ax.axvline(Nq / 2, color=PALETTE["grey"], lw=0.8)
ax.annotate("Page point", xy=(Nq / 2, 0.3), xytext=(Nq / 2 + 0.15, 0.3), color=PALETTE["grey"])
ax.set_xlabel("qubits in radiation $k$")
ax.set_ylabel(r"entropy of the radiation [nats]")
ax.set_title("Hawking's monotonic entropy versus the Page curve (N = 12)")
ax.legend(loc="upper left", fontsize=9)
save(fig, "fig29_hawking_vs_page_reference",
     "The monotonic semiclassical reference S = k ln 2 (each emitted quantum is maximally entangled with a partner that stays out of reach) violates the dimension bound for k > N/2. "
     "The entropy of the radiation in any globally pure state of the finite system must turn over at or before the Page point, because S(R) = S(B) <= (N-k) ln 2.",
     {"N": Nq})
save_results("09_page_curve", {"ensemble_stats": stats, "page_deficit_N40": float(deficit[-1])})
