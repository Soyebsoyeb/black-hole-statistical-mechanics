"""Experiment 10: random-unitary evaporation toy versus a non-unitary (information-loss) reference."""

from _common import *  # noqa: F401,F403
from bhstatmech.evaporation import simulate_nonunitary_evaporation, simulate_unitary_evaporation
from bhstatmech.page_curve import qubit_page_exact

save = figure_saver(__file__)
ecfg = CFG["evaporation"]
N, nseed = int(ecfg["n_qubits"]), int(ecfg["n_seeds"])
n_nonuni = 4
print("Experiment 10: unitary vs non-unitary evaporation")
uni = [simulate_unitary_evaporation(N, SEED + s, keep_spectra=(s == 0)) for s in range(nseed)]
non = [simulate_nonunitary_evaporation(N, SEED + 1000 + s) for s in range(n_nonuni)]
steps = uni[0].steps
S_u = np.array([r.s_radiation for r in uni])
mi_u = np.array([r.mutual_information for r in uni])
S_n = np.array([r.s_radiation for r in non]); B_n = np.array([r.s_black_hole for r in non]); G_n = np.array([r.s_global for r in non])
pur_n = np.array([r.purity_global for r in non])
exact = qubit_page_exact(N)
norm_dev = max(r.norm_deviation for r in uni)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
m, s = S_u.mean(0), S_u.std(0, ddof=1)
a.fill_between(steps, m - s, m + s, color=PALETTE["sky"], alpha=0.5, label=r"unitary: $\pm1$ s.d. over seeds")
a.plot(steps, m, color=PALETTE["blue"], label=f"unitary: S(R), mean of {nseed} seeds")
a.plot(steps, exact, color=PALETTE["black"], ls=":", label="exact Haar average (Page)")
a.plot(steps, S_n.mean(0), color=PALETTE["vermilion"], ls="--", label=r"non-unitary: $S(R)=k\ln2$")
a.set_xlabel("emission step k"); a.set_ylabel("S(R) [nats]"); a.set_title("(a) Radiation entropy"); a.legend(fontsize=8)
b.plot(steps, B_n.mean(0), color=PALETTE["green"], label="non-unitary: S(BH)")
b.plot(steps, G_n.mean(0), color=PALETTE["vermilion"], ls="--", label="non-unitary: S(R,BH)")
b.plot(steps, np.zeros_like(steps, dtype=float), color=PALETTE["blue"], label="unitary: S(R,BH) = 0")
b.plot(steps, pur_n.mean(0), color=PALETTE["grey"], ls="-.", label="non-unitary: purity of global state")
b.set_xlabel("emission step k"); b.set_title("(b) Global state: pure vs mixed"); b.legend(fontsize=8)
c3.plot(steps, mi_u.mean(0), color=PALETTE["blue"], label=r"unitary: $I(R\!:\!BH)=2S(R)$")
c3.plot(steps, non[0].mutual_information, color=PALETTE["vermilion"], ls="--", label="non-unitary: I = 0 (no correlations)")
c3.set_xlabel("emission step k"); c3.set_ylabel("mutual information [nats]"); c3.set_title("(c) Correlations radiation / black hole"); c3.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig32_unitary_vs_nonunitary_evaporation",
     f"Toy evaporation of an N = {N} qubit register: each step applies a Haar-random unitary to the remaining black-hole qubits and then emits one qubit. (a) Unitary evolution reproduces the Page curve "
     f"(mean over {nseed} seeds vs the exact average); the non-unitary reference (each emitted qubit is traced out of a mixed register and discarded from correlations) gives the monotonic Hawking-like S(R) = k ln 2. "
     f"(b) The unitary global state stays pure (max norm deviation {norm_dev:.1e}); the non-unitary one becomes mixed. (c) The mutual information between radiation and black hole is 2 S(R) in the unitary model and zero in the reference. "
     f"Non-unitary averages use {n_nonuni} seeds. This is a qubit toy model, not a gravitational calculation.",
     {"n_qubits": N, "n_seeds": nseed, "n_nonunitary_seeds": n_nonuni}, seed=SEED)

# entanglement spectrum evolution (seed 0)
sp = uni[0].spectra
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
for j in range(1, N):
    p = np.asarray(sp[j - 1]); p = p[p > 1e-14]
    a.semilogy(np.arange(1, len(p) + 1), p, ".-", ms=2.5, lw=0.8, color=plt.cm.viridis(j / N), label=f"k = {j}" if j in (1, 3, 5, 7, 9) else None)
a.set_xlabel("eigenvalue rank"); a.set_ylabel(r"entanglement spectrum $p_i$"); a.set_title("(a) Spectrum of the radiation state"); a.legend(fontsize=8, ncol=2)
rank = [min(2 ** j, 2 ** (N - j)) for j in range(1, N)]
eff = [np.exp(-np.sum(np.asarray(sp[j - 1])[np.asarray(sp[j - 1]) > 1e-14] * np.log(np.asarray(sp[j - 1])[np.asarray(sp[j - 1]) > 1e-14]))) for j in range(1, N)]
b.semilogy(range(1, N), rank, "s--", color=PALETTE["black"], label="maximal rank")
b.semilogy(range(1, N), eff, "o-", color=PALETTE["blue"], label=r"effective rank $e^{S}$")
b.set_xlabel("emission step k"); b.set_title("(b) Effective rank rises, then falls"); b.legend()
save(fig, "fig33_entanglement_spectrum_evolution",
     "Entanglement spectrum of the radiation for one seed of the unitary toy. The number of nonzero eigenvalues is bounded by min(2^k, 2^(N-k)); the effective rank exp(S) follows it up to the Page point "
     "and then decreases, which is the spectral statement of the Page turnover.", {"n_qubits": N}, seed=SEED)
save_results("10_unitary_evaporation", {"n_qubits": N, "mean_peak_S_R": float(m.max()), "exact_peak": float(exact.max()), "max_norm_deviation": float(norm_dev),
                                          "nonunitary_final_S_R": float(S_n.mean(0)[-1]), "unitary_final_S_R": float(m[-1])})
