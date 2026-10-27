"""Experiment 07: toy models of entanglement across a horizon (finite-dimensional, not a field calculation)."""

from _common import *  # noqa: F401,F403
from bhstatmech.entanglement import horizon
from bhstatmech.quantum_states import reduced_density_from_ket, von_neumann_entropy

save = figure_saver(__file__)
print("Experiment 07: horizon entanglement")

x = np.linspace(0.05, 6, 120)
exact = np.array([horizon.bose_mode_entropy(v) for v in x])
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
a.plot(x, exact, color=PALETTE["black"], lw=2.2, label="thermal Bose mode (analytic)")
for dcut, col, ls in [(2, PALETTE["vermilion"], "--"), (4, PALETTE["green"], "-."), (8, PALETTE["orange"], ":")]:
    s = [von_neumann_entropy(reduced_density_from_ket(horizon.thermofield_pair_state(v, dcut), (dcut, dcut), [1])) for v in x]
    a.plot(x, s, color=col, ls=ls, label=f"truncated pair state, d = {dcut}")
a.set_xlabel(r"$x=\hbar\omega/(k_BT_H)$")
a.set_ylabel(r"entropy of the outgoing mode [nats]")
a.set_title("(a) Interior-exterior entanglement of one mode")
a.legend(fontsize=9)
conv = {}
for v, col, ls in [(0.3, PALETTE["blue"], "-"), (1.0, PALETTE["vermilion"], "--"), (3.0, PALETTE["green"], "-.")]:
    ds = np.array([2, 4, 8, 16, 32, 64, 128])
    err = [abs(von_neumann_entropy(reduced_density_from_ket(horizon.thermofield_pair_state(v, int(dd)), (int(dd), int(dd)), [1])) - horizon.bose_mode_entropy(v)) for dd in ds]
    b.loglog(ds, np.maximum(err, 1e-17), color=col, ls=ls, marker="o", ms=4, label=rf"$x={v}$")
    conv[str(v)] = [float(e) for e in err]
b.set_xlabel("truncation dimension d")
b.set_ylabel("|S(d) - S(analytic)|")
b.set_title("(b) Convergence of the truncated state")
b.legend()
save(fig, "fig24_thermofield_pair_entropy",
     "Truncated two-mode squeezed (Bogoliubov-pair) state |Psi> = sqrt(1-q) sum_n q^{n/2}|n>_in|n>_out with q = exp(-hbar omega/k_B T_H). The global state is pure; tracing out the interior "
     "gives the thermal Bose distribution with <n> = 1/(e^x - 1) and entropy (n+1)ln(n+1) - n ln n. d = 2 reproduces the large-x limit (a qubit) only. "
     "Convergence in d is geometric, faster at large x. This models the mode structure, not the near-horizon field dynamics that produce it.",
     {"d_values": [2, 4, 8, 16, 32, 64, 128]})

N = 8
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
res = {}
for th, col, ls, lab in [(np.pi / 4, PALETTE["blue"], "-", r"maximally entangled pairs ($\theta=\pi/4$)"), (0.3, PALETTE["vermilion"], "--", r"weakly entangled pairs ($\theta=0.3$)")]:
    k, s_r, s_i, mi = horizon.pair_model_entropies(N, th)
    a.plot(k, s_r, color=col, ls=ls, marker="o", ms=4, label=lab)
    b.plot(k, mi, color=col, ls=ls, marker="o", ms=4, label=lab)
    res[f"{th:.3f}"] = {"S_rad": s_r.tolist(), "MI": mi.tolist(), "S_interior": s_i.tolist()}
a.plot(k, np.full_like(k, N * np.log(2), dtype=float), color=PALETTE["grey"], ls=":", lw=1)
a.annotate(r"$S(\mathrm{interior})=N\ln2$ (maximally entangled pairs)", xy=(0.3, N * np.log(2)), xytext=(0.3, N * np.log(2) - 0.55), color=PALETTE["grey"], fontsize=9)
a.set_xlabel("number of emitted quanta $k$")
a.set_ylabel("$S(R_k)$ [nats]")
a.set_title("(a) Radiation entropy grows linearly (no Page turnover)")
a.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.85))
b.set_xlabel("number of emitted quanta $k$")
b.set_ylabel(r"$I(R_k:\mathrm{interior})$ [nats]")
b.set_title("(b) Correlation with the interior partners")
b.legend(fontsize=9)
save(fig, "fig25_pair_emission_model",
     "N = 8 independent horizon pairs. Radiation R_k is the first k outgoing qubits; the interior holds all N partners. S(R_k) = k S_pair and I(R_k:interior) = 2 S(R_k): the emission "
     "entropy grows monotonically because the model never lets the radiation interact with the interior, i.e. it encodes the semiclassical (Hawking) assumption and has no mechanism to return information. "
     "Compare Figs 28-31, where unitary dynamics among the remaining degrees of freedom produces a turnover.",
     {"N": N, "theta": [np.pi / 4, 0.3]})
save_results("07_horizon_entanglement", {"pair_model": res, "truncation_errors": conv})
