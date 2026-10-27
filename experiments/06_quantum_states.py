"""Experiment 06: canonical states, entropies, eigenvalue distributions, mutual information versus entanglement."""

from _common import *  # noqa: F401,F403
from bhstatmech.entanglement import concurrence_two_qubit, mutual_information, negativity
from bhstatmech.numerical_methods import spawn_rngs
from bhstatmech.quantum_states import (
    K0, K1, bell_state, classically_correlated_state, density_from_ket, ghz_state, partial_trace,
    partially_entangled_state, product_state, purity, random_density_matrix, random_pure_state,
    reduced_density_from_ket, separable_mixture, spectrum, thermal_state, von_neumann_entropy, werner_state,
    SX, SZ,
)

save = figure_saver(__file__)
print("Experiment 06: quantum states")
LN2 = np.log(2.0)

# ---- Fig 20: Bell and product state entropies --------------------------------------------------------------------
theta = np.linspace(0, np.pi / 2, 200)
S_A = [von_neumann_entropy(reduced_density_from_ket(partially_entangled_state(t), (2, 2), [0])) for t in theta]
S_AB = [von_neumann_entropy(density_from_ket(partially_entangled_state(t))) for t in theta]
states = {
    "product": density_from_ket(product_state(K0, (K0 + K1) / np.sqrt(2))),
    "Bell": density_from_ket(bell_state()),
    "classically\ncorrelated": classically_correlated_state(),
    "Werner\n(p = 0.5)": werner_state(0.5),
}
rows = {}
for k, rho in states.items():
    rows[k] = {"S(AB)": von_neumann_entropy(rho), "S(A)": von_neumann_entropy(partial_trace(rho, (2, 2), [0])),
               "I(A:B)": mutual_information(rho, (2, 2)), "N": negativity(rho, (2, 2))}
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
a.plot(theta / np.pi, S_A, color=PALETTE["blue"], label=r"$S(A)$ of $\cos\theta|00\rangle+\sin\theta|11\rangle$")
a.plot(theta / np.pi, S_AB, color=PALETTE["vermilion"], ls="--", label=r"$S(AB)$ (global state is pure)")
a.axhline(LN2, color=PALETTE["grey"], ls=":", lw=1)
a.annotate(r"$\ln 2$", xy=(0.02, LN2), xytext=(0.02, LN2 + 0.02), color=PALETTE["grey"])
a.set_xlabel(r"$\theta/\pi$ (product at 0, Bell at 1/4)")
a.set_ylabel("entropy [nats]")
a.set_title("(a) Entanglement entropy of a two-qubit pure state")
a.legend(loc="upper center", fontsize=8.5)
names = list(rows)
w = 0.2
for i, (m, col) in enumerate(zip(["S(AB)", "S(A)", "I(A:B)", "N"], [PALETTE["vermilion"], PALETTE["blue"], PALETTE["green"], PALETTE["orange"]])):
    vals = [rows[n][m] for n in names]
    bars = b.bar(np.arange(len(names)) + (i - 1.5) * w, vals, w, color=col, label=m if m != "N" else "negativity N", edgecolor="white", linewidth=0.8)
    for xx, v in zip(np.arange(len(names)) + (i - 1.5) * w, vals):
        b.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
b.set_xticks(np.arange(len(names)))
b.set_xticklabels(names)
b.set_ylabel("nats")
b.set_title("(b) Entropies, mutual information and negativity")
b.legend(fontsize=8.5, loc="upper left")
b.set_ylim(0, 1.6)
save(fig, "fig20_bell_and_product_entropies",
     "(a) Single-qubit entropy of cos(t)|00> + sin(t)|11>: zero for the product state, ln 2 for the Bell state, while the global entropy is zero throughout. "
     "(b) Product state: all quantities vanish. Bell state: S(AB) = 0, S(A) = ln 2, I = 2 ln 2, N = 1/2. Classically correlated mixture: I = ln 2 but N = 0 "
     "(correlation without entanglement). Werner state p = 0.5: mixed, entangled.",
     {"theta_grid": 200})

# ---- Fig 21: eigenvalue distributions of random density matrices -----------------------------------------------------------
rng_list = spawn_rngs(SEED, 120)
d = 64
ev_hs = np.concatenate([d * spectrum(random_density_matrix(d, seed=r)) for r in rng_list])
m_sub, n_env = 16, 64
ev_red = np.concatenate([m_sub * np.linalg.eigvalsh(reduced_density_from_ket(random_pure_state(m_sub * n_env, r), (m_sub, n_env), [0])) for r in rng_list])
def mp_density(xv, cr):
    lo, hi = (1 - np.sqrt(cr)) ** 2, (1 + np.sqrt(cr)) ** 2
    out = np.zeros_like(xv)
    ok = (xv > lo) & (xv < hi)
    out[ok] = np.sqrt((hi - xv[ok]) * (xv[ok] - lo)) / (2 * np.pi * cr * xv[ok])
    return out
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
xg = np.linspace(0.01, 4.2, 600)
a.hist(ev_hs, bins=60, density=True, color=PALETTE["sky"], edgecolor="white", label=f"sampled, d = {d}, 120 matrices")
a.plot(xg, mp_density(xg, 1.0), color=PALETTE["vermilion"], label="Marchenko-Pastur, c = 1")
a.set_xlim(0, 4.2)
a.set_xlabel(r"rescaled eigenvalue $d\,\lambda$")
a.set_ylabel("density")
a.set_title("(a) Hilbert-Schmidt ensemble")
a.legend()
xg2 = np.linspace(0.3, 2.3, 600)
b.hist(ev_red, bins=50, density=True, color=PALETTE["sky"], edgecolor="white", label=f"sampled, {m_sub}x{n_env}, 120 states")
b.plot(xg2, mp_density(xg2, m_sub / n_env), color=PALETTE["vermilion"], label=f"Marchenko-Pastur, c = {m_sub}/{n_env}")
b.set_xlabel(r"rescaled eigenvalue $m\,\lambda$")
b.set_ylabel("density")
b.set_title("(b) Reduced state of a Haar-random pure state")
b.legend()
save(fig, "fig21_density_matrix_eigenvalues",
     "Eigenvalue distributions of random density matrices. (a) Hilbert-Schmidt measure (G G^dagger normalised, square Ginibre G) for d = 64 follows the Marchenko-Pastur "
     "law with c = 1 (support [0, 4]). (b) The reduced state of a Haar-random 16 x 64 bipartite pure state follows Marchenko-Pastur with c = m/n = 1/4 (support [1/4, 9/4]); "
     "the gap at zero is the signature of a subsystem much smaller than its complement (Page regime).",
     {"d": d, "m": m_sub, "n": n_env, "n_samples": 120}, seed=SEED)

# ---- Fig 22: density-matrix visualisation -----------------------------------------------------------------------------------
H = np.kron(SZ, SZ) + 0.6 * (np.kron(SX, np.eye(2)) + np.kron(np.eye(2), SX))
vis = {
    "product |0>|+>": density_from_ket(product_state(K0, (K0 + K1) / np.sqrt(2))),
    "Bell |phi+>": density_from_ket(bell_state()),
    "separable mixture\n(p = 0.5)": separable_mixture(0.5),
    "thermal state\n(beta = 1)": thermal_state(H.astype(complex), 1.0),
}
fig, axs = plt.subplots(1, 4, figsize=(14, 3.7))
for axx, (k, rho) in zip(axs, vis.items()):
    im = axx.imshow(np.abs(rho), cmap="Blues", vmin=0, vmax=1)
    axx.set_xticks(range(4)); axx.set_yticks(range(4))
    axx.set_xticklabels(["00", "01", "10", "11"]); axx.set_yticklabels(["00", "01", "10", "11"])
    axx.grid(False)
    ev = spectrum(rho)
    axx.set_title(f"{k}\nS={von_neumann_entropy(rho):.2f}, S(A)={von_neumann_entropy(partial_trace(rho,(2,2),[0])):.2f}, Tr$\\rho^2$={purity(rho):.2f}", fontsize=9)
    for i in range(4):
        for j in range(4):
            axx.text(j, i, f"{abs(rho[i, j]):.2f}", ha="center", va="center", fontsize=7, color="white" if abs(rho[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=axs, shrink=0.8, label=r"$|\rho_{ij}|$")
save(fig, "fig22_density_matrix_gallery",
     "Magnitudes |rho_ij| of two-qubit density matrices in the computational basis, with global entropy S, single-qubit entropy S(A) and purity Tr(rho^2) in the panel "
     "titles. The Bell state has off-diagonal coherences between |00> and |11>; the separable mixture has coherences too but is not entangled (negativity 0).",
     {"states": list(vis)})

# ---- Fig 23: mutual information vs entanglement and purity (Werner family) ------------------------------------------------------------
p = np.linspace(0, 1, 201)
MI = np.array([mutual_information(werner_state(q), (2, 2)) for q in p])
NEG = np.array([negativity(werner_state(q), (2, 2)) for q in p])
CON = np.array([concurrence_two_qubit(werner_state(q)) for q in p])
PUR = np.array([purity(werner_state(q)) for q in p])
SG = np.array([von_neumann_entropy(werner_state(q)) for q in p])
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
a.axvspan(0, 1 / 3, color=PALETTE["grey"], alpha=0.12)
a.plot(p, MI, color=PALETTE["blue"], label=r"mutual information $I(A:B)$ [nats]")
a.plot(p, NEG, color=PALETTE["vermilion"], ls="--", label="negativity")
a.plot(p, CON, color=PALETTE["green"], ls="-.", label="concurrence")
a.annotate("separable ($p \\leq 1/3$):\ncorrelated, not entangled", xy=(0.17, 0.5), xytext=(0.02, 0.75), color=PALETTE["grey"], fontsize=9)
a.set_xlabel(r"Werner parameter $p$ in $p|\Phi^+\rangle\langle\Phi^+| + (1-p) I/4$")
a.set_ylabel("measure")
a.set_title("(a) Correlation versus entanglement")
a.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.62))
b.plot(p, PUR, color=PALETTE["blue"], label=r"purity $\mathrm{Tr}\,\rho^2$")
b.plot(p, SG / (2 * LN2), color=PALETTE["vermilion"], ls="--", label=r"global entropy $S(AB)/2\ln2$")
b.set_xlabel(r"Werner parameter $p$")
b.set_ylabel("value")
b.set_title("(b) Purity and global entropy")
b.legend()
thr_neg = float(p[np.argmax(NEG > 1e-12)])
save(fig, "fig23_mutual_information_purity",
     "Werner states. Mutual information is non-zero for every p > 0 (it counts classical and quantum correlations) whereas negativity and concurrence vanish for p <= 1/3 "
     f"(computed threshold on the grid: p = {thr_neg:.3f}). Purity decreases monotonically as p decreases; the global entropy is 0 only at p = 1.",
     {"p_grid": 201})
save_results("06_states", {"rows": rows, "negativity_threshold_grid": thr_neg,
                           "mp_c1_support_max_sampled": float(ev_hs.max()), "mp_reduced_support_sampled": [float(ev_red.min()), float(ev_red.max())]})
