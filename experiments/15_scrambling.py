"""Experiment 15: OTOC light cone and entanglement growth in Haar brickwork circuits."""

from _common import *  # noqa: F401,F403
from bhstatmech.scrambling import entanglement_growth, otoc_vs_depth

save = figure_saver(__file__)
scfg = CFG["scrambling"]
n, nr = int(scfg["n_qubits_otoc"]), int(scfg["n_realizations"])
ne, nre = int(scfg["n_qubits_entanglement"]), int(scfg["n_realizations_entanglement"])
print("Experiment 15: scrambling")
depth = 20
dists = [1, 2, 3, 5, 7]
res = {d: otoc_vs_depth(n, d, depth, nr, SEED + d) for d in dists}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
t = np.arange(depth + 1)
for d, col in zip(dists, CYCLE + [PALETTE["grey"]]):
    m, s = res[d]
    a.plot(t, m, color=col, label=f"distance {d}"); a.fill_between(t, m - 2 * s, m + 2 * s, color=col, alpha=0.2)
a.axhline(0, color=PALETTE["black"], lw=0.6)
a.set_xlabel("circuit layers"); a.set_ylabel(r"OTOC $F(t)$"); a.set_title(f"(a) OTOC, n = {n} qubits"); a.legend(fontsize=8)
# front arrival time
thr = 0.5
arr = [int(np.argmax(res[d][0] < thr)) if (res[d][0] < thr).any() else np.nan for d in dists]
b.plot(dists, arr, "o-", color=PALETTE["blue"], label="first layer with F < 0.5")
b.plot(dists, [d for d in dists], color=PALETTE["black"], ls=":", label="light cone: 1 site per layer")
b.set_xlabel("operator separation"); b.set_ylabel("layers"); b.set_title("(b) Butterfly-velocity light cone"); b.legend(fontsize=8)
dm = 30
mE, sE, page = entanglement_growth(ne, dm, nre, SEED + 77)
tt = np.arange(dm + 1)
c3.plot(tt, mE, color=PALETTE["blue"], label=f"half-chain entropy, n = {ne}")
c3.fill_between(tt, mE - 2 * sE, mE + 2 * sE, color=PALETTE["blue"], alpha=0.3)
c3.axhline(page, color=PALETTE["vermilion"], ls="--", label=f"Page value {page:.4f}")
c3.axhline((ne // 2) * np.log(2), color=PALETTE["grey"], ls=":", label="maximal (N/2) ln 2")
c3.set_xlabel("circuit layers"); c3.set_ylabel("S [nats]"); c3.set_title("(c) Entanglement growth and saturation"); c3.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig37_scrambling_otoc_entanglement",
     f"Haar-random two-qubit brickwork circuits. (a) The OTOC F = Re Tr[W(t)^dag V^dag W(t) V]/d between Z on site 0 and Z at the stated distance stays at 1 until the operator front arrives and then decays toward ~0 (mean over {nr} circuits, bands +/- 2 s.e.m.). "
     f"(b) The arrival layer grows linearly with distance, a discrete light cone (open boundary conditions, so edge effects exist). (c) Half-chain entanglement from a product state saturates at {mE[-1]:.3f} +/- {sE[-1]:.3f} versus the Page value {page:.3f}. "
     "Exact state-vector calculations; small sizes only.", {"n_otoc": n, "n_ent": ne, "realizations": [nr, nre]}, seed=SEED)
save_results("15_scrambling", {"otoc_arrival_layers": dict(zip(map(str, dists), map(float, arr))), "final_entropy": float(mE[-1]), "final_sem": float(sE[-1]), "page_value": float(page)})
