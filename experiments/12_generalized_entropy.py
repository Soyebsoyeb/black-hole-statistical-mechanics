"""Experiment 12: generalized-entropy branches and the island envelope (toy model)."""

from _common import *  # noqa: F401,F403
from bhstatmech.generalized_entropy import IslandToyModel

save = figure_saver(__file__)
mdl = IslandToyModel()
tp = mdl.page_time()
print("Experiment 12: generalized entropy; page time", tp)
taus = np.linspace(0, 0.95, 400)
u = np.linspace(0, 0.6, 400)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
for t, col in zip([0.3, 0.55, tp, 0.8], CYCLE):
    a.plot(u, [float(mdl.S_gen(x, t)) for x in u], color=col, label=rf"$\tau$ = {t:.3f}")
    if mdl.island_exists(t):
        us = mdl.u_star_analytic(t)
        a.plot([us], [float(mdl.S_gen(us, t))], "o", color=col, ms=5)
a.set_xlabel("island boundary position u"); a.set_ylabel(r"$S_{gen}(u;\tau)$"); a.set_title("(a) Generalized entropy vs boundary position"); a.legend(fontsize=8)
no = np.array([float(mdl.S_no_island(t)) for t in taus])
isl = np.array([mdl.S_island_analytic(float(t)) for t in taus])
env = mdl.S_radiation(taus)
b.plot(taus, no, color=PALETTE["vermilion"], ls="--", label="no island (Hawking-like, grows)")
b.plot(taus, isl, color=PALETTE["green"], ls="-.", label="island extremum")
b.plot(taus, env, color=PALETTE["blue"], lw=2, label="minimum (island formula)")
b.axvline(tp, color=PALETTE["grey"], lw=0.8); b.annotate(rf"$\tau_P$ = {tp:.3f}", (tp, env.max() * 0.2), (tp + 0.02, env.max() * 0.2), fontsize=9)
b.set_xlabel(r"evaporation time $\tau$"); b.set_ylabel("S(radiation)"); b.set_title("(b) Page transition"); b.legend(fontsize=8)
c3.plot(taus, [float(mdl.S_H(t)) for t in taus], color=PALETTE["black"], label=r"$S_H(\tau)$ (black hole)")
c3.plot(taus, env, color=PALETTE["blue"], label="S(radiation)")
c3.set_xlabel(r"$\tau$"); c3.set_title("(c) Radiation entropy tracks the shrinking BH after the transition"); c3.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig34_generalized_entropy_branches",
     f"Phenomenological island toy model S_gen(u) = S_H(1 + mu u) + S_rad exp(-u/xi) with S_H = S0 (1-tau)^(2/3) and S_rad = r (S0 - S_H); parameters S0 = {mdl.S0:g}, r = {mdl.r:.3f}, mu = {mdl.mu:g}, xi = {mdl.xi:g}. "
     f"(a) The island candidate is a minimum of S_gen in u (dots). (b) The entropy of the radiation is the smaller of the no-island and island branches; the switch at tau_P = {tp:.4f} is a kink, i.e. a first-order-like transition. "
     "(c) After tau_P the radiation entropy follows the black-hole entropy. The functional form and parameters are illustrative assumptions, not derived from gravity.",
     {"S0": mdl.S0, "r": mdl.r, "mu": mdl.mu, "xi": mdl.xi})
save_results("12_generalized_entropy", {"page_time": float(tp), "model": dict(S0=mdl.S0, r=mdl.r, mu=mdl.mu, xi=mdl.xi)})
