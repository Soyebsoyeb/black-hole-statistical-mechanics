"""Experiment 13: island toy model parameter study and analytic-vs-numeric extremisation."""

from _common import *  # noqa: F401,F403
from bhstatmech.generalized_entropy import IslandToyModel, page_time_no_correction

save = figure_saver(__file__)
print("Experiment 13: island parameter study")
base = IslandToyModel()
taus = np.linspace(0.3, 0.9, 61)
ua = np.array([base.u_star_analytic(t) for t in taus]); un = np.array([base.u_star_numeric(t)[0] if base.island_exists(t) else 0.0 for t in taus])
err_u = np.abs(ua - un)
Sa = np.array([base.S_island_analytic(t) for t in taus]); Sn = np.array([base.S_island_numeric(t) for t in taus])
err_S = np.nanmax(np.abs(Sa - Sn))

rs = np.linspace(0.4, 4.0, 40)
xis = [0.01, 0.05, 0.2]
S0s = [100.0, 1000.0, 10000.0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
a.semilogy(taus[ua > 0], np.maximum(err_u[ua > 0], 1e-16), "o-", ms=3, color=PALETTE["blue"], label=r"$|u^*_{num}-u^*_{ana}|$")
a.set_xlabel(r"$\tau$"); a.set_ylabel("absolute difference"); a.set_title("(a) Numerical vs analytic extremum"); a.legend()
a.annotate(f"max |dS_gen| = {err_S:.1e}", (0.05, 0.1), xycoords="axes fraction", fontsize=9)
for xi, col, ls in zip(xis, CYCLE, LINESTYLES):
    tps = [IslandToyModel(r=r, xi=xi).page_time() for r in rs]
    b.plot(rs, tps, color=col, ls=ls, label=rf"$\xi$ = {xi:g}")
b.plot(rs, [page_time_no_correction(r) for r in rs], color=PALETTE["black"], ls=":", label=r"$\kappa\to0$ closed form")
b.set_xlabel("radiation-to-horizon entropy ratio r"); b.set_ylabel(r"Page time $\tau_P$"); b.set_title("(b) Page time vs r and xi"); b.legend(fontsize=8)
for S0, col, ls in zip(S0s, CYCLE, LINESTYLES):
    mm = IslandToyModel(S0=S0)
    tt = np.linspace(0, 0.95, 300)
    c3.plot(tt, mm.S_radiation(tt) / S0, color=col, ls=ls, label=rf"$S_0$ = {S0:g}")
c3.set_xlabel(r"$\tau$"); c3.set_ylabel(r"$S(\mathrm{rad})/S_0$"); c3.set_title("(c) Dependence on S0 (curves nearly collapse)"); c3.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig35_island_parameter_study",
     f"(a) Bounded-minimiser and closed-form island locations agree to ~1e-8 (max difference in S_gen {err_S:.1e}). (b) Page time of the toy model against the entropy ratio r for several xi; xi (through kappa = mu xi) shifts it only slightly "
     "from the kappa -> 0 closed form 1 - (r/(1+r))^(3/2). (c) The normalised curve depends weakly on S0, so the shape is set by r and xi. All parameters are illustrative.",
     {"xis": xis, "S0s": S0s})
save_results("13_island_model", {"max_u_error": float(err_u.max()), "max_S_error": float(err_S), "page_time_base": float(base.page_time())})
