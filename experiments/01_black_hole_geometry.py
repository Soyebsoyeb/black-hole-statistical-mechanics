"""Experiment 01: Schwarzschild geometry (metric function, curvature, tortoise coordinate, null rays)."""

from _common import *  # noqa: F401,F403
from bhstatmech.spacetimes import Schwarzschild

save = figure_saver(__file__)
bh = Schwarzschild(c.M_SUN)
rs = bh.schwarzschild_radius
print("Experiment 01: geometry")

# ---- Fig 01: metric function and coordinate-singularity regularisation ------------------
x = np.concatenate([np.linspace(0.05, 0.999, 300), np.linspace(1.001, 6.0, 400)])
r = x * rs
f = bh.f(r)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
a.plot(x, f, color=PALETTE["blue"])
a.axhline(0, color=PALETTE["grey"], lw=0.8)
a.axvline(1, color=PALETTE["vermilion"], ls="--", lw=1.2)
a.set_xlabel(r"$r/r_s$")
a.set_ylabel(r"$f(r) = 1 - r_s/r$")
a.set_ylim(-8, 1.2)
a.set_title("(a) Metric function")
a.annotate(r"horizon $r = r_s$", xy=(1, -4), xytext=(1.15, -4.5), color=PALETTE["vermilion"])
a.annotate(r"$f<0$: $r$ is timelike", xy=(0.3, -3), xytext=(1.3, -7.2), color=PALETTE["grey"])
g_rr = 1.0 / f
b.semilogy(x, np.abs(g_rr), color=PALETTE["blue"], label=r"$|g_{rr}| = 1/|f|$ (Schwarzschild coordinates)")
b.axhline(1.0, color=PALETTE["green"], ls="--", label=r"$|\det g_{(v,r)}|/c^2 = 1$ (ingoing Eddington-Finkelstein)")
block, _, _ = bh.ingoing_ef_metric(r[(x > 0.9) & (x < 1.1)])
det = np.linalg.det(block) / c.C**2
assert np.allclose(det, -1.0)
b.axvline(1, color=PALETTE["vermilion"], ls="--", lw=1.2)
b.set_ylim(0.3, 1e3)
b.set_xlabel(r"$r/r_s$")
b.set_ylabel("metric component (dimensionless)")
b.set_title("(b) Coordinate singularity is removable")
b.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
save(fig, "fig01_metric_function",
     "Schwarzschild metric function f(r) (left). The component g_rr = 1/f diverges at the horizon in "
     "Schwarzschild coordinates but the determinant of the (v, r) block of the ingoing Eddington-Finkelstein "
     "metric equals -c^2 everywhere, so the horizon is a coordinate artefact there (right).",
     {"M_kg": c.M_SUN, "r_s_m": rs})

# ---- Fig 02: Kretschmann scalar --------------------------------------------------------------
xs = np.logspace(-2, 1, 400)
K = bh.kretschmann(xs * rs) * rs**4
fig, ax = plt.subplots()
ax.loglog(xs, K, color=PALETTE["blue"])
ax.plot([1], [bh.kretschmann(rs) * rs**4], "o", color=PALETTE["vermilion"], ms=7, zorder=3)
ax.annotate(r"horizon: $K r_s^4 = 12$ (finite)", xy=(1, 12), xytext=(0.03, 3e-1), color=PALETTE["vermilion"],
            arrowprops=dict(arrowstyle="-", color=PALETTE["vermilion"], lw=0.8))
ax.annotate(r"$r \to 0$: $K \propto r^{-6}$ diverges", xy=(0.02, 12 / 0.02**6), xytext=(0.07, 2e9),
            color=PALETTE["black"])
ax.set_xlabel(r"$r/r_s$")
ax.set_ylabel(r"Kretschmann scalar $K\, r_s^4$")
ax.set_title("Curvature singularity at r = 0, regular horizon")
save(fig, "fig02_kretschmann_scalar",
     "K = 12 r_s^2 / r^6 = 48 G^2 M^2 / (c^4 r^6), verified symbolically (tests/test_spacetimes.py). K is finite at "
     "r = r_s, so the horizon is not a curvature singularity; K diverges only at r = 0.",
     {"M_kg": c.M_SUN, "x_range": [1e-2, 10]})

# ---- Fig 03: tortoise coordinate ---------------------------------------------------------------
eps = np.logspace(-10, 1.3, 500)
rstar = bh.tortoise(rs * (1 + eps)) / rs
fig, ax = plt.subplots()
ax.semilogx(eps, rstar, color=PALETTE["blue"], label=r"$r_*/r_s = r/r_s + \ln(r/r_s - 1)$")
ax.semilogx(eps, 1 + np.log(eps), color=PALETTE["vermilion"], ls="--", label=r"near horizon: $1 + \ln(r/r_s - 1)$")
ax.semilogx(eps, 1 + eps, color=PALETTE["green"], ls=":", label=r"far field: $r/r_s$")
ax.set_ylim(-26, 12)
ax.set_xlabel(r"$r/r_s - 1$")
ax.set_ylabel(r"$r_*/r_s$")
ax.set_title("Tortoise coordinate: logarithmic divergence at the horizon")
ax.legend(loc="lower right")
slope = np.polyfit(np.log(eps[eps < 1e-4]), rstar[eps < 1e-4], 1)[0]
save(fig, "fig03_tortoise_coordinate",
     f"r_* = r + r_s ln(r/r_s - 1). Fitted slope d(r_*/r_s)/d ln(r/r_s - 1) for r/r_s - 1 < 1e-4 is {slope:.6f} "
     "(expected 1). The horizon is infinitely far away in r_*, which underlies the mode definitions used for "
     "Hawking radiation.",
     {"eps_range": [1e-10, 20]})

# ---- Fig 04: radial null rays ---------------------------------------------------------------------
fig, ax = plt.subplots()
for i, r0 in enumerate([1.2, 2.0, 3.5, 6.0]):
    rr = rs * np.linspace(r0, 1 + 2e-7, 800)
    tin = -(bh.tortoise(rr) - bh.tortoise(rs * r0)) / c.C
    ax.plot(rr / rs, tin * c.C / rs, color=PALETTE["blue"], ls=LINESTYLES[0], lw=1.4,
            label="ingoing rays" if i == 0 else None)
    rr2 = rs * np.linspace(1 + 2e-7, r0, 800)
    ax.plot(rr2 / rs, (bh.tortoise(rr2) - bh.tortoise(rs * r0)) * 1 / rs, color=PALETTE["vermilion"], ls="--", lw=1.4,
            label="outgoing rays (traced back in time)" if i == 0 else None)
ax.axvline(1, color=PALETTE["grey"], lw=1)
ax.set_xlim(0.9, 6.5)
ax.set_ylim(-12, 12)
ax.set_xlabel(r"$r/r_s$")
ax.set_ylabel(r"$c\,(t - t_0)/r_s$")
ax.set_title("Radial null rays: the horizon is reached only as t -> infinity")
ax.legend(loc="upper right")
save(fig, "fig04_radial_null_rays",
     "Radial null geodesics t = t0 +/- (r_*(r) - r_*(r0))/c in Schwarzschild time. Rays approach r = r_s only "
     "asymptotically in t, illustrating the coordinate singularity of Schwarzschild time at the horizon.",
     {"r0_over_rs": [1.2, 2.0, 3.5, 6.0]})

save_results("01_geometry", {
    "r_s_m": rs, "kretschmann_at_horizon_times_rs4": float(bh.kretschmann(rs) * rs**4),
    "tortoise_log_slope_fit": float(slope),
    "ef_determinant_over_c2": float(det[0]),
})
