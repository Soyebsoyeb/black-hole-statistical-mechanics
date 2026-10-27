"""Experiment 05: Regge-Wheeler potentials, greybody factors, absorption cross section, emission rate."""

import time

from _common import *  # noqa: F401,F403
from bhstatmech.hawking_radiation import (
    absorption_cross_section, bose_occupation, greybody_factor, greybody_table, number_flux,
)
from bhstatmech.spacetimes import Schwarzschild

save = figure_saver(__file__)
print("Experiment 05: greybody factors")
gcfg = CFG["greybody"]
ELL_MAX = int(gcfg["ell_max"]) + 1  # one extra multipole as a truncation check
y = np.geomspace(gcfg["y_min"], gcfg["y_max"], int(gcfg["n_omega"]))
bh = Schwarzschild(1.0e30)
rs = bh.schwarzschild_radius

# ---- Fig 16: potentials -----------------------------------------------------------------------------
r = np.linspace(1.0005, 8, 800)
fig, ax = plt.subplots()
specs = [(0, 0, PALETTE["blue"], "-"), (1, 0, PALETTE["blue"], "--"), (2, 0, PALETTE["blue"], "-."),
         (2, 1, PALETTE["vermilion"], "-"), (2, 2, PALETTE["green"], "-")]
peaks = {}
for ell, s, col, ls in specs:
    V = bh.effective_potential(r * rs, ell, s) * rs**2
    ax.plot(r, V, color=col, ls=ls, label=rf"$s={s},\ \ell={ell}$")
    peaks[f"s{s}_l{ell}"] = {"r_peak_over_rs": float(r[np.argmax(V)]), "V_peak_rs2": float(V.max())}
ax.axvline(1.5, color=PALETTE["grey"], ls=":", lw=1)
ax.annotate("photon sphere\n$r=3M=1.5\\,r_s$", xy=(1.5, 0.02), xytext=(1.62, 0.02), color=PALETTE["grey"])
ax.set_xlabel(r"$r/r_s$")
ax.set_ylabel(r"$V(r)\,r_s^2$")
ax.set_title("Regge-Wheeler effective potentials on Schwarzschild")
ax.legend(loc="upper right")
save(fig, "fig16_effective_potentials",
     "Effective potentials V = f [l(l+1)/r^2 + (1 - s^2) r_s/r^3] for massless fields of spin s = 0, 1, 2 (units r_s = 1). The barrier peak "
     "lies near the photon sphere and rises like l(l+1); the scattering of waves off it determines the greybody factor. All potentials "
     "vanish at the horizon and decay as l(l+1)/r^2 at infinity.",
     {"fields": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]})

# ---- Compute transmission tables (real scattering calculation) -----------------------------------------
t0 = time.time()
G0 = greybody_table(y, ELL_MAX, spin=0)
defects = [greybody_factor(float(yy), l).flux_defect for yy in y[::10] for l in (0, 2)]
print(f"  scalar table {G0.shape} in {time.time() - t0:.1f} s; max flux defect {max(defects):.1e}")
np.savetxt(RESULTS / "greybody_scalar.csv", np.vstack([y, G0]).T, delimiter=",",
           header="y=omega*r_s/c, Gamma_l for l=0.." + str(ELL_MAX), comments="")

# ---- Fig 17: greybody factors ------------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ell, col, ls in zip(range(5), CYCLE + [PALETTE["grey"]], LINESTYLES):
    a.semilogx(y, G0[ell], color=col, ls=ls, label=rf"$\ell={ell}$")
a.set_xlabel(r"$y=\omega r_s/c$ ($x=4\pi y$)")
a.set_ylabel(r"transmission probability $\Gamma_\ell(y)$")
a.set_title("(a) Scalar field, $s=0$")
a.legend(loc="upper left")
a.set_ylim(-0.02, 1.05)
yy = np.geomspace(0.02, 1.6, 40)
for s, col, ls, lab in [(0, PALETTE["blue"], "-", "scalar"), (1, PALETTE["vermilion"], "--", "electromagnetic"), (2, PALETTE["green"], "-.", "gravitational (axial)")]:
    b.semilogx(yy, [greybody_factor(float(v), 2, s).gamma for v in yy], color=col, ls=ls, label=f"s = {s} ({lab})")
b.set_xlabel(r"$y=\omega r_s/c$")
b.set_ylabel(r"$\Gamma_{\ell=2}(y)$")
b.set_title(r"(b) Dependence on spin at $\ell=2$")
b.legend(loc="upper left", fontsize=9)
b.set_ylim(-0.02, 1.05)
save(fig, "fig17_greybody_factors",
     "Transmission probabilities obtained by integrating the radial wave equation from the horizon (pure ingoing wave) to a large radius and "
     "decomposing into Riccati-Hankel waves. All values lie in [0, 1] and Wronskian flux conservation holds to "
     f"{max(defects):.0e} (relative to the incident flux). Low multipoles transmit well above y ~ 0.3 while high multipoles are "
     "suppressed by the centrifugal barrier. Spin-1 and spin-2 fields have a stronger barrier at the same l.",
     {"ell_max": ELL_MAX, "n_omega": len(y), "rtol": 1e-10, "x_match": 400.0})

# ---- Fig 18: absorption cross section ------------------------------------------------------------------------------
sigma = absorption_cross_section(y, G0)
sigma_trunc = absorption_cross_section(y, G0[:-1])
trunc_err = np.max(np.abs(sigma - sigma_trunc) / sigma)
geo = 27 * np.pi / 4
fig, ax = plt.subplots()
ax.semilogx(y, sigma / (4 * np.pi), color=PALETTE["blue"], label=r"computed $\sigma_{abs}=\frac{\pi}{k^2}\sum(2\ell+1)\Gamma_\ell$")
ax.axhline(1.0, color=PALETTE["vermilion"], ls="--", label=r"$\omega\to0$: horizon area $4\pi r_s^2$")
ax.axhline(geo / (4 * np.pi), color=PALETTE["green"], ls="-.", label=r"$\omega\to\infty$: geometric optics $\frac{27}{4}\pi r_s^2$")
ax.set_xlabel(r"$y=\omega r_s/c$")
ax.set_ylabel(r"$\sigma_{abs}/(4\pi r_s^2)$")
ax.set_title("Scalar absorption cross section: both analytic limits recovered")
ax.legend(loc="center right", fontsize=9)
ax.set_ylim(0.5, 2.1)
save(fig, "fig18_absorption_cross_section",
     "Absorption cross-section of the Schwarzschild black hole for a massless scalar from the computed transmission probabilities. It "
     "tends to the horizon area as omega -> 0 (Das, Gibbons and Mathur 1997) and oscillates around the geometric-optics value (27/4) pi r_s^2 "
     f"at high frequency (Sanchez 1978). Adding one further multipole changes sigma by at most {trunc_err:.1e} (relative) over the range.",
     {"ell_max": ELL_MAX})

# ---- Fig 19: emission rate with and without greybody factor ------------------------------------------------------------------------
flux = number_flux(y, G0)
bb_area = 2 * y**2 / np.pi * bose_occupation(4 * np.pi * y)
bb_geo = 27 * y**2 / (8 * np.pi) * bose_occupation(4 * np.pi * y)
# power (in units hbar c^2 / r_s^2) and ratio to blackbody power with the horizon area, one bosonic dof
trap = lambda f: np.trapezoid(f, y)
P_grey = trap(y * flux)
P_bb = 1.0 / (7680 * np.pi)
ratio = P_grey / P_bb
P_geo = trap(y * bb_geo)
fig, ax = plt.subplots()
ax.loglog(y, flux, color=PALETTE["blue"], label="with greybody factors (computed)")
ax.loglog(y, bb_area, color=PALETTE["vermilion"], ls="--", label=r"blackbody, horizon area $4\pi r_s^2$")
ax.loglog(y, bb_geo, color=PALETTE["green"], ls="-.", label=r"blackbody, geometric-optics area $\frac{27}{4}\pi r_s^2$")
ax.set_xlabel(r"$y=\omega r_s/c$")
ax.set_ylabel(r"$d^2N/(dt\,d\omega)$ (dimensionless)")
ax.set_ylim(1e-8, 2)
ax.set_title("Scalar emission rate versus blackbody approximations")
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig19_emission_rate_greybody",
     "d^2N/(dt d omega) = (1/2 pi) sum (2l+1) Gamma_l / (exp(hbar omega/k_B T_H) - 1) for a single real scalar, computed from the transmission "
     "probabilities, compared with blackbody emission from the horizon area and from the geometric-optics capture area. At low frequency the "
     f"s-wave reproduces the horizon-area blackbody; the integrated scalar power is {ratio:.2f} times the horizon-area blackbody power for one "
     "bosonic degree of freedom (the geometric-optics blackbody, with emitting area 4 pi b_c^2 = 27 pi r_s^2, gives 27/4 = 6.75 and is reproduced numerically).",
     {"ell_max": ELL_MAX, "n_omega": len(y)})

# ---- Validation numbers -----------------------------------------------------------------------------------------------------------
lowf = [greybody_factor(v, 0).gamma / (4 * v * v) for v in (0.02, 0.01, 0.005)]
save_results("05_greybody", {
    "n_omega": len(y), "ell_max_used": ELL_MAX, "max_flux_defect_relative": float(max(defects)),
    "gamma_min": float(G0.min()), "gamma_max": float(G0.max()),
    "low_frequency_ratio_Gamma0_over_4y2_at_y_0.02_0.01_0.005": lowf,
    "sigma_over_geometric_optics_at_y_max": float(sigma[-1] / geo),
    "sigma_over_geometric_optics_range": [float(np.min(sigma[y > 0.6] / geo)), float(np.max(sigma[y > 0.6] / geo))],
    "truncation_max_relative_change_sigma": float(trunc_err),
    "scalar_power_over_horizon_area_blackbody": float(ratio),
    "geometric_optics_blackbody_power_over_horizon_area_blackbody": float(P_geo / P_bb),
    "potential_peaks": peaks,
})
