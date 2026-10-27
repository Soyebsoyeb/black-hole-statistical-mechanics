"""Experiment 03: area-law entropy scaling, log corrections, comparison with ordinary systems, ensembles."""

from _common import *  # noqa: F401,F403
from bhstatmech.statistical_mechanics import (
    bh_canonical_exponent, bh_saddle_energy, entropy_with_log_correction, two_level_heat_capacity_exact,
    two_level_system,
)
from bhstatmech.thermodynamics import laws

save = figure_saver(__file__)
print("Experiment 03: entropy scaling and statistical mechanics")

# ---- Fig 12: S(A) with parameterised log correction --------------------------------------------------------------
A = np.logspace(2, 10, 300)
alphas = [(0.0, "alpha = 0 (area law only)", PALETTE["blue"], "-"), (-1.5, "alpha = -3/2", PALETTE["vermilion"], "--"),
          (0.5, "alpha = +1/2", PALETTE["green"], "-.")]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
for al, lab, col, ls in alphas:
    S = entropy_with_log_correction(A, al)
    a.loglog(A, S, color=col, ls=ls, label=lab)
    b.semilogx(A, S - A / 4, color=col, ls=ls, label=lab)
a.set_xlabel(r"horizon area $A/\ell_P^2$")
a.set_ylabel(r"$S/k_B$")
a.set_title(r"(a) $S/k_B = A/4\ell_P^2 + \alpha \ln(A/\ell_P^2)$")
a.legend(loc="upper left")
b.set_xlabel(r"horizon area $A/\ell_P^2$")
b.set_ylabel(r"correction $S/k_B - A/4\ell_P^2$")
b.set_title("(b) Sub-leading logarithmic term")
b.legend(loc="lower left")
slope_fit = np.polyfit(np.log(A), entropy_with_log_correction(A, -1.5) - A / 4, 1)[0]
save(fig, "fig12_entropy_vs_area_log_correction",
     "Bekenstein-Hawking entropy with a parameterised logarithmic correction. The coefficient alpha is model dependent and the three "
     "values shown are illustrative only (different microscopic approaches give different values; none is assumed here). The fitted slope "
     f"of the correction against ln A recovers alpha = {slope_fit:.6f} for the alpha = -3/2 curve.",
     {"alphas": [0.0, -1.5, 0.5], "A_range_lP2": [1e2, 1e10]})

# ---- Fig 13: area vs volume scaling ----------------------------------------------------------------------------------
Tg = 300.0
a_rad = 4 * c.SIGMA_SB / c.C
s_gas = (4.0 / 3.0) * a_rad * Tg**3 / c.K_B  # entropy density / k_B, m^-3
R = np.logspace(-6, 62, 400)
S_gas = s_gas * (4 / 3) * np.pi * R**3
S_bh = c.C**3 * 4 * np.pi * R**2 / (4 * c.G * c.HBAR)
# crossing: s_gas (4/3) pi R^3 = c^3 pi R^2 /(G hbar)  ->  R_x = 3 c^3 / (4 G hbar s_gas)
Rx = 3 * c.C**3 / (4 * c.G * c.HBAR * s_gas)
fig, ax = plt.subplots()
ax.loglog(R, S_gas, color=PALETTE["vermilion"], ls="--", label=r"thermal photon gas at 300 K: $S\propto R^3$")
ax.loglog(R, S_bh, color=PALETTE["blue"], label=r"black hole with $r_s = R$: $S_{BH}\propto R^2$")
ax.axvline(Rx, color=PALETTE["grey"], lw=0.8)
R_UNIV = 4.4e26  # m, radius of the observable universe (for scale)
ax.axvline(R_UNIV, color=PALETTE["green"], ls="-.", lw=1)
ax.annotate("observable universe", xy=(R_UNIV, 1e20), xytext=(R_UNIV * 1e-17, 1e12), color=PALETTE["green"])
ax.annotate(rf"crossover $R\approx{Rx:.1e}$ m", xy=(Rx, 1e170), xytext=(1e33, 1e60), color=PALETTE["grey"], arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["grey"]))
ax.set_xlabel("radius $R$ [m]")
ax.set_ylabel(r"entropy $S/k_B$")
ax.set_title("Volume scaling of ordinary matter versus area scaling of black holes")
ax.legend(loc="upper left")
save(fig, "fig13_area_vs_volume_scaling",
     "An ordinary thermal system (photon gas at fixed temperature) has extensive entropy S ~ R^3, whereas the Bekenstein-Hawking entropy "
     "of a black hole of radius R scales as R^2. At 300 K the crossover lies at R ~ 1.5e54 m, far beyond the observable universe (4.4e26 m), so the plot illustrates only the "
     "different scaling exponents, not a bound; it does not identify the black-hole microstates.",
     {"gas_temperature_K": Tg})

# ---- Fig 14: statistical mechanics of finite vs gravitating systems ------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
beta = np.linspace(0.15, 20, 400)
for N, col, ls in [(10, PALETTE["blue"], "-"), (40, PALETTE["vermilion"], "--"), (160, PALETTE["green"], "-.")]:
    sp = two_level_system(N, 1.0)
    cv = np.array([sp.heat_capacity(b_) for b_ in beta])
    a.plot(1 / beta, cv / N, color=col, ls=ls, label=f"N = {N} (enumerated levels)")
exact = two_level_heat_capacity_exact(beta, 40) / 40
a.plot(1 / beta, exact, color=PALETTE["black"], lw=0.8, ls=":", label="Schottky closed form (N-independent per atom)")
a.set_xlim(0, 6)
a.set_xlabel(r"temperature $k_BT/\epsilon$")
a.set_ylabel(r"heat capacity per atom $C/(N k_B)$")
a.set_title("(a) Finite system: $C = \\beta^2 \\mathrm{Var}(E) \\geq 0$")
a.legend(fontsize=8)
E = np.linspace(0.02, 1.2, 400)
for bt, col, ls in [(8 * np.pi * 0.3, PALETTE["blue"], "-"), (8 * np.pi * 0.5, PALETTE["vermilion"], "--"), (8 * np.pi * 0.8, PALETTE["green"], "-.")]:
    y = bh_canonical_exponent(E, bt)
    b.plot(E, y - y.min(), color=col, ls=ls, label=rf"$\beta = 8\pi\cdot{bt/(8*np.pi):.1f}$")
    b.plot([bh_saddle_energy(bt)], [0 if True else 0], "o", color=col, ms=5)
b.set_xlabel("energy $E$ (Planck units)")
b.set_ylabel(r"$[S(E) - \beta E] - \min$")
b.set_title("(b) Black hole: canonical saddle is a minimum")
b.legend()
save(fig, "fig14_ensembles_finite_vs_gravitating",
     "(a) Heat capacity of finite two-level systems computed directly from the density of states (binomial degeneracies) and compared with "
     "the closed form; it is non-negative in the canonical ensemble. (b) For S(E) = 4 pi E^2 the exponent S(E) - beta E of the canonical "
     "integral has its stationary point at E = beta/(8 pi), which is a minimum (second derivative 8 pi > 0): the canonical integral does "
     "not converge as a dominated saddle and the canonical and microcanonical descriptions disagree (Planck units, k_B = 1).",
     {"N": [10, 40, 160]})

save_results("03_entropy_scaling", {
    "log_correction_slope_alpha_minus_1p5": float(slope_fit),
    "gas_bh_crossover_radius_m": float(Rx),
    "log10_Omega_eff_solar_mass": float(laws.bekenstein_hawking_entropy(c.M_SUN, True) / np.log(10)),
    "max_schottky_deviation_N40": float(np.max(np.abs(np.array([two_level_system(40).heat_capacity(b_) for b_ in beta]) / 40 - exact))),
})
