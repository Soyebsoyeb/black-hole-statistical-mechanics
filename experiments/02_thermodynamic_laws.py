"""Experiment 02: Hawking temperature, entropy, first law, heat capacity, free energy, evaporation."""

from _common import *  # noqa: F401,F403
from bhstatmech.evaporation import mass_loss as ml
from bhstatmech.spacetimes import Kerr, ReissnerNordstrom, SchwarzschildAdS
from bhstatmech.thermodynamics import first_law, laws

save = figure_saver(__file__)
print("Experiment 02: thermodynamic laws")
M = np.logspace(8, 33, 400)
T_CMB = 2.725  # K

# ---- Fig 05: T_H vs M ---------------------------------------------------------------------
T = laws.hawking_temperature(M)
M_cmb = float(c.HBAR * c.C**3 / (8 * np.pi * c.G * c.K_B * T_CMB))
fig, ax = plt.subplots()
ax.loglog(M / c.M_SUN, T, color=PALETTE["blue"])
ax.axhline(T_CMB, color=PALETTE["grey"], ls="--", lw=1)
ax.annotate("CMB 2.725 K", xy=(1e-24, T_CMB), xytext=(1e-24, T_CMB * 3), color=PALETTE["grey"])
ax.plot([1], [float(laws.hawking_temperature(c.M_SUN))], "o", color=PALETTE["vermilion"])
ax.annotate(r"$1\,M_\odot$: $6.2\times10^{-8}$ K", xy=(1, 6.2e-8), xytext=(3e-6, 1e-9), color=PALETTE["vermilion"])
ax.annotate(rf"$T_H = T_{{CMB}}$ at $M \approx {M_cmb:.1e}$ kg", xy=(M_cmb / c.M_SUN, T_CMB),
            xytext=(1e-20, 1e-4), color=PALETTE["black"], arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["black"]))
ax.set_xlabel(r"mass $M / M_\odot$")
ax.set_ylabel(r"Hawking temperature $T_H$ [K]")
ax.set_title(r"$T_H = \hbar c^3 / (8\pi G M k_B) \propto M^{-1}$")
save(fig, "fig05_hawking_temperature",
     "Hawking temperature versus mass for Schwarzschild black holes. A black hole is hotter than the 2.725 K CMB only below "
     f"about {M_cmb:.1e} kg; stellar-mass holes (T ~ 1e-7 K) absorb more than they radiate in the present universe.",
     {"M_range_kg": [1e8, 1e33]})

# ---- Fig 06: S_BH vs M ----------------------------------------------------------------------
S = laws.bekenstein_hawking_entropy(M, dimensionless=True)
slope = np.polyfit(np.log(M), np.log(S), 1)[0]
A = laws.horizon_area(M)
ratio = S / (A / (4 * c.PLANCK_LENGTH**2))
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
a.loglog(M / c.M_SUN, S, color=PALETTE["blue"])
a.set_xlabel(r"mass $M / M_\odot$")
a.set_ylabel(r"$S_{BH}/k_B$")
a.set_title(f"(a) $S_{{BH}} \\propto M^2$ (fitted exponent {slope:.6f})")
a.plot([1], [float(laws.bekenstein_hawking_entropy(c.M_SUN, True))], "o", color=PALETTE["vermilion"])
a.annotate(r"$1\,M_\odot$: $1.05\times10^{77}$", xy=(1, 1.05e77), xytext=(1e-14, 1e75), color=PALETTE["vermilion"])
b.semilogx(M / c.M_SUN, ratio - 1.0, color=PALETTE["blue"])
b.set_xlabel(r"mass $M / M_\odot$")
b.set_ylabel(r"$S_{BH}/k_B \,/\, (A/4\ell_P^2) - 1$")
b.set_title("(b) Area law: two independent evaluations agree")
b.set_ylim(-1e-12, 1e-12)
save(fig, "fig06_entropy_vs_mass",
     f"Bekenstein-Hawking entropy S_BH = 4 pi k_B G M^2/(hbar c). Fitted log-log slope {slope:.8f}; the ratio of the closed form "
     f"to A/(4 l_P^2) deviates from 1 by at most {np.max(np.abs(ratio - 1)):.1e} (floating-point round-off).",
     {"M_range_kg": [1e8, 1e33]})

# ---- Fig 07: first-law finite-difference residual ----------------------------------------------
hs = np.logspace(-9, -1, 40)
res_s = np.array([laws.first_law_residual(c.M_SUN, h) for h in hs])
res_rn = np.array([first_law.rn_first_law_residual(2.0, 1.0, h) for h in hs])
res_k = np.array([first_law.kerr_first_law_residual(1.0, 0.6, h) for h in hs])
res_ads = np.array([first_law.ads_first_law_residual(1.0, 1.0, h) for h in hs])
fig, ax = plt.subplots()
for (lab, y), ls, col in zip([("Schwarzschild", res_s), ("Reissner-Nordstrom", res_rn), ("Kerr", res_k), ("Schwarzschild-AdS", res_ads)],
                              LINESTYLES, CYCLE):
    ax.loglog(hs, np.maximum(y, 1e-18), ls=ls, color=col, label=lab)
ax.loglog(hs, 3e-2 * hs**2, color=PALETTE["grey"], lw=0.8, ls=":")
ax.annotate(r"$\propto h^2$ guide (truncation)", xy=(1e-2, 3e-6), xytext=(1.5e-4, 1e-5), color=PALETTE["grey"])
ax.set_xlabel("finite-difference step h (relative to the scale of the variable)")
ax.set_ylabel("relative first-law residual")
ax.set_title(r"First law $dM = T\,dS + \Phi\,dQ + \Omega\,dJ$: numerical check")
ax.legend(loc="lower right")
save(fig, "fig07_first_law_residual",
     "Relative residual of the first law of black-hole mechanics evaluated with central finite differences of the analytic "
     "M(S, Q, J). The residual follows the O(h^2) truncation error at large steps and is limited by round-off at small steps "
     "(the Schwarzschild entropy is exactly quadratic in M, so it is dominated by round-off throughout).",
     {"steps": [1e-9, 1e-1], "RN": [2.0, 1.0], "Kerr": [1.0, 0.6], "AdS": [1.0, 1.0]})

# ---- Fig 08: heat capacity and stability ----------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
a, b, d = axs
Ms = np.logspace(10, 32, 5)
a.loglog(M / c.M_SUN, np.abs(laws.heat_capacity(M)) / c.K_B, color=PALETTE["blue"])
a.set_xlabel(r"$M/M_\odot$")
a.set_ylabel(r"$|C|/k_B$ with $C = d(Mc^2)/dT_H < 0$")
a.set_title("(a) Schwarzschild: $C = -2 S_{BH} < 0$")
Q = 1.0
rp = np.linspace(1.0001, 6, 800) * Q
CS = []
for r_ in rp:
    bh = ReissnerNordstrom(ReissnerNordstrom.mass_from(r_, Q), Q)
    CS.append(bh.heat_capacity_fixed_Q / bh.entropy)
CS = np.array(CS)
CSm = np.where(np.abs(CS) > 80, np.nan, CS)
b.plot(rp / Q, CSm, color=PALETTE["blue"])
b.axvline(np.sqrt(3), color=PALETTE["vermilion"], ls="--")
b.annotate(r"$r_+ = \sqrt{3}\,|Q|$: $C_Q$ diverges", xy=(np.sqrt(3), 20), xytext=(2.0, 40), color=PALETTE["vermilion"])
b.set_ylim(-60, 60)
b.axhline(0, color=PALETTE["grey"], lw=0.8)
b.set_xlabel(r"$r_+/|Q|$")
b.set_ylabel(r"$C_Q / S$")
b.set_title("(b) Reissner-Nordstrom (fixed $Q$)")
L = 1.0
rpa = np.linspace(0.05, 3, 800) * L
Ca = np.array([SchwarzschildAdS(r_, L).heat_capacity / SchwarzschildAdS(r_, L).entropy for r_ in rpa])
Cam = np.where(np.abs(Ca) > 40, np.nan, Ca)
d.plot(rpa / L, Cam, color=PALETTE["blue"])
d.axvline(1 / np.sqrt(3), color=PALETTE["vermilion"], ls="--")
d.annotate(r"$r_+ = L/\sqrt{3}$", xy=(1 / np.sqrt(3), 15), xytext=(0.75, 25), color=PALETTE["vermilion"])
d.axhline(0, color=PALETTE["grey"], lw=0.8)
d.set_ylim(-30, 40)
d.set_xlabel(r"$r_+/L$")
d.set_ylabel(r"$C / S$")
d.set_title("(c) Schwarzschild-AdS")
a.text(0.04, 0.88, "C < 0 for every mass\n(plotted as |C|)", transform=a.transAxes, color=PALETTE["grey"])
b.text(0.42, 0.12, "C>0 (stable) for $r_+<\\sqrt{3}|Q|$,\nC<0 beyond", transform=b.transAxes, color=PALETTE["grey"])
d.text(0.42, 0.12, "C<0 small BH,\nC>0 large BH", transform=d.transAxes, color=PALETTE["grey"])
save(fig, "fig08_heat_capacity_stability",
     "Heat capacities in their stated ensembles. Schwarzschild: C = -2 S_BH < 0 (local thermodynamic instability). RN at fixed charge: "
     "positive for r_+ < sqrt(3)|Q| (near-extremal), diverging and negative beyond. Schwarzschild-AdS at fixed L: negative for "
     "r_+ < L/sqrt(3) (small BH), positive for large BH. Local thermodynamic stability is distinct from global preference "
     "(Fig 09) and from dynamical stability, which is not analysed here.",
     {"Q": Q, "L": L})

# ---- Fig 09: free-energy comparison (Hawking-Page) ----------------------------------------------------
L = 1.0
rp = np.linspace(0.02, 4, 2000) * L
bhs = [SchwarzschildAdS(r_, L) for r_ in rp]
Tt = np.array([b_.temperature for b_ in bhs])
F = np.array([b_.free_energy for b_ in bhs])
small = rp < L / np.sqrt(3)
Tmin, THP = SchwarzschildAdS.minimum_temperature(L), SchwarzschildAdS.hawking_page_temperature(L)
fig, ax = plt.subplots()
ax.plot(Tt[~small], F[~small], color=PALETTE["blue"], label="large AdS black hole (C > 0)")
ax.plot(Tt[small], F[small], color=PALETTE["vermilion"], ls="--", label="small AdS black hole (C < 0)")
ax.axhline(0, color=PALETTE["green"], ls="-.", label="thermal AdS (F = 0)")
Tf = np.linspace(0.06, 1.2, 300)
ax.plot(Tf, 1 / (16 * np.pi * Tf), color=PALETTE["orange"], ls=":", label=r"flat-space Schwarzschild: $F = 1/(16\pi T)$")
ax.axvline(Tmin, color=PALETTE["grey"], lw=0.8)
ax.axvline(THP, color=PALETTE["grey"], lw=0.8)
ax.annotate(r"$T_{min}=\sqrt{3}/(2\pi L)$", xy=(Tmin, 0.5), xytext=(Tmin - 0.1, 0.55), color=PALETTE["grey"], rotation=90, va="top")
ax.annotate(r"$T_{HP}=1/(\pi L)$", xy=(THP, 0.5), xytext=(THP + 0.01, 0.55), color=PALETTE["grey"], rotation=90, va="top")
ax.set_xlim(0.1, 0.9)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel(r"temperature $T$ ($\hbar = c = G = k_B = 1$, $L=1$)")
ax.set_ylabel("free energy F = M - T S")
ax.set_title("Hawking-Page transition in Schwarzschild-AdS$_4$")
ax.legend(loc="upper right")
save(fig, "fig09_free_energy_hawking_page",
     "Free energy of Schwarzschild-AdS black holes (large branch solid, small branch dashed) compared with thermal AdS (F = 0). "
     "Black holes exist only above T_min = sqrt(3)/(2 pi L); the large branch crosses F = 0 at T_HP = 1/(pi L) (r_+ = L), "
     "reproducing the Hawking-Page transition (Hawking and Page 1983). Asymptotically flat Schwarzschild has F = M/2 > 0 at all "
     "temperatures and no preferred equilibrium (negative heat capacity).",
     {"L": L})

# ---- Fig 10: extremal limits -------------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
q = np.linspace(0, 1, 400)
a.plot(q, [ReissnerNordstrom(1.0, x).temperature * 8 * np.pi for x in q], color=PALETTE["blue"], label=r"$T/T_{Schw}$")
a.plot(q, [ReissnerNordstrom(1.0, x).entropy / (4 * np.pi) for x in q], color=PALETTE["vermilion"], ls="--", label=r"$S / S_{Schw}$")
a.set_xlabel(r"$|Q|/M$")
a.set_title("(a) Reissner-Nordstrom: $T\\to0$ with finite $S = \\pi M^2$")
a.legend()
ac = np.linspace(0, 1, 400)
b.plot(ac, [Kerr(1.0, x).temperature * 8 * np.pi for x in ac], color=PALETTE["blue"], label=r"$T/T_{Schw}$")
b.plot(ac, [Kerr(1.0, x).horizon_angular_velocity * 2 for x in ac], color=PALETTE["green"], ls="-.", label=r"$2 M\Omega_H$")
b.plot(ac, [Kerr(1.0, x).entropy / (4 * np.pi) for x in ac], color=PALETTE["vermilion"], ls="--", label=r"$S/S_{Schw}$")
b.set_xlabel(r"$a/M$")
b.set_title("(b) Kerr: extremal $a = M$, $T = 0$, $\\Omega_H = 1/2M$")
b.legend()
save(fig, "fig10_extremal_limits",
     "Temperature, entropy and horizon angular velocity as functions of charge-to-mass ratio (RN) and spin (Kerr), normalised to the "
     "Schwarzschild values at equal M. T vanishes continuously at extremality with finite entropy; the Kerr Smarr relation "
     "M = 2TS + 2 Omega J holds to round-off (tests/test_spacetimes.py). The horizon radius is computed as sqrt((M-Q)(M+Q)) to avoid "
     "cancellation near extremality.",
     {"M": 1.0})

# ---- Fig 11: quasi-static evaporation ---------------------------------------------------------------------------
m0 = 1e12
t, Mt = ml.integrate_mass_loss(m0, cutoff_planck_masses=1e-3 * m0 / c.PLANCK_MASS)
t_ev = ml.evaporation_time_analytic(m0)
Mgrid = np.logspace(8, 14, 200)
Meps_grid = np.logspace(-11, 14, 300)
M_age = float((c.AGE_OF_UNIVERSE * 3 * ml.alpha_coefficient()) ** (1 / 3))
eps = np.array([ml.adiabaticity_parameter(m) for m in Meps_grid])
M_eps = ml.mass_at_adiabaticity(0.1)
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
axs[0].plot(t / t_ev, Mt / m0, color=PALETTE["blue"], label="ODE integration (DOP853)")
axs[0].plot(t / t_ev, (1 - t / t_ev) ** (1 / 3), color=PALETTE["vermilion"], ls="--", label=r"$(1 - t/t_{evap})^{1/3}$")
axs[0].set_xlabel(r"$t / t_{evap}$")
axs[0].set_ylabel(r"$M/M_0$")
axs[0].set_title(r"(a) $dM/dt = -\alpha/M^2$")
axs[0].legend()
tev = np.array([ml.evaporation_time_analytic(m) for m in Mgrid])
axs[1].loglog(Mgrid, tev / (365.25 * 86400), color=PALETTE["blue"])
axs[1].axhline(13.8e9, color=PALETTE["grey"], ls="--")
axs[1].annotate("age of universe", xy=(1e9, 13.8e9), xytext=(1e8, 4e10), color=PALETTE["grey"])
axs[1].set_xlabel("initial mass $M_0$ [kg]")
axs[1].set_ylabel(r"$t_{evap}$ [yr] (photons only, no greybody)")
axs[1].set_title(r"(b) $t_{evap}\propto M_0^3$")
axs[2].loglog(Meps_grid, eps, color=PALETTE["blue"])
axs[2].axhline(0.1, color=PALETTE["vermilion"], ls="--")
axs[2].axvline(c.PLANCK_MASS, color=PALETTE["grey"], ls=":")
axs[2].annotate("Planck mass", xy=(c.PLANCK_MASS, 1e-30), xytext=(c.PLANCK_MASS * 2, 1e-30), color=PALETTE["grey"])
axs[2].annotate(rf"$\epsilon = 0.1$ at $M \approx {M_eps:.1e}$ kg" "\n" rf"$(= {M_eps / c.PLANCK_MASS:.2f}\,M_P)$", xy=(M_eps, 0.1), xytext=(3e-11, 1e-12), color=PALETTE["vermilion"], arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["vermilion"]))
axs[2].set_xlim(1e-11, 1e14)
axs[2].set_xlabel("black-hole mass $M$ [kg]")
axs[2].set_ylabel(r"adiabaticity $\epsilon=\tau_q|\dot M|/M$")
axs[2].set_title(r"(c) Validity of the quasi-static picture")
save(fig, "fig11_evaporation",
     "Quasi-static photon-only evaporation (L = sigma_SB A T_H^4, no greybody factors or other species): numerical integration vs the "
     f"closed form, lifetime scaling and the adiabaticity parameter epsilon = (hbar/k_B T_H)|dM/dt|/M. The photon-only lifetime equals "
     f"the age of the universe at M ~ {M_age:.2e} kg. These lifetimes are approximations. epsilon grows as M^-2 but is still only "
     f"{ml.adiabaticity_parameter(c.PLANCK_MASS):.1e} at the Planck mass (it reaches 0.1 at {M_eps / c.PLANCK_MASS:.2f} M_P), so this heuristic criterion "
     "does not locate the breakdown of the semiclassical picture; Planck-scale effects are expected earlier and are not modelled. "
     "The model is not extended to the endpoint.",
     {"m0_kg": m0, "cross_section": "horizon"})

save_results("02_thermodynamics", {
    "T_sun_K": float(laws.hawking_temperature(c.M_SUN)), "S_sun_over_kB": float(laws.bekenstein_hawking_entropy(c.M_SUN, True)),
    "M_where_T_equals_CMB_kg": M_cmb, "entropy_loglog_slope": float(slope),
    "max_area_law_relative_deviation": float(np.max(np.abs(ratio - 1))),
    "hawking_page": {"T_min": Tmin, "T_HP": THP},
    "photon_only_lifetime_equals_age_of_universe_at_kg": M_age,
    "epsilon_0p1_mass_kg": M_eps,
    "epsilon_at_planck_mass": float(ml.adiabaticity_parameter(c.PLANCK_MASS)),
    "epsilon_at_1e12_kg": float(ml.adiabaticity_parameter(1e12)),
    "evaporation_ode_max_time_error": float(np.max(np.abs(t / t_ev - (1 - (Mt / m0) ** 3)))),
    "t_evap_1e12kg_seconds": t_ev,
})
