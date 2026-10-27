"""Experiment 08: entanglement entropy of a Gaussian harmonic chain (lattice scalar field)."""

from _common import *  # noqa: F401,F403
from bhstatmech.lattice_models import block_entropy, ground_state_covariance

save = figure_saver(__file__)
lcfg = CFG["lattice"]
N0, M0 = int(lcfg["chain_sites"]), float(lcfg["mass_small"])
print("Experiment 08: lattice entanglement")

# ---- (a) S vs block size with CFT fit -------------------------------------------------------------------------------
X, P = ground_state_covariance(N0, 1.0, M0, "periodic")
ls = np.arange(2, N0 // 2 + 1, 2)
S = np.array([block_entropy(X, P, range(int(l))) for l in ls])
chord = np.log(N0 / np.pi * np.sin(np.pi * ls / N0))
fit_window = (ls >= 4) & (ls <= N0 // 2)
slope, icpt = np.polyfit(chord[fit_window], S[fit_window], 1)
# ---- (b) lattice refinement at fixed physical size --------------------------------------------------------------------
Ns = np.array([32, 64, 128, 256, 512])
S_ref = []
for N in Ns:
    Xr, Pr = ground_state_covariance(int(N), 1.0 / N, 1e-2, "periodic")  # fixed physical length 1, mass 1e-2
    S_ref.append(block_entropy(Xr, Pr, range(int(N) // 4)))
S_ref = np.array(S_ref)
slope_ref = np.polyfit(np.log(Ns), S_ref, 1)[0]
sub = S_ref - (1 / 3) * np.log(Ns)  # remove the UV-divergent ln(1/a) piece
# ---- (c) boundary conditions ---------------------------------------------------------------------------------------------
Nb = 128
bc_data = {}
for bc in ("periodic", "dirichlet", "neumann"):
    Xb, Pb = ground_state_covariance(Nb, 1.0, 0.05, bc)
    bc_data[bc] = np.array([block_entropy(Xb, Pb, range(Nb // 2 - l // 2, Nb // 2 + l // 2)) for l in range(2, 41, 2)])
    bc_data[bc + "_end"] = np.array([block_entropy(Xb, Pb, range(0, l)) for l in range(2, 41, 2)])
# ---- (d) mass (IR regulator) and total size ------------------------------------------------------------------------------
masses = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
S_m = np.array([block_entropy(*ground_state_covariance(128, 1.0, m, "periodic"), range(16)) for m in masses])
Ntot = np.array([32, 48, 64, 96, 128, 192, 256])
S_N = np.array([block_entropy(*ground_state_covariance(int(n), 1.0, 0.05, "periodic"), range(10)) for n in Ntot])

fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
a, b, d, e = axs.ravel()
a.plot(chord, S, "o", ms=3.5, color=PALETTE["blue"], label=f"lattice, N = {N0}, m a = {M0:g}")
a.plot(chord, slope * chord + icpt, color=PALETTE["vermilion"], ls="--", label=rf"linear fit, slope {slope:.4f} ($c/3$ with $c=1$: 0.3333)")
a.set_xlabel(r"$\ln[(N/\pi)\sin(\pi\ell/N)]$")
a.set_ylabel(r"$S_A$ [nats]")
a.set_title("(a) Entropy vs block size (periodic chain)")
a.legend(fontsize=8.5)
b.plot(np.log(Ns), S_ref, "o-", color=PALETTE["blue"], label=f"S, fitted slope {slope_ref:.4f}")
b.plot(np.log(Ns), (1 / 3) * np.log(Ns) + sub[-1], color=PALETTE["vermilion"], ls="--", label=r"$\frac{1}{3}\ln(1/a)$ + const")
b.set_xlabel(r"$\ln(1/a)=\ln N$ at fixed physical length")
b.set_ylabel(r"$S_A$ [nats]")
b.set_title("(b) Ultraviolet divergence: lattice spacing as regulator")
b.legend(fontsize=8.5)
for bc, col, ls_ in [("periodic", PALETTE["blue"], "-"), ("dirichlet", PALETTE["vermilion"], "--"), ("neumann", PALETTE["green"], "-.")]:
    d.plot(np.arange(2, 41, 2), bc_data[bc], color=col, ls=ls_, label=f"{bc}: interior block")
    d.plot(np.arange(2, 41, 2), bc_data[bc + "_end"], color=col, ls=ls_, alpha=0.45, lw=1.1, label=f"{bc}: block at an end" if bc != "periodic" else None)
d.set_xlabel(r"block size $\ell$ (sites)")
d.set_ylabel(r"$S_A$ [nats]")
d.set_title(f"(c) Boundary conditions (N = {Nb}, ma = 0.05)")
d.legend(fontsize=8, ncol=1)
e.semilogx(masses, S_m, "o-", color=PALETTE["blue"], label=r"$\ell=16$, $N=128$ (vs $ma$)")
e.set_xlabel(r"mass $ma$ (infrared regulator)")
e.set_ylabel(r"$S_A$ [nats]", color=PALETTE["blue"])
e2 = e.twiny()
e2.plot(Ntot, S_N, "s--", color=PALETTE["vermilion"], label=r"$\ell=10$, $ma=0.05$ (vs $N$)")
e2.set_xlabel("total sites $N$", color=PALETTE["vermilion"])
e2.grid(False)
e.set_title("(d) Mass and finite-size dependence")
h1, l1 = e.get_legend_handles_labels(); h2, l2 = e2.get_legend_handles_labels()
e.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower left")
fig.tight_layout()
save(fig, "fig26_lattice_entanglement",
     "Ground-state entanglement entropy of a block of the massless-limit harmonic chain (Gaussian state, symplectic spectrum of the block covariance). "
     f"(a) The slope against ln[(N/pi) sin(pi l/N)] is {slope:.4f}, about {abs(slope * 3 - 1) * 100:.0f}% below the CFT value c/3 = 1/3 (c = 1). The deficit is an infrared effect: the correlation length 1/(ma) = {1/M0:.0f} sites is only {1/M0/N0:.1f} times the chain length, and diagnostics outside this script show the fitted slope falls further for larger ma (about 0.14 at ma = 0.01, N = 256) and approaches 1/3 as m N -> 0. "
     f"(b) At fixed physical size the entropy grows as {slope_ref:.4f} ln(1/a): the lattice spacing regulates the ultraviolet divergence. "
     "(c) Dirichlet/Neumann ends change the entropy of an end block (one entangling point instead of two) but not the interior block far from the boundary. "
     "(d) The IR mass lowers the entropy; finite-size corrections shrink as N grows. This is a lattice-field result, not a black-hole entropy calculation.",
     {"N": N0, "m": M0, "refine_N": Ns.tolist(), "bc_N": Nb})
save_results("08_lattice", {"cft_slope_fit": float(slope), "refinement_slope": float(slope_ref),
                            "entropy_minus_third_log_N": sub.tolist(), "N_refine": Ns.tolist(),
                            "two_point_exact_check_note": "tests/test_lattice_models.py::test_two_oscillator_matches_analytic_formula"})
