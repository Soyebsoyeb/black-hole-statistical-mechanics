"""Experiment 11: Renyi entropies and the replica identity S = -d/dn ln Tr rho^n at n = 1."""

from _common import *  # noqa: F401,F403
from bhstatmech.numerical_methods import make_rng
from bhstatmech.quantum_states import (
    random_density_matrix, random_pure_state, reduced_density_from_ket, renyi_entropy, spectrum,
    thermal_state, von_neumann_entropy,
)
from bhstatmech.replica_methods import (
    extrapolate_from_integers, log_moment, replica_entropy_fd, replica_entropy_richardson,
)

save = figure_saver(__file__)
print("Experiment 11: replica / Renyi")
d = 8
rho_rand = random_density_matrix(d, seed=SEED)
psi = random_pure_state(8 * 32, make_rng(SEED + 1))
rho_red = reduced_density_from_ket(psi, (8, 32), [0])
Hosc = np.diag(np.arange(d, dtype=float)).astype(complex)
rho_th = thermal_state(Hosc, 0.5)
rho_flat = np.eye(d) / d
cases = [("maximally mixed", rho_flat, PALETTE["grey"], ":"), ("Hilbert-Schmidt random", rho_rand, PALETTE["blue"], "-"),
         (r"reduced state of a random $8\times32$ pure state", rho_red, PALETTE["vermilion"], "--"), (r"thermal oscillator, $\beta=0.5$", rho_th, PALETTE["green"], "-.")]

# ---- Fig 30: Renyi entropy vs index --------------------------------------------------------------------------------------------
ns = np.logspace(-1.5, 1.7, 160)
fig, ax = plt.subplots()
for lab, rho, col, ls in cases:
    ax.semilogx(ns, [renyi_entropy(rho, n) for n in ns], color=col, ls=ls, label=lab)
    ax.plot([1], [von_neumann_entropy(rho)], "o", color=col, ms=5)
ax.axvline(1, color=PALETTE["grey"], lw=0.7)
ax.axvline(2, color=PALETTE["grey"], lw=0.7)
ax.annotate("von Neumann\n(n = 1)", xy=(1, 0.1), xytext=(0.35, 0.1), color=PALETTE["grey"], fontsize=9)
ax.annotate("purity\n(n = 2)", xy=(2, 0.1), xytext=(2.2, 0.1), color=PALETTE["grey"], fontsize=9)
ax.set_xlabel("Renyi index n")
ax.set_ylabel(r"$S_n$ [nats]")
ax.set_title(r"Renyi entropies: $S_n = \ln\mathrm{Tr}\,\rho^n/(1-n)$, non-increasing in $n$")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "fig30_renyi_entropy_vs_index",
     "Renyi entropies of four d = 8 states computed from the eigenvalues. S_0 = ln rank, S_1 is the von Neumann entropy (markers), S_2 = -ln Tr rho^2 and S_infinity = -ln lambda_max. "
     "The curves are non-increasing in n and flat only for the maximally mixed state, whose spectrum is flat.",
     {"d": d, "n_range": [10 ** -1.5, 10 ** 1.7]}, seed=SEED)

# ---- Fig 31: replica derivative vs direct entropy ------------------------------------------------------------------------------------
p = spectrum(rho_rand)
S_exact = von_neumann_entropy(rho_rand)
fig, (a, b, c3) = plt.subplots(1, 3, figsize=(14, 4.2))
nn = np.linspace(0.4, 1.8, 100)
a.plot(nn, [log_moment(p, n) for n in nn], color=PALETTE["blue"], label=r"$\ln\,\mathrm{Tr}\,\rho^n$")
a.plot(nn, -S_exact * (nn - 1), color=PALETTE["vermilion"], ls="--", label=rf"tangent at $n=1$: slope $-S = {-S_exact:.4f}$")
a.set_xlabel("replica number n")
a.set_title("(a) Analytic continuation in n")
a.legend(fontsize=9)
hs = np.logspace(-1, -8, 36)
e_fd = np.array([abs(replica_entropy_fd(p, h) - S_exact) for h in hs])
e_rich = np.array([abs(replica_entropy_richardson(p, h) - S_exact) for h in hs])
b.loglog(hs, np.maximum(e_fd, 1e-17), color=PALETTE["blue"], label="central difference, $O(h^2)$")
b.loglog(hs, np.maximum(e_rich, 1e-17), color=PALETTE["vermilion"], ls="--", label="Richardson, $O(h^4)$")
b.set_xlabel("step h in n")
b.set_ylabel(r"$|S_{replica} - S_{vN}|$")
b.set_title("(b) Derivative error versus step")
b.legend(fontsize=9)
nmax = np.arange(2, 13)
e_int = np.array([abs(extrapolate_from_integers(p, int(n)) - S_exact) for n in nmax])
c3.semilogy(nmax, e_int, "o-", color=PALETTE["blue"], label="polynomial through integer replicas")
c3.axhline(e_fd.min(), color=PALETTE["vermilion"], ls="--", label="derivative of the exact moments")
c3.set_xlabel("highest integer replica $n_{max}$")
c3.set_ylabel(r"$|S - S_{vN}|$")
c3.set_title("(c) Continuation from integers only")
c3.legend(fontsize=9)
save(fig, "fig31_replica_vs_direct_entropy",
     f"Replica identity for a random d = 8 density matrix (S_vN = {S_exact:.6f}). (a) ln Tr rho^n and its tangent at n = 1. (b) Finite-difference error: quadratic in h for the central difference "
     "and quartic for Richardson down to a round-off floor of ~1e-10..1e-11. (c) In finite dimension the continuation from integer replicas alone (polynomial interpolation) converges only slowly; "
     "gravitational path integrals provide only integer-n data, which is why the analytic continuation is the hard step there. A finite-dimensional calculation is not a gravitational path integral.",
     {"d": d, "h_range": [1e-8, 1e-1]}, seed=SEED)
save_results("11_replica", {"S_vn": S_exact, "min_fd_error": float(e_fd.min()), "min_richardson_error": float(e_rich.min()),
                            "integer_extrapolation_errors": dict(zip(nmax.tolist(), e_int.tolist()))})
