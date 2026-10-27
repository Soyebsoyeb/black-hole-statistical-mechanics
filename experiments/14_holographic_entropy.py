"""Experiment 14: Ryu-Takayanagi geodesics in AdS3 and BTZ."""

from _common import *  # noqa: F401,F403
from bhstatmech.holography import (
    ads3_entropy, btz_connected_length, btz_crossover_angle, btz_disconnected_length, btz_entropy,
    btz_geodesic_numeric, numeric_semicircle_length, ads3_vacuum_length,
)

save = figure_saver(__file__)
print("Experiment 14: holographic entropy")
eps = 1e-3
ls_ = np.linspace(0.05, 2.0, 60)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
a.plot(ls_, [ads3_vacuum_length(l, eps) for l in ls_], color=PALETTE["blue"], label=r"closed form $2L\ln(\ell/\epsilon)$")
a.plot(ls_[::4], [numeric_semicircle_length(l, eps) for l in ls_[::4]], "o", color=PALETTE["vermilion"], ms=4, label="numerical quadrature")
a.set_xlabel(r"interval length $\ell$"); a.set_ylabel("geodesic length"); a.set_title("(a) Poincare AdS3 semicircle"); a.legend(fontsize=9)
rp = 1.0; epsr = 1e-4
dphis = np.linspace(0.3, 2 * np.pi - 0.3, 200)
con = np.array([btz_connected_length(d, rp, epsr) for d in dphis]); dis = np.array([btz_disconnected_length(d, rp, epsr) for d in dphis])
cross = btz_crossover_angle(rp)
b.plot(dphis, con, color=PALETTE["blue"], label="connected")
b.plot(dphis, dis, color=PALETTE["vermilion"], ls="--", label="horizon-wrapping")
b.plot(dphis, np.minimum(con, dis), color=PALETTE["black"], lw=0.9, ls=":", label="minimum (RT)")
b.axvline(cross, color=PALETTE["grey"], lw=0.8); b.annotate(rf"crossover {cross:.3f}", (cross, con.min() + 1), (cross - 2.3, dis.max() - 3), fontsize=9)
b.set_xlabel(r"boundary interval $\Delta\phi$"); b.set_ylabel("length"); b.set_title(r"(b) BTZ candidates ($r_+=1$)"); b.legend(fontsize=8)
sel = np.linspace(0.4, 3.0, 12)
num = np.array([btz_geodesic_numeric(d, rp, epsr) for d in sel]); ana = np.array([btz_connected_length(d, rp, epsr) for d in sel])
err = np.abs(num - ana)
for rpl, col in zip([0.5, 1.0, 2.0], CYCLE):
    Ss = [btz_entropy(d, rpl, epsr)[0] for d in dphis]
    c3.plot(dphis, np.array(Ss) - 0, color=col, label=rf"$r_+$ = {rpl:g}")
c3.set_xlabel(r"$\Delta\phi$"); c3.set_ylabel(r"$S = \mathrm{Length}/4G_N$"); c3.set_title("(c) Minimal entropy: switch of branch"); c3.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig36_holographic_entropy_ads3_btz",
     f"Ryu-Takayanagi lengths with L = 1, G_N = 1. (a) The closed-form Poincare AdS3 geodesic length and a numerical quadrature of the semicircle agree. (b) In BTZ the connected geodesic competes with the horizon-wrapping candidate; "
     f"they cross at Delta phi = {cross:.4f} (the cutoff cancels). The minimum encodes the Araki-Lieb behaviour S(A) - S(A^c) = (horizon length)/4G. (c) Entropy S = min(length)/4G_N; the kink marks the switch to the horizon-wrapping candidate and moves toward 2 pi as r_+ increases (about 5.59 at r_+ = 1), so for large r_+ the entropy is the connected-geodesic (extensive thermal) result over almost the whole circle. "
     f"An independent turning-radius integration reproduces the connected closed form to {err.max():.1e}. Central charge c = 3L/2G_N; only the 2D vacuum and BTZ geometries are treated.",
     {"eps": eps, "eps_r": epsr, "r_plus": [0.5, 1.0, 2.0]})
save_results("14_holography", {"crossover_angle_rplus1": float(cross), "max_numeric_vs_closed_form": float(err.max()),
                               "brown_henneaux_c_L1_G1": 1.5})
