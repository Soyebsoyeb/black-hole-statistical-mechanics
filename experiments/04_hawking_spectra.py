"""Experiment 04: thermal (Bose-Einstein / Fermi-Dirac) occupation and Hawking spectra for several masses."""

from _common import *  # noqa: F401,F403
from bhstatmech.hawking_radiation import bose_occupation, dimensionless_frequency, fermi_occupation, planck_energy_flux_per_omega
from bhstatmech.thermodynamics import laws

save = figure_saver(__file__)
print("Experiment 04: Hawking spectra")

x = np.logspace(-2, 1.4, 400)
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
a.loglog(x, bose_occupation(x), color=PALETTE["blue"], label=r"bosons: $1/(e^x - 1)$")
a.loglog(x, fermi_occupation(x), color=PALETTE["vermilion"], ls="--", label=r"fermions: $1/(e^x + 1)$")
a.loglog(x, 1 / x, color=PALETTE["grey"], ls=":", lw=1, label=r"classical limit $1/x$")
a.set_ylim(1e-12, 1e3)
a.set_xlabel(r"$x = \hbar\omega / (k_B T_H)$")
a.set_ylabel(r"mean occupation $\langle n_\omega \rangle$")
a.set_title("(a) Thermal occupation numbers")
a.legend(loc="lower left")

masses = [1e11, 3e11, 1e12]
peaks = {}
for m, col, ls in zip(masses, [PALETTE["blue"], PALETTE["vermilion"], PALETTE["green"]], ["-", "--", "-."]):
    T = float(laws.hawking_temperature(m))
    kT_over_hbar = c.K_B * T / c.HBAR
    omega = kT_over_hbar * np.logspace(-2, 1.3, 600)
    # power per unit angular frequency: horizon area times the two-polarisation blackbody flux
    P_omega = float(laws.horizon_area(m)) * planck_energy_flux_per_omega(omega, T)
    hw_MeV = c.HBAR * omega / (1e6 * c.EV)
    b.loglog(hw_MeV, P_omega, color=col, ls=ls, label=rf"$M={m:.0e}$ kg, $k_BT_H={c.K_B*T/(1e6*c.EV):.0f}$ MeV")
    peaks[f"{m:.0e}"] = {"T_K": T, "peak_hbar_omega_MeV": float(hw_MeV[np.argmax(P_omega)]), "x_peak": float(c.HBAR * omega[np.argmax(P_omega)] / (c.K_B * T))}
b.set_xlabel(r"photon energy $\hbar\omega$ [MeV]")
b.set_ylabel(r"$dP/d\omega$ [W s] (blackbody, horizon area)")
b.set_ylim(1e8, 1e28)
b.set_title("(b) Spectra for different masses (no greybody factor)")
b.legend(loc="lower right", fontsize=8)
save(fig, "fig15_hawking_spectra",
     "(a) Bose-Einstein and Fermi-Dirac occupation as functions of x = hbar omega/(k_B T_H). (b) Photon-only blackbody spectra with the horizon "
     "area for three masses; the peak sits near x = 2.82 (Wien, in terms of dP/domega) and moves to lower energy as M grows. A thermal curve illustrates the "
     "spectrum; it is not a derivation of Hawking radiation (see docs/hawking_radiation.md), and greybody factors (Figs 18-20) modify it.",
     {"masses_kg": masses})
save_results("04_spectra", {"peaks": peaks, "wien_x_peak_expected": 2.8214})
