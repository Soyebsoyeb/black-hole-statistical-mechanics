"""Experiment 16: error-correcting codes: logical failure rates and erasure recoverability."""

from itertools import combinations
from _common import *  # noqa: F401,F403
from bhstatmech.error_correction import (
    encode_five_qubit, encode_repetition, erasure_recoverability, five_qubit_codewords, five_qubit_recover,
    logical_fidelity_five, pauli_string, repetition_decode, repetition_recover,
)
from bhstatmech.numerical_methods import make_rng

save = figure_saver(__file__)
print("Experiment 16: error correction")
rng = make_rng(SEED + 16)
ps = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
ntr = 6000
alpha, beta = np.cos(0.4), np.sin(0.4) * np.exp(0.7j)

def mc_rep(p):
    fails = 0
    for _ in range(ntr):
        psi = encode_repetition(alpha, beta)
        flips = rng.random(3) < p
        for q in np.flatnonzero(flips):
            e = ["I"] * 3; e[q] = "X"; psi = pauli_string("".join(e)) @ psi
        a2, b2 = repetition_decode(repetition_recover(psi))
        fails += abs(np.vdot([alpha, beta], [a2, b2])) ** 2 < 0.999999
    return fails / ntr

def mc_five(p):
    fails = 0
    for _ in range(ntr):
        psi = encode_five_qubit(alpha, beta)
        s = ["I"] * 5
        for q in range(5):
            if rng.random() < p:
                s[q] = "XYZ"[rng.integers(3)]
        psi = pauli_string("".join(s)) @ psi
        try:
            rec = five_qubit_recover(psi)
            fid = logical_fidelity_five((alpha, beta), rec)
        except ValueError:
            fid = 0.0
        fails += fid < 0.999999
    return fails / ntr

f3 = np.array([mc_rep(p) for p in ps]); f5 = np.array([mc_five(p) for p in ps])
# note: multi-qubit errors can be miscorrected "luckily" only up to stabiliser equivalence; the MC counts actual logical fidelity
th3 = 3 * ps**2 - 2 * ps**3
# erasure
z, o = five_qubit_codewords()
cw5 = np.stack([z, o], axis=1)
cw3 = np.zeros((8, 2), dtype=complex); cw3[0, 0] = 1; cw3[7, 1] = 1
def erasure_curve(cw, n):
    out = []
    for k in range(n + 1):
        vals = [erasure_recoverability(cw, n, list(c)) for c in combinations(range(n), k)]
        out.append((np.min(vals), np.max(vals)))
    return np.array(out)
e5, e3 = erasure_curve(cw5, 5), erasure_curve(cw3, 3)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
a, b, c3 = axs
m3 = f3 > 0; m5 = f5 > 0
a.loglog(ps[m3], f3[m3], "o-", color=PALETTE["blue"], label="3-qubit repetition (X errors): Monte Carlo")
a.loglog(ps, th3, color=PALETTE["blue"], ls=":", label=r"$3p^2-2p^3$")
a.loglog(ps[m5], f5[m5], "s-", color=PALETTE["vermilion"], label="5-qubit code (depolarising): Monte Carlo")
a.loglog(ps, ps, color=PALETTE["black"], ls="--", lw=0.8, label="reference line: failure = p")
a.set_xlabel("physical error probability p"); a.set_ylabel("logical failure probability"); a.set_title(f"(a) Logical failure ({ntr} trials per point)"); a.legend(fontsize=7.5)
ks = np.arange(6)
b.plot(ks, e5[:, 0] / np.log(2), "s-", color=PALETTE["vermilion"], label="5-qubit code: worst subset")
b.plot(ks, e5[:, 1] / np.log(2), "s--", color=PALETTE["vermilion"], alpha=0.5, label="5-qubit code: best subset")
b.plot(np.arange(4), e3[:, 0] / np.log(2), "o-", color=PALETTE["blue"], label="3-qubit code: worst subset")
b.set_xlabel("number of erased qubits"); b.set_ylabel(r"$I(\mathrm{ref}\!:\!\mathrm{kept})/\ln 2$ (2 = fully recoverable)"); b.set_title("(b) Erasure recoverability"); b.legend(fontsize=8)
c3.bar(["3-qubit\n(any 1 erased)", "3-qubit\n(any 2 erased)", "5-qubit\n(any 2 erased)", "5-qubit\n(any 3 erased)"],
       [e3[1, 0] / np.log(2), e3[2, 0] / np.log(2), e5[2, 0] / np.log(2), e5[3, 0] / np.log(2)], color=[PALETTE["blue"], PALETTE["blue"], PALETTE["vermilion"], PALETTE["vermilion"]])
c3.axhline(2, color=PALETTE["black"], ls=":"); c3.set_ylabel(r"worst-case $I/\ln 2$"); c3.set_title("(c) Holographic-code analogue: subregion recovery")
fig.tight_layout()
save(fig, "fig38_quantum_error_correction",
     f"(a) Monte Carlo logical failure of the 3-qubit repetition code under independent bit flips (agrees with 3p^2 - 2p^3) and the 5-qubit perfect code under independent depolarising noise (failure ~ O(p^2)); points with zero observed failures are omitted; counts are over {ntr} trials, binomial error ~ sqrt(f(1-f)/{ntr}). "
     "(b) Coherent-information-style diagnostic I(ref:kept) for a maximally entangled reference: the 5-qubit code keeps the logical qubit recoverable (2 ln 2) after erasing any 2 qubits and loses it at 3, matching its distance 3; the repetition code protects only against X errors and is not an erasure-correcting code for quantum information. "
     "(c) Worst-case recoverability. This is a finite-code toy analogue of subregion reconstruction in holography, not a derivation of it.",
     {"p": ps.tolist(), "trials": ntr}, seed=SEED + 16)
save_results("16_qec", {"p": ps.tolist(), "rep_failure": f3.tolist(), "five_failure": f5.tolist(), "erasure_five_worst_over_ln2": (e5[:, 0] / np.log(2)).tolist(), "erasure_three_worst_over_ln2": (e3[:, 0] / np.log(2)).tolist()})
