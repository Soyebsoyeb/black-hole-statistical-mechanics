# Quantum Information Toolkit

## States and ensembles (`bhstatmech.quantum_states`)

Partial trace and reduced states of pure vectors (computed from the Schmidt decomposition without forming the full density matrix), Haar-random pure states and unitaries, Hilbert-Schmidt and Bures random density matrices, thermal states. Entropy routines use eigenvalues with $0\ln0=0$.

For a Hilbert-Schmidt random state of dimension $d$ the rescaled eigenvalue density approaches a Marchenko-Pastur law whose support edges are $(1\pm\sqrt c)^2$ for aspect ratio $c$; the sampled eigenvalue ranges are compared with these edges in Fig 21 and 22.

## Renyi entropies

$S_n=\ln\mathrm{Tr}\rho^n/(1-n)$ is non-increasing in $n$; $S_1$ is the von Neumann entropy, $S_2$ the purity entropy, $S_\infty=-\ln\lambda_{max}$ (Fig 30).

## Scrambling (`bhstatmech.scrambling`)

Haar-random two-qubit brickwork circuits with open boundaries. The out-of-time-order correlator $F(t)=\mathrm{Re}\,\mathrm{Tr}[W(t)^\dagger V^\dagger W(t)V]/d$ with $W=Z_0$, $V=Z_j$ equals 1 until the operator front reaches site $j$. For $n=8$ the first layer with $F<0.5$ is 1, 3, 3, 7, 9 for distances 1, 2, 3, 5, 7: a roughly linear light cone with velocity near one site per layer, slowed by the brickwork geometry (Fig 37). The half-chain entropy of a 12-qubit register starting from a product state saturates at the Page value (see [validation.md](validation.md)).

## Error correction (`bhstatmech.error_correction`)

- Three-qubit repetition code: corrects one bit flip; logical failure $3p^2-2p^3$. It does *not* protect against phase errors, and erasing any single qubit already loses the quantum information (worst-case $I(\mathrm{ref}\!:\!\mathrm{kept})=\ln2$ instead of $2\ln2$).
- Five-qubit perfect code, stabilisers cyclic shifts of $XZZXI$: corrects any single-qubit Pauli error (tested exhaustively), keeps the logical qubit recoverable after erasure of any two physical qubits (worst-case $I(\mathrm{ref}\!:\!\mathrm{kept})=2\ln2$) and loses it after any three (0), matching code distance 3 (Fig 38).

Relation to holography is analogical only: redundancy of encoding mirrors the fact that a bulk operator can be reconstructed from several boundary regions (see [holography.md](holography.md)).
