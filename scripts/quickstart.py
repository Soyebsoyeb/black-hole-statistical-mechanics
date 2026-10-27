"""Quick-start example (also executed by tests/test_quickstart.py)."""

import numpy as np

from bhstatmech import constants as c
from bhstatmech.page_curve import qubit_page_exact
from bhstatmech.quantum_states import bell_state, density_from_ket, partial_trace, von_neumann_entropy
from bhstatmech.spacetimes import Schwarzschild
from bhstatmech.thermodynamics import first_law_residual

# 1. A solar-mass Schwarzschild black hole
bh = Schwarzschild(c.M_SUN)
print(f"r_s            = {bh.schwarzschild_radius:.2f} m")
print(f"T_H            = {bh.hawking_temperature:.3e} K")
print(f"S_BH / k_B     = {bh.bekenstein_hawking_entropy / c.K_B:.3e}")
print(f"first-law rel. residual = {first_law_residual(c.M_SUN):.1e}")

# 2. A Bell pair: pure globally, maximally mixed locally
rho = density_from_ket(bell_state("phi+"))
print(f"S(AB)          = {von_neumann_entropy(rho):.3f} nats")
print(f"S(A)           = {von_neumann_entropy(partial_trace(rho, (2, 2), [0])):.3f} nats (ln 2 = {np.log(2):.3f})")

# 3. Exact Haar-average Page curve for 10 qubits
page = qubit_page_exact(10)
print("Page curve S(R_k) [nats], k = 0..10:", np.round(page, 3).tolist())
print(f"peak at k = {int(np.argmax(page))}")
