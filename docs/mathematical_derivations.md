# Mathematical Derivations

## Schwarzschild geometry

$$ds^2 = -f\,c^2dt^2 + f^{-1}dr^2 + r^2d\Omega^2,\qquad f = 1 - \frac{r_s}{r},\quad r_s = \frac{2GM}{c^2}.$$

The Kretschmann scalar is $K = R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = 12\,r_s^2/r^6$, finite at $r=r_s$ (the horizon is a coordinate singularity) and divergent at $r=0$. The tortoise coordinate $dr_*=dr/f$ gives $r_* = r + r_s\ln|r/r_s - 1|$. Ingoing Eddington-Finkelstein coordinates $v = t + r_*/c$ have $\det g = -r^4\sin^2\theta$ (regular).

## Hawking temperature

Surface gravity from $\kappa = \tfrac12 f'(r_h)c^2$ gives, for Schwarzschild, $\kappa = c^4/(4GM)$ and

$$T_H = \frac{\hbar c^3}{8\pi G M k_B},\qquad S_{BH} = \frac{k_B c^3 A}{4G\hbar} = \frac{4\pi k_B G M^2}{\hbar c}.$$

Then $dM\,c^2 = T_H\,dS$ is the first law, and the heat capacity is

$$C = \frac{dM c^2}{dT_H} = -\frac{8\pi G M^2 k_B}{\hbar c} = -2S_{BH}\quad(\text{in units of }k_B).$$

## Evaporation (photon-only blackbody model)

With $L = \sigma_{SB}A\,T_H^4$ and $A = 16\pi G^2M^2/c^4$:

$$\frac{dM}{dt} = -\frac{\alpha}{M^2},\qquad \alpha = \frac{\hbar c^4}{15360\pi G^2},\qquad t_{evap} = \frac{M_0^3}{3\alpha} = \frac{5120\pi G^2M_0^3}{\hbar c^4}.$$

Since $d(M^3)/dt = -3\alpha$ is constant, the numerical integration is checked against $M(t)^3 = M_0^3 - 3\alpha t$. For the geometric-optics variant the emitting area is the sphere of critical impact parameter $b_c = 3\sqrt3\,GM/c^2$, so $4\pi b_c^2 = 27\pi r_s^2$, which is $27/4$ times the horizon area.

Adiabaticity: $\varepsilon = (\hbar/k_BT_H)\,|\dot M|/M \propto M^{-2}$.

## Reissner-Nordstrom, Kerr, (A)dS (Planck units)

RN: $f = 1 - 2M/r + Q^2/r^2$, $r_\pm = M\pm\sqrt{M^2-Q^2}$, $T = (r_+-r_-)/(4\pi r_+^2)$, $S=\pi r_+^2$, $\Phi = Q/r_+$.
Kerr: $r_\pm = M\pm\sqrt{M^2-a^2}$, $S = \pi(r_+^2+a^2)$, $T=(r_+-r_-)/(4\pi(r_+^2+a^2))$, $\Omega = a/(r_+^2+a^2)$. Smarr: $M = 2TS + 2\Omega J + \Phi Q$.
SAdS: $f = 1 - 2M/r + r^2/L^2$; $T = (1+3r_+^2/L^2)/(4\pi r_+)$ has a minimum $T_{min}=\sqrt3/(2\pi L)$ at $r_+=L/\sqrt3$. The free energy $F = M - TS = (r_+/4)(1 - r_+^2/L^2)$ vanishes at $r_+=L$, i.e. $T_{HP} = 1/(\pi L)$.
SdS has a black-hole and a cosmological horizon; the Nariai limit is $9M^2\Lambda = 1$.

## Entropy with logarithmic correction

$S = A/4 + \alpha\ln A$ (Planck units). The coefficient $\alpha$ is a free parameter in this code; values quoted in the literature depend on the approach and are not endorsed here.

## Page's formula

For a Haar-random pure state on $\mathbb C^m\otimes\mathbb C^n$ ($m\le n$), the average entropy of the $m$-dimensional factor is

$$\langle S_{m,n}\rangle = \sum_{k=n+1}^{mn}\frac1k - \frac{m-1}{2n}\;\approx\;\ln m - \frac{m}{2n}\quad(1\ll m\le n).$$

At $m=n=2^{N/2}$ the deficit from $\ln m$ tends to $1/2$ nat. The code evaluates the digamma form $\psi(mn+1)-\psi(n+1)-(m-1)/(2n)$.

## Harmonic chain

$H=\tfrac12\sum_i(p_i^2 + m^2x_i^2) + \tfrac12\sum_i(x_{i+1}-x_i)^2 = \tfrac12 p^2 + \tfrac12 x^TKx$. The ground state has $\langle xx\rangle = \tfrac12K^{-1/2}$, $\langle pp\rangle=\tfrac12K^{1/2}$. For a block $A$ restrict both matrices; the symplectic eigenvalues $\nu_j\ge\tfrac12$ are the square roots of the eigenvalues of $X_AP_A$ and

$$S_A = \sum_j\Big[(\nu_j+\tfrac12)\ln(\nu_j+\tfrac12) - (\nu_j-\tfrac12)\ln(\nu_j-\tfrac12)\Big].$$

## Island toy model

$S_{gen}(u;\tau)=S_H(\tau)(1+\mu u) + S_{rad}(\tau)e^{-u/\xi}$. Setting $\partial_uS_{gen}=0$ gives $u^*=\xi\ln\!\big[S_{rad}/(\kappa S_H)\big]$ with $\kappa=\mu\xi$ and $S_{gen}(u^*)=S_H[1+\kappa(1+\ln(S_{rad}/\kappa S_H))]$. The model is phenomenological (see [generalized_entropy.md](generalized_entropy.md)).

## BTZ geodesics

On a constant-time slice $ds^2 = L^2dr^2/(r^2-r_+^2)+r^2d\phi^2$. The connected geodesic has length $2L\ln\!\big[(2r_\infty/r_+)\sinh(r_+\Delta\phi/2L)\big]$; the competing candidate is the connected geodesic for the complement plus the horizon circumference $2\pi r_+$. Entropy is $S=\text{Length}/(4G_N)$ and $c=3L/2G_N$.
