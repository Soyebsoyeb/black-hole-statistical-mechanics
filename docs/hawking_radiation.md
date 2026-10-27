# Hawking Radiation

## Thermal spectra

The code takes the Hawking temperature as given and evaluates Planck/Bose and Fermi occupations, number flux and the two-polarisation energy flux (which integrates to $\sigma_{SB}T^4$). The spectrum per unit $\omega$ peaks at $\hbar\omega/k_BT\approx2.82$ (Wien-type value 2.8214; the grid-based maximum found in the experiment is 2.829, within the grid resolution). For $10^{11},3\times10^{11},10^{12}$ kg the peak photon energies are about $299,100,29.9$ MeV (Fig 15). No derivation of the emission from field theory in the black-hole background is implemented.

## Greybody factors

A scalar field mode $\phi=r^{-1}\psi(r)Y_{\ell m}e^{-i\omega t}$ obeys $\psi'' + (\omega^2-V)\psi=0$ in $r_*$ with Regge-Wheeler potential $V=f[\ell(\ell+1)/r^2+(1-s^2)r_s/r^3]$ (Fig 16). The solver integrates the radial equation from the horizon with an ingoing wave in $s=\ln(r-1)$ (units $r_s=1$), matches to Hankel-type asymptotics, and obtains $\Gamma_\ell(\omega)=1/|a_{in}|^2$.

Checks (see [validation.md](validation.md)): flux conservation (Wronskian defect), the low-frequency limit $\sigma\to$ horizon area for $\ell=0$, the high-frequency limit $\sigma\to27\pi r_s^2/4$ (the cross-section oscillates about it, Fig 18), and an independent WKB transmission formula near the barrier top.

## Emission

The energy emission rate with greybody factors is $\sum_\ell(2\ell+1)\Gamma_\ell/(e^{\hbar\omega/kT}-1)$ (Fig 19). The geometric-optics approximation to the total emitting area is $27/4$ times the horizon area.
