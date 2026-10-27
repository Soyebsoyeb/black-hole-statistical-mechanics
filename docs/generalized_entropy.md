# Generalized Entropy and the Island Toy Model

## Prescription

In semiclassical gravity the entropy of a radiation region $R$ is proposed to be

$$S(R)=\min_{I}\;\mathrm{ext}_{\partial I}\Big[\frac{\mathrm{Area}(\partial I)}{4G_N}+S_{\rm bulk}(R\cup I)\Big],$$

where $I$ ranges over "islands" in the gravitating region. The minimum includes the empty island, which gives the Hawking-like growing entropy; a non-empty island contributes an area term that tracks the shrinking black hole.

## Toy model used (`bhstatmech.generalized_entropy`)

$S_H(\tau)=S_0(1-\tau)^{2/3}$, $S_{rad}(\tau)=r\,(S_0-S_H)$, $S_{gen}(u)=S_H(1+\mu u)+S_{rad}e^{-u/\xi}$.

The functional forms are chosen to have the qualitative structure above, not derived from a gravitational path integral. The parameters ($S_0=1000$, $r=4/3$, $\mu=1$, $\xi=0.05$) are illustrative.

## Results

- The island extremum exists when $S_{rad}>\kappa S_H$ and is a minimum in $u$ (Fig 34a).
- The minimum over the two branches has a kink at $\tau_P=0.6203$ (Page time of the toy), after which the radiation entropy follows $S_H$ (Fig 34b,c).
- The bounded numerical minimiser and the closed form agree: $|\Delta u^*|\le4.5\times10^{-9}$, $|\Delta S_{gen}|\le1.1\times10^{-13}$ (Fig 35a).
- $\tau_P$ varies with the entropy ratio $r$ and weakly with $\xi$, approaching $1-(r/(1+r))^{3/2}$ as $\kappa\to0$ (Fig 35b).

## Caveat

Because the model is built to exhibit a transition, the existence of a Page time here is an *input*, not evidence for the island proposal.
