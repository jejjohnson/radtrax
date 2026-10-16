---
title: "Plume vetting — a spectral test for matched-filter candidates"
---

# Plume vetting

:::{important} Status: target
This page describes `radtrax.retrieval.vetting`, which Wave 5 ([#146](https://github.com/jejjohnson/radtrax/issues/146)) adds. Every function named here has an issue with its signature and tests, and the [plume vetting notebook](../notebooks/plume_vetting.ipynb) runs all of them on one scene, defined in the notebook until each issue lands. The numbers below are that notebook's.
:::

A matched filter scores each pixel by one projection of its spectrum onto methane's target, so anything that removes light under methane's 2.3 μm band scores: carbonate and clay soils, salt pans, dark water, shadows, striping. A segmentation network on the matched-filter map learns plume shapes, which a plume-shaped confounder also has. **Plume vetting** (Xiang et al. [^xiang2024]; Růžička et al. [^ruzicka2026]) goes back to the radiances behind a candidate. It pairs the candidate's strongest pixels with pixels outside it that have the same surface, divides them, and asks whether the ratio has the shape of methane's transmittance. EMIT's daily methane digest alerts without review when the misfit score $D_{\text{norm}}$ is below 0.3.

Vetting is a second test on the same data, not an independent one: its pixels are chosen by the matched filter and come from the same cube. It is more selective because it tests the shape of the absorption across a window of bands, where the matched filter tests one projection.

[^xiang2024]: Xiang, C., et al. (2024). A novel spectroscopic method for vetting methane point sources. AGU Fall Meeting, abstract A21K-1903.

[^ruzicka2026]: Růžička, V., Thompson, D. R., Fahlen, J. E., et al. (2026). Fully automatic trace gas plume detection. arXiv:2605.03372.

[^pv]: EMIT Science Data System (2025). plume-vetting, revision `47fc6ad`. https://github.com/emit-sds/plume-vetting (Apache-2.0).

## 1. The ratio of a pixel pair

Write $L_{\text{in}}(\lambda)$ for the radiance of a pixel under the plume and $L_{\text{out}}(\lambda)$ for a pixel outside it, $\rho$ and $\rho'$ for their surface reflectances, $F_0$ for the solar irradiance, $\theta_s$ for the solar zenith angle, $T_{\text{atm}}$ for the one-way transmittance of everything but the plume, $m$ for the two-way air mass and $\Delta X$ for the plume's column enhancement in mol/m². For a Lambertian surface,

$$
L_{\text{in}}(\lambda) = \frac{F_0 \cos\theta_s}{\pi} \rho(\lambda) T_{\text{atm}}(\lambda)^{m} E(\lambda; \Delta X) + \epsilon,
\qquad
L_{\text{out}}(\lambda) = \frac{F_0 \cos\theta_s}{\pi} \rho'(\lambda) T_{\text{atm}}(\lambda)^{m} + \epsilon',
$$ (eq-pv-forward)

with $E(\lambda; \Delta X)$ the band transmittance of the enhancement along the two-way path and $\epsilon, \epsilon'$ sensor noise. Averaged over $N$ pairs, the ratio keeps only the plume and what the two surfaces do not share:

$$
R(\lambda) = \frac{\frac{1}{N} \sum_n L_{\text{in}, n}(\lambda)}{\frac{1}{N} \sum_n L_{\text{out}, n}(\lambda)} \approx \eta(\lambda) E(\lambda; \Delta X),
\qquad \eta = \bar\rho / \bar\rho' .
$$ (eq-pv-ratio)

Pairing exists to make $\eta$ smooth: an in-plume pixel matched to an out-of-plume pixel of the same material makes $\eta$ close to one and slowly varying in $\lambda$.

## 2. The transmittance from the forward model

The reference implementation [^pv] reads $E$ from MODTRAN tables over solar and view zenith, elevation, water vapour and the plume's strength. radtrax computes it for the scene itself: with $y_b$ the band radiance of [`ForwardModel`](xref:api#radtrax.ForwardModel) for unit albedo,

$$
E_b(\Delta X) = \frac{y_b(x_{\text{atm}} + \Delta X)}{y_b(x_{\text{atm}})},
$$ (eq-pv-table)

on a grid of $\Delta X$, with the enhancement in the bottom layer by default. The albedo cancels because Beer–Lambert radiance is linear in it. The curve of growth is built in: lines saturate, so a strong plume absorbs less per molecule than the weak-absorption slope $\varepsilon_b = -\partial \ln y_b / \partial \Delta X$ at 0 predicts, the matched filter's target from [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target). Fitting with a constant coefficient, $E_b \approx \exp(-\varepsilon_b \Delta X)$, biases the column **low**: by 1.4, 2.8 and 7.7 % at 0.05, 0.1 and 0.3 mol/m² on the synthetic spectrum.

## 3. Seeds, pairs and the ratio

Inside a candidate mask $M$, the seeds $I$ are the $k = 40$ pixels of largest matched-filter score below the mask's 99th percentile, grown by one pixel inside $M$. The background pool $O$ is every pixel within a box of the seeds, outside every candidate, with a small matched-filter score. Pairing is one to one, minimising the total Euclidean distance between radiances on bands methane does not absorb, and keeps the best-matched half:

$$
\pi^\star = \arg\min_{\pi: I \to O \text{ injective}} \sum_{i \in I} \left\lVert s(i) - s(\pi(i)) \right\rVert_2 .
$$ (eq-pv-pairing)

These are the reference package's settings, `pv/config/config.yaml`.

## 4. The fit and the score

The continuum is a Legendre series on the window, $P(\lambda; c) = \sum_{k=0}^{K-1} c_k P_k(x)$ with $x \in [-1, 1]$, the basis of [`PolynomialAlbedo`](xref:api#radtrax.surface.PolynomialAlbedo):

$$
(\widehat{\Delta X}, \hat c) = \arg\min_{\Delta X, c} \sum_{\lambda} \left( R(\lambda) - P(\lambda; c) E(\lambda; \Delta X) \right)^2 .
$$ (eq-pv-fit)

For fixed $\Delta X$ the model is linear in $c$, so $\hat c(\Delta X)$ is a least-squares solve; profiling over the grid of $\Delta X$ gives $\widehat{\Delta X}$ with no starting point. The score divides out the continuum, $q = R / P(\hat c)$, and compares $q$ with the fitted transmittance relative to $q$'s own spread (Xiang et al.'s eq. 11):

$$
D_{\text{norm}} = \frac{\operatorname{mean}_\lambda \left| q(\lambda) - E(\lambda; \widehat{\Delta X}) \right|}{\operatorname{mean}_\lambda \left| q(\lambda) - \bar q \right|} .
$$ (eq-pv-dnorm)

A perfect fit gives 0. With no methane, $\widehat{\Delta X} = 0$, $E = 1$ and $\bar q \approx 1$, so numerator and denominator agree and $D_{\text{norm}} \approx 1$.

$\widehat{\Delta X}$ is the column of the *mean* transmittance over the pairs, not the mean column, since $E$ is convex in $\Delta X$; it is a score, not a retrieval.

## 5. A null from the same scene

The honest null is this scene's clutter with no plume. Vet $S$ copies $M_s$ of the mask, each rotated and translated to where there is no candidate, and report

$$
\hat p = \frac{1}{S} \sum_{s=1}^{S} \mathbb{1}\left[ D_{\text{norm}}(M_s) \le D_{\text{norm}}(M) \right] .
$$ (eq-pv-null)

$D_{\text{norm}}$'s threshold depends on the band set and the continuum's degree, so $\hat p$ travels between sensors where 0.3 does not. Which to decide on is [#160](https://github.com/jejjohnson/radtrax/issues/160).

## 6. One scene

The [notebook](../notebooks/plume_vetting.ipynb) builds a 64 × 64 EMIT-like scene: 30 bands outside methane's band, 9 inside it (2296–2356 nm), three surface classes, noise at SNR 300. Candidate A is a methane plume; candidate B is a plume-shaped patch of a surface whose reflectance dips by 6 % at 2335 nm, as a carbonate's does.

| | Matched filter, mean | $\widehat{\Delta X}$ | $D_{\text{norm}}$ | $\hat p$ |
|---|---|---|---|---|
| A, plume | 593 ppm·m | 1,737 ppm·m | 0.14 | 0.00 |
| B, carbonate | 1,162 ppm·m | 0 | 1.00 | 0.43 |

The matched filter scores B at twice A; vetting reverses the order. A's column is 93 % of the truth over its kept pixels: about 3 points go to averaging transmittances, the rest to paired surfaces that do not match exactly.

## 7. What radtrax computes

| Step | Equation | Function | Issue |
|---|---|---|---|
| Transmittance on a grid of $\Delta X$ | {eq}`eq-pv-table` | `transmittance_table` | [#154](https://github.com/jejjohnson/radtrax/issues/154) |
| A gas's fitting and pairing bands | | `VettingWindow`, `VettingWindow.ch4_emit` | [#155](https://github.com/jejjohnson/radtrax/issues/155) |
| Seeds and background pool | | `select_seeds`, `background_pool` | [#148](https://github.com/jejjohnson/radtrax/issues/148) |
| One-to-one pairs and the ratio | {eq}`eq-pv-pairing`, {eq}`eq-pv-ratio` | `pair_pixels`, `ratio_spectrum` | [#149](https://github.com/jejjohnson/radtrax/issues/149) |
| Profile fit and score | {eq}`eq-pv-fit`, {eq}`eq-pv-dnorm` | `fit_transmittance`, `TransmittanceFit`, `d_norm` | [#150](https://github.com/jejjohnson/radtrax/issues/150) |
| One candidate | | `vet_candidate`, `VettingResult` | [#151](https://github.com/jejjohnson/radtrax/issues/151) |
| The scene null | {eq}`eq-pv-null` | `null_masks`, `null_pvalue` | [#152](https://github.com/jejjohnson/radtrax/issues/152) |
| Every candidate of a GeoTensor scene | | `radtrax.geo.PlumeVetting` (`[geo]`) | [#157](https://github.com/jejjohnson/radtrax/issues/157) |

All of them live in `radtrax.retrieval.vetting`, beside the matched filter, except the operator. Vetting knows nothing about plume physics, so it does not belong in plumax; plumax takes vetted masks for the integrated mass enhancement.

## 8. Where radtrax departs from the reference code

- **The continuum.** The reference package fits degree 10 in raw monomials of wavelength in nanometres; $\lambda^{10}$ near 2300 nm makes the least squares badly conditioned, and eleven continuum terms on 46 bands leave little of methane's shape to score. radtrax uses a Legendre series, degree 2 by default.
- **The fit.** The reference uses Levenberg–Marquardt from a fixed start; radtrax profiles over $\Delta X$ with a closed-form solve for the continuum, which cannot wander at saturation.
- **The transmittance.** The reference package reads a fixed target file, marked in its code as a test placeholder; radtrax computes {eq}`eq-pv-table` for each scene.
- **The null.** The reference package draws 100 rotated and translated copies that avoid only their own plume, and does not yet decide on them; radtrax's copies avoid every candidate, and $\hat p$ is returned with every score.
