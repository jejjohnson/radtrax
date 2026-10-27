---
title: "The absorption-shape test — does a ratio spectrum have a gas's shape?"
---

# The absorption-shape test

:::{important} Status: in progress
This page describes `radtrax.retrieval.absorption`, which Wave 5 ([#146](https://github.com/jejjohnson/radtrax/issues/146)) adds: [`AbsorptionWindow`](xref:api#radtrax.retrieval.absorption.AbsorptionWindow) ([#155](https://github.com/jejjohnson/radtrax/issues/155)), [`transmittance_table`](xref:api#radtrax.retrieval.absorption.transmittance_table) ([#154](https://github.com/jejjohnson/radtrax/issues/154)), [`pair_pixels`](xref:api#radtrax.retrieval.absorption.pair_pixels) and [`ratio_spectrum`](xref:api#radtrax.retrieval.absorption.ratio_spectrum) ([#149](https://github.com/jejjohnson/radtrax/issues/149)), [`fit_transmittance`](xref:api#radtrax.retrieval.absorption.fit_transmittance) and [`d_norm`](xref:api#radtrax.retrieval.absorption.d_norm) ([#150](https://github.com/jejjohnson/radtrax/issues/150)). Plume vetting, which runs this test on matched-filter candidates, is `plumax.vetting` ({ref}`sec-abs-plumax`).
:::

A matched filter scores each pixel by one projection of its spectrum onto methane's target, so anything that removes light under methane's 2.3 μm band scores: carbonate and clay soils, salt pans, dark water, shadows, striping. A segmentation network on the matched-filter map learns plume shapes, which a plume-shaped confounder also has. The **absorption-shape test** (Xiang et al. [^xiang2024]; Růžička et al. [^ruzicka2026]) goes back to the radiances. Given two sets of pixels, the in-pixels whose absorption is tested and the out-pixels they are compared with, it pairs each in-pixel with an out-pixel of the same surface, divides them, and asks whether the ratio has the shape of the gas's transmittance. It returns the fitted column $\widehat{\Delta X}$ and the misfit $D_{\text{norm}}$: a similarity between a ratio spectrum and a transmittance, another retrieval product beside the matched filter's column. EMIT's daily methane digest alerts without review when $D_{\text{norm}}$ is below 0.3.

The test knows nothing of plumes. Which pixels form the two sets, and what to decide from the scores, is the caller's: `plumax.vetting` takes a candidate's strongest pixels as the in-pixels and a background pool around it as the out-pixels.

The test is more selective than the matched filter because it tests the shape of the absorption across a window of bands, where the matched filter tests one projection. That projection is the [matched filter](matched_filter.md)'s $\hat\alpha(x) = (x - \mu)^\top \Sigma^{-1} t \,/\, t^\top \Sigma^{-1} t$. A surface whose spectrum departs from the background mean by $d$ scores $d^\top \Sigma^{-1} t \,/\, t^\top \Sigma^{-1} t$, whatever the shape of $d$, so a carbonate's dip that overlaps methane's band scores positive. Many spectra with different shapes give the same score, because the filter keeps one number from the $B$ bands it sees. The absorption-shape test keeps the shape: it builds the in-pixels' absorption spectrum band by band and asks whether all of it agrees with methane.

[^xiang2024]: Xiang, C., et al. (2024). A novel spectroscopic method for vetting methane point sources. AGU Fall Meeting, abstract A21K-1903.

[^ruzicka2026]: Růžička, V., Thompson, D. R., Fahlen, J. E., et al. (2026). Fully automatic trace gas plume detection. arXiv:2605.03372.

[^pv]: EMIT Science Data System (2025). plume-vetting, revision `47fc6ad`. https://github.com/emit-sds/plume-vetting (Apache-2.0).

## 1. The ratio of a pixel pair

Write $L_{\text{in}}(\lambda)$ for the radiance of an in-pixel, under the gas's enhancement, and $L_{\text{out}}(\lambda)$ for an out-pixel, $\rho$ and $\rho'$ for their surface reflectances, $F_0$ for the solar irradiance, $\theta_s$ for the solar zenith angle, $T_{\text{atm}}$ for the vertical one-way transmittance of everything but the enhancement, $m$ for the two-way air mass and $\Delta X$ for the column enhancement in mol/m². Columns are also quoted in ppm·m, as EMIT quotes them: 1 ppm·m is $10^{-6}\, n_{\text{air}} \times 1$ m, with $n_{\text{air}} = p / (k_B T N_A)$ the molar density of air where the enhancement is, which is $4.03 \times 10^{-5}$ mol/m² in the bottom layer of the US standard atmosphere (≈954.8 hPa and 285.0 K, ~500 m), where the windows' presets take it. Other conventions give other constants: $4.23 \times 10^{-5}$ at the surface (1013.25 hPa, 288.15 K) and $4.46 \times 10^{-5}$ at STP (273.15 K). For a Lambertian surface,

$$
L_{\text{in}}(\lambda) = \frac{F_0 \cos\theta_s}{\pi} \rho(\lambda) T_{\text{atm}}(\lambda)^{m} E(\lambda; \Delta X) + \epsilon,
\qquad
L_{\text{out}}(\lambda) = \frac{F_0 \cos\theta_s}{\pi} \rho'(\lambda) T_{\text{atm}}(\lambda)^{m} + \epsilon',
$$ (eq-abs-forward)

with $E(\lambda; \Delta X)$ the band transmittance of the enhancement along the two-way path and $\epsilon, \epsilon'$ sensor noise. Averaged over $N$ pairs, the ratio keeps only the enhancement and what the two surfaces do not share:

$$
R(\lambda) = \frac{\frac{1}{N} \sum_n L_{\text{in}, n}(\lambda)}{\frac{1}{N} \sum_n L_{\text{out}, n}(\lambda)} \approx \eta(\lambda) E(\lambda; \Delta X),
\qquad \eta = \bar\rho / \bar\rho' .
$$ (eq-abs-ratio)

Pairing exists to make $\eta$ smooth: an in-pixel matched to an out-pixel of the same material makes $\eta$ close to one and slowly varying in $\lambda$.

Three assumptions sit inside the $\approx$: the two pixels of a pair share their sun, path and atmosphere; the $E$ in $R$ is an average over the pairs rather than the transmittance of one column; and the noise is small once the pairs are averaged. The derivation below makes each one explicit.

:::{dropdown} Derivation: what survives the ratio
Write $C_n = F_0 \cos\theta_s\, T_{\text{atm}}^{m} / \pi$ for the factor that pair $n$'s two pixels share. They share it when the out-pixels lie near the in-pixels, so both pixels see the same sun, elevation and water vapour; different pairs need not share it. Leaving out the noise,

$$
R = \frac{\sum_n C_n \rho_n E_n}{\sum_n C_n \rho'_n}
  = \underbrace{\frac{\sum_n C_n \rho_n}{\sum_n C_n \rho'_n}}_{\eta}
    \cdot
    \underbrace{\frac{\sum_n w_n E_n}{\sum_n w_n}}_{\bar E},
\qquad w_n = C_n \rho_n ,
$$ (eq-abs-ratio-exact)

with $E_n = E(\lambda; \Delta X_n)$. This is an identity, not an approximation, and it says three things.

**$\eta$ is a ratio of brightness-weighted mean reflectances.** Pairing on bands that methane does not absorb makes $\rho_n \approx \rho'_n$ pair by pair outside methane's band. Pairing cannot see a mismatch that exists only inside methane's band, such as a carbonate's dip at 2.3 μm, so that mismatch stays in $\eta$, and it is the fit and the score that must reject it.

**$\bar E$ is a mean of transmittances, not the transmittance of the mean column.** $E$ is decreasing and convex in $\Delta X$, so Jensen's inequality gives $\bar E \ge E(\overline{\Delta X})$ with $\overline{\Delta X} = \sum_n w_n \Delta X_n / \sum_n w_n$. The column that reproduces $\bar E$ is therefore at most $\overline{\Delta X}$: an enhancement whose column varies across the pairs is fitted with a smaller column than its weighted mean.

**The ratio is taken after averaging.** For independent noisy quantities $a$ and $b$ with means $\mu_a, \mu_b$ and relative standard deviation $s_b$ of $b$, a second-order expansion of $1/b$ gives $\mathbb{E}[a/b] \approx (\mu_a / \mu_b)(1 + s_b^2)$. Averaging $N$ pairs first divides $s_b^2$ by $N$; dividing each pair and then averaging leaves the bias at the size of a single pixel's $s_b^2$. The bias is largest for dark pixels, whose $L_{\text{out}}$ has the lowest signal-to-noise ratio.
:::

## 2. The transmittance from the forward model

The reference implementation [^pv] reads $E$ from MODTRAN tables over solar and view zenith, elevation, water vapour and the enhancement's strength. radtrax computes it for the scene itself: with $y_b$ the band radiance of [`ForwardModel`](xref:api#radtrax.ForwardModel) for unit albedo,

$$
E_b(\Delta X) = \frac{y_b(x_{\text{atm}} + \Delta X)}{y_b(x_{\text{atm}})},
$$ (eq-abs-table)

on a grid of $\Delta X$, with the enhancement in the bottom layer by default ([`transmittance_table`](xref:api#radtrax.retrieval.absorption.transmittance_table)). The albedo cancels because Beer–Lambert radiance is linear in it. That holds exactly for radtrax's `BeerLambert` solver, which has no path radiance or multiple scattering, and for an albedo that is constant across each band's response. A reflectance that varies inside one band reweights the wavenumbers the band averages over, which is a small effect when the reflectance is smooth over one band's width. The curve of growth is built in: lines saturate, so a strong enhancement absorbs less per molecule than the weak-absorption slope $\varepsilon_b = -\partial \ln y_b / \partial \Delta X$ at 0 predicts, the matched filter's target from [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target). Fitting with a constant coefficient, $E_b \approx \exp(-\varepsilon_b \Delta X)$, biases the column **low**: by 1.4, 2.8 and 7.7 % at 0.05, 0.1 and 0.3 mol/m² on the synthetic spectrum.

:::{dropdown} Derivation: why a constant coefficient biases the column low
Band $b$ averages the wavenumbers $\nu$ of its response with a weight $w_b(\nu) \ge 0$, normalised so that $\int w_b\, d\nu = 1$, that combines the response, the solar irradiance and the ambient transmittance. With $\sigma(\nu)$ methane's cross-section in m²/mol at the enhancement's layer,

$$
E_b(\Delta X) = \int w_b(\nu)\, e^{-m \sigma(\nu) \Delta X}\, d\nu ,
\qquad
\varepsilon_b = -\left.\frac{d \ln E_b}{d \Delta X}\right|_{\Delta X = 0} = m \int w_b(\nu)\, \sigma(\nu)\, d\nu .
$$ (eq-abs-band-transmittance)

The exponential is convex, so Jensen's inequality gives

$$
E_b(\Delta X) \ge \exp\left(-m \Delta X \int w_b \sigma\, d\nu\right) = e^{-\varepsilon_b \Delta X},
$$ (eq-abs-jensen)

with equality only if $\sigma$ is constant across the band. Physically, the line cores saturate first: once $e^{-m\sigma\Delta X} \approx 0$ at a line's centre, more methane removes no more light there, and the band darkens only through the line wings. A model $e^{-\varepsilon_b \Delta X}$ predicts more absorption than there is for every $\Delta X > 0$, so it reaches the observed transmittance with a smaller column, and the deficit grows with the enhancement's strength.

Taking logarithms does not remove the effect. $\ln E_b(\Delta X)$ is the cumulant generating function of $m\sigma$ under the weight $w_b$, evaluated at $-\Delta X$. That function is convex, and it is linear only when $\sigma$ is constant across the band. A model that is linear in $\Delta X$ in log-radiance, such as the matched filter on $\ln L$, has the same low bias for strong enhancements.
:::

## 3. Windows, pairs and the ratio

A gas's [`AbsorptionWindow`](xref:api#radtrax.retrieval.absorption.AbsorptionWindow) names the bands the transmittance is fitted on and the bands where the gas absorbs; pairing compares only the bands outside the absorbing ones. EMIT's CH₄ window fits 2100–2440 nm and keeps 1640–1690 and 2100–2440 nm out of pairing; its CO window fits and excludes 2278–2441 nm.

Given the in-pixels $I$ and the out-pixels $O$, as flat indices into the image ([`pair_pixels`](xref:api#radtrax.retrieval.absorption.pair_pixels)), with $s(p)$ pixel $p$'s radiance on the pairing bands, pairing is one to one, minimising the total Euclidean distance, and keeps the best-matched half:

$$
\pi^\star = \arg\min_{\pi: I \to O \text{ injective}} \sum_{i \in I} \left\lVert s(i) - s(\pi(i)) \right\rVert_2 .
$$ (eq-abs-pairing)

The kept pairs give the ratio {eq}`eq-abs-ratio` ([`ratio_spectrum`](xref:api#radtrax.retrieval.absorption.ratio_spectrum)). These are the reference package's settings, `pv/config/config.yaml`.

:::{dropdown} Why each setting
**Pairing uses only bands that methane does not absorb.** If methane's bands took part, pairing would favour out-pixels that happen to be dark there and cancel part of the absorption. The Euclidean distance on radiance matches brightness as well as shape, so it makes $\eta$ close to one and not only flat.

**One to one.** If one out-pixel served many in-pixels, its noise and any surface mismatch would enter many pairs at once. The injective assignment {eq}`eq-abs-pairing` is a linear assignment problem, solved exactly by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).

**The best-matched half.** Dropping the worst-matched pairs trades a factor of $\sqrt{2}$ in noise for a smaller and smoother surface mismatch in $\eta$. The reference code does not give its reason for choosing a half.

**Out-pixels free of the gas.** An out-pixel that carries the enhancement too cancels the signal being tested. Choosing out-pixels with a small matched-filter score, below the window's `mf_background_mol_m2`, keeps them out; that choice is the caller's.
:::

## 4. The fit and the score

The continuum is a Legendre series on the window, $P(\lambda; c) = \sum_{k=0}^{K-1} c_k P_k(x)$ with $x \in [-1, 1]$, the basis of [`PolynomialAlbedo`](xref:api#radtrax.surface.PolynomialAlbedo):

$$
(\widehat{\Delta X}, \hat c) = \arg\min_{\Delta X, c} \sum_{\lambda} \left( R(\lambda) - P(\lambda; c) E(\lambda; \Delta X) \right)^2 .
$$ (eq-abs-fit)

For fixed $\Delta X$ the model is linear in $c$, so $\hat c(\Delta X)$ is a least-squares solve; profiling over the grid of $\Delta X$ gives $\widehat{\Delta X}$ with no starting point ([`fit_transmittance`](xref:api#radtrax.retrieval.absorption.fit_transmittance)). The score divides out the continuum, $q = R / P(\hat c)$, and compares $q$ with the fitted transmittance relative to $q$'s own spread (Xiang et al.'s eq. 11, [`d_norm`](xref:api#radtrax.retrieval.absorption.d_norm)):

$$
D_{\text{norm}} = \frac{\operatorname{mean}_\lambda \left| q(\lambda) - E(\lambda; \widehat{\Delta X}) \right|}{\operatorname{mean}_\lambda \left| q(\lambda) - \bar q \right|} .
$$ (eq-abs-dnorm)

A perfect fit gives 0. With no methane, $\widehat{\Delta X} = 0$, $E = 1$ and $\bar q \approx 1$, so numerator and denominator agree and $D_{\text{norm}} \approx 1$.

$\widehat{\Delta X}$ is the column of the *mean* transmittance over the pairs, not the mean column, since $E$ is convex in $\Delta X$ ({eq}`eq-abs-ratio-exact`); it is a score, not a retrieval of the column.

:::{dropdown} The profile fit, written out
Let $V \in \mathbb{R}^{B \times K}$ hold the Legendre polynomials at the window's bands, $V_{\lambda k} = P_k(x_\lambda)$ with $x_\lambda = 2(\lambda - \lambda_{\min}) / (\lambda_{\max} - \lambda_{\min}) - 1$. For fixed $\Delta X$, the model in {eq}`eq-abs-fit` is $A(\Delta X)\, c$ with $A(\Delta X) = \operatorname{diag}(E(\Delta X))\, V$, so

$$
\hat c(\Delta X) = \left(A^\top A\right)^{-1} A^\top R,
\qquad
\mathrm{RSS}(\Delta X) = \left\lVert \left(I - \Pi_{A(\Delta X)}\right) R \right\rVert_2^2 ,
$$ (eq-abs-profile)

with $\Pi_A = A (A^\top A)^{-1} A^\top$ the orthogonal projector onto the columns of $A$. The grid point $k$ of least $\mathrm{RSS}$ needs no starting point: it is the global minimum over the grid. $\widehat{\Delta X}$ then refines it between the nodes: it is the vertex of the parabola through $\mathrm{RSS}$ at three neighbouring nodes around $k$ (the first or last three at a grid end), clipped to their span; $E(\widehat{\Delta X})$ interpolates $\ln E$ linearly between the two rows around it, and $\hat c$ is solved again there. The refined fit replaces the node's only where the parabola is convex and the refit lowers the residual, so a fit exact on a node stays there. If $\mathrm{RSS} \approx a u^2 + b u^3$ near its minimum, with $u$ the distance to it, and the nodes are $h$ apart, the vertex is off by $(b / 2a)(3\delta^2 - h^2)$, with $\delta$ the middle node's distance to the minimum: at most $|b| h^2 / a$, against $h / 2$ for the node alone. That matters for weak enhancements: on a 0.005 mol/m² grid, a noise-free 62 ppm·m (0.0025 mol/m²) ratio would otherwise sit on a node and score $D_{\text{norm}} \approx 0.5$; refined, it scores below $10^{-3}$. The grid starts at 0, which imposes $\Delta X \ge 0$: a ratio whose best fit wants no methane, or less than none, sits on that boundary at $\widehat{\Delta X} = 0$.

Least squares is the maximum-likelihood fit when the errors in $R$ are independent across bands and have equal variance. Neither holds exactly. The noise in $R$ is larger in bands with less light, and the leftover surface mismatch in $\eta$ is smooth in $\lambda$, so it is correlated across bands.
:::

:::{dropdown} How much of methane's shape the continuum leaves
Linearise for a weak enhancement in log space, where the continuum is additive: $\ln R \approx V a - \varepsilon\, \Delta X + n$, treating $\ln P$ as lying in the span of $V$ and taking $n \sim \mathcal{N}(0, \sigma^2 I)$. Profiling out $a$ (the Frisch–Waugh–Lovell theorem) leaves

$$
\widehat{\Delta X} = -\frac{\varepsilon_\perp^\top \ln R}{\varepsilon_\perp^\top \varepsilon_\perp},
\qquad
\operatorname{Var}\left(\widehat{\Delta X}\right) = \frac{\sigma^2}{\lVert \varepsilon_\perp \rVert_2^2},
\qquad
\varepsilon_\perp = \left(I - \Pi_V\right) \varepsilon .
$$ (eq-abs-identifiability)

Only the part of methane's band shape that a polynomial of degree $K - 1$ cannot reproduce informs $\widehat{\Delta X}$. Each added degree removes another direction from $\varepsilon_\perp$ and inflates the variance; with $K = B$ the continuum fits anything, and $\Delta X$ is not identifiable. This is the sense in which eleven continuum terms on 46 bands leave little of methane's shape to score ({ref}`sec-abs-departures`).

The basis does not change this. Legendre polynomials and monomials of the same degree span the same space, so they give the same $\varepsilon_\perp$ and, in exact arithmetic, the same fit. What the Legendre basis changes is conditioning: monomials of $\lambda$ near 2300 nm are nearly collinear, and the least-squares solve loses digits to that.

The same algebra says what the test can reject. A confounder that adds $d$ to $\ln R$ shifts $\widehat{\Delta X}$ by $-\varepsilon_\perp^\top d / \lVert \varepsilon_\perp \rVert^2$ and leaves the rest of $(I - \Pi_V) d$ in the residual. The test rejects a confounder only to the extent that its shape, once the continuum is removed, differs from methane's.
:::

:::{dropdown} What $D_{\text{norm}}$ measures
Write $q = E(\widehat{\Delta X}) + r$, with $r$ the residual. The numerator of {eq}`eq-abs-dnorm` is $\operatorname{mean}_\lambda |r|$, the part of $q$ the fit leaves unexplained. The denominator is $q$'s spread about its own mean, which is the absorption the fit explains plus the residual. $D_{\text{norm}}$ is an L1 analogue of the fraction of variance unexplained, $1 - R^2$.

**It is not purely a test of shape.** Let the residual have size $\sigma_q$ per band and the fitted transmittance vary across the window by $\delta$. When $\delta \gg \sigma_q$, $D_{\text{norm}}$ is of order $\sigma_q / \delta$; when $\delta \ll \sigma_q$, numerator and denominator are both about $\sigma_q$ and $D_{\text{norm}}$ tends to 1. A real but weak enhancement therefore scores like none. For $N$ pairs of pixels with signal-to-noise ratio SNR, $\sigma_q \approx \sqrt{2} / (\mathrm{SNR}\sqrt{N})$, so a fixed threshold such as 0.3 also depends on the sensor's noise, on the number of pairs and on the window, besides the continuum's degree.

**It can exceed 1**, when the fitted transmittance has structure that $q$ lacks.

**The fit and the score optimise different things.** The fit minimises the L2 norm of $R - P E$; the score is an L1 norm on $q = R / P$. $\widehat{\Delta X}$ therefore does not minimise $D_{\text{norm}}$'s numerator, though the difference is small when $P$ is close to 1.
:::

(sec-abs-plumax)=
## 5. Plume vetting lives in plumax

Plume vetting runs this test on matched-filter candidates, and that part is about plumes, so it lives in plumax as `plumax.vetting`, which imports radtrax: the in-pixels are a candidate's seeds, its strongest matched-filter pixels; the out-pixels are a background pool around them, outside every candidate, with a small matched-filter score; a null from the same scene (copies of the candidate's mask placed where there is no candidate) calibrates $D_{\text{norm}}$ into a p-value; and the decision on a candidate, with the worked example on a synthetic EMIT scene, is plumax's too. radtrax stays the plume-agnostic half: windows, transmittance, pairing given two pixel sets, the ratio, the fit and $D_{\text{norm}}$.

## 6. What radtrax computes

| Step | Equation | Function | Issue |
|---|---|---|---|
| A gas's fitting and pairing bands | | `AbsorptionWindow`, `AbsorptionWindow.ch4_emit`, `AbsorptionWindow.co_emit` | [#155](https://github.com/jejjohnson/radtrax/issues/155) |
| Transmittance on a grid of $\Delta X$ | {eq}`eq-abs-table` | `transmittance_table` | [#154](https://github.com/jejjohnson/radtrax/issues/154) |
| One-to-one pairs of two pixel sets and the ratio | {eq}`eq-abs-pairing`, {eq}`eq-abs-ratio` | `pair_pixels`, `ratio_spectrum` | [#149](https://github.com/jejjohnson/radtrax/issues/149) |
| Profile fit and score | {eq}`eq-abs-fit`, {eq}`eq-abs-dnorm` | `fit_transmittance`, `TransmittanceFit`, `d_norm` | [#150](https://github.com/jejjohnson/radtrax/issues/150) |

All of them live in `radtrax.retrieval.absorption`, beside the matched filter.

(sec-abs-departures)=
## 7. Where radtrax departs from the reference code

- **The continuum.** The reference package fits degree 10 in raw monomials of wavelength in nanometres; $\lambda^{10}$ near 2300 nm makes the least squares badly conditioned, and eleven continuum terms on 46 bands leave little of methane's shape to score. radtrax uses a Legendre series, degree 2 by default.
- **The fit.** The reference uses Levenberg–Marquardt from a fixed start; radtrax profiles over $\Delta X$ with a closed-form solve for the continuum, which cannot wander at saturation.
- **The transmittance.** The reference package reads a fixed target file, marked in its code as a test placeholder; radtrax computes {eq}`eq-abs-table` for each scene.
