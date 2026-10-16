---
title: "Plume vetting — a spectral test for matched-filter candidates"
---

# Plume vetting

:::{important} Status: target
This page describes `radtrax.retrieval.vetting`, which Wave 5 ([#146](https://github.com/jejjohnson/radtrax/issues/146)) adds. Every function named here has an issue with its signature and tests, and the [plume vetting notebook](../notebooks/plume_vetting.ipynb) runs all of them on one scene, defined in the notebook until each issue lands. The numbers below are that notebook's.
:::

A matched filter scores each pixel by one projection of its spectrum onto methane's target, so anything that removes light under methane's 2.3 μm band scores: carbonate and clay soils, salt pans, dark water, shadows, striping. A segmentation network on the matched-filter map learns plume shapes, which a plume-shaped confounder also has. **Plume vetting** (Xiang et al. [^xiang2024]; Růžička et al. [^ruzicka2026]) goes back to the radiances behind a candidate. It pairs the candidate's strongest pixels with pixels outside it that have the same surface, divides them, and asks whether the ratio has the shape of methane's transmittance. EMIT's daily methane digest alerts without review when the misfit score $D_{\text{norm}}$ is below 0.3.

Vetting is a second test on the same data, not an independent one: its pixels are chosen by the matched filter and come from the same cube. It is more selective because it tests the shape of the absorption across a window of bands, where the matched filter tests one projection.

That projection is the [matched filter](matched_filter.md)'s $\hat\alpha(x) = (x - \mu)^\top \Sigma^{-1} t \,/\, t^\top \Sigma^{-1} t$. A surface whose spectrum departs from the background mean by $d$ scores $d^\top \Sigma^{-1} t \,/\, t^\top \Sigma^{-1} t$, whatever the shape of $d$, so a carbonate's dip that overlaps methane's band scores positive. Many spectra with different shapes give the same score, because the filter keeps one number from the $B$ bands it sees. Vetting keeps the shape: it builds the candidate's absorption spectrum band by band and asks whether all of it agrees with methane.

[^xiang2024]: Xiang, C., et al. (2024). A novel spectroscopic method for vetting methane point sources. AGU Fall Meeting, abstract A21K-1903.

[^ruzicka2026]: Růžička, V., Thompson, D. R., Fahlen, J. E., et al. (2026). Fully automatic trace gas plume detection. arXiv:2605.03372.

[^pv]: EMIT Science Data System (2025). plume-vetting, revision `47fc6ad`. https://github.com/emit-sds/plume-vetting (Apache-2.0).

[^phipson2010]: Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

## 1. The ratio of a pixel pair

Write $L_{\text{in}}(\lambda)$ for the radiance of a pixel under the plume and $L_{\text{out}}(\lambda)$ for a pixel outside it, $\rho$ and $\rho'$ for their surface reflectances, $F_0$ for the solar irradiance, $\theta_s$ for the solar zenith angle, $T_{\text{atm}}$ for the vertical one-way transmittance of everything but the plume, $m$ for the two-way air mass and $\Delta X$ for the plume's column enhancement in mol/m². Columns are also quoted in ppm·m, as EMIT quotes them: 1 ppm·m is $10^{-6}\, n_{\text{air}} \times 1$ m, with $n_{\text{air}} = p / (k_B T N_A)$ the molar density of air where the plume is, which is $4.03 \times 10^{-5}$ mol/m² at the surface of the US standard atmosphere the notebook uses. For a Lambertian surface,

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

Three assumptions sit inside the $\approx$: the two pixels of a pair share their sun, path and atmosphere; the $E$ in $R$ is an average over the pairs rather than the transmittance of one column; and the noise is small once the pairs are averaged. The derivation below makes each one explicit.

:::{dropdown} Derivation: what survives the ratio
Write $C_n = F_0 \cos\theta_s\, T_{\text{atm}}^{m} / \pi$ for the factor that pair $n$'s two pixels share. They share it because the background pool lies in a box around the seeds, so both pixels see the same sun, elevation and water vapour; different pairs need not share it. Leaving out the noise,

$$
R = \frac{\sum_n C_n \rho_n E_n}{\sum_n C_n \rho'_n}
  = \underbrace{\frac{\sum_n C_n \rho_n}{\sum_n C_n \rho'_n}}_{\eta}
    \cdot
    \underbrace{\frac{\sum_n w_n E_n}{\sum_n w_n}}_{\bar E},
\qquad w_n = C_n \rho_n ,
$$ (eq-pv-ratio-exact)

with $E_n = E(\lambda; \Delta X_n)$. This is an identity, not an approximation, and it says three things.

**$\eta$ is a ratio of brightness-weighted mean reflectances.** Pairing on bands that methane does not absorb makes $\rho_n \approx \rho'_n$ pair by pair outside methane's band. Pairing cannot see a mismatch that exists only inside methane's band, such as a carbonate's dip at 2.3 μm, so that mismatch stays in $\eta$, and it is the fit and the score that must reject it.

**$\bar E$ is a mean of transmittances, not the transmittance of the mean column.** $E$ is decreasing and convex in $\Delta X$, so Jensen's inequality gives $\bar E \ge E(\overline{\Delta X})$ with $\overline{\Delta X} = \sum_n w_n \Delta X_n / \sum_n w_n$. The column that reproduces $\bar E$ is therefore at most $\overline{\Delta X}$: a plume whose column varies across the pairs is fitted with a smaller column than its weighted mean.

**The ratio is taken after averaging.** For independent noisy quantities $a$ and $b$ with means $\mu_a, \mu_b$ and relative standard deviation $s_b$ of $b$, a second-order expansion of $1/b$ gives $\mathbb{E}[a/b] \approx (\mu_a / \mu_b)(1 + s_b^2)$. Averaging $N$ pairs first divides $s_b^2$ by $N$; dividing each pair and then averaging leaves the bias at the size of a single pixel's $s_b^2$. The bias is largest for dark pixels, whose $L_{\text{out}}$ has the lowest signal-to-noise ratio.
:::

## 2. The transmittance from the forward model

The reference implementation [^pv] reads $E$ from MODTRAN tables over solar and view zenith, elevation, water vapour and the plume's strength. radtrax computes it for the scene itself: with $y_b$ the band radiance of [`ForwardModel`](xref:api#radtrax.ForwardModel) for unit albedo,

$$
E_b(\Delta X) = \frac{y_b(x_{\text{atm}} + \Delta X)}{y_b(x_{\text{atm}})},
$$ (eq-pv-table)

on a grid of $\Delta X$, with the enhancement in the bottom layer by default. The albedo cancels because Beer–Lambert radiance is linear in it. That holds exactly for radtrax's `BeerLambert` solver, which has no path radiance or multiple scattering, and for an albedo that is constant across each band's response. A reflectance that varies inside one band reweights the wavenumbers the band averages over, which is a small effect when the reflectance is smooth over one band's width. The curve of growth is built in: lines saturate, so a strong plume absorbs less per molecule than the weak-absorption slope $\varepsilon_b = -\partial \ln y_b / \partial \Delta X$ at 0 predicts, the matched filter's target from [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target). Fitting with a constant coefficient, $E_b \approx \exp(-\varepsilon_b \Delta X)$, biases the column **low**: by 1.4, 2.8 and 7.7 % at 0.05, 0.1 and 0.3 mol/m² on the synthetic spectrum.

:::{dropdown} Derivation: why a constant coefficient biases the column low
Band $b$ averages the wavenumbers $\nu$ of its response with a weight $w_b(\nu) \ge 0$, normalised so that $\int w_b\, d\nu = 1$, that combines the response, the solar irradiance and the ambient transmittance. With $\sigma(\nu)$ methane's cross-section in m²/mol at the plume's layer,

$$
E_b(\Delta X) = \int w_b(\nu)\, e^{-m \sigma(\nu) \Delta X}\, d\nu ,
\qquad
\varepsilon_b = -\left.\frac{d \ln E_b}{d \Delta X}\right|_{\Delta X = 0} = m \int w_b(\nu)\, \sigma(\nu)\, d\nu .
$$ (eq-pv-band-transmittance)

The exponential is convex, so Jensen's inequality gives

$$
E_b(\Delta X) \ge \exp\left(-m \Delta X \int w_b \sigma\, d\nu\right) = e^{-\varepsilon_b \Delta X},
$$ (eq-pv-jensen)

with equality only if $\sigma$ is constant across the band. Physically, the line cores saturate first: once $e^{-m\sigma\Delta X} \approx 0$ at a line's centre, more methane removes no more light there, and the band darkens only through the line wings. A model $e^{-\varepsilon_b \Delta X}$ predicts more absorption than there is for every $\Delta X > 0$, so it reaches the observed transmittance with a smaller column, and the deficit grows with the plume's strength.

Taking logarithms does not remove the effect. $\ln E_b(\Delta X)$ is the cumulant generating function of $m\sigma$ under the weight $w_b$, evaluated at $-\Delta X$. That function is convex, and it is linear only when $\sigma$ is constant across the band. A model that is linear in $\Delta X$ in log-radiance, such as the matched filter on $\ln L$, has the same low bias for strong plumes.
:::

## 3. Seeds, pairs and the ratio

Inside a candidate mask $M$, the seeds $I$ are the $k = 40$ pixels of largest matched-filter score below the mask's 99th percentile, grown by one pixel inside $M$. The background pool $O$ is every pixel within a box of the seeds, outside every candidate, with a small matched-filter score. Pairing is one to one, minimising the total Euclidean distance between radiances on bands methane does not absorb, and keeps the best-matched half:

$$
\pi^\star = \arg\min_{\pi: I \to O \text{ injective}} \sum_{i \in I} \left\lVert s(i) - s(\pi(i)) \right\rVert_2 .
$$ (eq-pv-pairing)

These are the reference package's settings, `pv/config/config.yaml`.

:::{dropdown} Why each setting
**Seeds come from the matched filter.** The depth of methane's absorption in $R$ grows with $\Delta X$, so the strongest pixels give the deepest and best-resolved transmittance.

**The seeds' noise is not innocent.** Choosing the $k$ largest scores selects pixels whose noise happens to point along methane's target, so the seeds' spectra are biased toward methane's shape by the selection itself. This is one reason vetting is not independent of the matched filter. Growing the seeds by one pixel adds neighbours that were not chosen for their own noise, and it keeps the in-set a connected region rather than a scatter of isolated pixels.

**The cap at the 99th percentile.** The reference code does not say why it caps there. The cap drops the mask's most extreme scores, which is where a single bad or saturated pixel would land.

**The background pool.** A small matched-filter score keeps plume pixels out of $L_{\text{out}}$, where they would cancel the signal being tested. The box keeps the pool near the seeds, so that each pair shares $C_n$ in {eq}`eq-pv-ratio-exact`.

**Pairing uses only bands that methane does not absorb.** If methane's bands took part, pairing would favour background pixels that happen to be dark there and cancel part of the absorption. The Euclidean distance on radiance matches brightness as well as shape, so it makes $\eta$ close to one and not only flat.

**One to one.** If one background pixel served many seeds, its noise and any surface mismatch would enter many pairs at once. The injective assignment {eq}`eq-pv-pairing` is a linear assignment problem, solved exactly by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` in the notebook).

**The best-matched half.** Dropping the worst-matched pairs trades a factor of $\sqrt{2}$ in noise for a smaller and smoother surface mismatch in $\eta$. The reference code does not give its reason for choosing a half.
:::

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

$\widehat{\Delta X}$ is the column of the *mean* transmittance over the pairs, not the mean column, since $E$ is convex in $\Delta X$ ({eq}`eq-pv-ratio-exact`); it is a score, not a retrieval.

:::{dropdown} The profile fit, written out
Let $V \in \mathbb{R}^{B \times K}$ hold the Legendre polynomials at the window's bands, $V_{\lambda k} = P_k(x_\lambda)$ with $x_\lambda = 2(\lambda - \lambda_{\min}) / (\lambda_{\max} - \lambda_{\min}) - 1$. For fixed $\Delta X$, the model in {eq}`eq-pv-fit` is $A(\Delta X)\, c$ with $A(\Delta X) = \operatorname{diag}(E(\Delta X))\, V$, so

$$
\hat c(\Delta X) = \left(A^\top A\right)^{-1} A^\top R,
\qquad
\mathrm{RSS}(\Delta X) = \left\lVert \left(I - \Pi_{A(\Delta X)}\right) R \right\rVert_2^2 ,
$$ (eq-pv-profile)

with $\Pi_A = A (A^\top A)^{-1} A^\top$ the orthogonal projector onto the columns of $A$. $\widehat{\Delta X}$ is the grid point of least $\mathrm{RSS}$. This needs no starting point, it finds the global minimum over the grid, and its resolution is the grid's spacing. The grid starts at 0, which imposes $\Delta X \ge 0$: candidate B's $\widehat{\Delta X} = 0$ sits on that boundary, so the best fit wanted no methane, or less than none.

Least squares is the maximum-likelihood fit when the errors in $R$ are independent across bands and have equal variance. Neither holds exactly. The noise in $R$ is larger in bands with less light, and the leftover surface mismatch in $\eta$ is smooth in $\lambda$, so it is correlated across bands.
:::

:::{dropdown} How much of methane's shape the continuum leaves
Linearise for a weak plume in log space, where the continuum is additive: $\ln R \approx V a - \varepsilon\, \Delta X + n$, treating $\ln P$ as lying in the span of $V$ and taking $n \sim \mathcal{N}(0, \sigma^2 I)$. Profiling out $a$ (the Frisch–Waugh–Lovell theorem) leaves

$$
\widehat{\Delta X} = -\frac{\varepsilon_\perp^\top \ln R}{\varepsilon_\perp^\top \varepsilon_\perp},
\qquad
\operatorname{Var}\left(\widehat{\Delta X}\right) = \frac{\sigma^2}{\lVert \varepsilon_\perp \rVert_2^2},
\qquad
\varepsilon_\perp = \left(I - \Pi_V\right) \varepsilon .
$$ (eq-pv-identifiability)

Only the part of methane's band shape that a polynomial of degree $K - 1$ cannot reproduce informs $\widehat{\Delta X}$. Each added degree removes another direction from $\varepsilon_\perp$ and inflates the variance; with $K = B$ the continuum fits anything, and $\Delta X$ is not identifiable. This is the sense in which eleven continuum terms on 46 bands leave little of methane's shape to score ({ref}`sec-pv-departures`).

The basis does not change this. Legendre polynomials and monomials of the same degree span the same space, so they give the same $\varepsilon_\perp$ and, in exact arithmetic, the same fit. What the Legendre basis changes is conditioning: monomials of $\lambda$ near 2300 nm are nearly collinear, and the least-squares solve loses digits to that.

The same algebra says what vetting can reject. A confounder that adds $d$ to $\ln R$ shifts $\widehat{\Delta X}$ by $-\varepsilon_\perp^\top d / \lVert \varepsilon_\perp \rVert^2$ and leaves the rest of $(I - \Pi_V) d$ in the residual. Vetting rejects a confounder only to the extent that its shape, once the continuum is removed, differs from methane's.
:::

:::{dropdown} What $D_{\text{norm}}$ measures
Write $q = E(\widehat{\Delta X}) + r$, with $r$ the residual. The numerator of {eq}`eq-pv-dnorm` is $\operatorname{mean}_\lambda |r|$, the part of $q$ the fit leaves unexplained. The denominator is $q$'s spread about its own mean, which is the absorption the fit explains plus the residual. $D_{\text{norm}}$ is an L1 analogue of the fraction of variance unexplained, $1 - R^2$.

**It is not purely a test of shape.** Let the residual have size $\sigma_q$ per band and the fitted transmittance vary across the window by $\delta$. When $\delta \gg \sigma_q$, $D_{\text{norm}}$ is of order $\sigma_q / \delta$; when $\delta \ll \sigma_q$, numerator and denominator are both about $\sigma_q$ and $D_{\text{norm}}$ tends to 1. A real but weak plume therefore scores like no plume. For $N$ pairs of pixels with signal-to-noise ratio SNR, $\sigma_q \approx \sqrt{2} / (\mathrm{SNR}\sqrt{N})$, so a fixed threshold such as 0.3 also depends on the sensor's noise, on the number of pairs and on the window, besides the continuum's degree.

**It can exceed 1**, when the fitted transmittance has structure that $q$ lacks.

**The fit and the score optimise different things.** The fit minimises the L2 norm of $R - P E$; the score is an L1 norm on $q = R / P$. $\widehat{\Delta X}$ therefore does not minimise $D_{\text{norm}}$'s numerator, though the difference is small when $P$ is close to 1.
:::

## 5. A null from the same scene

The honest null is this scene's clutter with no plume. Vet $S$ copies $M_s$ of the mask, each rotated and translated to where there is no candidate, and report

$$
\hat p = \frac{1}{S} \sum_{s=1}^{S} \mathbb{1}\left[ D_{\text{norm}}(M_s) \le D_{\text{norm}}(M) \right] .
$$ (eq-pv-null)

$D_{\text{norm}}$'s threshold depends on the band set, the continuum's degree and the ratio's noise, so $\hat p$ travels between sensors where 0.3 does not. Which to decide on is [#160](https://github.com/jejjohnson/radtrax/issues/160).

:::{dropdown} Reading $\hat p$
**What it estimates.** $\hat p$ is the fraction of null copies that score at least as well as the candidate. A small $\hat p$ says that a plume-free mask of the candidate's shape, placed elsewhere in this scene, rarely fits methane this well.

**Its resolution.** With $S$ copies, $\hat p$ moves in steps of $1/S$. $\hat p = 0$ says only that no copy did as well, not that the probability is zero. The usual Monte Carlo form, $(1 + \sum_s \mathbb{1}[\cdot]) / (1 + S)$, is never zero and is a valid p-value when the candidate and the copies are exchangeable under the null [^phipson2010]. With $S = 100$, candidate A's $\hat p = 0.00$ is $1/101 \approx 0.01$ in that form.

**What the null samples.** Each copy runs the whole of {eq}`eq-pv-pairing` to {eq}`eq-pv-dnorm`, so the selection of seeds by the matched filter happens inside the null too. The selection of the mask itself does not. Copies sit wherever there is no candidate, so they sample this scene's background in the mask's shape, not background that the matched filter flagged. A confounder is unusual background by construction, which is why the null calibrates the score but cannot replace the fit's test of shape.

**The copies are not independent.** They overlap one another and draw on the same background pool, so the tail of the null is estimated less precisely than $S$ independent draws would estimate it.
:::

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

(sec-pv-departures)=
## 8. Where radtrax departs from the reference code

- **The continuum.** The reference package fits degree 10 in raw monomials of wavelength in nanometres; $\lambda^{10}$ near 2300 nm makes the least squares badly conditioned, and eleven continuum terms on 46 bands leave little of methane's shape to score. radtrax uses a Legendre series, degree 2 by default.
- **The fit.** The reference uses Levenberg–Marquardt from a fixed start; radtrax profiles over $\Delta X$ with a closed-form solve for the continuum, which cannot wander at saturation.
- **The transmittance.** The reference package reads a fixed target file, marked in its code as a test placeholder; radtrax computes {eq}`eq-pv-table` for each scene.
- **The null.** The reference package draws 100 rotated and translated copies that avoid only their own plume, and does not yet decide on them; radtrax's copies avoid every candidate, and $\hat p$ is returned with every score.
