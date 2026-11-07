---
title: "Absorptive plume detection: change of variables, the LMP and the GLRT"
---

# Absorptive plume detection

:::{important} Status: theory, partly in code
The detectors on this page split into what radtrax computes today and what is still a recipe. The Tμ matched filter is [`apply_image`](xref:api#radtrax.retrieval.matched_filter.apply_image) with the target $\boldsymbol{\mu}_0 \odot \mathbf{t}$, and $\mathbf{t}$ is [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target). The quadratic matched filter and the GLRT have no function yet; {ref}`sec-ad-code` gives the few lines each one takes. Non-Gaussian clutter (the multivariate t, ACE and the lognormal matched filter) is on the companion page, [Beyond Gaussian clutter](clutter_models.md).
:::

The [matched filter](matched_filter.md) starts from an additive model, $\mathbf{y} = \boldsymbol{\mu} + \alpha \mathbf{t} + \boldsymbol{\varepsilon}$, which is what makes it linear and optimal. However, a gas does not add light to a pixel, it removes a fraction of it, so the plume's footprint in a spectrum depends on the spectrum it lands on. This page redoes the detection problem with the multiplicative model that Beer–Lambert actually gives, following Theiler & Schaum [^theiler2020] and Theiler [^theiler2021]. The tools are the ones a statistician would reach for: a change of variables to get the plume-on density, a likelihood ratio, and a Taylor series in the plume strength to deal with the fact that we do not know it. What comes out is a family of detectors (the clairvoyant detector, the quadratic matched filter, the GLRT) and a precise statement of where the linear matched filter sits among them.

[^theiler2020]: Theiler, J. & Schaum, A. (2020). Some closed-form expressions for absorptive plume detection. *Proc. IEEE IGARSS 2020*, 1786–1789. doi:10.1109/IGARSS39084.2020.9323752

[^theiler2021]: Theiler, J. (2021). Absorptive weak plume detection on Gaussian and non-Gaussian background clutter. *IEEE JSTARS*, 14, 6842–6854. doi:10.1109/JSTARS.2021.3093820 (CC-BY 4.0).

[^schaum2021]: Schaum, A. (2021). A uniformly most powerful detector of gas plumes against a cluttered background. *Remote Sensing of Environment*, 260, 112443. doi:10.1016/j.rse.2021.112443

[^foote2020]: Foote, M. D., Dennison, P. E., Thorpe, A. K., et al. (2020). Fast and accurate retrieval of methane concentration from imaging spectrometer data using sparsity prior. *IEEE TGRS*, 58(9), 6480–6492.

[^kay1998]: Kay, S. M. (1998). *Fundamentals of Statistical Signal Processing, Vol. II: Detection Theory*. Prentice Hall.

## TL;DR

A plume of strength $\alpha$ multiplies each band of the plume-free spectrum, $\mathbf{y} = e^{\alpha \mathbf{T}} \mathbf{y}_0$ with $\mathbf{T} = \mathrm{diag}(\mathbf{t})$. Changing variables from $\mathbf{y}_0$ to $\mathbf{y}$ gives the plume-on density, and its ratio to the background density is the optimal detector when $\alpha$ is known:

$$
\log \mathrm{LR}(\alpha; \mathbf{y}) = \log p_0(e^{-\alpha \mathbf{T}} \mathbf{y}) - \log p_0(\mathbf{y}) + \alpha \tau
$$

For Gaussian clutter, a second-order Taylor series in $\alpha$ turns the unknown strength into a closed-form estimate and a detector, the GLRT:

$$
\hat{\alpha}(\mathbf{y}) = \frac{Q(\mathbf{y}) + \tau}{E(\mathbf{y})}
$$

$$
D_\text{GLRT}(\mathbf{y}) = \frac{Q(\mathbf{y}) + \tau}{\sqrt{E(\mathbf{y})}}
$$

with $Q(\mathbf{y}) = (\mathbf{T}\mathbf{y})^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{y} - \boldsymbol{\mu}_0)$ the slope and $E(\mathbf{y})$ the curvature of the log likelihood ratio. Stopping at first order instead gives the "locally most powerful" quadratic matched filter, $Q + \tau$, which is optimal only for plumes too weak to detect. Replacing $\mathbf{T}\mathbf{y}_0$ by $\mathbf{T}\boldsymbol{\mu}_0$ gives the linear matched filter with target $\boldsymbol{\mu}_0 \odot \mathbf{t}$.

The assumptions are 1) Beer–Lambert per band, 2) a background distribution estimated from plume-free pixels, 3) Gaussian clutter and 4) a plume strength small enough for a second-order series. Practically speaking, on an EMIT-like 2.3 μm scene the Tμ matched filter is already within a factor of two of the clairvoyant bound and indistinguishable from the GLRT as a detector, whereas the GLRT's strength estimate stays within 5 % of the truth up to 25 detection limits, where the matched filter's has lost a quarter.

:::{tip} ELI5 · why can't a gas just be "added" to a spectrum?
Think of a plume as a pair of **sunglasses** on the ground. Sunglasses do not add darkness, they let through a **fraction** of the light, so they take more light from a bright white roof than from black asphalt. A detector that looks for a fixed amount of missing light (the additive model) expects the same dip everywhere, which is wrong on every pixel brighter or darker than average. The fix is to ask "how likely is this spectrum if it were some background spectrum seen **through** the sunglasses?", and that question is a change of variables.
:::

## The problem

An imaging spectrometer like EMIT or AVIRIS-NG gives one radiance spectrum per pixel, and we want to flag the pixels whose spectrum carries a gas's absorption. The signal is tiny: at the detection limit a methane plume changes the radiance by about two parts in ten thousand ({ref}`sec-ad-check`), far below the pixel-to-pixel variation of the surface. What makes detection possible is that we know the gas's absorption spectrum to high precision and see it across many bands, and that we can model the clutter (the statistics of plume-free pixels) from the scene itself.

So the question is a hypothesis test per pixel: is $\mathbf{y}$ a draw from the background distribution (no plume, $\alpha = 0$), or from the background seen through a plume of some unknown strength $\alpha > 0$? The unknown $\alpha$ makes this a composite hypothesis test [^kay1998], which is why there is no single "optimal" detector and why the papers end up with several.

**Running example · an EMIT-like 2.3 μm pixel**

| Quantity | Value |
|---|---|
| Bands | 9 Gaussian bands, 2296–2356 nm every 7.5 nm, FWHM 8.5 nm |
| Atmosphere | AFGL US Standard to 20 km, CH₄ and H₂O from radtrax's synthetic Voigt tables, Beer–Lambert |
| Geometry | sun at 35°, sensor at 10° |
| Plume | a column enhancement in the bottom layer, 1 ppm·m = 4.03 × 10⁻⁵ mol/m² |
| Surface | Lambertian, mean albedo 0.25, brightness spread 35 %, small spectral slope and curvature |
| Noise | additive Gaussian, SNR 300 at albedo 0.25 |
| Pixels | 200 000 plume-free, and the same pixels with a plume implanted |

The numbers come from radtrax itself: $\mathbf{t}$ from [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target) and the implanted plume from the exact band transmittance of [`transmittance_table`](xref:api#radtrax.retrieval.absorption.transmittance_table), so the implant includes the curve of growth that the derivation leaves out. Nine bands is the window of the synthetic table; a real EMIT scene has many more, which lowers the detection limit but changes none of the comparisons.

**Symbols**

| Symbol | Meaning | Shape | Unit |
|---|---|---|---|
| $B$ | number of bands | scalar | – |
| $\mathbf{y}$ | observed band radiance of one pixel | $\mathbb{R}^{B}$ | radiance |
| $\mathbf{y}_0$ | the same pixel without the plume | $\mathbb{R}^{B}$ | radiance |
| $\alpha$ | plume strength (column enhancement) | $\mathbb{R}_{\ge 0}$ | mol/m² |
| $\mathbf{t}$ | unit absorption spectrum, $t_b = \partial \ln y_b / \partial \alpha$ at 0; negative where the gas absorbs | $\mathbb{R}^{B}$ | m²/mol |
| $\mathbf{T}$ | $\mathrm{diag}(\mathbf{t})$ | $\mathbb{R}^{B \times B}$ | m²/mol |
| $\tau$ | $-\mathbf{1}^\top \mathbf{t}$, the unit optical depth summed over bands | $\mathbb{R}_{\ge 0}$ | m²/mol |
| $p_0$, $p_\alpha$ | density of plume-free and plume-on spectra | $\mathbb{R}^{B} \to \mathbb{R}_{\ge 0}$ | radiance⁻ᴮ |
| $\boldsymbol{\mu}_0$, $\boldsymbol{\Sigma}_0$ | mean and covariance of $p_0$ | $\mathbb{R}^{B}$, $\mathbb{R}^{B \times B}$ | radiance, radiance² |
| $\mathrm{LR}(\alpha; \mathbf{y})$ | likelihood ratio $p_\alpha(\mathbf{y}) / p_0(\mathbf{y})$ | scalar | – |
| $A$, $Q$, $E$ | Mahalanobis distance, slope and curvature of the log likelihood ratio | scalar | –, m²/mol, (m²/mol)² |
| $\alpha_0$ | characteristic plume strength | scalar | mol/m² |
| $D_\text{name}$ | a detector's statistic; larger means more plume-like | scalar | – |

The matched-filter page writes the pixel as $x$; here it is $\mathbf{y}$, the observation. Theiler writes the strength as $a$ and the absorption as a positive $T$; his $aT$ is our $-\alpha \mathbf{T}$, so every formula below is his with that sign flipped, and $\tau$, $Q$ and $E$ keep his meaning exactly.

## Part 1 · From Beer–Lambert to a likelihood ratio

In words: we first write what a plume does to a spectrum, then use that map to get the density of plume-on spectra from the density of plume-free ones. The only non-obvious ingredient is the Jacobian of the map, which is where the summed optical depth $\tau$ comes from. With both densities in hand, the Neyman–Pearson lemma says their ratio is the best detector, if we knew $\alpha$.

:::{prf:assumption} Beer–Lambert per band
:label: ad-a1
Let's assume the plume multiplies each band by its own transmittance, $e^{\alpha t_b}$, with the same unit absorption $t_b$ at every strength. This is exact per wavenumber, and holds per band when the band's lines are far from saturation, which is the weak-plume regime of methane in the 2.3 μm window.

$$
y_b = e^{\alpha t_b} y_{0,b}
$$

*Breaks when the band's strongest lines saturate (the curve of growth, see [the absorption-shape test](absorption_test.md)), or when path radiance or plume emission add light the plume does not attenuate; then the transmittance is a table, $E_b(\alpha)$, not an exponential.*
:::

### Step 1 · Write the plume as a map *(uses {prf:ref}`ad-a1`)*

Stacking the bands, the plume is a diagonal linear map on the plume-free spectrum, so we will have:

$$
\mathbf{y} = e^{\alpha \mathbf{T}} \mathbf{y}_0
$$ (eq-ad-map)

$$
\mathbf{y}_0 = e^{-\alpha \mathbf{T}} \mathbf{y}
$$

where $e^{\alpha \mathbf{T}}$ is the matrix exponential, which for a diagonal matrix is the diagonal of the elementwise exponentials. As seen above, the map is linear in $\mathbf{y}_0$ but not in $\alpha$, and that is the whole difficulty: for a fixed pixel the plume moves the spectrum along a curve, not a straight line, and the direction it moves in, $\mathbf{T}\mathbf{y}_0$, depends on the pixel.

:::{prf:assumption} The background is known
:label: ad-a2
Let's also assume that plume-free spectra are draws from a density $p_0$ that we know, in practice through a mean and covariance estimated from the scene. Plume pixels are rare, so they barely move those estimates.

$$
\mathbf{y}_0 \sim p_0
$$

*Breaks when plume pixels contaminate the estimates or the scene mixes surfaces with different statistics; then use robust means, clusters or local windows ([matched filter](matched_filter.md), §3).*
:::

### Step 2 · Change variables *(uses {prf:ref}`ad-a2`)*

If $\mathbf{y}_0$ has density $p_0$ and $\mathbf{y}$ is an invertible function of it, the density of $\mathbf{y}$ is $p_0$ at the preimage times the absolute Jacobian determinant of the inverse map, so we will have:

$$
p_\alpha(\mathbf{y}) = p_0(e^{-\alpha \mathbf{T}} \mathbf{y}) \left\vert \det \frac{\partial \mathbf{y}_0}{\partial \mathbf{y}} \right\vert
$$

$$
\det e^{-\alpha \mathbf{T}} = \prod_{b=1}^{B} e^{-\alpha t_b} = e^{-\alpha \mathrm{tr}(\mathbf{T})} = e^{\alpha \tau}
$$ (eq-ad-jacobian)

$$
p_\alpha(\mathbf{y}) = p_0(e^{-\alpha \mathbf{T}} \mathbf{y}) e^{\alpha \tau}
$$

The factor $e^{\alpha \tau}$ is easy to drop by accident, and Theiler credits Schaum with insisting on it. It has a plain meaning: the plume shrinks every axis of spectrum space by $e^{\alpha t_b} < 1$, so it packs the background's probability into a smaller volume, and the density goes up by the inverse of the volume ratio. Without it the likelihood ratio is biased against the plume hypothesis by exactly the amount the plume compresses the clutter. We will see on the [companion page](clutter_models.md) that the factor disappears in log space, where the plume is a translation and translations preserve volume.

### Step 3 · Take the ratio

With both densities, the ratio of plume-on to plume-free density at the observed spectrum is:

$$
\mathrm{LR}(\alpha; \mathbf{y}) = \frac{p_\alpha(\mathbf{y})}{p_0(\mathbf{y})} = \frac{p_0(e^{-\alpha \mathbf{T}} \mathbf{y}) e^{\alpha \tau}}{p_0(\mathbf{y})}
$$ (eq-ad-lr)

For a known $\alpha$ the Neyman–Pearson lemma makes this the most powerful test at every false-alarm rate. It is called the **clairvoyant** detector [^kay1998], because it knows the one thing we do not: the plume strength. It is still worth writing down, since every other detector on this page is an approximation to it, and in simulation (where we do know $\alpha$) it is the bound the others are measured against.

:::{prf:assumption} Gaussian clutter
:label: ad-a3
Lastly, for this page, we assume the background is Gaussian, $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$. It is the standard model, and the one under which the linear matched filter is optimal for an additive target.

$$
-2 \log p_0(\mathbf{y}) = A(\mathbf{y}) + \mathrm{const}
$$

$$
A(\mathbf{y}) = (\mathbf{y} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{y} - \boldsymbol{\mu}_0)
$$

*Breaks when the clutter has fat tails, which real hyperspectral scenes usually do; then use the multivariate t or the lognormal ([Beyond Gaussian clutter](clutter_models.md)).*
:::

### Step 4 · The clairvoyant detector for Gaussian clutter *(uses {prf:ref}`ad-a3`)*

Taking the logarithm of {eq}`eq-ad-lr` and inserting the Gaussian, the normalising constants cancel and we will have:

$$
\log \mathrm{LR}(\alpha; \mathbf{y}) = -\frac{1}{2} A(e^{-\alpha \mathbf{T}} \mathbf{y}) + \frac{1}{2} A(\mathbf{y}) + \alpha \tau
$$ (eq-ad-clairvoyant)

$A(\mathbf{y})$ is the Mahalanobis distance of the pixel from the background mean (the RX anomaly detector), so the clairvoyant detector asks how much closer to the background the pixel moves when the plume is "taken off", plus the volume term. It holds for every $\alpha > 0$, not just weak plumes.

:::{important} Result · the clairvoyant detector
{eq}`eq-ad-clairvoyant` is the optimal detector of an absorptive plume of known strength on Gaussian clutter. Every detector below approximates it for unknown $\alpha$.
:::

## Part 2 · Two ways around the unknown strength

In words: $\alpha$ appears inside $A(e^{-\alpha \mathbf{T}} \mathbf{y})$, so maximising over it is a transcendental problem with no closed form. Both papers handle this with a Taylor series in $\alpha$. Keeping only the first-order term gives the locally most powerful (LMP) detector, which is what "assume a weak plume" usually means. Keeping the second-order term as well lets us maximise over $\alpha$ in closed form, which is the GLRT, and the maximiser is an estimate of the plume strength we get for free.

:::{prf:assumption} Second order in the plume strength
:label: ad-a4
Let's assume $\alpha t_b$ is small enough in every band that the matrix exponential is well described to second order.

$$
e^{-\alpha \mathbf{T}} \mathbf{y} = \mathbf{y} - \alpha \mathbf{T} \mathbf{y} + \frac{\alpha^2}{2} \mathbf{T}^2 \mathbf{y} + O(\alpha^3)
$$

*Breaks when $\alpha \lvert t_b \rvert$ approaches one in the strongest band; in the running example the GLRT's estimate is still within 5 % at 25 detection limits, about 17 000 ppm·m.*
:::

### Step 5 · Expand the Mahalanobis distance *(uses {prf:ref}`ad-a4`)*

Write the perturbation as $\boldsymbol{\delta} = -\alpha \mathbf{T}\mathbf{y} + \frac{\alpha^2}{2} \mathbf{T}^2 \mathbf{y}$, expand the quadratic form $A(\mathbf{y} + \boldsymbol{\delta})$ and collect powers of $\alpha$, so we will have:

$$
A(e^{-\alpha \mathbf{T}} \mathbf{y}) = A(\mathbf{y}) - 2\alpha Q(\mathbf{y}) + \alpha^2 E(\mathbf{y}) + O(\alpha^3)
$$ (eq-ad-taylor)

$$
Q(\mathbf{y}) = (\mathbf{T}\mathbf{y})^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{y} - \boldsymbol{\mu}_0)
$$

$$
E(\mathbf{y}) = (\mathbf{T}\mathbf{y})^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{T}\mathbf{y}) + (\mathbf{T}^2 \mathbf{y})^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{y} - \boldsymbol{\mu}_0)
$$

The two terms of $E$ have different origins: the first is $\boldsymbol{\delta}^\top \boldsymbol{\Sigma}_0^{-1} \boldsymbol{\delta}$ at first order, and the second is the cross term of the second-order perturbation with the residual. That makes $E$ exactly a Newton curvature, and its first term alone the Gauss–Newton curvature, which is always positive. The full $E$ can go negative for a pixel far from $\boldsymbol{\mu}_0$, where the second-order series has no interior maximum; this never happened on the running example, but the code in {ref}`sec-ad-code` falls back to the Gauss–Newton term there, and on the running example the two give the same false-alarm rates to three digits. Inserting {eq}`eq-ad-taylor` into {eq}`eq-ad-clairvoyant`:

$$
2 \log \mathrm{LR}(\alpha; \mathbf{y}) = 2\alpha \left(Q(\mathbf{y}) + \tau\right) - \alpha^2 E(\mathbf{y}) + O(\alpha^3)
$$ (eq-ad-quadratic)

### Step 6 · First order: the quadratic matched filter

The locally most powerful detector keeps the slope of the log likelihood ratio at $\alpha = 0$. Since $\log \mathrm{LR}(0; \mathbf{y}) = 0$, we divide by $\alpha$ (a monotone rescaling for $\alpha > 0$, which does not change the detector) before taking the limit, so we will have:

$$
D_\text{QMF}(\mathbf{y}) = \lim_{\alpha \to 0} \frac{\log \mathrm{LR}(\alpha; \mathbf{y})}{\alpha} = Q(\mathbf{y}) + \tau
$$ (eq-ad-qmf)

This is a "matched filter" matched to $\mathbf{T}\mathbf{y}$, the pixel's own plume direction, rather than to a fixed target, which makes it quadratic in $\mathbf{y}$. The constant $\tau$ does not change it as a detector; it only makes its mean zero over background pixels. It is the result most people have in mind when they say the weak-plume limit justifies a linear filter, and it shows that it does not: even as $\alpha \to 0$, the optimal detector is quadratic. {ref}`sec-ad-amf` explains why it is still the worst detector here.

### Step 7 · Second order: the GLRT

The GLRT replaces the unknown $\alpha$ by its maximum-likelihood estimate and evaluates the likelihood ratio there. {eq}`eq-ad-quadratic` is a parabola in $\alpha$, of the form $2\alpha a - \alpha^2 b$, maximised at $a / b$ with maximum $a^2 / b$, so we will have:

$$
\hat{\alpha}(\mathbf{y}) = \frac{\max(0, Q(\mathbf{y}) + \tau)}{E(\mathbf{y})}
$$ (eq-ad-alpha-hat)

$$
D_\text{GLRT}(\mathbf{y}) = \frac{Q(\mathbf{y}) + \tau}{\sqrt{E(\mathbf{y})}}
$$ (eq-ad-glrt)

The $\max(0, \cdot)$ imposes $\alpha \ge 0$: an absorptive plume cannot brighten a pixel. The detector itself is the square root of the maximised $2 \log \mathrm{LR}$, signed, which is monotone in it, and Theiler drops the clipping in the detector, $(Q + \tau)/\sqrt{E}$, since it changes only negative values and no useful threshold is negative. In addition, $\hat{\alpha}$ is a plume-strength estimate in its own right, with a built-in albedo correction: numerator and denominator both scale with the pixel's brightness through $\mathbf{T}\mathbf{y}$, so a bright pixel is not read as a strong plume. Foote et al. [^foote2020] add that correction by hand to the matched filter, dividing its estimate by $r = \mathbf{y}^\top \boldsymbol{\mu}_0 / \boldsymbol{\mu}_0^\top \boldsymbol{\mu}_0$; here it falls out of the likelihood.

:::{important} Result · the GLRT and its plume strength
{eq}`eq-ad-glrt` detects and {eq}`eq-ad-alpha-hat` quantifies an absorptive plume of unknown strength on Gaussian clutter, to second order in $\alpha$, in closed form and per pixel. In the running example its estimate stays within 1 % of the true strength at 3–5 detection limits and within 5 % at 25, where the linear matched filter's is 26 % low.
:::

(sec-ad-amf)=
## Part 3 · Where the linear matched filter sits

In words: the linear matched filter needs an additive model, a fixed vector that the plume adds to every pixel. We get one by linearising Beer–Lambert and then replacing the pixel-dependent direction $\mathbf{T}\mathbf{y}_0$ by a constant. The choice of constant is the difference between a good and a bad matched filter. The same constant also defines the strength of a "just detectable" plume, and that strength is what explains why the LMP detector does so badly.

:::{prf:assumption} A constant plume direction
:label: ad-a5
Let's assume the plume's direction in spectrum space is the same for every pixel, the mean background's, $\mathbf{T}\mathbf{y}_0 \approx \mathbf{T}\boldsymbol{\mu}_0$. This is what turns the multiplicative model into an additive one.

$$
\mathbf{y} \approx \mathbf{y}_0 + \alpha \mathbf{T} \boldsymbol{\mu}_0
$$

*Breaks when the pixel's brightness differs a lot from the mean (bright roofs, dark water), where the matched filter reads the plume too strong or too weak; then use the GLRT's $\hat{\alpha}$ or an albedo-corrected estimate.*
:::

### Step 8 · The Tμ matched filter *(uses {prf:ref}`ad-a5`)*

With a constant target $\mathbf{T}\boldsymbol{\mu}_0 = \boldsymbol{\mu}_0 \odot \mathbf{t}$ on Gaussian clutter, the problem is the additive one of the [matched filter](matched_filter.md), so we will have:

$$
D_\text{AMF}(\mathbf{y}) = \frac{(\mathbf{T}\boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{y} - \boldsymbol{\mu}_0)}{(\mathbf{T}\boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{T}\boldsymbol{\mu}_0)}
$$ (eq-ad-amf)

Comparing with {eq}`eq-ad-qmf`, the QMF is this filter with $\boldsymbol{\mu}_0$ replaced by $\mathbf{y}$ in the target and no normalisation. Both papers also test the "naive" target $\mathbf{t}$ itself, which assumes a spectrally flat background; the Tμ target beat it in every case they tried, and in the running example it cuts the false-alarm rate at 80 % detection five- to six-fold for a plume at 5 $\alpha_0$ ({ref}`sec-ad-check`). In radtrax this is the `μ ⊙ t` option that [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target) documents: `apply_image(cube, mean, cov, mean * t)`.

### Step 9 · The characteristic plume strength

Under the background hypothesis the matched filter's estimate {eq}`eq-ad-amf` has mean zero and standard deviation $\alpha_0$, so a plume of strength $n \alpha_0$ is an $n$-sigma detection. Theiler takes $\alpha_0$ as the scale of a "just detectable" plume, so we will have:

$$
\alpha_0 = \frac{1}{\sqrt{(\mathbf{T}\boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\mathbf{T}\boldsymbol{\mu}_0)}} = \frac{1}{\left\Vert \boldsymbol{\Sigma}_0^{-1/2} \mathbf{T} \boldsymbol{\mu}_0 \right\Vert_2}
$$ (eq-ad-a0)

This is the inverse of the per-unit-strength SNR that [`matched_filter_snr`](xref:api#radtrax.retrieval.matched_filter.matched_filter_snr) computes, $\mathrm{SNR}(\alpha) = \alpha / \alpha_0$, with the target $\boldsymbol{\mu}_0 \odot \mathbf{t}$. The same scale also says how weak a detectable plume is in the raw data: at $\alpha_0$ the spectrum changes by a fraction $\alpha_0 \Vert \mathbf{T}\boldsymbol{\mu}_0 \Vert_2 / \Vert \boldsymbol{\mu}_0 \Vert_2$, which is $2 \times 10^{-4}$ in the running example and below $10^{-3}$ on all four of Theiler's real scenes. The matched filter's gain comes from whitening, from looking along the directions where the clutter is quiet.

### Step 10 · Why the "locally most powerful" detector fails

The QMF dropped the $\alpha^2 E$ term of {eq}`eq-ad-quadratic` on the grounds that $\alpha$ is small. However, small compared with what? Averaging $E$ over background pixels (the expectation of a quadratic form, $\mathbb{E}[\mathbf{u}^\top \mathbf{M} \mathbf{u}] = \mathbb{E}[\mathbf{u}]^\top \mathbf{M} \mathbb{E}[\mathbf{u}] + \mathrm{tr}(\mathbf{M} \boldsymbol{\Sigma}_0)$), we will have:

$$
\mathbb{E}_{p_0}[E(\mathbf{y})] = \frac{1}{\alpha_0^2} + \mathrm{tr}(\mathbf{T} \boldsymbol{\Sigma}_0^{-1} \mathbf{T} \boldsymbol{\Sigma}_0) + \mathrm{tr}(\mathbf{T}^2) \ge \frac{1}{\alpha_0^2}
$$

so the dropped term is at least $(\alpha / \alpha_0)^2$, the squared detection SNR. A plume weak enough for the first-order term to dominate is weaker than $\alpha_0$, which is to say undetectable. At a pixel that does carry a plume, $Q + \tau \approx \alpha E$, so the dropped term is half the kept one whatever the strength (0.50–0.53 in the running example from 3 to 25 detection limits). The LMP detector is optimal in exactly the regime where no detector works, which is Theiler's Appendix B. Theiler states the bound per pixel; it holds on average over background pixels, and at a typical pixel in the running example $E \alpha_0^2 = 1.12$.

:::{important} Result · the hierarchy of detectors
On Gaussian clutter: clairvoyant ≥ GLRT ≳ Tμ-AMF > t-AMF > QMF, in false alarms at a fixed detection rate. The linear matched filter is a good detector because it approximates the GLRT, not because the weak-plume limit makes it optimal; the strength estimate is where the GLRT pulls ahead.
:::

(sec-ad-check)=
## Check

**Units**

| Quantity | Units | |
|---|---|---|
| $\alpha t_b$, $\alpha \tau$ | mol/m² × m²/mol | ✓ dimensionless |
| $Q$ | (m²/mol × radiance) × radiance⁻² × radiance | m²/mol, like $\tau$ ✓ |
| $E$ | (m²/mol)² | ✓ |
| $\hat{\alpha} = (Q + \tau)/E$ | (m²/mol) / (m²/mol)² | ✓ mol/m² |
| $D_\text{GLRT} = (Q + \tau)/\sqrt{E}$ | (m²/mol) / (m²/mol) | ✓ dimensionless, an SNR |
| $\alpha_0$ | 1 / (m²/mol) | ✓ mol/m² |

**Numbers · the running example** (Gaussian albedo; lognormal brightness in parentheses)

| Quantity | Value |
|---|---|
| strongest band, $\min_b t_b$ | −0.19 m²/mol |
| $\tau$ | 0.54 m²/mol |
| $\alpha_0$ | 0.028 mol/m² ≈ 690 ppm·m (0.027 mol/m², 660 ppm·m) |
| fractional radiance change at $\alpha_0$ | 2.0 × 10⁻⁴ |
| $\mathbb{E}[E] \alpha_0^2$ over background pixels | 1.12 (1.13) ✓ ≥ 1 |
| median $\hat{\alpha}_\text{GLRT} / \alpha$ at 3, 5, 25 $\alpha_0$ | 0.99, 0.99, 0.95 |
| median $\hat{\alpha}_\text{AMF} / \alpha$ at 3, 5, 25 $\alpha_0$ | 0.96, 0.93, 0.74 (0.93, 0.90, 0.71) |

**False-alarm rate at 80 % detection, plume at 5 $\alpha_0$**

| Detector | Gaussian albedo | Lognormal brightness |
|---|---|---|
| clairvoyant {eq}`eq-ad-clairvoyant` | 7.2 × 10⁻⁴ | 6.2 × 10⁻⁴ |
| GLRT {eq}`eq-ad-glrt` | 1.1 × 10⁻³ | 1.0 × 10⁻³ |
| Tμ-AMF {eq}`eq-ad-amf` | 1.2 × 10⁻³ | 1.0 × 10⁻³ |
| t-AMF | 6.3 × 10⁻³ | 6.5 × 10⁻³ |
| QMF {eq}`eq-ad-qmf` | 2.7 × 10⁻² | 2.6 × 10⁻² |

Each rate is over 200 000 background pixels, so the smallest is 120–140 false alarms and good to roughly ±10 %. The Tμ target beating $\mathbf{t}$ and the QMF in last place match Theiler throughout. One difference: on his simulated Gaussian backgrounds (his Table VI) the GLRT is substantially better than the Tμ filter, whereas here they tie, as they do on his real AVIRIS-NG scene (his Fig. 1a). The synthetic scene has nine bands and mostly brightness clutter, so it leaves the GLRT little to gain as a detector; its strength estimate is where it still pulls ahead.

**Limits.** At $\alpha = 0$, {eq}`eq-ad-clairvoyant` is zero, as a log likelihood ratio must be. As $\boldsymbol{\mu}_0$ becomes spectrally flat, $\mathbf{T}\boldsymbol{\mu}_0 \propto \mathbf{t}$ and the Tμ and t filters coincide, which is when the naive target is fine. If the pixel equals the mean, $Q = 0$ and $\hat{\alpha} = \tau / E$, a small positive bias from the volume term that the clipping at zero then dominates.

(sec-ad-code)=
## Code

The Tμ matched filter is one call; the GLRT is a handful of lines on the same operators. The snippet below ran against radtrax `main`.

```{code-block} python
import gaussx as gx
import jax
import jax.numpy as jnp

from radtrax.retrieval.matched_filter import apply_image, estimate_cov_shrunk, estimate_mean

# μ₀, Σ₀ from the scene;  t = ∂ ln y / ∂α at 0 from unit_absorption_target
mean = estimate_mean(cube)                                   # (B,)
cov = estimate_cov_shrunk(cube, mean)                        # lineax operator, (B, B)

# α̂_AMF = (Tμ)ᵀΣ⁻¹(y − μ) / (Tμ)ᵀΣ⁻¹(Tμ)
alpha_amf = apply_image(cube, mean, cov, mean * t)           # (H, W)  [mol/m²]


def glrt_image(cube, mean, cov_op, t):
    # τ = −1ᵀt
    tau = -jnp.sum(t)                                                    # ()
    # r = y − μ;  Ty = t ⊙ y
    r = cube - mean                                                      # (H, W, B)
    ty = t * cube                                                        # (H, W, B)
    # Σ⁻¹(Ty), one structured solve per pixel
    s_ty = jax.vmap(jax.vmap(lambda v: gx.solve(cov_op, v)))(ty)         # (H, W, B)
    # Q = (Ty)ᵀΣ⁻¹(y − μ)
    q = jnp.einsum("hwb,hwb->hw", s_ty, r)                               # (H, W)
    # E_GN = (Ty)ᵀΣ⁻¹(Ty);  E = E_GN + (T²y)ᵀΣ⁻¹(y − μ)
    e_gn = jnp.einsum("hwb,hwb->hw", s_ty, ty)                          # (H, W)
    e = e_gn + jnp.einsum("hwb,hwb->hw", s_ty * t, r)                    # (H, W)
    # far from μ the Newton curvature can turn negative: fall back to Gauss–Newton
    e = jnp.where(e > 0, e, e_gn)                                        # (H, W)
    # α̂ = max(0, Q + τ) / E;  D_GLRT = (Q + τ) / √E
    alpha_hat = jnp.maximum(q + tau, 0.0) / e                            # (H, W)
    return (q + tau) / jnp.sqrt(e), alpha_hat
```

`apply_image` is {eq}`eq-ad-amf` (Step 8); `glrt_image` is {eq}`eq-ad-taylor`, {eq}`eq-ad-alpha-hat` and {eq}`eq-ad-glrt` (Steps 5 and 7). The double `vmap` solves per pixel, which keeps the sketch structure-agnostic through `gx.solve`; for a dense or low-rank $\boldsymbol{\Sigma}_0$ it is cheaper to factorise once and solve the whole cube as one block of right-hand sides, which is what a library version would do.

## When the assumptions break

| Assumption | Fails when | Then use |
|---|---|---|
| {prf:ref}`ad-a1` Beer–Lambert per band | lines saturate within a band; path radiance; emission | the exact band transmittance [`transmittance_table`](xref:api#radtrax.retrieval.absorption.transmittance_table) in the clairvoyant detector, or a finite-amplitude target, [`nonlinear_target_from_obs`](xref:api#radtrax.retrieval.matched_filter.nonlinear_target_from_obs) |
| {prf:ref}`ad-a2` known background | plume contamination; mixed surfaces | robust means, GMM clusters, adaptive windows ([matched filter](matched_filter.md), §3) |
| {prf:ref}`ad-a3` Gaussian clutter | fat tails, multiplicative albedo variation | the multivariate t, ACE, the lognormal matched filter ([Beyond Gaussian clutter](clutter_models.md)) |
| {prf:ref}`ad-a4` second order in $\alpha$ | $\alpha \lvert t_b \rvert$ near one | maximise {eq}`eq-ad-clairvoyant` over $\alpha$ numerically (a scalar problem per pixel) |
| {prf:ref}`ad-a5` constant plume direction | strong brightness variation | the GLRT's $\hat{\alpha}$, or an albedo-corrected matched filter |

*Draft 1, November 2026. Running-example numbers are from a synthetic 9-band scene built with radtrax; they illustrate the comparisons, not EMIT's detection limit. Next: [Beyond Gaussian clutter](clutter_models.md).*
