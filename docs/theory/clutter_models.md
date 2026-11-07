---
title: "Beyond Gaussian clutter: the multivariate t, ACE and the lognormal matched filter"
---

# Beyond Gaussian clutter

:::{important} Status: theory, partly in code
The log-space matched filter needs no new code: [`apply_image`](xref:api#radtrax.retrieval.matched_filter.apply_image) on $\ln \vect{y}$ with the target $\vect{t}$, and the mean and covariance estimated on the same $\ln \vect{y}$, as [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target) documents. The elliptically contoured and ACE variants, the second-order GLRT on t clutter and the multi-gas detector are recipes on this page; none is a function yet.
:::

The [absorptive-detection page](absorptive_detection.md) derived the detectors of an absorptive plume on Gaussian clutter. Real hyperspectral clutter is not Gaussian: it has fat tails, and its spread is mostly multiplicative, since a surface's albedo scales the whole spectrum. This page keeps the plume physics and changes the background, in two directions. Theiler [^theiler2021] replaces the Gaussian with the multivariate t, an elliptically contoured distribution with fat tails, and gets the adaptive cosine estimator (ACE) as a limit. Schaum [^schaum2021] replaces it with the lognormal, which turns the problem into an additive one in log space, where the matched filter is uniformly most powerful. We derive both, fix one coefficient in the t-distributed GLRT, and say what each optimality claim needs, because on real data neither model is right and the papers' own experiments show it.

[^theiler2021]: Theiler, J. (2021). Absorptive weak plume detection on Gaussian and non-Gaussian background clutter. *IEEE JSTARS*, 14, 6842–6854. doi:10.1109/JSTARS.2021.3093820 (CC-BY 4.0).

[^schaum2021]: Schaum, A. (2021). A uniformly most powerful detector of gas plumes against a cluttered background. *Remote Sensing of Environment*, 260, 112443. doi:10.1016/j.rse.2021.112443

[^theilerfoy2008]: Theiler, J. & Foy, B. R. (2008). EC-GLRT: detecting weak plumes in non-Gaussian hyperspectral clutter using an elliptically-contoured generalized likelihood ratio test. *Proc. IEEE IGARSS 2008*, I-221–I-224.

[^kraut2005]: Kraut, S., Scharf, L. L. & Butler, R. W. (2005). The adaptive coherence estimator: a uniformly most-powerful-invariant adaptive detection statistic. *IEEE Trans. Signal Process.*, 53(2), 427–438.

[^marden2004]: Marden, D. B. & Manolakis, D. (2004). Using elliptically contoured distributions to model hyperspectral imaging data and generate statistically similar synthetic data. *Proc. SPIE*, 5425, 558–572.

[^lehmann2005]: Lehmann, E. L. & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3rd ed.). Springer.

## TL;DR

On multivariate-t clutter with $\nu$ degrees of freedom, every Gaussian detector picks up a "fattening" factor that discounts pixels far from the background mean, and the $\nu \to 2$ limit of the matched filter is ACE:

$$
F_\nu^2(\vect{y}) = \frac{\nu - 1}{\nu - 2 + A(\vect{y})}
$$

$$
D_\text{ACE}(\vect{y}) = \frac{(\mat{T}\gvect{\mu}_y)^\top \gmat{\Sigma}_y^{-1} (\vect{y} - \gvect{\mu}_y)}{\sqrt{A(\vect{y})}}
$$

On lognormal clutter, $\ln \vect{y} = \ln \vect{y}_\text{bg} + \alpha \vect{t}$ is an additive target on a Gaussian, so the matched filter on log radiance is uniformly most powerful, one test for every plume strength:

$$
D_\text{LMF}(\vect{y}) = \frac{\vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} (\ln \vect{y} - \gvect{\mu}_{\ln y})}{\vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} \vect{t}}
$$

The assumptions are 1) Beer–Lambert per band and a background estimated from the scene, as before, 2) multivariate-t clutter for the first family, 3) lognormal clutter for the second, and 4) for several gases, a known spectrum per gas. Practically speaking, which detector wins is decided by the background model, not the plume model. On Theiler's real scenes the ACE and EC variants beat their Gaussian counterparts by large factors, and the simple Tμ-ACE was competitive with the best GLRT. Schaum's UMP property, on the other hand, needs a log-linearity that band integration and additive sensor noise break, so in our running example the log matched filter loses to the plain Tμ filter even on a lognormal-brightness scene.

:::{tip} ELI5 · why do fat tails and logarithms change the detector?
A Gaussian background says pixels far from average are **vanishingly rare**, so any big departure looks like a plume. Real scenes have plenty of odd pixels (roofs, water, shadows), so the fat-tailed detector **discounts** distance from average and asks only whether the departure points the plume's way; that is the cosine in "adaptive cosine estimator". Logarithms help for a different reason. Sunglasses multiply light, and the logarithm turns multiplying into **adding**, so in log space a plume is a fixed shift and the textbook detector for a fixed shift applies exactly.
:::

**Running example.** The scene of the [absorptive-detection page](absorptive_detection.md): nine EMIT-like bands at 2.3 μm, 200 000 background pixels with mean albedo 0.25, SNR 300, and two clutter models, a Gaussian albedo and a lognormal brightness (35 % spread in both). Its numbers below are from radtrax; the real-data results are Theiler's.

**Symbols** (in addition to those of the [absorptive-detection page](absorptive_detection.md))

| Symbol | Meaning | Shape | Unit |
|---|---|---|---|
| $\nu$ | degrees of freedom of the multivariate t | $\mathbb{R}_{>2}$ | none |
| $F_\nu(\vect{y})$ | fattening factor | scalar | none |
| $\Theta_\nu$ | $(\nu - 1)\tau / (\nu + B)$ | scalar | m²/mol |
| $\gvect{\mu}_{\ln y}$, $\gmat{\Sigma}_{\ln y}$ | mean and covariance of $\ln \vect{y}_\text{bg}$ | $\mathbb{R}^{B}$, $\mathbb{R}^{B \times B}$ | none |
| $G$ | number of gases | scalar | none |
| $\vect{t}_g$ | unit absorption spectrum of gas $g$ | $\mathbb{R}^{B}$ | m²/mol |
| $\gvect{\alpha}$ | column enhancements of the $G$ gases | $\mathbb{R}^{G}_{\ge 0}$ | mol/m² |
| $\mat{S}$ | whitened targets, $\gmat{\Sigma}_{\ln y}^{-1/2} [\vect{t}_1, \ldots, \vect{t}_G]$ | $\mathbb{R}^{B \times G}$ | m²/mol |
| $\vect{w}$ | whitened log residual, $\gmat{\Sigma}_{\ln y}^{-1/2} (\ln \vect{y} - \gvect{\mu}_{\ln y})$ | $\mathbb{R}^{B}$ | none |

Logarithms of radiance carry the radiance unit only as an additive constant, which the mean $\gvect{\mu}_{\ln y}$ absorbs, so everything in log space is dimensionless.

## Part 1 · Fat tails: the multivariate t and ACE

In words: we swap the Gaussian for a multivariate t that has the same mean and covariance but fatter tails. Its density still depends on the pixel only through the Mahalanobis distance $A(\vect{y})$, so the likelihood ratio of the plume problem keeps the same ingredients, $A$, $Q$ and $E$, and only the way they are combined changes. The linear filter picks up a factor; the GLRT picks up the same factor and an extra constant; and letting the tails get as fat as they can ($\nu \to 2$) gives ACE.

:::{prf:assumption} Multivariate-t clutter
:label: cm-a1
Let's assume plume-free spectra follow an elliptically contoured multivariate t with mean $\gvect{\mu}_y$, covariance $\gmat{\Sigma}_y$ and $\nu > 2$ degrees of freedom, which fits hyperspectral clutter better than a Gaussian [^marden2004]. $\nu$ is estimated from the scene by an exceedance plot or a moment method; $\nu \to \infty$ is the Gaussian.

$$
p_0(\vect{y}) \propto \left[\nu - 2 + A(\vect{y})\right]^{-(B + \nu)/2}
$$

*Breaks when the clutter is not elliptical, for example when it is mostly brightness variation (a ray, not an ellipsoid), or a mixture of distinct surfaces; then cluster first or model the brightness in log space (Part 2).*
:::

### Step 1 · The fattened matched filter *(uses {prf:ref}`cm-a1`)*

For an additive target on t clutter, Theiler & Foy [^theilerfoy2008] show the GLRT is the Gaussian matched filter scaled by a factor that shrinks with the Mahalanobis distance, so we will have:

$$
F_\nu(\vect{y}) = \sqrt{\frac{\nu - 1}{\nu - 2 + A(\vect{y})}}
$$ (eq-cm-fattening)

$$
D_\text{EC}(\vect{y}) = F_\nu(\vect{y}) D_\text{AMF}(\vect{y})
$$

$$
D_\text{ACE}(\vect{y}) = \lim_{\nu \to 2} D_\text{EC}(\vect{y}) = \frac{D_\text{AMF}(\vect{y})}{\sqrt{A(\vect{y})}}
$$ (eq-cm-ace)

As $\nu \to \infty$, $F_\nu \to 1$ and the matched filter is back. At the other end, ACE divides the matched-filter projection by the pixel's whitened length, which makes it the cosine of the angle between the whitened residual and the whitened target; it is usually derived on its own terms as a uniformly most powerful invariant test [^kraut2005]. On real scenes $\nu$ is small next to $B$, so $F_\nu \approx F_2$ and the EC and ACE detectors are nearly the same, which is what Theiler found.

### Step 2 · The absorptive likelihood ratio on t clutter

The change of variables of the [absorptive-detection page](absorptive_detection.md) does not depend on the background model, so the plume-on density is again $p_0(e^{-\alpha \mat{T}} \vect{y}) e^{\alpha \tau}$. Raising the likelihood ratio to the power $-2/(\nu + B)$ (monotone, so the detector does not change), we will have:

$$
\mathrm{LR}(\alpha; \vect{y})^{-2/(\nu + B)} = \left(1 + \frac{F_\nu^2(\vect{y})}{\nu - 1} \left[A(e^{-\alpha \mat{T}} \vect{y}) - A(\vect{y})\right]\right) e^{-2\alpha\tau/(\nu + B)}
$$ (eq-cm-lr)

For known $\alpha$ the exponential is a constant, so the clairvoyant detector on t clutter is $F_\nu^2(\vect{y}) [A(\vect{y}) - A(e^{-\alpha \mat{T}} \vect{y})]$: the Gaussian one without the volume term, fattened.

### Step 3 · Second order in the strength

Inserting the expansion $A(e^{-\alpha \mat{T}} \vect{y}) - A(\vect{y}) = -2\alpha Q + \alpha^2 E$ and $e^{-\alpha c} = 1 - \alpha c + \alpha^2 c^2 / 2$ with $c = 2\Theta_\nu / (\nu - 1)$, the rescaled statistic $(\nu - 1)(1 - \mathrm{LR}^{-2/(\nu + B)})$ is a parabola in $\alpha$, so we will have:

$$
(\nu - 1)\left(1 - \mathrm{LR}^{-2/(\nu + B)}\right) = 2\alpha a_\nu(\vect{y}) - \alpha^2 b_\nu(\vect{y}) + O(\alpha^3)
$$

$$
a_\nu(\vect{y}) = F_\nu^2(\vect{y}) Q(\vect{y}) + \Theta_\nu
$$

$$
b_\nu(\vect{y}) = F_\nu^2(\vect{y}) E(\vect{y}) + \frac{4 F_\nu^2(\vect{y}) Q(\vect{y}) \Theta_\nu + 2\Theta_\nu^2}{\nu - 1}
$$ (eq-cm-coeffs)

So the LMP detector is $F_\nu^2 Q$ (the QMF, fattened; its ACE limit is $Q / A$), the strength estimate is $\hat{\alpha} = a_\nu / b_\nu$ and the GLRT is $a_\nu / \sqrt{b_\nu}$. As $\nu \to \infty$, $F_\nu \to 1$, $\Theta_\nu \to \tau$ and the last fraction vanishes, which gives back the Gaussian $(Q + \tau)/\sqrt{E}$.

:::{dropdown} A correction to Theiler (2021), eqs. 31–38
As printed, Theiler's eq. 31 has the cross term $4 Q \Theta_\nu$ without the factor $F_\nu^2$, and eqs. 33, 36, 37 and 38 inherit it. The term comes from multiplying the linear part of the bracket in {eq}`eq-cm-lr`, $-2\alpha F_\nu^2 Q / (\nu - 1)$, by the linear part of the exponential, $-\alpha c$, so the factor $F_\nu^2$ belongs to it:

$$
\frac{-2\alpha F_\nu^2 Q}{\nu - 1} \cdot \left(-\frac{2\alpha \Theta_\nu}{\nu - 1}\right) = \frac{4\alpha^2 F_\nu^2 Q \Theta_\nu}{(\nu - 1)^2}
$$

A numerical second derivative of the exact statistic agrees with {eq}`eq-cm-coeffs` to seven digits and not with the printed form (12 bands, $\nu = 5$: 3156.298 against 3274.950). The Gaussian limit is unaffected, since the term vanishes there, but the ACE limit (his eq. 38) should read $\sqrt{E/A + 4Q\Theta_2/A + 2\Theta_2^2}$ in the denominator, with $\Theta_2 = \tau / (B + 2)$.
:::

:::{important} Result · detectors on t clutter
The linear filter becomes $F_\nu D_\text{AMF}$ ({eq}`eq-cm-fattening`), and ACE in the fattest limit ({eq}`eq-cm-ace`); the absorptive GLRT becomes $a_\nu / \sqrt{b_\nu}$ with {eq}`eq-cm-coeffs`. All of them need only $A$, $Q$ and $E$, the same three quadratic forms as on Gaussian clutter.
:::

## Part 2 · Multiplicative clutter: the lognormal and the log matched filter

In words: a Lambertian pixel's radiance is its albedo times what the sun and the atmosphere deliver, so most of the clutter is a multiplication, and so is the plume. Schaum's move is to model the background as lognormal, which makes the logarithm of a plume-free spectrum Gaussian. The plume then becomes a constant shift in log space, which is the additive problem the matched filter solves, and solves for every strength at once. We also see that the volume factor $e^{\alpha \tau}$ disappears, and why.

:::{prf:assumption} Lognormal clutter
:label: cm-a2
Let's assume the logarithm of a plume-free spectrum is Gaussian. Radiance is positive, and a brightness change adds a constant to every band of $\ln \vect{y}_\text{bg}$, so the model has the right support and the right kind of variation; where the clutter's spread is small next to its mean, the lognormal and the Gaussian are nearly the same distribution.

$$
\ln \vect{y}_\text{bg} \sim \mathcal{N}(\gvect{\mu}_{\ln y}, \gmat{\Sigma}_{\ln y})
$$

*Breaks when additive sensor noise matters, for dark pixels or weak bands, since noise adds to $\vect{y}$ and not to $\ln \vect{y}$, and for radiances at or below zero, which have no logarithm; then use the radiance-space detectors.*
:::

### Step 4 · The plume is a shift in log space *(uses {prf:ref}`cm-a2`)*

Taking the logarithm of the Beer–Lambert map $\vect{y} = e^{\alpha \mat{T}} \vect{y}_\text{bg}$ elementwise, we will have:

$$
\ln \vect{y} = \ln \vect{y}_\text{bg} + \alpha \vect{t}
$$ (eq-cm-logshift)

This is the additive target model, exactly, with target $\vect{t}$ and Gaussian clutter. The plume map in log space is a translation, whose Jacobian determinant is one, which is why the volume factor $e^{\alpha \tau}$ of the radiance-space derivation is gone. That factor was the volume change of a multiplication. The logarithm has turned it into the volume change of the transform itself, $\prod_b 1/y_b$, which is the same under both hypotheses and so cancels in the ratio. Schaum makes the same point the other way round: the likelihood ratio is invariant under any invertible transform applied to both hypotheses, so we may compute it wherever it is easiest.

### Step 5 · One test for every strength

For a Gaussian with an additive target, the log likelihood ratio is linear in one statistic, so we will have:

$$
\log \mathrm{LR}(\alpha; \vect{y}) = \alpha \vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} (\ln \vect{y} - \gvect{\mu}_{\ln y}) - \frac{\alpha^2}{2} \vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} \vect{t}
$$

$$
D_\text{LMF}(\vect{y}) = \frac{\vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} (\ln \vect{y} - \gvect{\mu}_{\ln y})}{\vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} \vect{t}}
$$ (eq-cm-lmf)

For every $\alpha > 0$ the likelihood ratio is increasing in $D_\text{LMF}$, so thresholding $D_\text{LMF}$ is the Neyman–Pearson test for every $\alpha$ at once. That is the definition of uniformly most powerful (the Karlin–Rubin theorem for a monotone likelihood ratio [^lehmann2005]), and it is the lognormal matched filter (LMF) of Schaum's eq. 20. It is also the maximum-likelihood estimate of $\alpha$, and its null distribution, $\mathcal{N}(0, 1 / \vect{t}^\top \gmat{\Sigma}_{\ln y}^{-1} \vect{t})$, does not depend on the background's parameters, so a threshold set for one false-alarm rate keeps it (the CFAR property). Note the sign: Schaum's target $A$ is positive with $\ln \vect{y} = \ln \vect{y}_\text{bg} - c\vect{A}$; radtrax's $\vect{t}$ is negative, so his $-\vect{A}^\top$ is our $\vect{t}^\top$.

### Step 6 · Estimate the background in log space

The lognormal's parameters are the mean and covariance of $\ln \vect{y}_\text{bg}$, and since $\ln \vect{y}_\text{bg}$ is Gaussian their maximum-likelihood estimates are the sample mean and covariance of the log pixels. If only the radiance-space moments are at hand, the two sets are related by (Schaum's eq. 15, with the exponential elementwise):

$$
\gvect{\mu}_y = \exp\left(\gvect{\mu}_{\ln y} + \frac{1}{2} \op{diag}(\gmat{\Sigma}_{\ln y})\right)
$$

$$
\gmat{\Sigma}_y = \gvect{\mu}_y \gvect{\mu}_y^\top \odot \left(\exp(\gmat{\Sigma}_{\ln y}) - \vect{1}\vect{1}^\top\right)
$$

where $\op{diag}(\cdot)$ here takes the diagonal of a matrix as a vector. The sample moments of radiance are the Gaussian estimates, not the lognormal ones, so estimating on $\ln \vect{y}$ directly is both simpler and right; in radtrax it is `estimate_mean(np.log(cube))` and `estimate_cov_shrunk(np.log(cube), mean_ln)`. The same fattening carries over: log-EC and log-ACE are {eq}`eq-cm-fattening` and {eq}`eq-cm-ace` with $\ln \vect{y}$, $\gvect{\mu}_{\ln y}$, $\gmat{\Sigma}_{\ln y}$ and the target $\vect{t}$, though they lose the UMP property.

:::{important} Result · the lognormal matched filter
{eq}`eq-cm-lmf` is the uniformly most powerful detector of an absorptive plume on lognormal clutter, for every plume strength, under exact per-band Beer–Lambert. It is the existing matched filter applied to $\ln \vect{y}$ with the target $\vect{t}$.
:::

### Step 7 · What "for every strength" needs

The UMP claim is exact under its model, and the model has two parts that real data break, so we need to be careful about where it holds. First, log-linearity: {eq}`eq-cm-logshift` needs $\ln y_b$ linear in $\alpha$, which holds per wavenumber, whereas after band integration $\ln E_b(\alpha)$ is convex (the cumulant generating function argument on [the absorption-shape test page](absorption_test.md)). The target then bends with strength, so the log filter is no longer the same test at every $\alpha$, and its strength estimate reads low for strong plumes, which is precisely the regime the UMP property was meant to cover. Second, the clutter: sensor noise is additive in radiance, so even a background whose brightness is exactly lognormal is not lognormal once noise is added, and the log transform inflates the noise of dark pixels and weak bands.

In the running example, which has a lognormal brightness and SNR 300, the LMF ties the Tμ filter for a plume at 3 detection limits (false-alarm rate at 80 % detection $4.5 \times 10^{-2}$ against $4.6 \times 10^{-2}$) and loses at 5 ($3.5 \times 10^{-3}$ against $1.0 \times 10^{-3}$). On the Gaussian-albedo scene it loses at both ($1.0 \times 10^{-2}$ against $1.2 \times 10^{-3}$ at 5). Theiler's simulations show the same weakness: on Gaussian backgrounds his log detectors had false-alarm rates hundreds of times those of the GLRT, and they were poor on t backgrounds too. However, on his four real scenes log-ACE and log-EC were among the best detectors, beating even the GLRT-ACE in three of them. So the honest summary is that log space is a good place to put multiplicative clutter, and that its optimality is a property of a model, not of data.

## Part 3 · Several gases at once

In words: in log space a mixture of gases is a sum of shifts, one per gas, so detecting "any of these gases" is detecting a target in a subspace with nonnegative coefficients. Schaum solves it with an oblique projection followed by a nonnegativity fix, and the coefficients are estimates of each gas's column for free.

:::{prf:assumption} Known gas spectra
:label: cm-a3
Lastly, we assume each gas $g$ has a known unit absorption spectrum $\vect{t}_g$ and acts independently (Beer–Lambert multiplies, so log-space shifts add).

$$
\ln \vect{y} = \ln \vect{y}_\text{bg} + \sum_{g=1}^{G} \alpha_g \vect{t}_g
$$

*Breaks when the gases' spectra are nearly collinear in the whitened space (the coefficients become unidentifiable) or overlap in saturated lines; then fix one gas or use a full retrieval ([3D-Var](var3d.md)).*
:::

### Step 8 · Whiten and fit with nonnegative coefficients *(uses {prf:ref}`cm-a3`)*

Whitening the log residual makes the clutter white, and the GLRT over $\gvect{\alpha} \ge 0$ becomes a nonnegative least-squares fit, so we will have:

$$
\hat{\gvect{\alpha}} = \arg\min_{\gvect{\alpha} \ge \vect{0}} \Vert \vect{w} - \mat{S}\gvect{\alpha} \Vert_2^2
$$ (eq-cm-nnls)

$$
2 \log \mathrm{LR}(\hat{\gvect{\alpha}}; \vect{y}) = \Vert \vect{w} \Vert_2^2 - \Vert \vect{w} - \mat{S}\hat{\gvect{\alpha}} \Vert_2^2
$$

Without the constraint the minimiser is $(\mat{S}^\top \mat{S})^{-1} \mat{S}^\top \vect{w}$, the coefficients of the projection of $\vect{w}$ onto the span of the targets in the (generally non-orthogonal) target basis; Schaum's eq. 26 prints it as $(\mat{S}\mat{S}^\top)^{-1} \mat{S}^\top$, which is a dimension slip, since $\mat{S}$ is $B \times G$. For two gases Schaum enumerates the cases: both coefficients positive, keep them; both negative, return zero; opposite signs, set the negative one to zero and refit the other alone, clipping it at zero. For more gases it is a small NNLS problem per pixel. The coefficients $\hat{\alpha}_g$ are column estimates, in the units of $\vect{t}_g$, without a radiative-transfer inversion.

:::{important} Result · multi-gas detection
{eq}`eq-cm-nnls` detects a nonnegative mixture of known gases on lognormal clutter and estimates each column, by nonnegative least squares in whitened log space.
:::

## What the papers found

Theiler implants plumes into four real scenes (SO₂ and NO₂ on OMI, CH₄ on two AVIRIS-NG scenes) and into simulated Gaussian, t and lognormal backgrounds with the same mean and covariance. In our own words, his conclusions are:

1. Detectors differ by orders of magnitude in false alarms, and the biggest cause is how well the background model fits the scene.
2. The clairvoyant detector knows the plume strength, not the background, so it can lose on a background it does not model.
3. On Gaussian clutter the AMF variants win, but only narrowly over EC and ACE; on non-Gaussian clutter, including all the real scenes, the EC and ACE variants win, often by large factors.
4. EC is barely better than ACE, because the fitted $\nu$ is small next to the number of bands.
5. The log-AMF is good only on lognormal clutter; log-EC and log-ACE do poorly on simulated Gaussian and t clutter but well on the real scenes.
6. The Tμ-ACE, one line of algebra, is often competitive with the best GLRT, and the Tμ targets beat the t targets almost everywhere.
7. The LMP quadratic filters are not recommended.

Our running example has elliptical-looking but really brightness-dominated clutter, so ACE gains nothing there (false-alarm rate $6.5 \times 10^{-3}$ against the Tμ filter's $1.2 \times 10^{-3}$ at 5 detection limits): fat tails along the brightness direction are not the fat tails ACE expects. Which family wins on EMIT scenes is an empirical question radtrax can answer with the detectors on these two pages.

## Check

**Limits.** As $\nu \to \infty$: $F_\nu \to 1$, $\Theta_\nu \to \tau$ and {eq}`eq-cm-coeffs` gives the Gaussian GLRT. For $G = 1$, {eq}`eq-cm-nnls` gives $\hat{\alpha} = \max(0, \mat{S}^\top \vect{w}) / \mat{S}^\top \mat{S}$, which is {eq}`eq-cm-lmf` clipped at zero. For a narrow lognormal ($\gmat{\Sigma}_{\ln y}$ small), $\ln \vect{y} - \gvect{\mu}_{\ln y} \approx (\vect{y} - \gvect{\mu}_y) \oslash \gvect{\mu}_y$ and $\gmat{\Sigma}_{\ln y} \approx \gmat{\Sigma}_y \oslash \gvect{\mu}_y \gvect{\mu}_y^\top$, so the LMF becomes the Tμ filter: they differ only where the clutter's spread is a sizeable fraction of its mean.

**Numbers.** The moment relations of Step 6 match a 400 000-sample lognormal to 2 × 10⁻⁴ in the mean and 4 × 10⁻³ in the covariance (relative), and the coefficient check is in the dropdown of Step 3.

## What radtrax computes

| Detector | Equation | In radtrax |
|---|---|---|
| Lognormal matched filter | {eq}`eq-cm-lmf` | `apply_image(np.log(cube), mean_ln, cov_ln, t)` with `mean_ln`, `cov_ln` from [`estimate_mean`](xref:api#radtrax.retrieval.matched_filter.estimate_mean) and [`estimate_cov_shrunk`](xref:api#radtrax.retrieval.matched_filter.estimate_cov_shrunk) on `np.log(cube)` |
| EC and ACE matched filters | {eq}`eq-cm-fattening`, {eq}`eq-cm-ace` | not yet: one extra solve for $A(\vect{y})$ |
| EC and ACE GLRT | {eq}`eq-cm-coeffs` | not yet: the GLRT sketch of the [absorptive-detection page](absorptive_detection.md) plus $A$ and $\nu$ |
| Multi-gas NNLS | {eq}`eq-cm-nnls` | not yet |

## When the assumptions break

| Assumption | Fails when | Then use |
|---|---|---|
| {prf:ref}`cm-a1` multivariate t | clutter is a ray (brightness) or a mixture | clusters ([matched filter](matched_filter.md), §3), or log space |
| {prf:ref}`cm-a2` lognormal | additive noise at dark pixels; nonpositive radiance; band-integrated saturation | the radiance-space GLRT or Tμ filter; the exact transmittance for strong plumes |
| {prf:ref}`cm-a3` known, separable gas spectra | collinear or saturated spectra | fix a gas, or a joint retrieval ([3D-Var](var3d.md)) |

*Draft 1, November 2026. Running-example numbers are from a synthetic 9-band scene built with radtrax; real-data findings are Theiler's (2021), paraphrased.*
