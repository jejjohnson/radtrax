---
title: "Matched filter: derivation, covariance flavours and the link to 3D-Var"
---

# Matched filter for hyperspectral methane retrieval

*Ported from plumax 0.1.2 (`28f257e`); module and link paths updated for radtrax.*

The matched filter (MF) is the maximum-SNR linear detector for a known target signature in additive Gaussian noise. In hyperspectral methane retrieval it is the workhorse baseline: it is fast, it has a closed form, and it is surprisingly competitive with full variational retrievals when the signal is weak and the scene is statistically homogeneous. This note derives the MF from scratch, lays out the four covariance flavours implemented in [radtrax.retrieval.matched_filter](https://github.com/jejjohnson/radtrax/tree/main/src/radtrax/retrieval/matched_filter), and shows the exact sense in which the MF is the limit of the **dual (PSAS) 3D-Var solution** of the companion [3D-Var derivation](var3d.md) for a one-dimensional state with a flat prior.

The derivation borrows from Funk et al. [^funk2001], Theiler & Foy [^theiler2006] and Thompson et al. [^thompson2015]; the operator-based implementation follows the structured linear algebra of Golub & Van Loan [^gvl2013] and the Woodbury-friendly design of [`gaussx`](https://github.com/jejjohnson/gaussx).

[^funk2001]: Funk, C. C., Theiler, J., Roberts, D. A., & Borel, C. C. (2001). Clustering to improve matched filter detection of weak gas plumes in hyperspectral thermal imagery. *IEEE TGRS*, 39(7).

[^theiler2006]: Theiler, J. & Foy, B. R. (2006). Effect of signal contamination in matched-filter detection of the signal on a cluttered background. *IEEE GRSL*, 3(1).

[^thompson2015]: Thompson, D. R., et al. (2015). Real-time remote detection and measurement for airborne imaging spectroscopy: a case study with methane. *AMT*, 8.

[^gvl2013]: Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins.

[^theiler2021]: Theiler, J. (2021). Absorptive weak plume detection on Gaussian and non-Gaussian background clutter. *IEEE JSTARS*, 14, 6842–6854. doi:10.1109/JSTARS.2021.3093820

[^schaum2021]: Schaum, A. (2021). A uniformly most powerful detector of gas plumes against a cluttered background. *Remote Sensing of Environment*, 260, 112443. doi:10.1016/j.rse.2021.112443

**Symbols**

| Symbol | Meaning | Shape | Unit |
|---|---|---|---|
| $B$ | number of instrument bands | scalar | dimensionless |
| $N$ | number of background pixels the clutter is estimated from | scalar | dimensionless |
| $\vect{y}$ | one pixel's band radiance (the observation) | $\mathbb{R}^{B}$ | radiance |
| $\gvect{\mu}_y$ | mean of plume-free spectra (the background mean) | $\mathbb{R}^{B}$ | radiance |
| $\gmat{\Sigma}_y$ | covariance of plume-free spectra (the background clutter: noise, surface variability, calibration residual) | $\mathbb{R}^{B \times B}$ | radiance² |
| $\gvect{\varepsilon}$ | the pixel's departure from the background mean | $\mathbb{R}^{B}$ | radiance |
| $\alpha$ | plume strength (the unknown) | $\mathbb{R}_{\ge 0}$ | unit of the target, e.g. mol/m² |
| $\mat{J}_h$ | target signature: the change in $\vect{y}$ per unit $\alpha$, a one-column Jacobian of the forward model | $\mathbb{R}^{B}$ | radiance per unit $\alpha$ |
| $\vect{t}$ | unit absorption spectrum, $t_b = \partial \ln y_b / \partial \alpha$ at $\alpha = 0$; negative where the gas absorbs | $\mathbb{R}^{B}$ | m²/mol |
| $\hat\alpha$ | the matched-filter score, an estimate of $\alpha$ | scalar | unit of $\alpha$ |
| $\alpha_0$ | characteristic plume strength, the score's standard deviation under the null | scalar | unit of $\alpha$ |
| $\vect{w}$, $z$ | the precomputed filter $\gmat{\Sigma}_y^{-1} \mat{J}_h$ and its energy $\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h$ | $\mathbb{R}^{B}$, scalar | dimensionless |
| $\Phi$ | standard normal CDF | $\mathbb{R} \to [0, 1]$ | dimensionless |

The symbols of §4 are those of the [3D-Var page](var3d.md) and are defined where they first appear.

## 1. The signal model

We observe one radiance spectrum $\vect{y} \in \mathbb{R}^{B}$ per pixel of the scene. The additive model says that a plume of strength $\alpha$ adds a fixed multiple of the target signature to the background, so we will have:

$$
\vect{y} = \gvect{\mu}_y + \alpha \mat{J}_h + \gvect{\varepsilon}
$$ (eq-signal-model)

$$
\gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_y)
$$

where $\gvect{\mu}_y$ is the **background mean spectrum** (what we would observe at a pixel free of any plume enhancement), $\mat{J}_h$ is the **target signature** (the spectral shape that a unit enhancement leaves in $\vect{y}$), $\alpha \ge 0$ is the **plume strength** (pixels with $\alpha = 0$ are background, $\alpha > 0$ is a plume) and $\gmat{\Sigma}_y$ is the background clutter covariance, assumed Gaussian. Under this model the log-likelihood of a pixel is, up to additive constants,

$$
\ell(\alpha \mid \vect{y}) = -\frac{1}{2} (\vect{y} - \gvect{\mu}_y - \alpha \mat{J}_h)^\top \gmat{\Sigma}_y^{-1} (\vect{y} - \gvect{\mu}_y - \alpha \mat{J}_h)
$$ (eq-loglik)

Setting $\partial \ell / \partial \alpha = 0$ gives the maximum-likelihood estimator of the strength:

$$
\hat\alpha(\vect{y}) = \frac{(\vect{y} - \gvect{\mu}_y)^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}
$$ (eq-mf)

:::{important} Result · the matched filter
{eq}`eq-mf` is the matched filter. It is unbiased ($\mathbb{E}[\hat\alpha] = \alpha$), it has the minimum variance among all unbiased linear estimators (Gauss–Markov), and it maximises the SNR in the Neyman–Pearson sense for detecting the alternative $\alpha > 0$ against the null $\alpha = 0$.
:::

:::{note} The additive model is an approximation
All of this holds for the model {eq}`eq-signal-model`, in which the plume *adds* a fixed vector. A gas absorbs instead: Beer–Lambert multiplies each band of the plume-free spectrum by $e^{\alpha t_b}$, so the plume's footprint depends on the pixel it lands on. Theiler [^theiler2021] shows that the optimal detector of that multiplicative model is quadratic even in the weak-plume limit, and that the matched filter is a good detector because it approximates the GLRT, provided the target is the mean pixel's plume direction, $\gvect{\mu}_y \odot \vect{t}$ ({ref}`sec-mf-targets`). The derivation, with the change of variables, the GLRT and its built-in albedo correction, is on [Absorptive plume detection](absorptive_detection.md); non-Gaussian clutter, ACE and the log-space matched filter of Schaum [^schaum2021] are on [Beyond Gaussian clutter](clutter_models.md).
:::

### 1.1 The SNR and detection threshold

Under the null ($\alpha = 0$) the score $\hat\alpha(\vect{y})$ is $\mathcal{N}(0, 1 / \mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h)$, and under an alternative with strength $\alpha$ it is $\mathcal{N}(\alpha, 1 / \mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h)$. So the **detection SNR** is

$$
\mathrm{SNR}(\alpha) = \alpha \sqrt{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}
$$ (eq-snr)

and the Gaussian threshold on $\hat\alpha$ for a false-alarm rate (FAR) is

$$
\alpha_\text{FAR} = \frac{\Phi^{-1}(1 - \mathrm{FAR})}{\sqrt{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}}
$$ (eq-threshold)

Both are implemented in [`matched_filter.core`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/core.py) as `matched_filter_snr` and `detection_threshold`.

The score's standard deviation under the null, $\alpha_0 = 1 / \sqrt{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}$, is what Theiler [^theiler2021] calls the **characteristic plume strength**: a plume of strength $n \alpha_0$ is an $n$-sigma detection, $\mathrm{SNR}(\alpha) = \alpha / \alpha_0$. It also shows how much whitening buys. At $\alpha_0$ the radiance itself changes by a fraction $\alpha_0 \Vert \mat{J}_h \Vert_2 / \Vert \gvect{\mu}_y \Vert_2$, which is below $10^{-3}$ on Theiler's AVIRIS-NG and OMI scenes and about $2 \times 10^{-4}$ on the synthetic EMIT-like scene of [Absorptive plume detection](absorptive_detection.md). So plumes that are invisible in the raw spectra are clear in the matched-filter map.

## 2. Practical decomposition: precompute once, apply per pixel

Equation {eq}`eq-mf` is deceptively cheap. Per pixel it is a dot product of length $B$, except that $\gmat{\Sigma}_y^{-1}$ appears. Done naively we would invert a $B \times B$ matrix once, but that throws away the chance to exploit any structure in $\gmat{\Sigma}_y$. The right decomposition precomputes the filter and its energy once per scene, so we will have:

$$
\vect{w} = \gmat{\Sigma}_y^{-1} \mat{J}_h
$$ (eq-precompute)

$$
z = \mat{J}_h^\top \vect{w} = \mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h
$$

and then, per pixel,

$$
\hat\alpha(\vect{y}) = \frac{(\vect{y} - \gvect{\mu}_y)^\top \vect{w}}{z}
$$ (eq-mf-apply)

That is one linear solve and one dot product per scene, followed by one dot product per pixel (a single BLAS call when vectorised over the image). This is what `apply_image` in `core.py` does; the sketch below leaves out its check that $z > 0$, and it gives the same scores as `apply_image` to the last bit on a random cube:

```python
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

# Shapes: H, W image rows and columns; B bands


def apply_image(
    cube: Float[Array, "H W B"],
    mean: Float[Array, " B"],
    cov_op: lx.AbstractLinearOperator,
    target: Float[Array, " B"],
) -> Float[Array, "H W"]:
    # today (gaussx)
    # w = Σ_y⁻¹ J_h, one structured solve per scene
    w = gx.solve(cov_op, target)                             # (B,) → (B,)
    # z = J_hᵀ Σ_y⁻¹ J_h
    z = jnp.dot(target, w)                                   # (B,) → ()
    # r = y − μ_y, for every pixel
    residual = cube - mean                                   # (H, W, B) → (H, W, B)
    # α̂ = rᵀ w / z
    scores = jnp.einsum("hwb,b->hw", residual, w)            # (H, W, B) → (H, W)
    return scores / z                                        # (H, W) → (H, W)
```

The key move is `gx.solve(cov_op, target)`. Since `cov_op` is an abstract `lineax.AbstractLinearOperator`, the solve dispatches on structure: a dense operator triggers a Cholesky, a diagonal operator a per-element divide, and a low-rank update $\sigma_\text{tik}^2 \mat{I} + \mat{V} \op{diag}(\gvect{\sigma}^2) \mat{V}^\top$ triggers Woodbury, with no change to the kernel above.

## 3. Four flavours of $\gmat{\Sigma}_y$

The four reference files in `jej_vc_snippets/methane_retrieval/` originally wrote out four separate implementations of {eq}`eq-mf`, one per covariance model. In the operator-based view they collapse into four *estimation* routines that all return the same type, a `lineax.AbstractLinearOperator`. Below, $\vect{y}_n$ for $n = 1, \ldots, N$ are the background pixels, $\beta \in [0, 1]$ is the shrinkage weight, $\mat{V} \in \mathbb{R}^{B \times r}$ holds the top $r$ spectral directions with variances $\gvect{\sigma}^2 \in \mathbb{R}^{r}_{\ge 0}$, $\sigma_\text{tik}^2 > 0$ is the Tikhonov floor (the `tikhonov` argument) and $K$ is the number of clusters.

:::{list-table} Covariance estimators and the operators they produce.
:header-rows: 1

* - Estimator
  - Formula
  - Operator type
  - Per-pixel cost
* - `estimate_cov_empirical`
  - $\hat{\gmat{\Sigma}}_y = \frac{1}{N} \sum_{n=1}^{N} (\vect{y}_n - \gvect{\mu}_y)(\vect{y}_n - \gvect{\mu}_y)^\top$
  - `MatrixLinearOperator` (dense)
  - $\mathcal{O}(B^2)$ once, then $\mathcal{O}(B)$
* - `estimate_cov_shrunk` (Ledoit–Wolf, OAS)
  - $(1 - \beta) \hat{\gmat{\Sigma}}_y + \beta \frac{\op{tr}(\hat{\gmat{\Sigma}}_y)}{B} \mat{I}$
  - `MatrixLinearOperator` (dense, PD)
  - same
* - `estimate_cov_lowrank`
  - $\sigma_\text{tik}^2 \mat{I} + \mat{V} \op{diag}(\gvect{\sigma}^2) \mat{V}^\top$, $\mat{V}$ from a randomised SVD
  - `gaussx.LowRankUpdate` (Woodbury)
  - $\mathcal{O}(B r + r^3)$ once, then $\mathcal{O}(B)$
* - `gmm_cluster_background`
  - $(\gvect{\mu}_y^{(k)}, \gmat{\Sigma}_y^{(k)})$ per GMM cluster $k = 1, \ldots, K$
  - list of the operators above, one per cluster
  - per-cluster amortised
:::

The Ledoit–Wolf and OAS estimators come from `sklearn.covariance`, and both shrink toward the scaled identity in the table; the randomised SVD of the low-rank estimator is `gaussx.randomized_svd` (Halko–Martinsson–Tropp, in JAX). The background estimation is off the JAX gradient path (the background is treated as a fixed input), so we lean on scikit-learn freely there. Only the per-pixel *application* stays in JAX, vectorised with `jax.vmap` or `einsum` for GPU-friendly batching.

### 3.1 Why low-rank plus Tikhonov matters

For $N \gg B$ and a homogeneous scene, the spectral covariance is close to low rank: a handful of surface-reflectance principal components explain most of the variance. Writing $\gmat{\Sigma}_y = \sigma_\text{tik}^2 \mat{I} + \mat{V} \op{diag}(\gvect{\sigma}^2) \mat{V}^\top$ lets us apply Woodbury, so we will have:

$$
\gmat{\Sigma}_y^{-1} = \sigma_\text{tik}^{-2} \mat{I} - \sigma_\text{tik}^{-4} \mat{V} \left(\op{diag}(\gvect{\sigma}^2)^{-1} + \sigma_\text{tik}^{-2} \mat{V}^\top \mat{V}\right)^{-1} \mat{V}^\top
$$ (eq-woodbury)

The inner inverse is $r \times r$, so $\vect{w} = \gmat{\Sigma}_y^{-1} \mat{J}_h$ costs $\mathcal{O}(B r + r^3)$ instead of $\mathcal{O}(B^3)$. For a typical hyperspectral methane setup ($B = 400$, $r = 16$) this is a ~1000× speedup on the MF precompute, which the user only sees in that [`estimate_cov_lowrank`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/background.py) returns a `gaussx.LowRankUpdate` instead of a dense operator.

## 4. Relationship to 3D-Var

In the notation of the [3D-Var page](var3d.md), $\vect{x} \in \mathbb{R}^{N_x}$ is the state with prior mean $\gvect{\mu}_x$ and covariance $\gmat{\Sigma}_x$, $\vect{y} \in \mathbb{R}^{M}$ the observations with error covariance $\gmat{\Sigma}_\varepsilon$, $h$ the observation operator and $\mat{H} = \mat{J}_h(\gvect{\mu}_x)$ its tangent-linear, $\vect{d} = \vect{y} - h(\gvect{\mu}_x)$ the innovation and $\delta\vect{x} = \vect{x} - \gvect{\mu}_x$ the increment. (Here $N_x$ and $M$ are the sizes of the state and of the observations, written $N$ and $M$ on that page; $N$ on this page counts background pixels.) Incremental 3D-Var minimises

$$
\mathcal{J}(\delta\vect{x}) = \frac{1}{2} \delta\vect{x}^\top \gmat{\Sigma}_x^{-1} \delta\vect{x} + \frac{1}{2} (\vect{d} - \mat{H} \delta\vect{x})^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{d} - \mat{H} \delta\vect{x})
$$ (eq-3dvar)

and the PSAS (dual) path solves for a dual variable $\vect{y}^\dagger \in \mathbb{R}^{M}$ in observation space, then maps it back to the state:

$$
(\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon) \vect{y}^\dagger = \vect{d}
$$ (eq-psas)

$$
\delta\hat{\vect{x}} = \gmat{\Sigma}_x \mat{H}^\top \vect{y}^\dagger
$$

**The MF is the PSAS analysis of a one-dimensional state, the plume strength itself, under a flat prior.** Let's take $\delta\vect{x} = \alpha$, so the tangent-linear is the target signature, $\mat{H} = \mat{J}_h$, and the innovation of one pixel is $\vect{d} = \vect{y} - \gvect{\mu}_y$. Let's also take the prior variance $\gmat{\Sigma}_x = \sigma_x^2$ and the observation error $\gmat{\Sigma}_\varepsilon = \gmat{\Sigma}_y$. Then {eq}`eq-psas` is a rank-one update of $\gmat{\Sigma}_y$, and the Sherman–Morrison formula solves it in closed form, so we will have:

$$
(\sigma_x^2 \mat{J}_h \mat{J}_h^\top + \gmat{\Sigma}_y) \vect{y}^\dagger = \vect{d}
$$

$$
\vect{y}^\dagger = \gmat{\Sigma}_y^{-1} \vect{d} - \frac{\gmat{\Sigma}_y^{-1} \mat{J}_h \mat{J}_h^\top \gmat{\Sigma}_y^{-1} \vect{d}}{\sigma_x^{-2} + \mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h}
$$

$$
\hat\alpha = \sigma_x^2 \mat{J}_h^\top \vect{y}^\dagger = \frac{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \vect{d}}{\mat{J}_h^\top \gmat{\Sigma}_y^{-1} \mat{J}_h + \sigma_x^{-2}}
$$

As $\sigma_x \to \infty$ the $\sigma_x^{-2}$ in the denominator vanishes and $\hat\alpha$ is the matched filter {eq}`eq-mf` at $\vect{y} = \gvect{\mu}_y + \vect{d}$. For a finite $\sigma_x$ the same algebra gives the matched filter shrunk toward zero by its prior, which is the one-pixel version of what the full retrieval does. So the MF is *not* a heuristic: it is the exact, closed-form, flat-prior dual 3D-Var solution, and for this one-dimensional problem the dual solve needs no iterations at all; the SNR and threshold formulas follow without further work. [`run_dual_psas`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/solve.py) solves the same system with CG for a full field.

The corollary is that when $\gmat{\Sigma}_x$ is *not* flat, i.e. when we have real prior information about the VMR field (spatial smoothness, a climatological mean, etc.), the full 3D-Var retrieval uses information the MF throws away, and it does better whenever that prior is right. See plumax's [`01_mf_vs_3dvar.ipynb`](https://github.com/jejjohnson/plumax/blob/main/docs/notebooks/matched_filter/01_mf_vs_3dvar.ipynb) for a numerical comparison on the same scene (it lives in plumax because it simulates the plume).

## 5. Where the target signature comes from

For methane, $\mat{J}_h$ is the spectral signature of a unit VMR enhancement. Let's write the Beer–Lambert forward model per wavenumber $\nu$ as $L(\nu) = L_0(\nu) \exp(-a(\nu) \Delta\mathrm{VMR})$, where $L_0(\nu)$ is the background radiance and $a(\nu) \ge 0$ the absorption per unit enhancement. Its Taylor expansion at the background state gives

$$
J_h(\nu) = \left. \frac{\partial L}{\partial \Delta\mathrm{VMR}} \right|_{\Delta\mathrm{VMR} = 0} = -L_0(\nu) a(\nu)
$$ (eq-target)

where $-a(\nu)$ is the unit absorption $t$ at that wavenumber, so the target is the background radiance times the unit absorption. Rather than hard-coding this, we take the **JVP of the existing observation operator**: `linear_target_from_obs` differentiates the forward model with `jax.jvp` along a perturbation pattern of the state and reads the result at one output pixel.

```python
from radtrax.retrieval.matched_filter import linear_target_from_obs

# Shapes: H, W rows and columns of the state grid; B bands

# today (radtrax)
# J_h = ∂h(μ_x + α δx) / ∂α at α = 0, with δx uniform over the scene
target = linear_target_from_obs(                             # (H, W) → (B,)
    obs, vmr_background, pattern="uniform"
)
```

This guarantees the target is exactly consistent with whatever forward model (PSF, GSD, SRF) the retrieval uses, so there is no drift between a "target generator" and the forward model. For strong plumes, where Beer–Lambert saturation matters, `nonlinear_target_from_obs(obs, vmr_background, amplitude=α)` computes the exact finite-amplitude response $h(\gvect{\mu}_x + \alpha \delta\vect{x}) - h(\gvect{\mu}_x)$ instead. That turns the filter into the additive approximation of the *clairvoyant* detector at $\alpha$, the optimal detector when the strength is known ([Absorptive plume detection](absorptive_detection.md), Step 4), which is best near the chosen amplitude and not uniformly.

(sec-mf-targets)=
### 5.1 Three targets for one gas

On the layered [`ForwardModel`](xref:api#radtrax.ForwardModel), [`unit_absorption_target`](xref:api#radtrax.unit_absorption_target) returns the gas's unit absorption in log radiance, $\vect{t}$ with $t_b = \partial \ln y_b / \partial \alpha$, which is negative where the gas absorbs. Since Beer–Lambert is multiplicative, the additive target built from it depends on what the filter is applied to, and Theiler [^theiler2021] and Schaum [^schaum2021] compare three:

:::{list-table} Matched-filter targets for an absorbing gas.
:header-rows: 1

* - Target
  - Applied to
  - Assumes
  - radtrax
* - $\vect{t}$ on radiance ("t")
  - $\vect{y}$
  - a spectrally flat background
  - `apply_image(cube, mean, cov, t)`; not recommended
* - $\gvect{\mu}_y \odot \vect{t}$ ("Tμ")
  - $\vect{y}$
  - every pixel's plume direction is the mean pixel's
  - `apply_image(cube, mean, cov, mean * t)`
* - $\vect{t}$ on log radiance
  - $\ln \vect{y}$
  - lognormal clutter, so the plume is a shift in log space
  - `apply_image(np.log(cube), mean_ln, cov_ln, t)`, with `mean_ln`, `cov_ln` estimated on `np.log(cube)`
:::

The Tμ target beat the flat one on every scene in both papers, and the Beer–Lambert target {eq}`eq-target` above is of the Tμ kind. The log-space filter is uniformly most powerful when the clutter is lognormal, but band integration and additive sensor noise break that model; on Theiler's real scenes its fattened variants (log-ACE, log-EC) did well, while on the synthetic EMIT-like scene the plain Tμ filter beat it. [Beyond Gaussian clutter](clutter_models.md) has the details.

## 6. Summary

The pieces of [`radtrax.retrieval.matched_filter`](https://github.com/jejjohnson/radtrax/tree/main/src/radtrax/retrieval/matched_filter):

:::{list-table}
:header-rows: 1

* - Module
  - Role
* - [`core.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/core.py)
  - `apply_pixel`, `apply_image`, `matched_filter_snr`, `detection_threshold`: {eq}`eq-mf`, {eq}`eq-snr` and {eq}`eq-threshold`. Dispatches on the structure of `cov_op` via `gaussx.solve`.
* - [`target.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/target.py)
  - `linear_target_from_obs` (JVP), `nonlinear_target_from_obs` (finite amplitude) and `unit_absorption_target` (log radiance on the layered `ForwardModel`).
* - [`background.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/background.py)
  - `estimate_mean` (robust variants) and `estimate_cov_{empirical,shrunk,lowrank}`: scikit-learn or `gaussx.randomized_svd` in, `lineax`/`gaussx` operators out.
* - [`cluster.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/cluster.py)
  - `gmm_cluster_background` and `adaptive_window_background`: a spatially varying $(\gvect{\mu}_y, \gmat{\Sigma}_y)$ when the scene is heterogeneous.
* - [`streaming.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/streaming.py)
  - `WelfordAccumulator` and `streaming_background`, for background aggregation over many scenes or files.
* - [`io.py`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/io.py)
  - `apply_image_xarray` and `open_multi_scene`: the xarray glue (dimensional metadata, Dask chunks, `open_mfdataset`).
:::

The two companion notebooks:

- [`01_mf_vs_3dvar.ipynb`](https://github.com/jejjohnson/plumax/blob/main/docs/notebooks/matched_filter/01_mf_vs_3dvar.ipynb): the MF against full 3D-Var on the same HAPI methane scene used in the assimilation tutorial, so we can see where the flat-prior MF gives up information.
- [`02_covariance_flavours.ipynb`](../notebooks/mf_covariance_flavours.ipynb): dense, shrunk, low-rank and GMM-clustered $\gmat{\Sigma}_y$ on a heterogeneous scene, comparing detection power and runtime.
