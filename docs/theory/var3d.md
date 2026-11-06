---
title: "3D-Var: derivation, preconditioning and adjoint methods with lineax and optimistix"
---

# 3D-Var for methane retrieval

*Ported from plumax 0.1.2 (`28f257e`); module and link paths updated for radtrax.*

This note derives the cost function used by `radtrax.retrieval.var3d`, walks through the three control-variable formulations we implement, and then maps each piece of the maths onto a concrete `jax`, `lineax`, `optimistix` or `gaussx` call. The aim is that you can read it side by side with [solve.py](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/solve.py) and see which line implements which equation.

The derivation follows Bouttier & Courtier's *Data assimilation concepts and methods* [^bc1999] and Rodgers' *Inverse methods for atmospheric sounding* [^rodgers2000], although the letters are not theirs: we write the GeoStack notation, and the crosswalk table below maps their letters (and the code's argument names) onto ours. The JAX-side implementation tricks are from the lineax docs [^lineaxdocs] and the optimistix paper [^optimistix].

[^bc1999]: Bouttier, F. & Courtier, P. (1999). *Data assimilation concepts and methods*. Meteorological Training Course Lecture Series, ECMWF. https://www.ecmwf.int/sites/default/files/elibrary/2003/16928-data-assimilation-concepts-and-methods.pdf

[^rodgers2000]: Rodgers, C. D. (2000). *Inverse methods for atmospheric sounding: theory and practice*. World Scientific.

[^lineaxdocs]: Rader, J., Lyons, T., & Kidger, P. (2024). *Lineax: a library for linear solvers in JAX.* https://docs.kidger.site/lineax/

[^optimistix]: Rader, J., Lyons, T., & Kidger, P. (2024). *Optimistix: modular optimisation in JAX and Equinox.* arXiv:2402.09983.

**Symbols**

| Symbol | Meaning | Shape | Unit |
|---|---|---|---|
| $N$ | number of state pixels, $N = H W$ for an $H \times W$ field | scalar | dimensionless |
| $M$ | number of observations, the flattened band-radiance cube ($h w B$) | scalar | dimensionless |
| $\vect{x}$ | state: the per-pixel VMR field, flattened | $\mathbb{R}^{N}$ | VMR |
| $\vect{y}$ | observations: the band-integrated radiance cube, flattened | $\mathbb{R}^{M}$ | radiance |
| $\gvect{\mu}_x$ | prior (background) mean of the state | $\mathbb{R}^{N}$ | VMR |
| $\gmat{\Sigma}_x$ | prior covariance of the state | $\mathbb{R}^{N \times N}$ | VMR² |
| $\gmat{\Sigma}_\varepsilon$ | observation-error covariance | $\mathbb{R}^{M \times M}$ | radiance² |
| $h$ | nonlinear observation operator | $\mathbb{R}^{N} \to \mathbb{R}^{M}$ | VMR to radiance |
| $\mat{H} = \mat{J}_h(\gvect{\mu}_x)$ | tangent-linear of $h$ at the prior mean | $\mathbb{R}^{M \times N}$ | radiance / VMR |
| $\vect{d}$ | innovation, $\vect{y} - h(\gvect{\mu}_x)$ | $\mathbb{R}^{M}$ | radiance |
| $\delta\vect{x}$ | increment, $\vect{x} - \gvect{\mu}_x$ | $\mathbb{R}^{N}$ | VMR |
| $\mathcal{J}$ | 3D-Var cost over the state | scalar | dimensionless |
| $\gmat{\Lambda}_{x \mid y}$ | posterior precision, the Hessian of $\mathcal{J}$ | $\mathbb{R}^{N \times N}$ | VMR⁻² |
| $\mat{L}_x$ | square root of the prior covariance, $\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top$ | $\mathbb{R}^{N \times N}$ | VMR |
| $\gvect{\xi}$ | whitened control variable, $\delta\vect{x} = \mat{L}_x \gvect{\xi}$ | $\mathbb{R}^{N}$ | dimensionless |
| $\vect{y}^\dagger$ | dual (PSAS) variable | $\mathbb{R}^{M}$ | radiance⁻¹ |
| $\gvect{\theta}$ | hyperparameters of the retrieval | $\mathbb{R}^{D_\theta}$ | various |

**Crosswalk** from the classic data-assimilation letters, and to the argument names in `radtrax.retrieval.var3d`:

| Classic (Bouttier & Courtier) | Rodgers | This page | radtrax argument |
|---|---|---|---|
| $x_b$ | $x_a$ | $\gvect{\mu}_x$ | `background_state` |
| $B$ | $S_a$ | $\gmat{\Sigma}_x$ | `background_op`, `background` |
| $R$ | $S_\epsilon$ | $\gmat{\Sigma}_\varepsilon$ | `obs_inv_variance` (passes $\gmat{\Sigma}_\varepsilon^{-1}$, scalar or diagonal) |
| $H(x)$ | $F(x)$ | $h(\vect{x})$ | `forward_fn` |
| $H'$ | $K$ | $\mat{H}$ | none (from `jax.jvp` / `jax.vjp` of `forward_fn`) |
| $y$ | $y$ | $\vect{y}$ | `observation` |
| $J$ | n/a | $\mathcal{J}$ | `Cost` (`build_cost_x`, `build_cost_xi`) |
| $U$ | n/a | $\mat{L}_x$ | `WhiteningTransform.cholesky_op` |
| $\lambda$ | n/a | $\vect{y}^\dagger$ | `lam` inside `run_dual_psas` |

## 1. From Bayes to the 3D-Var cost

We have a state vector $\vect{x} \in \mathbb{R}^{N}$, which for our methane retrieval is the per-pixel VMR field flattened to length $N = H W$, and an observation vector $\vect{y} \in \mathbb{R}^{M}$, the band-integrated radiance cube. Both come with Gaussian uncertainty:

- **Prior**: $\vect{x} \sim \mathcal{N}(\gvect{\mu}_x, \gmat{\Sigma}_x)$. The background mean $\gvect{\mu}_x$ is whatever we knew before the observation arrived (typically a uniform climatological VMR), and $\gmat{\Sigma}_x$ encodes the per-pixel variance and any spatial correlations.
- **Likelihood**: $\vect{y} \mid \vect{x} \sim \mathcal{N}(h(\vect{x}), \gmat{\Sigma}_\varepsilon)$, with the (possibly nonlinear) observation operator $h$ and the observation-error covariance $\gmat{\Sigma}_\varepsilon$. In our case $h = \mathrm{SRF} \circ \mathrm{GSD} \circ \mathrm{PSF} \circ \exp(-a \Delta\mathrm{VMR})$.

By Bayes' rule the posterior $p(\vect{x} \mid \vect{y}) \propto p(\vect{y} \mid \vect{x}) p(\vect{x})$ is proportional to $\exp(-\mathcal{J}(\vect{x}))$, so we will have:

$$
\mathcal{J}(\vect{x}) = \frac{1}{2} (\vect{x} - \gvect{\mu}_x)^\top \gmat{\Sigma}_x^{-1} (\vect{x} - \gvect{\mu}_x) + \frac{1}{2} (\vect{y} - h(\vect{x}))^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{y} - h(\vect{x}))
$$ (eq-J)

The MAP estimate $\hat{\vect{x}} = \mathrm{argmin}_{\vect{x}} \mathcal{J}(\vect{x})$ is the 3D-Var analysis. The "3D" is historical: in weather prediction it refers to the three spatial dimensions of the state, and what it really says is that there is no time axis, which is what distinguishes 3D-Var from 4D-Var (our state is a 2-D field).

### 1.1 Incremental form

In practice we minimise over the **increment** $\delta\vect{x} = \vect{x} - \gvect{\mu}_x$, for two reasons: 1) the background term then has a clean quadratic form, and 2) the observation operator is often linearised around $\gvect{\mu}_x$ (the *inner loop* of incremental 3D-Var). Substituting $\vect{x} = \gvect{\mu}_x + \delta\vect{x}$ and writing $\vect{d} := \vect{y} - h(\gvect{\mu}_x)$ for the *innovation*, we will have:

$$
\mathcal{J}(\delta\vect{x}) = \frac{1}{2} \delta\vect{x}^\top \gmat{\Sigma}_x^{-1} \delta\vect{x} + \frac{1}{2} (\vect{d} - \mat{H} \delta\vect{x})^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{d} - \mat{H} \delta\vect{x})
$$ (eq-J-incremental)

where $\mat{H} := \mat{J}_h(\gvect{\mu}_x) = \partial h / \partial \vect{x}$ evaluated at $\gvect{\mu}_x$ is the **tangent-linear** operator. Setting the gradient to zero gives the normal equations:

$$
(\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}) \delta\hat{\vect{x}} = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \vect{d}
$$ (eq-normal)

The matrix on the left is the Hessian of {eq}`eq-J-incremental`, which is the posterior precision, $\gmat{\Lambda}_{x \mid y} = \nabla^2_{\vect{x}} \mathcal{J} = \gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}$ (the prior precision plus the information the observations add). Its inverse $\gmat{\Lambda}_{x \mid y}^{-1}$ is the textbook posterior covariance for the linear-Gaussian case.

> **In code.** Equation {eq}`eq-J` is built by [`build_cost_x`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/cost.py). The prior term is `gaussx.quadratic_form(background_op, δx_flat)`, one structurally dispatched solve, so any structured $\gmat{\Sigma}_x$ from [background.py](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/background.py) (diagonal, Kronecker, low-rank) is plugged in without being materialised. The observation term differentiates through the JAX `forward_fn` by reverse-mode AD, so we never write $\mat{H}^\top$ by hand.

## 2. Why preconditioning matters

The catch in {eq}`eq-normal` is that the conditioning of $\gmat{\Lambda}_{x \mid y}$ tracks the spread of its eigenvalues. In a typical retrieval $\gmat{\Sigma}_x^{-1}$ has a large dynamic range (some pixels are tightly constrained by the prior, others very loosely), so a plain minimiser like L-BFGS can take hundreds of iterations to make progress along the floppy directions.

The **control variable transform** (CVT) removes this by changing variables to $\gvect{\xi} := \mat{L}_x^{-1} \delta\vect{x}$, where $\mat{L}_x$ is *any* square root of the prior covariance:

$$
\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top
$$ (eq-B-factor)

We pick the Cholesky factor, but other choices (an eigendecomposition, balance operators) work the same way. Substituting $\delta\vect{x} = \mat{L}_x \gvect{\xi}$ into {eq}`eq-J-incremental` gives:

$$
\mathcal{J}(\gvect{\xi}) = \frac{1}{2} \gvect{\xi}^\top \gvect{\xi} + \frac{1}{2} (\vect{d} - \mat{H} \mat{L}_x \gvect{\xi})^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{d} - \mat{H} \mat{L}_x \gvect{\xi})
$$ (eq-J-whitened)

The prior term collapses to $\frac{1}{2} \Vert \gvect{\xi} \Vert_2^2$, a sphere, and the Hessian in $\gvect{\xi}$ becomes:

$$
\nabla^2_{\gvect{\xi}} \mathcal{J} = \mat{I}_N + (\mat{H} \mat{L}_x)^\top \gmat{\Sigma}_\varepsilon^{-1} (\mat{H} \mat{L}_x)
$$ (eq-hess-whitened)

This is the identity plus a positive-semidefinite term of rank at most $M$, so every eigenvalue is at least one and at most $M$ of them differ from one. So the badly conditioned prior has dropped out entirely: what is left is set by how informative the observations are (the largest eigenvalue is one plus the largest whitened signal-to-noise ratio squared), and in exact arithmetic conjugate gradients on this Hessian finishes in at most $M + 1$ iterations. In practice this is the biggest lever we have in variational DA, and L-BFGS benefits from it the same way.

> **In code.** Equation {eq}`eq-J-whitened` is built by [`build_cost_xi`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/cost.py); the operator $\mat{L}_x$ is constructed by [`WhiteningTransform.from_background`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/control.py), which calls `gaussx.cholesky(background)`. gaussx returns a *structured* operator (a Cholesky of a `Kronecker` is itself a Kronecker of Cholesky factors), so all the matvecs and inverse solves that follow stay in the cheap regime.

## 3. Primal vs dual: PSAS

The normal equations {eq}`eq-normal` are an $N \times N$ system. When the observation count is small ($M \ll N$, typical for sparse plumes seen by an instrument with a few bands), the **dual** form is a lot cheaper. Applying the matrix-inversion lemma to {eq}`eq-normal`, we will have:

$$
\delta\hat{\vect{x}} = \gmat{\Sigma}_x \mat{H}^\top (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} \vect{d}
$$ (eq-dual-solution)

The middle factor is an $M \times M$ system. Defining the *dual variable* $\vect{y}^\dagger := (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} \vect{d} \in \mathbb{R}^{M}$ (it is the Lagrange multiplier of the observation constraint, often written $\lambda$, a letter we keep for wavelength), the recipe is:

1. Solve $(\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon) \vect{y}^\dagger = \vect{d}$ for $\vect{y}^\dagger$ (CG-friendly, since the matrix is symmetric positive definite).
2. Set $\delta\hat{\vect{x}} = \gmat{\Sigma}_x \mat{H}^\top \vect{y}^\dagger$.

This is the Physical-Space Statistical Analysis System (PSAS) [^cohn1998]. For a *linear* $h$, the dual and primal give the same analysis. For a nonlinear $h$, both solve the same linearised problem, the inner loop of each outer iteration of incremental 3D-Var or 4D-Var, so the choice between them is about cost ($N$ against $M$) and not about the answer.

[^cohn1998]: Cohn, S. E., et al. (1998). *Assessing the effects of data selection with the DAO Physical-Space Statistical Analysis System.* Mon. Wea. Rev. **126**, 2913-2926.

> **In code.** [`run_dual_psas`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/solve.py) wraps the dual matrix as a `lineax.FunctionLinearOperator`, tags it symmetric and PSD, and feeds it to `lineax.linear_solve(..., solver=lineax.CG(...))`. Inside the matvec, $\mat{H} \delta\vect{x}$ is $h(\gvect{\mu}_x + \delta\vect{x}) - h(\gvect{\mu}_x)$, which is exact because the solver expects a forward that is linear in $\delta\vect{x}$ (callers pass `model.make_forward(linear=True)`), and $\mat{H}^\top \vect{u}$ comes from a single `jax.vjp` pullback hoisted out of the CG loop. There is no hand-written adjoint of the observation operator anywhere in the code.

### 3.1 The matched filter as the flat-prior limit of PSAS

Now take a single pixel and restrict the increment to one direction, a uniform enhancement of strength $\alpha$, so that $\mat{H} \delta\vect{x} = \alpha \vect{t}$ with $\vect{t}$ the target signature ($\mat{H}$ applied to a unit enhancement). In addition, let the prior on $\alpha$ be flat (its variance $\sigma_x^2 \to \infty$) and let $\gmat{\Sigma}_\varepsilon$ be the per-pixel clutter covariance $\gmat{\Sigma}_y$, with $\vect{d} = \vect{y} - \gvect{\mu}_y$ the departure of the pixel spectrum $\vect{y}$ from the background mean. Then the dual system is $(\sigma_x^2 \vect{t} \vect{t}^\top + \gmat{\Sigma}_y) \vect{y}^\dagger = \vect{d}$, and in the limit {eq}`eq-dual-solution` becomes the pixel-wise matched filter:

$$
\hat{\alpha} = \frac{(\vect{y} - \gvect{\mu}_y)^\top \gmat{\Sigma}_y^{-1} \vect{t}}{\vect{t}^\top \gmat{\Sigma}_y^{-1} \vect{t}}
$$ (eq-matched)

The full limit, with the Sherman–Morrison step, is on the [matched filter](matched_filter.md) page (§4). So the matched filter, a narrow-band retrieval and full 3D-Var are the same estimator with progressively richer prior assumptions, and the matched filter in [`radtrax.retrieval.matched_filter`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/matched_filter/core.py) is the simplest case of the framework here. Note that it is the exact, closed-form solution of the restricted one-parameter problem (a single scalar solve); it is not what one CG iteration on the full $M \times M$ dual system returns, since the first CG step moves along $\vect{d}$ rather than along $\gmat{\Sigma}_y^{-1} \vect{t}$.

## 4. Adjoint methods: four flavours

Computing the gradient $\nabla_{\delta\vect{x}} \mathcal{J}$ needs the *adjoint* (transpose) of the tangent-linear $\mat{H}$. Historically this was the part of variational DA you wrote and maintained by hand. With JAX we get four options out of the box, each with a different cost and memory profile:

| Method | When it applies | Cost | Memory | JAX call |
|---|---|---|---|---|
| **Discrete adjoint** (reverse-mode AD) | $h$ is a JAX program; one forward pass | one VJP is $\mathcal{O}(\text{forward})$; a full Jacobian is $M$ of them | $\mathcal{O}(\text{forward tape})$ | `jax.grad(loss)` |
| **Tangent-linear** (forward-mode AD) | sensitivities, ensemble propagation | one JVP is $\mathcal{O}(\text{forward})$; a full Jacobian is $N$ of them | $\mathcal{O}(\text{forward})$ | `jax.jvp(h, (x,), (v,))` |
| **Implicit-function adjoint** | differentiating *through* a converged inner solve | one linear solve | $\mathcal{O}(\text{linear solve})$ | `optimistix.ImplicitAdjoint(linear_solver=...)` |
| **Checkpointed adjoint** | memory-bounded long unrolls (ODE, iterative) | $\mathcal{O}(\text{forward} \log n)$ with $\mathcal{O}(\log n)$ checkpoints, about two forwards with $\mathcal{O}(\sqrt{n})$ | $\mathcal{O}(k)$ for $k$ checkpoints | `optimistix.RecursiveCheckpointAdjoint(...)` |

### 4.1 Discrete adjoint (`jax.grad`)

This is what we use for the observation-term gradient in {eq}`eq-J`. Concretely, [`build_cost_xi`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/cost.py) wraps the cost in `jax.value_and_grad(value)`, and reverse-mode AD walks back through `forward_fn` applying each elementary VJP, including the kernel-flipped convolution adjoint inside [`PointSpreadFunction`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/instrument/psf.py) and the spread-with-$1/f^2$ adjoint inside [`GroundSamplingDistance`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/instrument/psf.py). The unit tests in [test_psf.py](https://github.com/jejjohnson/radtrax/blob/main/tests/instrument/test_psf.py) verify the inner-product identity $\langle \mat{A} \vect{x}, \vect{u} \rangle = \langle \vect{x}, \mat{A}^\top \vect{u} \rangle$ for both operators, so the autodiff gradient is the one a hand-rolled adjoint would give.

### 4.2 Tangent-linear (`jax.jvp`)

`jax.jvp(h, (x,), (v,))` computes $\mat{J}_h(\vect{x}) \vect{v}$ in one forward pass without ever forming the Jacobian. It is useful for:

- The dual matvec inside {eq}`eq-dual-solution`, one per CG iteration. [`run_dual_psas`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/solve.py) gets the same product without `jax.jvp`, as the difference $h(\gvect{\mu}_x + \delta\vect{x}) - h(\gvect{\mu}_x)$, which is exact for the linear forward it expects; a `jax.jvp` would be the drop-in for a nonlinear $h$.
- Hessian-vector products by forward-over-reverse: `jax.jvp(jax.grad(J), (x,), (v,))` returns $\nabla^2_{\vect{x}} \mathcal{J} \vect{v}$ at $\mathcal{O}(\text{forward})$ cost. This is what [`Cost.hvp`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/cost.py) does, and it is what feeds the diagnostics in [diagnostics.py](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/diagnostics.py).

### 4.3 Implicit-function adjoint

Suppose we want $\partial \vect{x}^\star / \partial \gvect{\theta}$, the derivative of the converged 3D-Var solution with respect to a hyperparameter $\gvect{\theta}$ (e.g. a regularisation weight, an observation-error variance, a prior length scale). Differentiating through the L-BFGS unroll would be both expensive and numerically suspect.

The implicit-function theorem gives a closed form instead. The optimum $\vect{x}^\star(\gvect{\theta})$ satisfies $\nabla_{\vect{x}} \mathcal{J}(\vect{x}^\star; \gvect{\theta}) = \mathbf{0}$, and differentiating that identity in $\gvect{\theta}$ gives:

$$
\frac{\partial \vect{x}^\star}{\partial \gvect{\theta}} = -\left(\nabla^2_{\vect{x}} \mathcal{J}\right)^{-1} \nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J}
$$ (eq-implicit-adjoint)

where $\nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J} \in \mathbb{R}^{N \times D_\theta}$ is the mixed second derivative. So we only need **one** linear solve with the Hessian $\nabla^2_{\vect{x}} \mathcal{J}$ (the posterior precision again). For a loss $\mathcal{L}(\gvect{\theta})$ evaluated at the analysis, the total derivative then follows from the chain rule:

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\gvect{\theta}} = \nabla_{\gvect{\theta}} \mathcal{L} + \left(\frac{\partial \vect{x}^\star}{\partial \gvect{\theta}}\right)^\top \nabla_{\vect{x}} \mathcal{L}
$$

This is what `optimistix.ImplicitAdjoint(linear_solver=lineax.CG(...))` does internally: you write a normal `optimistix.minimise(...)` call, and gradients with respect to anything captured by the cost flow back through this single adjoint solve, with no tape and no checkpoints.

> **In code.** The solvers in [solve.py](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/solve.py) run optimistix with its default adjoint, `ImplicitAdjoint()`, but the wrappers do not expose an `adjoint` argument. To differentiate the analysis with respect to its inputs, call `optimistix.minimise` directly with a custom `linear_solver` (e.g. `lineax.CG(rtol=...)` for matrix-free Hessians, or `lineax.Cholesky()` for small dense ones).

### 4.4 Checkpointed adjoint

When the forward map is itself an iterative procedure (e.g. a chemistry solver inside the observation operator, or a Kalman-smoother prior), the reverse-mode tape can blow up memory. `optimistix.RecursiveCheckpointAdjoint(checkpoints=k)` does binomial (treeverse) checkpointing: only $k$ forward states are stored, and the rest are recomputed on the backward pass, so memory is traded for recomputation as in the table above. We don't currently need this, since the methane forward map is a single fused convolution chain, but it is the right tool the moment we couple the retrieval to a dynamical model.

## 5. Mapping maths to libraries

| Math object | Library call |
|---|---|
| $\gmat{\Sigma}_x$ (Kronecker, low-rank, diagonal) | `gaussx.Kronecker`, `gaussx.LowRankUpdate`, `lineax.DiagonalLinearOperator` |
| $\delta\vect{x}^\top \gmat{\Sigma}_x^{-1} \delta\vect{x}$ | `gaussx.quadratic_form(background_op, δx)` (structured dispatch) |
| $\gmat{\Sigma}_x^{-1} \delta\vect{x}$ | `gaussx.solve(background_op, δx)` (structured dispatch) |
| $\mat{L}_x$ such that $\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top$ | `gaussx.cholesky(background)` |
| $h$ (forward operator, JAX) | `RadianceObservationModel.forward(...)`, `make_forward(...)` |
| $\mat{H} \vect{v}$ (tangent-linear) | `jax.jvp(forward_fn, (x,), (v,))` |
| $\mat{H}^\top \vect{u}$ (adjoint) | `jax.vjp(forward_fn, x)[1](u)`, no manual adjoint |
| $\nabla \mathcal{J}$ (discrete adjoint) | `jax.grad(J)` |
| $\nabla^2 \mathcal{J} \vect{v}$ (Hessian-vector) | `jax.jvp(jax.grad(J), (x,), (v,))` |
| Outer minimiser (whitened L-BFGS) | `optimistix.LBFGS(...)` via `optimistix.minimise(...)` |
| Gauss–Newton with a custom inner solver | `optimistix.GaussNewton(...)` with `lineax.CG(...)` for $\mat{J}_r^\top \mat{J}_r$, $r$ the `residual_fn` |
| Dual (PSAS) solve | `lineax.linear_solve(dual_op, d, solver=lineax.CG(...))` |
| Adjoint *through* the converged solve | `optimistix.ImplicitAdjoint(linear_solver=lineax.CG(...))` |
| $\chi^2_\text{red}$, DFS, posterior covariance | [`reduced_chi_squared`, `degrees_of_freedom_for_signal`, `posterior_covariance_proxy`](https://github.com/jejjohnson/radtrax/blob/main/src/radtrax/retrieval/var3d/diagnostics.py) |

## 6. Worked example

The end-to-end demo lives in plumax's [06_variational_methane_retrieval.ipynb](https://github.com/jejjohnson/plumax/blob/main/docs/notebooks/assimilation/06_variational_methane_retrieval.ipynb). It builds a synthetic plume from the HAPI LUT, injects it through the JAX forward operator, and runs the retrieval three ways (naive primal, whitened L-BFGS and dual PSAS), comparing iteration counts, run times and posterior diagnostics. It lives in plumax because it simulates the plume with plumax's Gaussian plume model; radtrax computes everything after the column. The same notebook also shows the matched filter (plumax PR #24) falling out of the dual recipe, as the flat-prior limit of §3.1.
