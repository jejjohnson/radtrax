"""Posterior diagnostics for a converged 3D-Var solution.

After the optimiser has produced ``x̂``, these tell you whether the result is
statistically consistent with the prior and likelihood you specified, and how
much the observations decided:

- :func:`reduced_chi_squared` — ``χ² = ‖y − H(x̂)‖²_R⁻¹ / dim(y)``. ≈ 1 means
  the posterior fits the observations to the noise level. Much smaller →
  over-fitting (R too generous); much larger → systematic errors not in R.
- :func:`gauss_newton_posterior` — the Laplace posterior ``N(x̂, Ŝ)`` at the
  minimum, ``Ŝ = (Kᵀ R⁻¹ K + B⁻¹)⁻¹``, from the whitened residuals through
  :func:`argminax.curvature.gauss_newton`.
- :func:`averaging_kernel` — ``A = I − Ŝ B⁻¹`` (Rodgers, 2000) from a cost's
  Hessian-vector product, through :func:`argminax.diagnose.averaging_kernel`.
- :func:`degrees_of_freedom_for_signal` — ``DFS = tr A``, Hutchinson's
  estimate over Rademacher probes drawn from a JAX key, through
  :func:`argminax.diagnose.degrees_of_freedom`.
- :func:`posterior_covariance_proxy` — the inverse Hessian at the optimum,
  as a callable ``v → Ŝ v`` through ``argminax.curvature.Laplace.covariance``.

These are diagnostics, not exact statistics — for the linear-Gaussian case
they coincide with the textbook posterior; for the nonlinear case they're
Laplace approximations around the optimum. Every one returns JAX values and
runs under ``jit``, ``vmap`` and ``grad``; the only randomness is the key
passed to :func:`degrees_of_freedom_for_signal`.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/assimilation/diagnostics.py``); the
curvature, averaging kernel and trace estimate now come from argminax.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from argminax import curvature as amx_curvature, diagnose as amx_diagnose
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from radtrax.exceptions import ParameterError
from radtrax.retrieval.var3d.cost import ForwardFn, ObsInvVariance, StateVector


__all__ = [
    "averaging_kernel",
    "degrees_of_freedom_for_signal",
    "gauss_newton_posterior",
    "posterior_covariance_proxy",
    "reduced_chi_squared",
]


if TYPE_CHECKING:
    from radtrax.retrieval.var3d.control import (
        IdentityTransform,
        WhiteningTransform,
    )

_PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def reduced_chi_squared(
    *,
    forward_fn: ForwardFn,
    estimated_state: Float[ArrayLike, "H W"],
    observation: Float[ArrayLike, "h w B"],
    obs_inv_variance: ObsInvVariance,
) -> Float[Array, ""]:
    """``χ²_red = (y - H(x̂))ᵀ R⁻¹ (y - H(x̂)) / dim(y)``.

    Values near 1 indicate the posterior is consistent with the observation
    error model; ≪ 1 → noise overestimated; ≫ 1 → noise underestimated or
    systematic forward-model error.

    Returns:
        χ²_red as a JAX scalar, differentiable in the state and the forward
        model.
    """
    y = jnp.asarray(observation)
    R_inv = jnp.asarray(obs_inv_variance)
    residual = y - forward_fn(jnp.asarray(estimated_state))
    return jnp.sum(R_inv * residual * residual) / y.size


def gauss_newton_posterior(
    *,
    forward_fn: ForwardFn,
    whitening: WhiteningTransform | IdentityTransform,
    obs_inv_variance: ObsInvVariance,
    background_state: Float[ArrayLike, "H W"],
    observation: Float[ArrayLike, "h w B"],
    estimated_state: Float[ArrayLike, "H W"],
) -> amx_curvature.Laplace:
    """Return the Laplace posterior at a 3D-Var minimum, by Gauss–Newton.

        r(x) = (R^{-1/2} (y − H(x)),  U⁻¹ (x − x_b)),   B = U Uᵀ
        Ĥ = J_rᵀ J_r = Kᵀ R⁻¹ K + B⁻¹,   Ŝ = Ĥ⁻¹,   K = ∂H/∂x at x̂

    The whitening of both residuals stays here; argminax sees only r and
    builds the curvature (densely up to 1000 unknowns, matrix-free above).
    Gauss–Newton drops the second-order term of the forward model, so Ĥ is
    the exact Hessian of the cost for a linear H, and at any x̂ where the
    measurement residual vanishes.

    Args:
        forward_fn: The forward operator H, a JAX function of the (H, W)
            state.
        whitening: The transform with ``B = U Uᵀ``, from
            :meth:`~radtrax.retrieval.var3d.control.WhiteningTransform.from_background`
            (an ``IdentityTransform`` means ``B = I``).
        obs_inv_variance: R⁻¹, a scalar or an array broadcastable against
            the observation cube; non-negative.
        background_state: The prior mean x_b.
        observation: The measurement y.
        estimated_state: The minimum x̂.

    Returns:
        The posterior N(x̂, Ŝ) over the flattened state, shape (N,), as an
        :class:`argminax.curvature.Laplace`: ``.covariance()`` is Ŝ, and
        ``argminax.diagnose.averaging_kernel(posterior, B)`` is A.
    """
    state_shape = jnp.shape(estimated_state)
    x_b = jnp.ravel(jnp.asarray(background_state))
    y = jnp.asarray(observation)
    # R^{-1/2}, element-wise for a diagonal R
    sqrt_r_inv = jnp.sqrt(jnp.asarray(obs_inv_variance))

    def residuals(
        x: StateVector, args: None
    ) -> tuple[Float[Array, "h w B"], StateVector]:
        measurement = sqrt_r_inv * (y - forward_fn(jnp.reshape(x, state_shape)))
        prior = whitening.apply_inverse(x - x_b)
        return measurement, prior

    return amx_curvature.gauss_newton(
        residuals, jnp.ravel(jnp.asarray(estimated_state))
    )


def _hessian_posterior(
    hessian_vector_product: Callable[[StateVector], StateVector],
    state_size: int,
    *,
    cg_rtol: float,
    cg_atol: float,
    cg_max_steps: int,
) -> amx_curvature.Laplace:
    """The Laplace posterior whose precision is a cost's Hessian, solved by CG.

    The Hessian already carries the point it was taken at, and neither the
    covariance nor the averaging kernel reads the mode, so the mode is zero.
    """
    precision = lx.FunctionLinearOperator(
        lambda v: hessian_vector_product(jnp.asarray(v)),
        input_structure=jax.eval_shape(lambda: jnp.zeros(state_size)),
        tags=_PSD_TAGS,
    )
    # ``throw=False`` returns the last CG iterate at max_steps rather than
    # raising: for diagnostics a noisy estimate beats a crash on one stiff probe.
    solver = gx.CGSolver(
        rtol=cg_rtol, atol=cg_atol, max_steps=cg_max_steps, throw=False
    )
    return amx_curvature.Laplace(jnp.zeros(state_size), precision, solver)


def averaging_kernel(
    *,
    hessian_vector_product: Callable[[StateVector], StateVector],
    state_size: int,
    background_op: lx.AbstractLinearOperator | None = None,
    whitened: bool = False,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-10,
    cg_max_steps: int = 2000,
) -> lx.AbstractLinearOperator:
    """Return the averaging kernel ``A = I − Ŝ B⁻¹`` of a cost's Hessian, lazily.

        A = I − Ŝ B⁻¹,   Ŝ = Hess⁻¹;   whitened: A_ξ = I − Hess_ξ⁻¹

    A says how much a change in the true state moves the retrieved one
    (Rodgers 2000, ch. 2): A = I means the observations decide everything,
    A = 0 means the prior does. Each product with A is one CG solve with the
    Hessian (and, in model space, one ``gaussx.solve`` with B), through
    :func:`argminax.diagnose.averaging_kernel`; A is never formed.

    Pass the Hessian that matches the space it was built in:

    - **Model space** (``whitened=False``, the default) — the Hessian from
      :func:`radtrax.retrieval.var3d.cost.build_cost_x` and
      ``background_op = B``.
    - **Whitened space** (``whitened=True``) — the Hessian from
      :func:`radtrax.retrieval.var3d.cost.build_cost_xi`
      (``Hess_ξ = I + (HU)ᵀR⁻¹HU``), whose prior is the identity; omit
      ``background_op``.

    The trace of A is invariant under ``δx = U ξ``, but A itself is not:
    the whitened kernel is ``U⁻¹ A U``.

    Args:
        hessian_vector_product: ``v → Hess v`` at the optimum.
        state_size: N, the length of the flattened state.
        background_op: B, in model space only.
        whitened: Whether the Hessian is the ξ-space one.
        cg_rtol: Relative tolerance of each CG solve.
        cg_atol: Absolute tolerance of each CG solve.
        cg_max_steps: Step budget of each CG solve; at the budget the last
            iterate is used.

    Returns:
        A, an (N, N) lazy lineax operator; tagged symmetric and positive
        semidefinite when whitened.

    Raises:
        ParameterError: If ``background_op`` is passed with
            ``whitened=True``, or omitted with ``whitened=False``.
    """
    _check_space("averaging_kernel", background_op=background_op, whitened=whitened)
    posterior = _hessian_posterior(
        hessian_vector_product,
        state_size,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_max_steps=cg_max_steps,
    )
    return amx_diagnose.averaging_kernel(posterior, None if whitened else background_op)


def _check_space(
    caller: str, *, background_op: lx.AbstractLinearOperator | None, whitened: bool
) -> None:
    if whitened and background_op is not None:
        raise ParameterError(
            f"{caller}: `whitened=True` uses the "
            "whitened prior B_ξ = I; do not pass `background_op` (the ξ-space "
            "Hessian already absorbs B via δx = U ξ)."
        )
    if not whitened and background_op is None:
        raise ParameterError(
            f"{caller}: model-space mode "
            "(`whitened=False`) requires `background_op = B`; pass the "
            "Hessian from build_cost_x, or set `whitened=True` for a "
            "build_cost_xi (ξ-space) Hessian."
        )


def degrees_of_freedom_for_signal(
    *,
    hessian_vector_product: Callable[[StateVector], StateVector],
    state_size: int,
    background_op: lx.AbstractLinearOperator | None = None,
    whitened: bool = False,
    n_probes: int = 32,
    key: PRNGKeyArray | None = None,
    seed: int | None = None,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-10,
    cg_max_steps: int = 2000,
) -> amx_diagnose.TraceEstimate:
    """Estimate ``DFS = tr(I − Ŝ B⁻¹)`` (Rodgers 2000, §2.5) from a JAX key.

        d_s = tr A ≈ (1/K) ∑ₖ zₖᵀ A zₖ,   zₖ ∈ {−1, 1}ᴺ Rademacher

    DFS measures how many independent pieces of information the observations
    contributed (``DFS = 0`` → no information; ``DFS = N`` → observations
    fully pin the state). It is invariant under the control-variable change
    ``δx = U ξ``, but the estimate must be computed in the space the
    supplied ``hessian_vector_product`` lives in — mixing an ξ-space Hessian
    with a model-space ``B`` gives a wrong answer for any ``B ≠ I`` — so
    ``whitened`` and ``background_op`` follow :func:`averaging_kernel`. In
    whitened space ``zᵀz = N``, so this is Hutchinson's estimate of
    ``N − tr(Hess_ξ⁻¹)``. A is :func:`averaging_kernel`.

    The K probes come from ``key`` through
    :func:`argminax.diagnose.degrees_of_freedom`, so ``jax.vmap`` over
    pixels with split keys gives each pixel its own probes. Cost per probe:
    one CG solve, ``O(n_CG · forward)``. For the exact trace (N solves), call
    ``argminax.diagnose.degrees_of_freedom(averaging_kernel(...))`` without
    a key.

    Args:
        hessian_vector_product: ``v → Hess v`` at the optimum.
        state_size: N, the length of the flattened state.
        background_op: B, in model space only.
        whitened: Whether the Hessian is the ξ-space one.
        n_probes: Number of probes K, at least 2.
        key: Random key for the probes.
        seed: Deprecated; pass ``key=jax.random.key(seed)``. A call without
            a key uses ``jax.random.key(seed)``, ``seed = 0`` by default, and
            warns.
        cg_rtol: Relative tolerance of each CG solve.
        cg_atol: Absolute tolerance of each CG solve.
        cg_max_steps: Step budget of each CG solve.

    Returns:
        d_s and the standard error of its K probe terms, as an
        :class:`argminax.diagnose.TraceEstimate` of JAX scalars. The
        standard error does not see the bias of an unconverged CG solve.

    Raises:
        ParameterError: If ``background_op`` does not match ``whitened``, or
            both ``key`` and ``seed`` are passed.
        argminax.ValidationError: If ``n_probes`` is below 2.

    Warns:
        DeprecationWarning: When called without ``key``.
    """
    if key is not None and seed is not None:
        raise ParameterError(
            "degrees_of_freedom_for_signal: pass `key` or the deprecated `seed`, "
            "not both."
        )
    if key is None:
        warnings.warn(
            "degrees_of_freedom_for_signal without `key` (and its `seed` "
            "argument) is deprecated; pass key=jax.random.key(seed), and split "
            "it per pixel.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = jax.random.key(0 if seed is None else seed)
    _check_space(
        "degrees_of_freedom_for_signal", background_op=background_op, whitened=whitened
    )
    kernel = averaging_kernel(
        hessian_vector_product=hessian_vector_product,
        state_size=state_size,
        background_op=background_op,
        whitened=whitened,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_max_steps=cg_max_steps,
    )
    return amx_diagnose.degrees_of_freedom(kernel, num_probes=n_probes, key=key)


def posterior_covariance_proxy(
    *,
    hessian_vector_product: Callable[[StateVector], StateVector],
    state_size: int,
    whitening: WhiteningTransform | None = None,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-9,
    cg_max_steps: int = 200,
) -> Callable[[StateVector], StateVector]:
    """Return a callable ``v → Ŝ v`` (model-space posterior covariance) via CG.

        Ŝ = Hess⁻¹;   whitened: Ŝ = U Hess_ξ⁻¹ Uᵀ

    ``Ŝ ≈ Hess⁻¹`` is the Laplace-approximation posterior covariance, here
    ``argminax.curvature.Laplace.covariance`` of the Hessian with a CG
    solve. We expose it as a matvec (rather than materialising) so
    per-pixel variance estimates remain ``O(state · cg_iters)``. With the
    forward model and residuals at hand, :func:`gauss_newton_posterior`
    builds the same covariance from the Gauss–Newton curvature instead.

    Pass the Hessian that matches ``whitening``:

    - ``whitening=None`` (default) — a **model-space** Hessian (from
      :func:`radtrax.retrieval.var3d.cost.build_cost_x`); the CG solve returns
      ``Hess⁻¹ v = Ŝ v`` directly.
    - a :class:`~radtrax.retrieval.var3d.control.WhiteningTransform` — a **whitened**
      Hessian (from :func:`radtrax.retrieval.var3d.cost.build_cost_xi`), whose
      inverse is the ξ-space covariance, *not* ``Ŝ``. The model-space
      covariance is recovered as ``Ŝ = U Hess_ξ⁻¹ Uᵀ``, so each apply
      un-whitens on both sides: ``v → U (Hess_ξ⁻¹ (Uᵀ v))``.

    At ``max_steps`` the last CG iterate is returned rather than raising.
    """
    covariance = _hessian_posterior(
        hessian_vector_product,
        state_size,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_max_steps=cg_max_steps,
    ).covariance()

    def apply(v: StateVector) -> StateVector:
        v = jnp.asarray(v)
        # Whitened Hessian: Ŝ = U Hess_ξ⁻¹ Uᵀ, so map into ξ-space first.
        rhs = whitening.project_gradient(v) if whitening is not None else v
        sol = covariance.mv(rhs)
        return whitening.apply(sol) if whitening is not None else sol

    return apply
