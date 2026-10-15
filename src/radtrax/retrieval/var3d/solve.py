"""Optimisers for 3D-Var: primal LBFGS, Gauss-Newton, and a dual PSAS path.

Three entry points, each tuned to a different conditioning story:

- :func:`run_lbfgs` — black-box minimiser. Pass any cost+grad bundle from
  :mod:`.cost`. Robust, but iteration count tracks ``cond(Hessian)``, so use
  this *with* a :class:`~radtrax.retrieval.var3d.control.WhiteningTransform`.
- :func:`run_gauss_newton` — assumes the cost is a sum of squared residuals
  (which 3D-Var is, by construction). Each outer step solves a linear system
  ``(JᵀJ) δ = -Jᵀr`` via a user-chosen :mod:`lineax` solver (CG by default,
  so structured ``B`` keeps Woodbury / Kronecker on the hot path).
- :func:`run_dual_psas` — the **observation-space** reformulation. When
  ``dim(y) ≪ dim(x)`` (typical for compact plumes), solving
  ``(H'BH'ᵀ + R) λ = d`` and recovering ``δx = B H'ᵀ λ`` is dramatically
  cheaper than the primal — and exactly equivalent for the linear forward.

The dual path is also the "matched filter, generalised": the matched-filter
target spectrum ``t`` falls out as ``H'`` applied to a unit increment, and
the PSAS solution ``δx̂ = B H'ᵀ (H'BH'ᵀ + R)⁻¹ d`` is the BLUE estimator that
the pixel-wise matched filter approximates with ``B = ∞ I, R = Σ_pixel``.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/assimilation/solve.py``); its frozen
dataclasses became Equinox modules in radtrax#58.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from radtrax.retrieval.var3d.cost import ForwardFn


if TYPE_CHECKING:
    from radtrax.retrieval.var3d.cost import Cost


class SolveResult(eqx.Module):
    """Common result type for all three solvers.

    Attributes
    ----------
    state : np.ndarray
        Optimal control variable (``ξ`` for whitened, ``δx`` for model-space,
        ``δx`` for the dual after the back-transform).
    n_steps : int
        Number of iterations consumed (``-1`` if the solver doesn't report).
    cost_history : np.ndarray
        Cost trajectory; for solvers that don't expose it, this is empty.
    """

    state: np.ndarray
    n_steps: int
    cost_history: np.ndarray


# ── primal LBFGS ────────────────────────────────────────────────────────────


def run_lbfgs(
    cost: Cost,
    initial_state: Float[ArrayLike, " N"],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 200,
) -> SolveResult:
    """Minimise ``cost.value`` via :class:`optimistix.LBFGS`.

    The cost is wrapped to match optimistix's ``(y, args) → scalar`` signature.
    The implicit-function adjoint is the default — fine for this use case
    because we don't differentiate *through* the solve.
    """
    fn = lambda y, _args: cost.value(y)
    solver = optx.LBFGS(rtol=rtol, atol=atol)
    sol = optx.minimise(fn, solver, initial_state, max_steps=max_steps, throw=False)
    return SolveResult(
        state=np.asarray(sol.value),
        n_steps=int(sol.stats.get("num_steps", -1)),
        cost_history=np.asarray([]),
    )


# ── Gauss-Newton with custom linear solver ─────────────────────────────────


def run_gauss_newton(
    *,
    residual_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    initial_state: Float[ArrayLike, " N"],
    linear_solver: lx.AbstractLinearSolver | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 50,
) -> SolveResult:
    """Run :class:`optimistix.GaussNewton` for ``min ½ ‖r(y)‖²``.

    ``residual_fn`` should return a flat residual vector
    ``[B^{-½}(δx); R^{-½}(y - H(x_b + δx))]`` (the "augmented" 3D-Var residual);
    optimistix internally constructs the Jacobian and solves the normal
    equations ``(JᵀJ) δ = -Jᵀr`` using ``linear_solver``. Pass
    ``lineax.CG(...)`` for a matrix-free inner solve (useful when the
    Jacobian has Kronecker / low-rank structure that full Cholesky would
    materialise), or ``lineax.Cholesky()`` for small dense systems.
    """
    if linear_solver is None:
        linear_solver = lx.AutoLinearSolver(well_posed=None)
    fn = lambda y, _args: residual_fn(y)
    solver = optx.GaussNewton(rtol=rtol, atol=atol, linear_solver=linear_solver)
    sol = optx.least_squares(
        fn, solver, initial_state, max_steps=max_steps, throw=False
    )
    return SolveResult(
        state=np.asarray(sol.value),
        n_steps=int(sol.stats.get("num_steps", -1)),
        cost_history=np.asarray([]),
    )


# ── dual / PSAS ─────────────────────────────────────────────────────────────


def run_dual_psas(
    *,
    forward_fn: ForwardFn,
    background_op: lx.AbstractLinearOperator,
    obs_inv_variance: float,
    background_state: Float[ArrayLike, "H W"],
    observation: Float[ArrayLike, "h w B"],
    state_shape: tuple[int, ...],
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-9,
    cg_max_steps: int = 200,
) -> SolveResult:
    """Solve the **dual** 3D-Var system in observation space.

    For a forward operator that is linear in the state perturbation the
    optimum of ``J(δx) = ½ δxᵀ B⁻¹ δx + ½(y - H(x_b + δx))ᵀ R⁻¹(y - H(x_b + δx))``
    satisfies

        (H' B H'ᵀ + R) λ = d,           with d = y − H(x_b)
        δx̂ = B H'ᵀ λ.

    When ``dim(y) ≪ dim(x)`` this is much cheaper than primal — and it's
    exactly the estimator that the pixel-wise matched filter approximates.

    Parameters
    ----------
    forward_fn : Callable
        **Full-state forward** ``x → y`` (i.e. the same object returned by
        :meth:`RadianceObservationModel.make_forward`). For the dual to be
        correct, ``H`` must be linear in ``δx = x − x_b``. We extract the
        tangent-linear action by differencing at ``x_b`` once (see
        ``base_obs`` below) and the adjoint via :func:`jax.vjp`. Callers
        typically pass ``model.make_forward(linear=True)``.
    background_op : lineax operator
        ``B``.
    obs_inv_variance : float
        Scalar ``R⁻¹`` (heteroscedastic R can be folded into ``forward_fn``
        at the cost of clarity — keep it scalar for the demo).
    background_state, observation, state_shape
        Self-explanatory.
    cg_rtol, cg_atol, cg_max_steps
        Tolerances for the CG solve of the dual system.
    """
    x_b = jnp.asarray(background_state)
    y = jnp.asarray(observation)
    R_inv = float(obs_inv_variance)

    # Evaluate H(x_b) once — used both for the innovation and as the affine
    # anchor in the tangent-linear probe. Hoisting this out of the matvec
    # avoids a redundant forward pass per CG iteration.
    base_obs = forward_fn(x_b)
    d = (y - base_obs).reshape(-1)
    n_obs = d.size
    n_state = int(np.prod(state_shape))

    def H_op(delta_x_flat: jax.Array) -> jax.Array:
        """``H' · δx``: linearity ⇒ ``H(x_b + δx) − H(x_b)`` is exact."""
        delta_x = jnp.reshape(delta_x_flat, state_shape)
        return (forward_fn(x_b + delta_x) - base_obs).reshape(-1)

    # Build the VJP closure *once*, outside the CG matvec, so every iteration
    # reuses the traced pullback instead of re-tracing jax.vjp(H_op, 0) each
    # time.  This is materially faster — was the P2 perf finding in review.
    _, _Ht_pullback = jax.vjp(H_op, jnp.zeros(n_state))

    def Ht_op(lam: jax.Array) -> jax.Array:
        (out,) = _Ht_pullback(lam)
        return out

    def dual_matvec(lam: jax.Array) -> jax.Array:
        """``(H' B H'ᵀ + R) λ``, with ``R λ = λ / R_inv`` for scalar R."""
        BHt_lam = background_op.mv(Ht_op(lam))
        return H_op(BHt_lam) + lam / R_inv

    dual_op = lx.FunctionLinearOperator(
        dual_matvec,
        input_structure=jax.eval_shape(lambda: jnp.zeros(n_obs)),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    sol = lx.linear_solve(
        dual_op,
        d,
        solver=lx.CG(rtol=cg_rtol, atol=cg_atol, max_steps=cg_max_steps),
        throw=False,  # surface non-convergence in `sol.result` instead of raising
    )
    lam_star = sol.value
    delta_x_flat = background_op.mv(Ht_op(lam_star))
    return SolveResult(
        state=np.asarray(delta_x_flat),
        n_steps=int(sol.stats.get("num_steps", -1)) if hasattr(sol, "stats") else -1,
        cost_history=np.asarray([]),
    )
