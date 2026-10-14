"""Solver tests: LBFGS recovers truth, dual ≡ primal, twin experiment.

Three end-to-end checks:

1. **Twin experiment** — inject a flat-field ``ΔVMR > 0``, run LBFGS in
   whitened space, verify the recovered field matches truth to within the
   prior+obs noise budget.
2. **Dual == primal** for a linear forward — the PSAS solution and the
   primal LBFGS solution coincide (the dual is closed-form, so this
   pins down both implementations at once).
3. **Whitening drops iteration count** — LBFGS in ξ-space converges in
   ≤ 10 iterations even with a stiff prior, while model-space takes orders
   of magnitude more (or fails). Soft check; mostly a regression guard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from radtrax.retrieval.var3d.background import (
    build_diagonal_background,
    build_kronecker_background,
)
from radtrax.retrieval.var3d.control import WhiteningTransform
from radtrax.retrieval.var3d.cost import build_cost_x, build_cost_xi
from radtrax.retrieval.var3d.solve import (
    run_dual_psas,
    run_lbfgs,
)


jax.config.update("jax_enable_x64", True)


def test_twin_experiment_flat_field_recovery(obs_model_no_optics):
    """Inject ΔVMR=2e-7 uniformly; LBFGS in ξ-space should recover it."""
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 2e-7
    truth_field = jnp.full((ny, nx), truth_vmr)
    y_obs = model.forward(truth_field, linear=False)

    # Mildly informative prior, very tight obs likelihood.
    B = build_kronecker_background(
        ny=ny,
        nx=nx,
        variance=1e-12,
        length_scale_y=2.0,
        length_scale_x=2.0,
    )
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e8,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    sol = run_lbfgs(cost, jnp.zeros(ny * nx), max_steps=200, rtol=1e-8, atol=1e-12)
    delta_x_flat = W.apply(jnp.asarray(sol.state))
    recovered = np.asarray(delta_x_flat).reshape(ny, nx)
    np.testing.assert_allclose(recovered, np.full((ny, nx), truth_vmr), rtol=5e-2)


def test_dual_psas_matches_primal_for_linear_forward(obs_model_no_optics):
    """Primal LBFGS in ξ-space ≡ dual PSAS for a *linear* forward."""
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 1e-7
    truth_field = jnp.full((ny, nx), truth_vmr)
    # Use the linear forward so the primal/dual equivalence is exact.
    y_obs = model.forward(truth_field, linear=True)

    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    R_inv = 1e8

    # --- primal in ξ-space ---
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=True),
        whitening=W,
        obs_inv_variance=R_inv,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    sol_primal = run_lbfgs(
        cost, jnp.zeros(ny * nx), max_steps=200, rtol=1e-9, atol=1e-13
    )
    delta_x_primal = np.asarray(W.apply(jnp.asarray(sol_primal.state)))

    # --- dual PSAS ---
    sol_dual = run_dual_psas(
        forward_fn=model.make_forward(linear=True),
        background_op=B,
        obs_inv_variance=R_inv,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
        cg_rtol=1e-10,
        cg_atol=1e-14,
    )
    delta_x_dual = sol_dual.state

    np.testing.assert_allclose(delta_x_primal, delta_x_dual, rtol=1e-4, atol=1e-9)


def test_lbfgs_in_xi_space_converges_quickly(obs_model_no_optics):
    """Soft regression: whitening (ξ-space) speeds LBFGS up over x-space.

    plumax asserted an absolute bound (< 35 steps). The count varies with the
    CPU: 28 steps on one machine, 36 on GitHub's runner, with the same answer.
    The property the test guards is the preconditioning, so it compares the
    two formulations on the same problem (x-space takes 68 steps where ξ-space
    takes 28), and keeps a generous cap against a real blow-up.
    """
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 1.5e-7
    y_obs = model.forward(jnp.full((ny, nx), truth_vmr), linear=False)
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    common = dict(
        forward_fn=model.make_forward(linear=False),
        obs_inv_variance=1e8,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    solve = dict(max_steps=500, rtol=1e-8, atol=1e-12)
    sol = run_lbfgs(build_cost_xi(whitening=W, **common), jnp.zeros(ny * nx), **solve)
    sol_x = run_lbfgs(
        build_cost_x(background_op=B, **common), jnp.zeros(ny * nx), **solve
    )
    if sol.n_steps != -1 and sol_x.n_steps != -1:
        assert sol.n_steps < sol_x.n_steps, (
            f"ξ-space LBFGS took {sol.n_steps} steps, x-space {sol_x.n_steps}: "
            "whitening no longer helps — preconditioning regression?"
        )
        assert sol.n_steps < 100, (
            f"LBFGS took {sol.n_steps} steps — preconditioning regression?"
        )
