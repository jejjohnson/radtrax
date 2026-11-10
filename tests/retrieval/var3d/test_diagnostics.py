"""Posterior-diagnostics tests.

Each diagnostic is checked against a closed form on a 3 × 3 toy scene: the
dense Hessian (or the Gauss–Newton precision Kᵀ R⁻¹ K + B⁻¹ from a dense
Jacobian) gives Ŝ, A = I − Ŝ B⁻¹ and d_s = tr A exactly. The textbook
identities (χ²_red ≈ 0 at the truth, 0 ≤ DFS ≤ N) are easy to break with a
sign flip or transposed index, so we test those even on toy inputs. The
probed DFS is a Monte Carlo estimate, so it is bounded by its own sampling
distribution (see `_hutchinson_sd`), not by a tolerance tuned on one draw.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match
from argminax.curvature import Laplace
from argminax.diagnose import TraceEstimate

from radtrax.exceptions import ParameterError
from radtrax.retrieval.var3d.background import build_diagonal_background
from radtrax.retrieval.var3d.control import WhiteningTransform
from radtrax.retrieval.var3d.cost import build_cost_x, build_cost_xi
from radtrax.retrieval.var3d.diagnostics import (
    averaging_kernel,
    degrees_of_freedom_for_signal,
    gauss_newton_posterior,
    posterior_covariance_proxy,
    reduced_chi_squared,
)


jax.config.update("jax_enable_x64", True)

NY, NX = 3, 3
N = NY * NX
# Wide, non-uniform prior variances → B ≠ I in a way that the space-mixing
# bug cannot hide behind.
VARIANCES = np.array([0.05, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0, 12.0])
# A partial-information regime (0 < DFS < N) for the synthetic table.
OBS_INV = 1e-11
# CG run to machine precision: a 9-unknown SPD system converges in at most 9
# steps, so these diagnostics equal their direct-solve values and can be
# compared with closed forms and finite differences.
TIGHT_CG = dict(cg_rtol=1e-12, cg_atol=1e-14, cg_max_steps=200)


def _dense(matvec, n: int = N) -> np.ndarray:
    """The matrix of a linear map ℝⁿ → ℝⁿ, column by column."""
    return np.stack([np.asarray(matvec(jnp.asarray(e))) for e in np.eye(n)], axis=1)


def _hutchinson_sd(kernel: np.ndarray) -> float:
    """The standard deviation of one Rademacher probe term zᵀ A z.

    With S = (A + Aᵀ)/2, zᵀ A z = tr A + ∑_{i≠j} S_ij zᵢ zⱼ, whose products
    zᵢ zⱼ (i < j) are pairwise uncorrelated with unit variance, so
    Var(zᵀ A z) = 2 ∑_{i≠j} S_ij². The mean of K terms has σ/√K.
    """
    sym = 0.5 * (kernel + kernel.T)
    off = sym - np.diag(np.diag(sym))
    return float(np.sqrt(2.0 * np.sum(off**2)))


def _problem(model, obs_inv: float = OBS_INV, variances=VARIANCES):
    """The model-space and whitened costs of one toy retrieval, with B and U."""
    B = build_diagonal_background(variances)
    W = WhiteningTransform.from_background(B)
    fwd = model.make_forward(linear=False)
    y = model.forward(jnp.full((NY, NX), 1e-7), linear=False)
    kw = dict(
        forward_fn=fwd,
        obs_inv_variance=obs_inv,
        background_state=jnp.zeros((NY, NX)),
        observation=y,
        state_shape=(NY, NX),
    )
    cost_x = build_cost_x(background_op=B, **kw)
    cost_xi = build_cost_xi(whitening=W, **kw)
    return dict(B=B, W=W, fwd=fwd, y=y, cost_x=cost_x, cost_xi=cost_xi)


def _hvp(cost):
    return lambda v: cost.hvp(jnp.zeros(N), v)


# ── reduced χ² ──────────────────────────────────────────────────────────────


def test_chi2_at_truth_is_zero(obs_model_no_optics):
    model = obs_model_no_optics
    truth = jnp.full((3, 3), 1e-7)
    y = model.forward(truth, linear=False)
    chi2 = reduced_chi_squared(
        forward_fn=model.make_forward(linear=False),
        estimated_state=truth,
        observation=y,
        obs_inv_variance=1e8,
    )
    assert chi2 < 1e-10


def test_chi2_takes_a_numpy_state(obs_model_no_optics):
    """A NumPy state reaches the JAX-only forward closure as a JAX array."""
    model = obs_model_no_optics
    truth = np.full((3, 3), 1e-7)
    chi2 = reduced_chi_squared(
        forward_fn=model.make_forward(linear=False),
        estimated_state=truth,
        observation=model.forward(jnp.asarray(truth), linear=False),
        obs_inv_variance=1e8,
    )
    assert chi2 < 1e-10


def test_chi2_is_a_jax_scalar_with_its_closed_form(obs_model_no_optics):
    """χ²_red = ∑ R⁻¹ δ² / M for an observation offset by a known δ."""
    model = obs_model_no_optics
    truth = jnp.full((3, 3), 1e-7)
    y_true = model.forward(truth, linear=False)
    delta = 1e-3 * jnp.cos(jnp.arange(y_true.size)).reshape(y_true.shape)
    chi2 = reduced_chi_squared(
        forward_fn=model.make_forward(linear=False),
        estimated_state=truth,
        observation=y_true + delta,
        obs_inv_variance=4.0,
    )
    assert isinstance(chi2, jax.Array)
    assert chi2.shape == ()
    np.testing.assert_allclose(chi2, 4.0 * np.sum(delta**2) / y_true.size, rtol=1e-12)


# ── Gauss–Newton posterior ──────────────────────────────────────────────────


def test_gauss_newton_posterior_matches_the_closed_form(obs_model_with_psf):
    """Ŝ = (Kᵀ R⁻¹ K + B⁻¹)⁻¹ with K the dense Jacobian at x̂."""
    p = _problem(obs_model_with_psf)
    estimate = jnp.full((NY, NX), 5e-8)
    posterior = gauss_newton_posterior(
        forward_fn=p["fwd"],
        whitening=p["W"],
        obs_inv_variance=OBS_INV,
        background_state=jnp.zeros((NY, NX)),
        observation=p["y"],
        estimated_state=estimate,
    )
    assert isinstance(posterior, Laplace)
    np.testing.assert_array_equal(posterior.mode, np.ravel(estimate))
    K = jax.jacfwd(lambda x: p["fwd"](x.reshape(NY, NX)).ravel())(jnp.ravel(estimate))
    precision = OBS_INV * K.T @ K + np.diag(1.0 / VARIANCES)
    np.testing.assert_allclose(
        posterior.covariance().as_matrix(), np.linalg.inv(precision), rtol=1e-10
    )


def test_gauss_newton_posterior_matches_the_proxy_at_a_zero_residual(
    obs_model_with_psf,
):
    """Where y = H(x̂), Gauss–Newton is the exact Hessian: Ŝ equals the proxy's."""
    p = _problem(obs_model_with_psf)
    truth = jnp.full((NY, NX), 1e-7)
    posterior = gauss_newton_posterior(
        forward_fn=p["fwd"],
        whitening=p["W"],
        obs_inv_variance=OBS_INV,
        background_state=jnp.zeros((NY, NX)),
        observation=p["y"],
        estimated_state=truth,
    )
    proxy = posterior_covariance_proxy(
        hessian_vector_product=lambda v: p["cost_x"].hvp(jnp.ravel(truth), v),
        state_size=N,
        **TIGHT_CG,
    )
    np.testing.assert_allclose(
        posterior.covariance().as_matrix(), _dense(proxy), rtol=1e-8
    )


# ── averaging kernel ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_averaging_kernel_matches_the_closed_form(obs_model_with_psf):
    """A = I − Hess⁻¹ B⁻¹ in model space, A_ξ = I − Hess_ξ⁻¹ whitened; tr agrees."""
    p = _problem(obs_model_with_psf)
    hess_x = _dense(_hvp(p["cost_x"]))
    hess_xi = _dense(_hvp(p["cost_xi"]))
    kernel_x = averaging_kernel(
        hessian_vector_product=_hvp(p["cost_x"]),
        state_size=N,
        background_op=p["B"],
        **TIGHT_CG,
    ).as_matrix()
    kernel_xi = averaging_kernel(
        hessian_vector_product=_hvp(p["cost_xi"]),
        state_size=N,
        whitened=True,
        **TIGHT_CG,
    ).as_matrix()
    exact_x = np.eye(N) - np.linalg.inv(hess_x) @ np.diag(1.0 / VARIANCES)
    exact_xi = np.eye(N) - np.linalg.inv(hess_xi)
    np.testing.assert_allclose(kernel_x, exact_x, atol=1e-10)
    np.testing.assert_allclose(kernel_xi, exact_xi, atol=1e-10)
    np.testing.assert_allclose(np.trace(kernel_x), np.trace(kernel_xi), rtol=1e-8)


# ── degrees of freedom for signal ───────────────────────────────────────────


def test_dfs_zero_in_no_information_limit(obs_model_no_optics):
    """When the prior dominates (obs infinitely noisy), Hess → B⁻¹ and DFS → 0.

    This is the key regression test: the old implementation computed
    ``trace(B · Hess)`` which returned ``state_size`` instead of ``0`` in
    this limit, so the sanity check was systematically biased high.
    """
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)  # well-scaled prior
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e-12,  # effectively no obs info
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    dfs = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitened=True,  # cost is build_cost_xi → ξ-space Hessian, prior = I
        n_probes=8,
        key=jax.random.key(0),
    )
    assert isinstance(dfs, TraceEstimate)
    assert np.isfinite(dfs.value)
    # No observations → posterior == prior → DFS ≈ 0. Hutchinson variance
    # plus CG tolerance give ~5% of state_size as a comfortable bound.
    assert abs(dfs.value) < 0.5


def test_dfs_approaches_state_size_in_high_information_limit(obs_model_no_optics):
    """When obs dominate (prior very loose), Hess → H'ᵀR⁻¹H' and DFS → state_size."""
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)  # well-scaled prior
    W = WhiteningTransform.from_background(B)
    truth = jnp.full((ny, nx), 1e-7)
    y = model.forward(truth, linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e12,  # tight observations
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    dfs = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitened=True,  # cost is build_cost_xi → ξ-space Hessian, prior = I
        n_probes=16,
        key=jax.random.key(0),
    )
    # Should be close to state_size = 9; allow Hutchinson + CG slack.
    assert 7.0 < dfs.value < ny * nx + 0.5


def test_dfs_nonidentity_background(obs_model_no_optics):
    """DFS agrees between the model-space and whitened paths for B ≠ I.

    The old code combined a ξ-space Hessian with a model-space B⁻¹ solve, which
    is only correct when B = I. Here B is strongly heteroscedastic, so the two
    paths must be computed in their own spaces (``whitened`` flag) to agree —
    and both must match an exact dense reference: within four standard errors
    of their sampling distribution (plus the CG tolerance), and within the
    old test's ``atol=0.6``. Without optics each pixel sees only its own bands
    and B is diagonal, so A is diagonal and Rademacher probes are exact.
    """
    p = _problem(obs_model_no_optics)
    keys = jax.random.split(jax.random.key(0))
    dfs_x = degrees_of_freedom_for_signal(
        hessian_vector_product=_hvp(p["cost_x"]),
        state_size=N,
        background_op=p["B"],
        whitened=False,
        n_probes=64,
        key=keys[0],
    )
    dfs_xi = degrees_of_freedom_for_signal(
        hessian_vector_product=_hvp(p["cost_xi"]),
        state_size=N,
        whitened=True,
        n_probes=64,
        key=keys[1],
    )

    # Exact reference: dense ξ-space Hessian, DFS = n − trace(Hess_ξ⁻¹).
    hess_xi_inv = np.linalg.inv(_dense(_hvp(p["cost_xi"])))
    exact = N - np.trace(hess_xi_inv)
    kernel_x = np.eye(N) - np.linalg.inv(_dense(_hvp(p["cost_x"]))) @ np.diag(
        1.0 / VARIANCES
    )

    # The setup must be genuinely partial for the test to exercise the bug.
    assert 0.5 < exact < N - 0.5
    for estimate, kernel in ((dfs_x, kernel_x), (dfs_xi, np.eye(N) - hess_xi_inv)):
        # 4σ/√K of sampling error, plus the bias of nine CG solves at the
        # default rtol = 1e-6 on terms of order 1, well inside 1e-4.
        bound = 4.0 * _hutchinson_sd(kernel) / np.sqrt(64) + 1e-4
        assert abs(float(estimate.value) - exact) <= bound
        np.testing.assert_allclose(estimate.value, exact, atol=0.6)
    np.testing.assert_allclose(dfs_x.value, dfs_xi.value, atol=0.8)

    # Non-vacuous: the old ξ-hvp + B⁻¹ mix gives a materially different number.
    buggy = N - np.trace(hess_xi_inv @ np.diag(1.0 / VARIANCES))
    assert abs(exact - buggy) > 0.5


@pytest.mark.parametrize("whitened", [False, True])
def test_dfs_within_four_standard_errors_of_the_exact_trace(
    obs_model_with_psf, whitened
):
    """A coupled A (PSF): the probed DFS lies within 4σ/√K of tr A.

    σ is the exact standard deviation of one probe term (`_hutchinson_sd`) for
    the dense A; at 4σ a correct estimator fails about once in 16 000 draws
    (normal approximation of the mean of K = 64 terms). The reported standard
    error is the sample estimate of the same σ/√K, so it is positive here.
    """
    p = _problem(obs_model_with_psf)
    cost = p["cost_xi"] if whitened else p["cost_x"]
    kw = dict(whitened=True) if whitened else dict(background_op=p["B"])
    num_probes = 64
    estimate = degrees_of_freedom_for_signal(
        hessian_vector_product=_hvp(cost),
        state_size=N,
        n_probes=num_probes,
        key=jax.random.key(3),
        **kw,
    )
    hess = _dense(_hvp(cost))
    kernel = (
        np.eye(N) - np.linalg.inv(hess)
        if whitened
        else np.eye(N) - np.linalg.inv(hess) @ np.diag(1.0 / VARIANCES)
    )
    sd = _hutchinson_sd(kernel)
    assert sd > 1e-3  # the probes are not trivially exact
    assert abs(float(estimate.value) - np.trace(kernel)) <= 4.0 * sd / np.sqrt(
        num_probes
    )
    assert estimate.stderr > 0.0


@pytest.mark.slow
def test_dfs_vmap_over_pixels_with_split_keys_gives_independent_probes(
    obs_model_with_psf,
):
    """Split keys give each batch member its own probes; one key repeats.

    Today's NumPy generator drew the same seed-0 probes for every pixel; with
    keys, the same problem under four keys gives four different estimates.
    """
    p = _problem(obs_model_with_psf)
    hvp = _hvp(p["cost_xi"])

    @jax.jit
    def dfs(key):
        return degrees_of_freedom_for_signal(
            hessian_vector_product=hvp,
            state_size=N,
            whitened=True,
            n_probes=4,
            key=key,
        ).value

    split = jax.vmap(dfs)(jax.random.split(jax.random.key(0), 4))
    repeated = jax.vmap(dfs)(jnp.stack([jax.random.key(0)] * 4))
    assert len(np.unique(np.asarray(split))) == 4
    assert len(np.unique(np.asarray(repeated))) == 1
    np.testing.assert_allclose(split[1], dfs(jax.random.split(jax.random.key(0), 4)[1]))


@pytest.mark.slow
def test_dfs_seed_is_deprecated_and_maps_to_a_key(obs_model_with_psf):
    """``seed=s`` (and no key at all) warn and equal ``key=jax.random.key(s)``."""
    p = _problem(obs_model_with_psf)
    kw = dict(hessian_vector_product=_hvp(p["cost_xi"]), state_size=N, whitened=True)
    with pytest.warns(DeprecationWarning, match="key=jax.random.key"):
        seeded = degrees_of_freedom_for_signal(seed=5, **kw)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        unkeyed = degrees_of_freedom_for_signal(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        keyed = degrees_of_freedom_for_signal(key=jax.random.key(5), **kw)
        default = degrees_of_freedom_for_signal(key=jax.random.key(0), **kw)
    np.testing.assert_array_equal(seeded.value, keyed.value)
    np.testing.assert_array_equal(unkeyed.value, default.value)
    with pytest.raises(ParameterError, match="not both"):
        degrees_of_freedom_for_signal(key=jax.random.key(5), seed=5, **kw)


@pytest.mark.parametrize(
    "diagnostic",
    [averaging_kernel, degrees_of_freedom_for_signal],
    ids=["averaging_kernel", "degrees_of_freedom_for_signal"],
)
def test_rejects_inconsistent_background_args(obs_model_no_optics, diagnostic):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    hvp = lambda v: cost.hvp(jnp.zeros(ny * nx), v)
    extra = (
        {"key": jax.random.key(0), "n_probes": 2}
        if diagnostic is degrees_of_freedom_for_signal
        else {}
    )
    with pytest.raises(ValueError, match=r"whitened=True.*do not pass `background_op`"):
        diagnostic(
            hessian_vector_product=hvp,
            state_size=ny * nx,
            background_op=B,
            whitened=True,
            **extra,
        )
    with pytest.raises(ValueError, match=r"model-space mode.*requires"):
        diagnostic(
            hessian_vector_product=hvp,
            state_size=ny * nx,
            whitened=False,
            **extra,
        )


# ── posterior covariance proxy ──────────────────────────────────────────────


def test_posterior_covariance_proxy_whitened_matches_model_space(obs_model_no_optics):
    """Ŝ = U Hess_ξ⁻¹ Uᵀ: the whitened proxy reproduces the model-space one."""
    model = obs_model_no_optics
    ny, nx = 3, 3
    n = ny * nx
    variances = np.array([0.05, 0.2, 0.5, 0.8, 1.0, 1.3, 2.0, 4.0, 9.0])
    p = _problem(model, variances=variances)

    proxy_x = posterior_covariance_proxy(
        hessian_vector_product=_hvp(p["cost_x"]),
        state_size=n,
        cg_max_steps=500,
    )
    proxy_xi = posterior_covariance_proxy(
        hessian_vector_product=_hvp(p["cost_xi"]),
        state_size=n,
        whitening=p["W"],
        cg_max_steps=500,
    )
    v = jnp.asarray(np.linspace(-1.0, 1.0, n))
    np.testing.assert_allclose(
        np.asarray(proxy_xi(v)), np.asarray(proxy_x(v)), rtol=1e-3, atol=1e-8
    )


def test_posterior_covariance_proxy_matches_the_inverse_hessian(obs_model_with_psf):
    """Both forms equal the dense Hess⁻¹ (model space) and U Hess_ξ⁻¹ Uᵀ."""
    p = _problem(obs_model_with_psf)
    hess_x_inv = np.linalg.inv(_dense(_hvp(p["cost_x"])))
    proxy_x = posterior_covariance_proxy(
        hessian_vector_product=_hvp(p["cost_x"]), state_size=N, **TIGHT_CG
    )
    proxy_xi = posterior_covariance_proxy(
        hessian_vector_product=_hvp(p["cost_xi"]),
        state_size=N,
        whitening=p["W"],
        **TIGHT_CG,
    )
    np.testing.assert_allclose(_dense(proxy_x), hess_x_inv, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(_dense(proxy_xi), hess_x_inv, rtol=1e-8, atol=1e-14)


def test_posterior_covariance_proxy_runs(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    # The HVP is whitened (build_cost_xi), so pass `whitening` to get the
    # model-space covariance Ŝ = U Hess_ξ⁻¹ Uᵀ rather than the raw ξ-space one.
    matvec = posterior_covariance_proxy(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitening=W,
        cg_max_steps=50,
    )
    Bv = matvec(jnp.ones(ny * nx))
    assert Bv.shape == (ny * nx,)
    assert np.all(np.isfinite(np.asarray(Bv)))


# ── jit, vmap and grad ──────────────────────────────────────────────────────


def _scalar_diagnostics(model):
    """One scalar function of t per public diagnostic, R⁻¹ = OBS_INV eᵗ.

    Each builds its cost inside, as a caller differentiating with respect to
    the noise level would, and runs CG to machine precision so that finite
    differences see a smooth function.
    """
    truth = jnp.full((NY, NX), 1e-7)
    fwd = model.make_forward(linear=False)
    y = model.forward(truth, linear=False)
    B = build_diagonal_background(VARIANCES)
    W = WhiteningTransform.from_background(B)
    probe = jnp.linspace(-1.0, 1.0, N)

    def costs(t):
        kw = dict(
            forward_fn=fwd,
            obs_inv_variance=OBS_INV * jnp.exp(t),
            background_state=jnp.zeros((NY, NX)),
            observation=y,
            state_shape=(NY, NX),
        )
        return build_cost_x(background_op=B, **kw), build_cost_xi(whitening=W, **kw)

    def chi2(t):
        return reduced_chi_squared(
            forward_fn=fwd,
            estimated_state=truth * (1.0 + t),
            observation=y,
            obs_inv_variance=OBS_INV,
        )

    def posterior_variance(t):
        posterior = gauss_newton_posterior(
            forward_fn=fwd,
            whitening=W,
            obs_inv_variance=OBS_INV * jnp.exp(t),
            background_state=jnp.zeros((NY, NX)),
            observation=y,
            estimated_state=0.5 * truth,
        )
        return jnp.vdot(probe, posterior.covariance().mv(probe))

    def kernel_trace(t):
        kernel = averaging_kernel(
            hessian_vector_product=_hvp(costs(t)[1]),
            state_size=N,
            whitened=True,
            **TIGHT_CG,
        )
        return jnp.vdot(probe, kernel.mv(probe))

    def dfs(t):
        return degrees_of_freedom_for_signal(
            hessian_vector_product=_hvp(costs(t)[0]),
            state_size=N,
            background_op=B,
            n_probes=4,
            key=jax.random.key(1),
            **TIGHT_CG,
        ).value

    def proxy(t):
        cov = posterior_covariance_proxy(
            hessian_vector_product=_hvp(costs(t)[1]),
            state_size=N,
            whitening=W,
            **TIGHT_CG,
        )
        return jnp.vdot(probe, cov(probe))

    return {
        "reduced_chi_squared": chi2,
        "gauss_newton_posterior": posterior_variance,
        "averaging_kernel": kernel_trace,
        "degrees_of_freedom_for_signal": dfs,
        "posterior_covariance_proxy": proxy,
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "reduced_chi_squared",
        "gauss_newton_posterior",
        "averaging_kernel",
        "degrees_of_freedom_for_signal",
        "posterior_covariance_proxy",
    ],
)
def test_diagnostic_under_jit_vmap_and_grad(obs_model_with_psf, name):
    """Eager, jit and vmap agree; grad matches central finite differences."""
    f = _scalar_diagnostics(obs_model_with_psf)[name]
    ts = jnp.array([-0.5, 0.0, 0.7])
    eager = np.array([float(f(t)) for t in ts])
    np.testing.assert_allclose(jax.jit(f)(ts[1]), eager[1], rtol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.vmap(f))(ts), eager, rtol=1e-10)
    assert_gradients_match(lambda x: f(x[0]), jnp.array([0.3]))
