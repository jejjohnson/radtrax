"""The var3d classes as Equinox modules (radtrax#58).

As a frozen dataclass the observation model was hashed as a static ``jit``
argument, so every new model recompiled, and it could not be differentiated as
an argument (only through a closure that rebuilt it). As an ``eqx.Module`` its
arrays, including ``vmr_reference`` and the PSF kernel, are leaves. These tests
pin that down.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.instrument import BandInstrument
from radtrax.retrieval.var3d import (
    Cost,
    IdentityTransform,
    RadianceObservationModel,
    SolveResult,
    WhiteningTransform,
    build_cost_x,
    build_diagonal_background,
)


NY = NX = 4
VMR = jnp.full((NY, NX), 1.5e-7)


@pytest.mark.parametrize(
    "cls",
    [
        RadianceObservationModel,
        IdentityTransform,
        WhiteningTransform,
        Cost,
        SolveResult,
    ],
)
def test_classes_are_equinox_modules(cls) -> None:
    assert issubclass(cls, eqx.Module)


def test_model_arrays_are_leaves(obs_model_with_psf) -> None:
    m = obs_model_with_psf
    leaves = jax.tree_util.tree_leaves(m)
    arrays = (
        m.absorption_coeff_hr,
        m.instrument.response,
        m.instrument.shift_cm,
        m.psf.kernel,
    )
    for array in arrays:
        assert any(leaf is array for leaf in leaves)
    assert (m.n_lambda, m.n_bands) == (
        m.absorption_coeff_hr.size,
        m.instrument.response.shape[0],
    )


def test_filter_grad_with_respect_to_the_model(obs_model_with_psf) -> None:
    """Gradients reach the model's leaves when it is passed as an argument.

    The frozen dataclass could not be an argument to a gradient transform;
    only a closure that rebuilt it carried the gradient.
    """
    m = obs_model_with_psf
    y_obs = m.forward(VMR) * 1.001  # a small misfit, so the gradient is not zero

    def loss(model: RadianceObservationModel) -> jax.Array:
        return jnp.sum((model.forward(VMR) - y_obs) ** 2)

    g = eqx.filter_grad(loss)(m)
    for leaf in (
        g.instrument.response,
        g.absorption_coeff_hr,
        g.vmr_reference,
        g.psf.kernel,
    ):
        assert leaf is not None and bool(jnp.any(leaf != 0))

    # The SRF entries of that gradient, against central differences.
    flat = m.instrument.response.ravel()
    idx = list(range(0, flat.size, flat.size // 12))[:12]

    def loss_of_srf(srf_flat: jax.Array) -> jax.Array:
        return loss(
            eqx.tree_at(
                lambda mm: mm.instrument.response,
                m,
                srf_flat.reshape(m.instrument.response.shape),
            )
        )

    result = assert_gradients_match(loss_of_srf, flat, indices=idx)
    np.testing.assert_allclose(
        result.jac_autodiff[0],
        np.asarray(g.instrument.response).ravel()[idx],
        rtol=1e-12,
    )
    # And the scalar background VMR, which was static as a Python float. In
    # ppm, so the variable is O(1): _gradcheck's step is ε^{1/3}·max(1, |x|),
    # far too large for a VMR of order 1e-7.
    assert_gradients_match(
        lambda ppm: loss(eqx.tree_at(lambda mm: mm.vmr_reference, m, 1e-6 * ppm[0])),
        jnp.array([0.01]),
    )


def test_cost_over_make_forward_still_grads(obs_model_no_optics) -> None:
    # The optimisers' path: a Cost over make_forward evaluates and grads.
    m = obs_model_no_optics
    cost = build_cost_x(
        forward_fn=m.make_forward(),
        background_op=build_diagonal_background(1e-12, n_pixels=NY * NX),
        obs_inv_variance=1e8,
        background_state=VMR,
        observation=m.forward(VMR) * 1.001,
        state_shape=(NY, NX),
    )
    assert bool(jnp.all(jnp.isfinite(cost.grad(jnp.zeros(NY * NX)))))


def test_one_compilation_serves_two_models(obs_model_no_optics) -> None:
    traces = []

    @eqx.filter_jit
    def forward(model: RadianceObservationModel, vmr: jax.Array) -> jax.Array:
        traces.append(1)  # runs once per trace, not per call
        return model.forward(vmr)

    m1 = obs_model_no_optics
    m2 = eqx.tree_at(
        lambda mm: mm.absorption_coeff_hr, m1, 2.0 * m1.absorption_coeff_hr
    )
    # A different background VMR, set the way a user would: through __init__.
    m3 = RadianceObservationModel(
        m1.absorption_coeff_hr, m1.instrument, vmr_reference=1e-8
    )
    outs = [forward(mm, VMR) for mm in (m1, m2, m3)]
    assert len(traces) == 1
    assert not np.allclose(outs[0], outs[1])  # really different models
    assert not np.allclose(outs[0], outs[2])


def test_filter_jit_of_the_forward_method(obs_model_no_optics) -> None:
    # jax.jit(model.forward) no longer works (not hashable); this replaces it.
    m = obs_model_no_optics
    np.testing.assert_allclose(
        eqx.filter_jit(m.forward)(VMR), m.forward(VMR), rtol=1e-15
    )
    with pytest.raises(TypeError):
        hash(m)


def test_vmap_over_a_stack_of_models_equals_a_loop(obs_model_with_psf) -> None:
    m = obs_model_with_psf
    models = [
        eqx.tree_at(lambda mm: mm.absorption_coeff_hr, m, s * m.absorption_coeff_hr)
        for s in (0.5, 1.0, 2.0)
    ]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *models)
    batched = eqx.filter_vmap(lambda mm: mm.forward(VMR))(stacked)
    looped = jnp.stack([mm.forward(VMR) for mm in models])
    # Same arithmetic; batching may reorder the sums on another backend.
    np.testing.assert_allclose(batched, looped, rtol=1e-14, atol=0)


@pytest.mark.usefixtures("shapes_unchecked")
def test_constructor_still_validates_shapes() -> None:
    instrument = BandInstrument(jnp.arange(5.0), jnp.full((3, 5), 0.2))
    with pytest.raises(ValueError, match="1-D"):
        RadianceObservationModel(jnp.ones((2, 2)), instrument)
    with pytest.raises(ValueError, match="does not"):
        RadianceObservationModel(jnp.ones(4), instrument)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="does not"),
    ):
        RadianceObservationModel(jnp.ones(4), jnp.ones((3, 5)))
    with pytest.raises(TypeError, match="BandInstrument"):
        RadianceObservationModel(jnp.ones(5), IdentityTransform())
