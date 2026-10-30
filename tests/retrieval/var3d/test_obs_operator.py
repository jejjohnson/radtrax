"""Tests for the JAX observation operator.

Three invariants:

1. **Forward shape + zero-VMR sanity** — at ``ΔVMR = 0`` the radiance equals
   the L1-normalised SRF acting on a flat unit spectrum, i.e. ``1.0`` per band.
2. **Linear == nonlinear at small ΔVMR** — Maclaurin-1 differs from
   ``exp(−Δτ)`` by O(Δτ²); for tiny ΔVMR they should match to ~1e-6.
3. **Adjoint identity via JAX VJP** — for the *linear* forward,
   ``⟨H δx, u⟩ = ⟨δx, Hᵀ u⟩`` to machine precision via :func:`jax.vjp`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


def test_forward_shape_no_optics(obs_model_no_optics):
    model = obs_model_no_optics
    x = jnp.zeros((6, 6))
    y = model.forward(x)
    assert y.shape == (6, 6, model.n_bands)


def test_forward_zero_vmr_is_unit_radiance(obs_model_no_optics):
    """At ΔVMR = 0, exp(0) = 1 and the SRF integrates to 1 per band."""
    model = obs_model_no_optics
    y = model.forward(jnp.zeros((4, 4)))
    np.testing.assert_allclose(np.asarray(y), 1.0, atol=1e-12)


def test_linear_approximates_nonlinear_at_small_vmr(obs_model_no_optics):
    model = obs_model_no_optics
    x = jnp.full((3, 3), 1e-9)  # small VMR perturbation
    y_nl = model.forward(x, linear=False)
    y_lin = model.forward(x, linear=True)
    # Δτ ≈ a · 1e-9 ≈ 1e-9 · O(1e-19 cross-section · 1e19 number density · 1e6 path)
    # → very small; linear and nonlinear agree to 1e-9 absolute.
    np.testing.assert_allclose(np.asarray(y_nl), np.asarray(y_lin), atol=1e-8)


def test_with_psf_gsd_shape(obs_model_with_gsd):
    model = obs_model_with_gsd
    y = model.forward(jnp.zeros((8, 8)))
    assert y.shape == (4, 4, model.n_bands)


def test_jax_vjp_adjoint_identity_linear(obs_model_no_optics):
    """For the linear forward, ⟨H δx, u⟩ = ⟨δx, Hᵀ u⟩ via jax.vjp."""
    model = obs_model_no_optics

    def H(delta_x_field):
        # Linear forward, shifted to remove the constant '1' so we get a pure
        # linear map: H(δx) = -a(ν) · δx (band-integrated).
        return model.forward(delta_x_field, linear=True) - model.forward(
            jnp.zeros_like(delta_x_field), linear=True
        )

    rng = np.random.default_rng(0)
    dx = jnp.asarray(rng.standard_normal((4, 4)))
    u = jnp.asarray(rng.standard_normal((4, 4, model.n_bands)))
    Hdx, vjp = jax.vjp(H, dx)
    (Htu,) = vjp(u)
    lhs = float(jnp.sum(Hdx * u))
    rhs = float(jnp.sum(dx * Htu))
    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)


def test_jit_runs(obs_model_no_optics):
    model = obs_model_no_optics
    fwd = jax.jit(model.make_forward(linear=False))
    y = fwd(jnp.zeros((5, 5)))
    assert y.shape == (5, 5, model.n_bands)


# ── the band integration is a BandInstrument (radtrax#95) ────────────────────


def _vmr_field() -> jnp.ndarray:
    rows, cols = jnp.meshgrid(jnp.arange(5.0), jnp.arange(4.0), indexing="ij")
    return 1e-7 * (1.0 + 0.3 * rows + 0.1 * cols)


def test_forward_equals_the_former_wavelength_ordered_srf_product(
    obs_model_with_psf_and_gsd, hyperspectral_srf
):
    """The einsum the model used before radtrax#95, on the same inputs."""
    model = obs_model_with_psf_and_gsd
    a_wl = model.absorption_coeff_hr[::-1]  # back to ascending wavelength
    x = _vmr_field()
    L_hr = jnp.exp(-(x - model.vmr_reference)[..., None] * a_wl)
    L_hr = model.gsd.apply(model.psf.apply(L_hr))
    former = jnp.einsum("bl, ijl -> ijb", jnp.asarray(hyperspectral_srf.matrix), L_hr)
    # Same products summed in reverse order: rounding only.
    np.testing.assert_allclose(model.forward(x), former, rtol=1e-12, atol=0)


def test_from_lut_puts_both_on_the_srf_wavenumbers(
    obs_model_no_optics, hyperspectral_srf
):
    model = obs_model_no_optics
    nu = np.asarray(model.instrument.nu_cm)
    np.testing.assert_allclose(1e7 / nu[::-1], hyperspectral_srf.wavelengths_hr_nm)
    assert np.all(np.diff(nu) > 0) and float(model.instrument.shift_cm) == 0.0
    assert model.absorption_coeff_hr.shape == nu.shape


def test_the_spectral_shift_is_differentiable(synthetic_lut, hyperspectral_srf):
    """A 3D-Var retrieval can estimate δ. y is piecewise linear in δ, so the
    check sits mid-segment of the 2.5 cm⁻¹ grid, as for BandInstrument, and at
    VMRs of 1e-5, where the lines are deep enough (τ ≈ 1) that every entry of
    ∂y/∂δ stands above the central difference's rounding."""
    import equinox as eqx
    from _gradcheck import assert_gradients_match

    from radtrax.retrieval.var3d import RadianceObservationModel

    model = RadianceObservationModel.from_lut(
        synthetic_lut,
        srf=hyperspectral_srf,
        T_K=280.0,
        p_atm=1.0,
        path_length_cm=8.4e5,
        amf=2.0,
        shift_cm=0.9,
    )
    assert float(model.instrument.shift_cm) == 0.9
    x = 100.0 * _vmr_field()

    def y(shift: jnp.ndarray) -> jnp.ndarray:
        return eqx.tree_at(lambda m: m.instrument.shift_cm, model, shift[0]).forward(x)

    assert_gradients_match(y, jnp.array([0.9]))


def test_the_deprecated_srf_matrix_forms_give_the_same_numbers(
    obs_model_no_optics, hyperspectral_srf
):
    import pytest

    from radtrax.retrieval.var3d import RadianceObservationModel

    model = obs_model_no_optics
    with pytest.warns(DeprecationWarning, match="srf_matrix"):
        srf_matrix = model.srf_matrix
    np.testing.assert_array_equal(srf_matrix, model.instrument.response[:, ::-1])
    np.testing.assert_allclose(srf_matrix, hyperspectral_srf.matrix, rtol=1e-15)
    with pytest.warns(DeprecationWarning, match="BandInstrument"):
        old = RadianceObservationModel(model.absorption_coeff_hr[::-1], srf_matrix)
    x = _vmr_field()
    np.testing.assert_allclose(old.forward(x), model.forward(x), rtol=1e-14, atol=0)
