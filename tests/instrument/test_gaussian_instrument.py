"""GaussianInstrument: Gaussian bands shifted analytically in cm⁻¹ or nm (radtrax#96)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _bands import emit_like_srf
from _gradcheck import assert_gradients_match

from radtrax import GridError, ParameterError, ShapeError
from radtrax.instrument import (
    BandInstrument,
    GaussianInstrument,
    SpectralResponseFunction,
)
from radtrax.spectroscopy.synthetic import NU_CM as NU


SRF = emit_like_srf(np.asarray(NU))
# A spectrum with structure on the bands' scale: a few absorption lines.
RADIANCE = 1.0 - 0.4 * jnp.exp(
    -(((NU[:, None] - jnp.array([4268.0, 4301.5, 4333.0])) / 0.8) ** 2)
).sum(1)


def test_at_zero_shift_it_is_the_band_instrument() -> None:
    band = BandInstrument.from_srf(SRF, NU)
    gaussian = GaussianInstrument.from_srf(SRF, NU)
    np.testing.assert_allclose(
        gaussian.response(), band.response, rtol=1e-13, atol=1e-18
    )
    np.testing.assert_allclose(gaussian(RADIANCE), band(RADIANCE), rtol=1e-13)
    assert gaussian.band_names == band.band_names


def _bands_at(shift: jax.Array) -> jax.Array:
    return GaussianInstrument.from_srf(SRF, NU, shift_cm=shift)(RADIANCE)


def test_the_shift_has_curvature() -> None:
    """∂²y/∂δ² is finite and nonzero, and matches central differences of ∂y/∂δ."""
    second = jax.jacfwd(jax.jacfwd(_bands_at))(0.0)
    assert bool(jnp.all(jnp.isfinite(second))) and float(jnp.max(jnp.abs(second))) > 0
    h = 1e-3
    first = jax.jacfwd(_bands_at)
    central = (first(h) - first(-h)) / (2 * h)
    # Central differences of a smooth derivative: O(h²) ≈ 1e-6 relative.
    np.testing.assert_allclose(
        second, central, rtol=1e-5, atol=1e-8 * float(jnp.max(jnp.abs(second)))
    )


def test_the_slope_in_the_shift_agrees_with_the_band_instrument() -> None:
    """BandInstrument's interpolated shift has a derivative within 0.2 % of an
    analytically shifted SRF's on this 0.01 cm⁻¹ grid (numerics review of #94)."""
    analytic = jax.jacfwd(_bands_at)(0.0)
    band = BandInstrument.from_srf(SRF, NU)
    interpolated = jax.jacfwd(
        lambda d: eqx.tree_at(lambda m: m.shift_cm, band, d)(RADIANCE)
    )(0.0)
    scale = float(jnp.max(jnp.abs(analytic)))
    np.testing.assert_allclose(interpolated, analytic, atol=2e-3 * scale)


def test_a_shift_in_nm_moves_the_centres() -> None:
    shifted = GaussianInstrument.from_srf(SRF, NU, shift_nm=0.3)
    moved = GaussianInstrument(NU, SRF.band_centers_nm + 0.3, SRF.band_widths_nm)
    np.testing.assert_allclose(shifted(RADIANCE), moved(RADIANCE), rtol=1e-12)


def test_a_shift_in_cm_moves_the_bands_to_higher_wavenumber() -> None:
    """A line 3 cm⁻¹ above a band's centre: a positive δ moves the band onto it."""
    instrument = GaussianInstrument.from_srf(SRF, NU)
    b = instrument.n_bands // 2
    center_nu = 1e7 / float(SRF.band_centers_nm[b])
    line = 1.0 - 0.5 * jnp.exp(-(((NU - (center_nu + 3.0)) / 0.3) ** 2))

    def band(shift: float) -> float:
        return float(GaussianInstrument.from_srf(SRF, NU, shift_cm=shift)(line)[b])

    assert band(1.0) < band(0.0)


def test_gradients_in_the_shifts_and_widths() -> None:
    def bands(theta: jax.Array) -> jax.Array:
        return GaussianInstrument(
            NU, SRF.band_centers_nm, theta[2:], shift_cm=theta[0], shift_nm=theta[1]
        )(RADIANCE)

    theta = jnp.concatenate([jnp.array([0.02, -0.1]), jnp.asarray(SRF.band_widths_nm)])
    assert_gradients_match(bands, theta)


def test_it_jits_and_vmaps_over_shifts() -> None:
    shifts = jnp.array([0.0, 0.05])
    batch = eqx.filter_jit(jax.vmap(_bands_at))(shifts)
    np.testing.assert_allclose(batch[1], _bands_at(0.05), rtol=1e-13)


def test_a_band_shifted_off_the_grid_fails_loudly() -> None:
    """About 160 nm off: the response underflows on the whole grid, so 0/0 would be NaN."""
    with pytest.raises(RuntimeError, match="no response on the grid"):
        _bands_at(300.0)


def test_from_srf_needs_gaussian_bands() -> None:
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=SRF.wavelengths_hr_nm,
        band_centers_nm=SRF.band_centers_nm,
        band_widths_nm=SRF.band_widths_nm,
        band_names=SRF.band_names,
        srf_type="rectangular",
    )
    with pytest.raises(ParameterError, match="Gaussian"):
        GaussianInstrument.from_srf(srf, NU)


@pytest.mark.usefixtures("shapes_unchecked")
def test_shapes_and_the_grid_are_checked() -> None:
    centers, widths = jnp.array([2320.0, 2330.0]), jnp.array([8.5, 8.5])
    with pytest.raises(ShapeError, match="alike"):
        GaussianInstrument(NU, centers, widths[:1])
    with pytest.raises(ShapeError, match="band names"):
        GaussianInstrument(NU, centers, widths, band_names=("a",))
    with pytest.raises(ShapeError, match="scalars"):
        GaussianInstrument(NU, centers, widths, shift_cm=jnp.zeros(2))
    with pytest.raises(GridError, match="GaussianInstrument: nu_cm must be strictly"):
        GaussianInstrument(NU[::-1], centers, widths)
