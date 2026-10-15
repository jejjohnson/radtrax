"""The Instrument protocol and BandInstrument (radtrax#36)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _bands import emit_like_srf
from _gradcheck import assert_gradients_match

from radtrax.instrument import BandInstrument, Instrument, SpectralResponseFunction
from radtrax.spectroscopy.synthetic import NU_CM as NU


STEP = 0.01  # the synthetic grid's spacing [cm⁻¹]


@pytest.fixture(scope="module")
def srf() -> SpectralResponseFunction:
    return emit_like_srf(NU)


@pytest.fixture(scope="module")
def instrument(srf: SpectralResponseFunction) -> BandInstrument:
    return BandInstrument.from_srf(srf, NU)


def _spectrum() -> jax.Array:
    """A smooth spectrum with structure inside every band            (G,)."""
    return jnp.asarray(1.0 + 0.3 * np.sin(NU / 3.0) + 0.1 * np.cos(NU / 0.7))


def test_the_protocol_cannot_be_built() -> None:
    with pytest.raises(TypeError):
        Instrument()  # type: ignore[abstract]


def test_unshifted_bands_equal_the_numpy_srf(
    instrument: BandInstrument, srf: SpectralResponseFunction
) -> None:
    radiance = _spectrum()
    # The NumPy SRF reads the spectrum in ascending wavelength.
    expected = srf.apply(np.asarray(radiance)[::-1])
    np.testing.assert_allclose(instrument(radiance), expected, rtol=1e-14)
    assert instrument.band_names == srf.band_names
    assert instrument.n_bands == srf.n_bands


def test_a_flat_spectrum_gives_flat_bands(instrument: BandInstrument) -> None:
    np.testing.assert_allclose(
        instrument(jnp.full(NU.size, 2.5)), np.full(instrument.n_bands, 2.5), rtol=1e-14
    )


@pytest.mark.parametrize("k", [-3, 1, 2])
def test_a_shift_of_whole_grid_steps_reads_the_neighbouring_samples(
    instrument: BandInstrument, k: int
) -> None:
    """At δ = kΔν, L(ν_g + δ) is the sample k steps away, and past the
    grid's end the edge value is held. The grid is arange(…, 0.01), so the
    nodes sit ~1e-12 cm⁻¹ off a uniform step: 1e-13 relative in L here."""
    radiance = np.asarray(_spectrum())
    shifted = eqx.tree_at(lambda i: i.shift_cm, instrument, jnp.asarray(k * STEP))
    held = np.pad(radiance, abs(k), mode="edge")  # edge values beyond the grid
    window = held[abs(k) + k : abs(k) + k + radiance.size]
    expected = np.asarray(instrument.response) @ window
    np.testing.assert_allclose(shifted(radiance), expected, rtol=1e-12)


def test_jit_and_vmap_over_shifts_match_a_loop(instrument: BandInstrument) -> None:
    radiance = _spectrum()

    def bands(shift: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda i: i.shift_cm, instrument, shift)(radiance)

    shifts = jnp.linspace(-0.02, 0.02, 5)
    batched = eqx.filter_jit(eqx.filter_vmap(bands))(shifts)
    looped = jnp.stack([bands(s) for s in shifts])
    # A batched matrix product may sum in another order: a few ulp.
    np.testing.assert_allclose(batched, looped, rtol=1e-13)


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradient_in_the_shift_between_grid_nodes(instrument: BandInstrument) -> None:
    """y is piecewise linear in δ, so the check sits mid-segment
    (δ = 0.0037 cm⁻¹, as the radtrax · 3 notebook does), where a central
    difference of step ~1e-8 cm⁻¹ never crosses a node."""
    radiance = _spectrum()

    def bands(shift: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda i: i.shift_cm, instrument, shift[0])(radiance)

    assert_gradients_match(bands, jnp.array([0.0037]))


def test_on_a_node_the_derivative_is_the_slope_above_it(
    instrument: BandInstrument,
) -> None:
    """At δ = 0, forward and reverse mode both give (y(Δν) − y(0)) / Δν,
    the rule a retrieval starting at δ = 0 relies on."""
    radiance = _spectrum()
    nu = np.asarray(instrument.nu_cm)
    step = nu[1] - nu[0]

    def bands(shift: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda i: i.shift_cm, instrument, shift)(radiance)

    slope = (
        np.asarray(bands(jnp.asarray(step))) - np.asarray(bands(jnp.asarray(0.0)))
    ) / step
    # The grid's last node takes the slope below it, the quotient sees the
    # held edge: 1e-7 relative in the band nearest that end (its response
    # there is 1.5e-5 of its peak).
    np.testing.assert_allclose(jax.jacfwd(bands)(jnp.asarray(0.0)), slope, rtol=1e-6)
    np.testing.assert_allclose(
        jax.jacrev(bands)(jnp.asarray(0.0)),
        jax.jacfwd(bands)(jnp.asarray(0.0)),
        rtol=1e-13,
    )


def test_gradient_in_the_radiance_is_the_response(instrument: BandInstrument) -> None:
    jac = jax.jacrev(instrument)(_spectrum())
    np.testing.assert_allclose(jac, instrument.response, rtol=1e-14)


# ── construction ─────────────────────────────────────────────────────────────


def test_from_srf_rejects_an_srf_on_another_grid() -> None:
    srf = emit_like_srf(NU[::2])
    with pytest.raises(ValueError, match="1e7 / nu_cm"):
        BandInstrument.from_srf(srf, NU)


def test_default_band_names() -> None:
    instrument = BandInstrument(jnp.arange(3.0), jnp.eye(3)[:2])
    assert instrument.band_names == ("B0", "B1")


@pytest.mark.usefixtures("shapes_unchecked")
@pytest.mark.parametrize(
    ("nu", "response", "shift", "names", "match"),
    [
        (jnp.zeros((2, 3)), jnp.eye(3), 0.0, None, "nu_cm must be"),
        (jnp.arange(3.0)[::-1], jnp.eye(3), 0.0, None, "strictly increasing"),
        (jnp.arange(3.0), jnp.eye(4), 0.0, None, "response must be"),
        (jnp.arange(3.0), jnp.eye(3), 0.0, ("a",), "band names"),
        (jnp.arange(3.0), jnp.eye(3), jnp.zeros(2), None, "shift_cm must be"),
    ],
)
def test_shape_errors(
    nu: jax.Array,
    response: jax.Array,
    shift: float | jax.Array,
    names: tuple[str, ...] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        BandInstrument(nu, response, shift, names)
