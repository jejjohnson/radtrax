"""The multi-pass family: SBMP, MBSP and MBMP with scale factors (radtrax#111)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from radtrax.retrieval.band_table import BandTable
from radtrax.retrieval.mbmp import (
    background_mask,
    mbmp_log_signal,
    mbmp_signal,
    mbsp_signal,
    sbmp_signal,
    scale_factor,
)


if TYPE_CHECKING:
    from _radtrax_notes import Note2


RNG = np.random.default_rng(0)
R11 = jnp.asarray(0.2 + 0.1 * RNG.random((20, 20)))
R12 = 0.8 * R11 * jnp.asarray(1 + 0.02 * RNG.normal(size=(20, 20)))


def test_scale_factor_is_least_squares_through_the_origin() -> None:
    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([2.1, 3.9, 6.2])
    np.testing.assert_allclose(
        scale_factor(x, y),
        np.linalg.lstsq(np.asarray(x)[:, None], np.asarray(y))[0][0],
        rtol=1e-14,
    )
    mask = jnp.array([True, True, False])
    np.testing.assert_allclose(scale_factor(x, y, mask), (2.1 + 7.8) / 5.0, rtol=1e-14)


def test_a_masked_nan_stays_out_of_the_fit_and_its_gradient() -> None:
    x, y = jnp.array([1.0, 2.0, 3.0, jnp.nan]), jnp.array([2.0, 4.0, 6.5, 5.0])
    keep = jnp.array([True, True, True, False])
    np.testing.assert_allclose(scale_factor(x, y, keep), 29.5 / 14.0, rtol=1e-14)
    grad = jax.grad(lambda v: scale_factor(v, y, keep))(x)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(grad[3]) == 0.0


def test_without_scale_factors_it_is_the_log_signal() -> None:
    ref11, ref12 = 1.03 * R11, 0.99 * R12
    np.testing.assert_array_equal(
        mbmp_signal(R11, R12, ref11, ref12, scale_factors=False),
        mbmp_log_signal(R11, R12, ref11, ref12),
    )


def test_scale_factors_remove_scene_wide_changes() -> None:
    """Brightening and a 1 % band-12 change between passes cancel; the plume stays
    (with ``eps = 0``: the log guard does not cancel between bands, ~1e-12)."""
    plume = jnp.zeros((20, 20)).at[8:12, 8:12].set(0.05)
    obs11, obs12 = 1.05 * R11, 1.05 * 0.99 * R12 * jnp.exp(-plume)
    background = plume == 0
    s = mbmp_signal(obs11, obs12, R11, R12, mask=background, eps=0.0)
    np.testing.assert_allclose(s[background], 0.0, atol=1e-12)
    np.testing.assert_allclose(s[~background], 0.05, rtol=1e-12)
    np.testing.assert_allclose(
        sbmp_signal(obs12, R12, mask=background, eps=0.0)[~background], 0.05, rtol=1e-12
    )
    # one pass: the band ratio's texture stays, but its scene-wide level does not
    assert abs(float(jnp.mean(mbsp_signal(R11, 0.9 * R12)))) < 1e-3


def test_nan_pixels_are_not_background() -> None:
    first = np.zeros((10, 10))
    first[2, 2] = np.nan
    mask = background_mask(first)
    assert not mask[2, 2] and mask[0, 0]


def test_background_mask_grows_the_detection() -> None:
    first = np.zeros((40, 40))
    first[18:22, 18:22] = 0.1
    mask = background_mask(first)
    assert not mask[18:22, 18:22].any()
    assert not mask[16, 20] and mask[0, 0]  # grown by the Gaussian, not everywhere


# ── radtrax · 2's numbers ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def note2() -> Note2:
    pytest.importorskip("georeader")
    from _radtrax_notes import note2

    return note2()


@pytest.mark.integration
@pytest.mark.slow
def test_a_background_mask_keeps_the_mass(note2) -> None:
    """c over every pixel moves with the plume (0.31 %): 89 % of the mass; c over
    the background mask recovers 98.8 % (radtrax · 2, stage 4)."""
    from _radtrax_notes import M_AMF

    table = BandTable(jnp.array([M_AMF]), note2.grid, (note2.tau12 - note2.tau11)[None])
    (r11p, r12p), (r11r, r12r) = note2.obs, note2.ref
    plume = note2.regions["plume"]

    def mass_fraction(mask) -> float:
        signal = mbmp_signal(r11p, r12p, r11r, r12r, mask=mask)
        omega = np.asarray(table.invert(signal, M_AMF))
        return float(omega[plume].sum() / note2.plume[plume].sum())

    all_pixels = mbmp_signal(r11p, r12p, r11r, r12r)
    mask = background_mask(np.asarray(table.invert(all_pixels, M_AMF)))
    assert round(mass_fraction(None), 2) == 0.89
    assert round(mass_fraction(mask), 3) == 0.988


@pytest.mark.integration
@pytest.mark.slow
def test_noise_at_one_percent_per_band(note2) -> None:
    """1 % multiplicative noise per band and pass, linear inversion: SBMP 0.016,
    MBSP 0.018 and MBMP 0.025 kg/m², each within 3 % of first-order theory
    (√2 ε / (m α₁₂), √2 ε / (m Δα), 2 ε / (m Δα)) (radtrax · 2, stage 5)."""
    from _radtrax_notes import M_AMF, SEED
    from scipy.ndimage import gaussian_filter

    eps, da = 0.01, note2.a12 - note2.a11
    nrng = np.random.default_rng(SEED + 2)
    ref_n = note2.ref * (1 + eps * nrng.normal(size=note2.ref.shape))
    obs_s = note2.scene(note2.a11_map, note2.ratio, note2.plume)
    obs_n = obs_s * (1 + eps * nrng.normal(size=obs_s.shape))

    def triple(o: np.ndarray, r: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "sbmp": np.asarray(sbmp_signal(o[1], r[1])) / (M_AMF * note2.a12),
            "mbsp": np.asarray(mbsp_signal(o[0], o[1])) / (M_AMF * da),
            "mbmp": np.asarray(mbmp_signal(o[0], o[1], r[0], r[1])) / (M_AMF * da),
        }

    clean, noisy = triple(obs_s, note2.ref), triple(obs_n, ref_n)
    off = ~(gaussian_filter(note2.regions["plume"].astype(float), 4) > 0.01)
    measured = {k: float(np.std((noisy[k] - clean[k])[off])) for k in clean}
    theory = {
        "sbmp": np.sqrt(2) * eps / (M_AMF * note2.a12),
        "mbsp": np.sqrt(2) * eps / (M_AMF * da),
        "mbmp": 2 * eps / (M_AMF * da),
    }
    for k in measured:
        assert abs(measured[k] / theory[k] - 1) < 0.03, (k, measured[k], theory[k])
    assert tuple(round(measured[k], 3) for k in ("sbmp", "mbsp", "mbmp")) == (
        0.016,
        0.018,
        0.025,
    )
