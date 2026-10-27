"""pair_pixels and ratio_spectrum, one-to-one pairing and the ratio (radtrax#149).

The pairing reference is the reference package's procedure written out in
NumPy and SciPy (``pv/src/pv/pv.py``): Euclidean distances on the pairing
bands, ``linear_sum_assignment``, the best-matched fraction by sorted
distance. Optimality is also checked against brute force on a problem small
enough to enumerate.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax import DegenerateInputError, ParameterError, ShapeError
from radtrax.retrieval.absorption import (
    AbsorptionWindow,
    pair_pixels,
    ratio_spectrum,
)


optimize = pytest.importorskip("scipy.optimize")

# EMIT's 285 band centres, uniformly spaced between its first and last
# (no EMIT wavelength table yet, #114), as in test_window.py.
EMIT_NM = np.linspace(381.00558, 2493.2576, 285)
PAIRING = AbsorptionWindow.ch4_emit().pairing_bands(EMIT_NM)
H, W, B = 24, 32, EMIT_NM.size


@pytest.fixture(scope="module")
def cube() -> np.ndarray:
    """Three surface materials at varying brightness, with noise, ``(H, W, B)``."""
    rng = np.random.default_rng(0)
    x = (EMIT_NM - EMIT_NM.min()) / np.ptp(EMIT_NM)
    materials = np.stack([0.2 + 0.3 * x, 0.5 - 0.2 * x, 0.3 + 0.1 * np.sin(6 * x)])
    label = rng.integers(0, 3, size=(H, W))
    brightness = rng.uniform(0.7, 1.3, size=(H, W, 1))
    noise = 0.01 * rng.standard_normal((H, W, B))
    return 10.0 * (materials[label] * brightness + noise)


IN_BOX = np.zeros((H, W), dtype=bool)
IN_BOX[8:14, 10:16] = True  # 36 in-pixels


@pytest.fixture(scope="module")
def in_pixels() -> np.ndarray:
    return np.flatnonzero(IN_BOX)


@pytest.fixture(scope="module")
def out_pixels() -> np.ndarray:
    return np.flatnonzero(~IN_BOX)


def _pv_pairs(
    cube: np.ndarray, i_in: np.ndarray, i_out: np.ndarray, bands: np.ndarray, keep
) -> set[tuple[int, int]]:
    """The reference package's pairing, written out."""
    flat = cube.reshape(-1, cube.shape[-1])[:, bands]
    diff = flat[i_in][:, None, :] - flat[i_out][None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=-1))
    rows, cols = optimize.linear_sum_assignment(distance)
    order = np.argsort(distance[rows, cols])[: int(keep * len(rows))]
    return {(int(i_in[rows[n]]), int(i_out[cols[n]])) for n in order}


# ── pair_pixels ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("keep", [0.5, 0.25, 1.0, 0.3])
def test_pairs_are_one_to_one_and_keep_the_stated_fraction(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray, keep: float
) -> None:
    in_idx, out_idx = pair_pixels(cube, in_pixels, out_pixels, PAIRING, keep=keep)
    assert in_idx.shape == out_idx.shape == (int(keep * in_pixels.size),)
    assert np.unique(out_idx).size == out_idx.size
    assert np.unique(in_idx).size == in_idx.size
    assert np.isin(in_idx, in_pixels).all() and np.isin(out_idx, out_pixels).all()


def test_vt3_pairs_equal_the_reference_packages(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> None:
    """Gate VT3, a parity check: the same pairs as the reference package.

    Euclidean distance, one-to-one assignment, best half, computed by the
    reference package's own steps (``_pv_pairs``) on the same cube.
    """
    in_idx, out_idx = pair_pixels(cube, in_pixels, out_pixels, PAIRING)
    got = set(zip(in_idx.tolist(), out_idx.tolist(), strict=True))
    assert got == _pv_pairs(cube, in_pixels, out_pixels, PAIRING, 0.5)


def test_kept_pairs_are_ordered_and_better_than_the_dropped(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> None:
    flat = cube.reshape(-1, B)[:, PAIRING]
    every_in, every_out = pair_pixels(cube, in_pixels, out_pixels, PAIRING, keep=1.0)
    d = np.linalg.norm(flat[every_in] - flat[every_out], axis=-1)
    assert np.all(np.diff(d) >= 0)
    in_idx, out_idx = pair_pixels(cube, in_pixels, out_pixels, PAIRING)
    np.testing.assert_array_equal(in_idx, every_in[: in_idx.size])
    np.testing.assert_array_equal(out_idx, every_out[: out_idx.size])


def test_every_in_pixel_finds_its_exact_copy_among_the_out_pixels(
    cube: np.ndarray,
) -> None:
    box = np.zeros((H, W), dtype=bool)
    box[2:6, 2:6] = True  # 16 in-pixels
    planted = cube.copy()
    # An absorber on the in-pixels' absorbing bands only: pairing must not
    # see it.
    planted[2:6, 2:6] = cube[16:20, 20:24] * np.where(PAIRING, 1.0, 0.8)
    # The copies among every other pixel.
    in_idx, out_idx = pair_pixels(
        planted, np.flatnonzero(box), np.flatnonzero(~box), PAIRING, keep=1.0
    )
    r_in, c_in = np.unravel_index(in_idx, (H, W))
    r_out, c_out = np.unravel_index(out_idx, (H, W))
    np.testing.assert_array_equal(r_out, r_in + 14)
    np.testing.assert_array_equal(c_out, c_in + 18)


def test_the_assignment_minimises_the_total_distance() -> None:
    """Brute force over every injection of 4 in-pixels into 6 out-pixels."""
    rng = np.random.default_rng(1)
    cube = rng.uniform(1.0, 2.0, size=(1, 10, 3))
    bands = np.array([True, True, False])
    in_idx, out_idx = pair_pixels(cube, np.arange(4), np.arange(4, 10), bands, keep=1.0)
    flat = cube[0][:, bands]
    total = np.linalg.norm(flat[in_idx] - flat[out_idx], axis=-1).sum()
    best = min(
        sum(np.linalg.norm(flat[i] - flat[4 + j]) for i, j in enumerate(perm))
        for perm in itertools.permutations(range(6), 4)
    )
    np.testing.assert_allclose(total, best, rtol=1e-12)  # exact up to rounding


def test_any_order_and_integer_type_of_the_sets_gives_the_same_pairs(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> None:
    rng = np.random.default_rng(3)
    shuffled_in = rng.permutation(in_pixels).astype(np.int32)
    shuffled_out = rng.permutation(out_pixels).astype(np.int32)
    expected = pair_pixels(cube, in_pixels, out_pixels, PAIRING)
    got = pair_pixels(cube, shuffled_in, shuffled_out, PAIRING)
    # The optimal assignment is unique for continuous noise, so the kept
    # pairs and their best-first order do not depend on the sets' order.
    for a, b in zip(expected, got, strict=True):
        np.testing.assert_array_equal(a, b)
        assert b.dtype == np.intp


def test_only_the_pairing_bands_decide_the_pairs(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> None:
    scrambled = cube.copy()
    rng = np.random.default_rng(2)
    scrambled[..., ~PAIRING] = rng.uniform(0.0, 50.0, size=(H, W, (~PAIRING).sum()))
    for a, b in zip(
        pair_pixels(cube, in_pixels, out_pixels, PAIRING),
        pair_pixels(scrambled, in_pixels, out_pixels, PAIRING),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("keep", [0.0, -0.1, 1.5, float("nan")])
def test_keep_outside_the_unit_interval_raises_parameter_error(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray, keep: float
) -> None:
    with pytest.raises(ParameterError, match="keep"):
        pair_pixels(cube, in_pixels, out_pixels, PAIRING, keep=keep)


def test_fewer_out_pixels_than_in_pixels_raises(
    cube: np.ndarray, in_pixels: np.ndarray
) -> None:
    few = np.arange(35)  # 35 < 36 in-pixels, row 0 (outside the box)
    with pytest.raises(DegenerateInputError, match="fewer than the 36 in-pixels"):
        pair_pixels(cube, in_pixels, few, PAIRING)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no in-pixels", "no in-pixels"),
        ("no pairing band", "no band"),
        ("no pair kept", "keeps no pair"),
        ("non-finite", "non-finite"),
    ],
)
def test_degenerate_inputs_raise(
    cube: np.ndarray,
    in_pixels: np.ndarray,
    out_pixels: np.ndarray,
    case: str,
    match: str,
) -> None:
    bands, keep, data = PAIRING, 0.5, cube
    if case == "no in-pixels":
        in_pixels = np.array([], dtype=int)
    elif case == "no pairing band":
        bands = np.zeros_like(PAIRING)
    elif case == "no pair kept":
        keep = 0.01  # int(0.01 · 36) = 0
    else:
        data = cube.copy()
        data[0, 0, np.flatnonzero(PAIRING)[0]] = np.nan
    with pytest.raises(DegenerateInputError, match=match):
        pair_pixels(data, in_pixels, out_pixels, bands, keep=keep)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("overlap", "share a pixel"),
        ("repeat in", "in_pixels repeats"),
        ("repeat out", "out_pixels repeats"),
        ("negative", r"in_pixels must lie in \[0, 768\)"),
        ("too large", r"out_pixels must lie in \[0, 768\)"),
    ],
)
def test_invalid_index_sets_raise_parameter_error(
    cube: np.ndarray,
    in_pixels: np.ndarray,
    out_pixels: np.ndarray,
    case: str,
    match: str,
) -> None:
    if case == "overlap":
        out_pixels = np.append(out_pixels, in_pixels[0])
    elif case == "repeat in":
        in_pixels = np.append(in_pixels, in_pixels[0])
    elif case == "repeat out":
        out_pixels = np.append(out_pixels, out_pixels[0])
    elif case == "negative":
        in_pixels = np.append(in_pixels, -1)
    else:
        out_pixels = np.append(out_pixels, H * W)
    with pytest.raises(ParameterError, match=match):
        pair_pixels(cube, in_pixels, out_pixels, PAIRING)


@pytest.mark.usefixtures("shapes_unchecked")
def test_non_integer_index_sets_raise_parameter_error(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> None:
    with pytest.raises(ParameterError, match="integer"):
        pair_pixels(cube, in_pixels.astype(float), out_pixels, PAIRING)
    with pytest.raises(ParameterError, match="integer"):
        pair_pixels(cube, in_pixels, IN_BOX.ravel(), PAIRING)  # a mask, not a set


@pytest.mark.usefixtures("shapes_unchecked")
@pytest.mark.parametrize("which", ["cube", "in", "out", "bands"])
def test_mismatched_shapes_raise_shape_error(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray, which: str
) -> None:
    args = [cube, in_pixels, out_pixels, PAIRING]
    args[["cube", "in", "out", "bands"].index(which)] = {
        "cube": cube[..., 0],
        "in": IN_BOX,  # a 2-D mask, not flat indices
        "out": out_pixels[None],
        "bands": PAIRING[:-1],
    }[which]
    with pytest.raises(ShapeError):
        pair_pixels(*args)


# ── ratio_spectrum ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pairs(
    cube: np.ndarray, in_pixels: np.ndarray, out_pixels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return pair_pixels(cube, in_pixels, out_pixels, PAIRING)


def test_ratio_is_the_ratio_of_the_means(
    cube: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    in_idx, out_idx = pairs
    flat = cube.reshape(-1, B)
    expected = flat[in_idx].mean(0) / flat[out_idx].mean(0)
    ratio = ratio_spectrum(cube, in_idx, out_idx)
    assert ratio.shape == (B,)
    np.testing.assert_allclose(ratio, expected, rtol=1e-13)  # rounding only


def test_ratio_of_a_uniform_transmittance_recovers_it(
    cube: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    in_idx, out_idx = pairs
    transmittance = np.linspace(0.6, 1.0, B)
    flat = cube.reshape(-1, B).copy()
    flat[in_idx] = flat[out_idx] * transmittance  # perfect partners, an absorber
    ratio = ratio_spectrum(flat.reshape(H, W, B), in_idx, out_idx)
    np.testing.assert_allclose(ratio, transmittance, rtol=1e-13)


def test_non_positive_out_pixel_radiance_leaves_the_pair_out_of_that_band() -> None:
    # Pixels 0, 1 are in-pixels; 2, 3 their partners. Pixel 3 is not positive in
    # band 1, and neither partner is in band 2.
    cube = np.array(
        [[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [2.0, 8.0, 0.0], [2.0, -1.0, -2.0]]]
    )
    ratio = np.asarray(ratio_spectrum(cube, np.array([0, 1]), np.array([2, 3])))
    np.testing.assert_allclose(ratio[:2], [4.0 / 4.0, 2.0 / 8.0], rtol=1e-15)
    assert np.isnan(ratio[2])


def test_a_non_finite_radiance_leaves_its_pair_out_of_that_band() -> None:
    # Pairs (0, 2) and (1, 3). Pixel 0 is NaN in band 1, pixel 3 inf in band 2.
    cube = np.array(
        [[[1.0, np.nan, 1.0], [3.0, 4.0, 2.0], [2.0, 8.0, 4.0], [2.0, 2.0, np.inf]]]
    )
    ratio = np.asarray(ratio_spectrum(cube, np.array([0, 1]), np.array([2, 3])))
    np.testing.assert_allclose(ratio, [4.0 / 4.0, 4.0 / 2.0, 1.0 / 4.0], rtol=1e-15)


@pytest.mark.slow
def test_a_nan_in_pixel_gives_finite_gradients_zero_where_masked() -> None:
    cube = np.array([[[1.0, 2.0], [3.0, 4.0], [2.0, 8.0], [2.0, 2.0]]])
    cube[0, 0, 1] = np.nan  # pair (0, 2) is unusable in band 1
    jac = jax.jacrev(lambda c: ratio_spectrum(c, np.array([0, 1]), np.array([2, 3])))(
        jnp.asarray(cube)
    )  # (B, 1, 4, B)
    assert np.all(np.isfinite(jac))
    np.testing.assert_array_equal(jac[1, 0, [0, 2], 1], 0.0)  # the masked pair
    # Band 1 is pixel 1 over pixel 3: ∂R/∂L₁ = 1/2, ∂R/∂L₃ = −4/2² = −1.
    np.testing.assert_allclose(jac[1, 0, [1, 3], 1], [0.5, -1.0], rtol=1e-15)


def test_ratio_gradients_match_central_differences(
    cube: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    in_idx, out_idx = pairs
    pixels = np.concatenate([in_idx[:3], out_idx[:3]])
    bands = np.array([0, 140, 284])
    sub = jnp.asarray(cube.reshape(-1, B)[np.ix_(pixels, bands)])

    def ratio_of(x: jnp.ndarray) -> jnp.ndarray:
        """R on three bands as a function of six pixels' radiances there."""
        small = x.reshape(1, 6, 3)
        return ratio_spectrum(small, np.arange(3), np.arange(3, 6))

    assert_gradients_match(ratio_of, sub.ravel())


@pytest.mark.slow
def test_unusable_band_has_a_zero_not_nan_gradient() -> None:
    cube = jnp.array([[[1.0, 2.0], [3.0, -1.0]]])
    jac = jax.jacrev(lambda c: ratio_spectrum(c, np.array([0]), np.array([1])))(cube)
    assert np.isnan(ratio_spectrum(cube, np.array([0]), np.array([1]))[1])
    assert np.all(np.isfinite(jac))
    np.testing.assert_array_equal(jac[1], 0.0)


def test_ratio_runs_under_jit(
    cube: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    in_idx, out_idx = pairs
    jitted = eqx.filter_jit(lambda c: ratio_spectrum(c, in_idx, out_idx))
    np.testing.assert_allclose(
        jitted(jnp.asarray(cube)), ratio_spectrum(cube, in_idx, out_idx), rtol=1e-15
    )


def test_ratio_without_pairs_raises(cube: np.ndarray) -> None:
    empty = np.array([], dtype=int)
    with pytest.raises(DegenerateInputError, match="no pairs"):
        ratio_spectrum(cube, empty, empty)


def test_ratio_index_out_of_range_raises(cube: np.ndarray) -> None:
    with pytest.raises(ParameterError, match="out_idx"):
        ratio_spectrum(cube, np.array([0]), np.array([H * W]))


@pytest.mark.usefixtures("shapes_unchecked")
def test_ratio_with_mismatched_indices_raises_shape_error(cube: np.ndarray) -> None:
    with pytest.raises(ShapeError, match="one length"):
        ratio_spectrum(cube, np.array([0, 1]), np.array([2]))
    with pytest.raises(ShapeError, match=r"\(H, W, B\)"):
        ratio_spectrum(cube[..., 0], np.array([0]), np.array([1]))
