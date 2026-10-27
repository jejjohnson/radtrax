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
from radtrax.retrieval.vetting import (
    VettingWindow,
    pair_pixels,
    ratio_spectrum,
)


optimize = pytest.importorskip("scipy.optimize")

# EMIT's 285 band centres, uniformly spaced between its first and last
# (no EMIT wavelength table yet, #114), as in test_window.py.
EMIT_NM = np.linspace(381.00558, 2493.2576, 285)
PAIRING = VettingWindow.ch4_emit().pairing_bands(EMIT_NM)
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


@pytest.fixture(scope="module")
def seeds() -> np.ndarray:
    out = np.zeros((H, W), dtype=bool)
    out[8:14, 10:16] = True  # 36 seeds
    return out


@pytest.fixture(scope="module")
def pool(seeds: np.ndarray) -> np.ndarray:
    return ~seeds


def _pv_pairs(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray, bands: np.ndarray, keep
) -> set[tuple[int, int]]:
    """The reference package's pairing, written out."""
    flat = cube.reshape(-1, cube.shape[-1])[:, bands]
    i_in, i_out = np.flatnonzero(seeds), np.flatnonzero(pool)
    diff = flat[i_in][:, None, :] - flat[i_out][None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=-1))
    rows, cols = optimize.linear_sum_assignment(distance)
    order = np.argsort(distance[rows, cols])[: int(keep * len(rows))]
    return {(int(i_in[rows[n]]), int(i_out[cols[n]])) for n in order}


# ── pair_pixels ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("keep", [0.5, 0.25, 1.0, 0.3])
def test_pairs_are_one_to_one_and_keep_the_stated_fraction(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray, keep: float
) -> None:
    in_idx, out_idx = pair_pixels(cube, seeds, pool, PAIRING, keep=keep)
    n_in = int(seeds.sum())
    assert in_idx.shape == out_idx.shape == (int(keep * n_in),)
    assert np.unique(out_idx).size == out_idx.size
    assert np.unique(in_idx).size == in_idx.size
    assert seeds.ravel()[in_idx].all() and pool.ravel()[out_idx].all()


def test_vt3_pairs_equal_the_reference_packages(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray
) -> None:
    """Gate VT3: Euclidean distance, one-to-one assignment, best half."""
    in_idx, out_idx = pair_pixels(cube, seeds, pool, PAIRING)
    got = set(zip(in_idx.tolist(), out_idx.tolist(), strict=True))
    assert got == _pv_pairs(cube, seeds, pool, PAIRING, 0.5)


def test_kept_pairs_are_ordered_and_better_than_the_dropped(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray
) -> None:
    flat = cube.reshape(-1, B)[:, PAIRING]
    every_in, every_out = pair_pixels(cube, seeds, pool, PAIRING, keep=1.0)
    d = np.linalg.norm(flat[every_in] - flat[every_out], axis=-1)
    assert np.all(np.diff(d) >= 0)
    in_idx, out_idx = pair_pixels(cube, seeds, pool, PAIRING)
    np.testing.assert_array_equal(in_idx, every_in[: in_idx.size])
    np.testing.assert_array_equal(out_idx, every_out[: out_idx.size])


def test_every_seed_finds_its_exact_copy_in_the_pool(cube: np.ndarray) -> None:
    seeds = np.zeros((H, W), dtype=bool)
    seeds[2:6, 2:6] = True  # 16 seeds
    copies = np.zeros((H, W), dtype=bool)
    copies[16:20, 20:24] = True
    planted = cube.copy()
    # A plume on the seeds' absorbing bands only: pairing must not see it.
    planted[2:6, 2:6] = cube[16:20, 20:24] * np.where(PAIRING, 1.0, 0.8)
    pool = ~seeds  # the copies among every other pixel
    in_idx, out_idx = pair_pixels(planted, seeds, pool, PAIRING, keep=1.0)
    r_in, c_in = np.unravel_index(in_idx, (H, W))
    r_out, c_out = np.unravel_index(out_idx, (H, W))
    np.testing.assert_array_equal(r_out, r_in + 14)
    np.testing.assert_array_equal(c_out, c_in + 18)


def test_the_assignment_minimises_the_total_distance() -> None:
    """Brute force over every injection of 4 seeds into a pool of 6."""
    rng = np.random.default_rng(1)
    cube = rng.uniform(1.0, 2.0, size=(1, 10, 3))
    seeds = np.zeros((1, 10), dtype=bool)
    seeds[0, :4] = True
    pool = np.zeros((1, 10), dtype=bool)
    pool[0, 4:] = True
    bands = np.array([True, True, False])
    in_idx, out_idx = pair_pixels(cube, seeds, pool, bands, keep=1.0)
    flat = cube[0][:, bands]
    total = np.linalg.norm(flat[in_idx] - flat[out_idx], axis=-1).sum()
    best = min(
        sum(np.linalg.norm(flat[i] - flat[4 + j]) for i, j in enumerate(perm))
        for perm in itertools.permutations(range(6), 4)
    )
    np.testing.assert_allclose(total, best, rtol=1e-12)  # exact up to rounding


def test_only_the_pairing_bands_decide_the_pairs(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray
) -> None:
    scrambled = cube.copy()
    rng = np.random.default_rng(2)
    scrambled[..., ~PAIRING] = rng.uniform(0.0, 50.0, size=(H, W, (~PAIRING).sum()))
    for a, b in zip(
        pair_pixels(cube, seeds, pool, PAIRING),
        pair_pixels(scrambled, seeds, pool, PAIRING),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("keep", [0.0, -0.1, 1.5, float("nan")])
def test_keep_outside_the_unit_interval_raises_parameter_error(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray, keep: float
) -> None:
    with pytest.raises(ParameterError, match="keep"):
        pair_pixels(cube, seeds, pool, PAIRING, keep=keep)


def test_a_pool_smaller_than_the_seeds_raises(
    cube: np.ndarray, seeds: np.ndarray
) -> None:
    small = np.zeros((H, W), dtype=bool)
    small[0, :35] = True  # 35 < 36 seeds
    with pytest.raises(DegenerateInputError, match="fewer than the 36 seeds"):
        pair_pixels(cube, seeds, small, PAIRING)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no seeds", "no seeds"),
        ("no pairing band", "no band"),
        ("no pair kept", "keeps no pair"),
        ("non-finite", "non-finite"),
    ],
)
def test_degenerate_inputs_raise(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray, case: str, match: str
) -> None:
    bands, keep, data = PAIRING, 0.5, cube
    if case == "no seeds":
        seeds = np.zeros_like(seeds)
    elif case == "no pairing band":
        bands = np.zeros_like(PAIRING)
    elif case == "no pair kept":
        keep = 0.01  # int(0.01 · 36) = 0
    else:
        data = cube.copy()
        data[0, 0, np.flatnonzero(PAIRING)[0]] = np.nan
    with pytest.raises(DegenerateInputError, match=match):
        pair_pixels(data, seeds, pool, bands, keep=keep)


@pytest.mark.usefixtures("shapes_unchecked")
@pytest.mark.parametrize("which", ["cube", "seeds", "pool", "bands"])
def test_mismatched_shapes_raise_shape_error(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray, which: str
) -> None:
    args = [cube, seeds, pool, PAIRING]
    args[["cube", "seeds", "pool", "bands"].index(which)] = {
        "cube": cube[..., 0],
        "seeds": seeds[:-1],
        "pool": pool[:, :-1],
        "bands": PAIRING[:-1],
    }[which]
    with pytest.raises(ShapeError):
        pair_pixels(*args)


# ── ratio_spectrum ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pairs(
    cube: np.ndarray, seeds: np.ndarray, pool: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return pair_pixels(cube, seeds, pool, PAIRING)


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
    flat[in_idx] = flat[out_idx] * transmittance  # perfect partners, a plume
    ratio = ratio_spectrum(flat.reshape(H, W, B), in_idx, out_idx)
    np.testing.assert_allclose(ratio, transmittance, rtol=1e-13)


def test_non_positive_background_radiance_leaves_the_pair_out_of_that_band() -> None:
    # Pixels 0, 1 are in; 2, 3 their partners. Pixel 3 is not positive in
    # band 1, and neither partner is in band 2.
    cube = np.array(
        [[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [2.0, 8.0, 0.0], [2.0, -1.0, -2.0]]]
    )
    ratio = np.asarray(ratio_spectrum(cube, np.array([0, 1]), np.array([2, 3])))
    np.testing.assert_allclose(ratio[:2], [4.0 / 4.0, 2.0 / 8.0], rtol=1e-15)
    assert np.isnan(ratio[2])


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
