"""GeoTensor matched-filter operators on the one core (radtrax#29).

Ported from geotoolz's operator-level ``tests/test_matched_filter.py``. The
expected values follow radtrax's conventions where they differ: covariances
are the ``1/n`` scatter, the adaptive window gives the population variance,
and the Gaussian mixture is scikit-learn's. They need the ``[geo]`` extra
(in the dev group).
"""

from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest


pytestmark = pytest.mark.integration
GeoTensor = pytest.importorskip("georeader.geotensor").GeoTensor
rasterio = pytest.importorskip("rasterio")

from _typecheck import shapes_unchecked
from pipekit import Operator

import radtrax.geo as rg
from geo._carriers import fill_pixel_mask, toy_geotensor
from radtrax.retrieval.matched_filter import (
    apply_image,
    apply_pixel,
    linear_target_from_obs,
    nonlinear_target_from_obs,
)
from radtrax.retrieval.var3d import RadianceObservationModel


def _matrix(op) -> np.ndarray:
    return np.asarray(op.as_matrix())


# ── scoring ──────────────────────────────────────────────────────────────────


def test_matched_filter_recovers_known_amplitude_and_preserves_metadata() -> None:
    mean = np.array([10.0, 20.0, 30.0])
    target = np.array([1.0, 2.0, -1.0])
    amplitudes = np.array([[0.0, 1.0], [2.0, -0.5]])
    gt = toy_geotensor(mean[:, None, None] + target[:, None, None] * amplitudes)
    out = rg.MatchedFilter(mean=mean, cov_op=np.eye(3), target=target)(gt)
    np.testing.assert_allclose(np.asarray(out), amplitudes, atol=1e-12)
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"
    assert np.isnan(out.fill_value_default)


def test_scores_are_float64_whatever_the_jax_setting() -> None:
    """The operators run the core in float64 for the call only."""
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(3, 5, 5))
    op = rg.MatchedFilter(mean=np.zeros(3), cov_op=np.eye(3), target=np.ones(3))
    with jax.enable_x64(False):
        out = op(cube)
        assert jnp.ones(1).dtype == jnp.float32  # the global setting is untouched
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, cube.sum(axis=0) / 3.0, rtol=1e-14)


def test_pixel_snr_threshold_and_validation() -> None:
    mean, target = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    cov = np.array([[2.0, 0.0], [0.0, 1.0]])
    pixel = mean + 3.0 * target
    assert rg.MatchedFilterPixel(mean=mean, cov_op=cov, target=target)(
        pixel
    ) == pytest.approx(3.0)

    t, eye = np.array([1.0, 2.0]), np.eye(2)
    snr = rg.MatchedFilterSNR(amplitude=3.0, cov_op=eye, target=t)()
    assert snr == pytest.approx(3.0 * np.sqrt(5.0))
    assert rg.DetectionThreshold(false_alarm_rate=0.5, cov_op=eye, target=t)() == (
        pytest.approx(0.0, abs=1e-15)
    )
    assert rg.ValidateMFInputs(cov_op=eye, target=t)("ok") == "ok"
    with pytest.raises(ValueError, match="all zero"):
        rg.ValidateMFInputs(cov_op=eye, target=np.zeros(2))()
    with pytest.raises(ValueError, match="singular"):
        rg.ValidateMFInputs(cov_op=np.diag([1.0, 0.0]), target=np.array([0.0, 1.0]))()


def test_null_false_alarm_rate_matches_threshold() -> None:
    """Exceedances of the FAR-p threshold under a Gaussian null; the bound is
    three binomial standard errors of the exceedance fraction."""
    rng = np.random.default_rng(123)
    bands, n = 6, 20_000
    target = np.array([1.0, -0.5, 0.3, 0.0, 0.2, -0.1])
    cube = rng.normal(size=(n, bands)).T.reshape(bands, n, 1)
    scores = np.asarray(
        rg.MatchedFilter(mean=np.zeros(bands), cov_op=np.eye(bands), target=target)(
            toy_geotensor(cube)
        )
    ).ravel()
    for far in (0.05, 0.01):
        threshold = rg.DetectionThreshold(
            false_alarm_rate=far, cov_op=np.eye(bands), target=target
        )()
        tol = 3.0 * np.sqrt(far * (1 - far) / n)
        assert abs(float((scores > threshold).mean()) - far) < tol


# ── fitted operator ──────────────────────────────────────────────────────────


def test_first_call_fit_populates_reusable_state() -> None:
    rng = np.random.default_rng(0)
    target = np.array([0.5, 1.0])
    first, second = rng.normal(size=(2, 4, 4)), rng.normal(loc=5.0, size=(2, 4, 4))
    op = rg.MatchedFilter(target=target, cov_method="empirical")
    assert np.asarray(op(toy_geotensor(first))).shape == (4, 4)
    fitted_mean, fitted_cov = op.mean_, op.cov_op_
    assert fitted_mean is not None and fitted_cov is not None
    reused = np.asarray(op(second))
    assert op.mean_ is fitted_mean and op.cov_op_ is fitted_cov
    np.testing.assert_allclose(reused, np.asarray(op.transform(second)))


def test_fit_on_call_estimates_per_call_without_storing() -> None:
    rng = np.random.default_rng(1)
    target, cube = np.array([0.5, 1.0]), rng.normal(size=(2, 4, 4))
    op = rg.MatchedFilter(target=target, fit_on_call=True, cov_method="empirical")
    config = op.get_config()
    out = np.asarray(op(cube))
    assert op.mean_ is None and op.cov_op_ is None
    assert op.get_config() == config
    expected = rg.MatchedFilter(target=target, cov_method="empirical").fit(cube)
    np.testing.assert_allclose(out, np.asarray(expected.transform(cube)))


def test_fit_on_call_false_preserves_explicit_mean() -> None:
    fixed_mean = np.array([100.0, 200.0])
    cube = np.ones((2, 4, 4)) * fixed_mean[:, None, None]
    op = rg.MatchedFilter(
        mean=fixed_mean, target=np.array([1.0, 0.0]), cov_method="empirical"
    )
    op(toy_geotensor(cube))
    np.testing.assert_allclose(op.mean_, fixed_mean)
    assert op.cov_op_ is not None
    assert op.cov_op is None


def test_fit_transform_and_config() -> None:
    """``fit`` learns the background once; it never reaches ``get_config()``."""
    rng = np.random.default_rng(2)
    target = np.array([0.5, 1.0, -0.2])
    background, scene = rng.normal(size=(2, 3, 5, 5))
    op = rg.MatchedFilter(target=target, cov_method="empirical")
    config = op.get_config()
    with pytest.raises(ValueError, match=r"fit\(\)"):
        op.transform(scene)
    assert op.fit(background) is op
    scores = np.asarray(op.transform(scene))
    assert op.get_config() == config
    assert config["mean"] is None and config["cov_op"] is None
    expected = apply_image(
        jnp.asarray(np.moveaxis(scene, 0, -1)),
        jnp.asarray(op.mean_),
        op.cov_op_,
        jnp.asarray(target),
    )
    np.testing.assert_allclose(scores, np.asarray(expected), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(op(scene)), scores)


def test_matched_filter_satisfies_fittable_transformer() -> None:
    from pipekit.protocols import FittableTransformer

    assert isinstance(rg.MatchedFilter(), FittableTransformer)


def test_first_call_fits_once_under_thread_map() -> None:
    import threading
    import time

    from pipekit.parallel import ThreadMap

    rng = np.random.default_rng(4)
    scenes = [rng.normal(loc=i, size=(2, 5, 5)) for i in range(8)]
    op = rg.MatchedFilter(target=np.array([0.5, 1.0]), cov_method="empirical")
    calls, lock, fit = [], threading.Lock(), op.fit

    def slow_fit(x):
        with lock:
            calls.append(x)
        time.sleep(0.05)
        return fit(x)

    op.fit = slow_fit
    outputs = ThreadMap(op, n_workers=8)(scenes)
    assert len(calls) == 1
    for scene, out in zip(scenes, outputs, strict=True):
        np.testing.assert_allclose(np.asarray(out), np.asarray(op.transform(scene)))


@pytest.mark.parametrize("cov_method", ["empirical", "ledoit_wolf", "oas", "lowrank"])
def test_every_cov_method_scores(cov_method: str) -> None:
    rng = np.random.default_rng(6)
    cube = rng.normal(size=(4, 6, 6))
    out = rg.MatchedFilter(target=np.ones(4), cov_method=cov_method)(cube)
    assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="cov_method"), shapes_unchecked():
        rg.MatchedFilter(target=np.ones(4), cov_method="nope")(cube)  # type: ignore[arg-type]


# ── estimators ───────────────────────────────────────────────────────────────


def test_estimators_return_expected_backgrounds() -> None:
    cube = np.array([[[1.0, 2.0], [3.0, 100.0]], [[10.0, 20.0], [30.0, 1000.0]]])
    gt = toy_geotensor(cube)
    np.testing.assert_allclose(rg.EstimateMean(method="median")(gt), [2.5, 25.0])
    cov = rg.EstimateCovEmpirical(mean=np.zeros(2), ridge=1e-6)(gt)
    assert isinstance(cov, lx.AbstractLinearOperator)
    assert np.all(np.linalg.eigvalsh(_matrix(cov)) > 0)
    # About a given mean it is the 1/n second moment, plus the ridge.
    samples = cube.reshape(2, -1).T
    np.testing.assert_allclose(
        _matrix(cov), samples.T @ samples / 4 + 1e-6 * np.eye(2), rtol=1e-12
    )


def test_estimator_ops_accept_plain_ndarray_input() -> None:
    cube = _ndarray_cube()
    gt = toy_geotensor(cube)
    np.testing.assert_allclose(
        rg.EstimateMean(method="median")(cube), rg.EstimateMean(method="median")(gt)
    )
    np.testing.assert_allclose(
        _matrix(rg.EstimateCovShrunk()(cube)), _matrix(rg.EstimateCovShrunk()(gt))
    )
    assert rg.AdaptiveWindowBackground(window=3)(cube).mean.shape == cube.shape
    streaming = rg.StreamingBackground(cov_kind="empirical")([cube, gt])
    np.testing.assert_allclose(streaming.mean, cube.reshape(3, -1).mean(axis=1))


def test_streaming_background_matches_batch_estimators() -> None:
    from sklearn.covariance import ledoit_wolf

    rng = np.random.default_rng(8)
    cube_a, cube_b = rng.normal(size=(2, 3, 4, 5))
    stacked = np.concatenate([cube_a.reshape(3, -1), cube_b.reshape(3, -1)], axis=1).T
    empirical = rg.StreamingBackground(cov_kind="empirical", ridge=0.0)(
        [toy_geotensor(cube_a), cube_b]
    )
    np.testing.assert_allclose(empirical.mean, stacked.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        _matrix(empirical.cov_op), np.cov(stacked, rowvar=False, ddof=0), rtol=1e-12
    )
    shrunk = rg.StreamingBackground()([cube_a, cube_b])  # Ledoit–Wolf, 1e-8 ridge
    ref, _ = ledoit_wolf(stacked)
    np.testing.assert_allclose(
        _matrix(shrunk.cov_op), ref + 1e-8 * np.eye(3), rtol=1e-10
    )
    with pytest.raises(ValueError, match="at least one cube"):
        rg.StreamingBackground()([])


def test_lowrank_estimator_matches_core() -> None:
    from radtrax.retrieval.matched_filter import estimate_cov_lowrank

    rng = np.random.default_rng(9)
    cube = rng.normal(size=(5, 6, 6))
    op = rg.EstimateCovLowRank(rank=2, tikhonov=1e-2)(cube)
    ref = estimate_cov_lowrank(np.moveaxis(cube, 0, -1), rank=2, tikhonov=1e-2)
    # Same rows, same key, same jitted SVD: equal up to rounding.
    np.testing.assert_allclose(_matrix(op), _matrix(ref), rtol=1e-12)


# ── cluster and adaptive backgrounds ─────────────────────────────────────────


def _ndarray_cube() -> np.ndarray:
    """Bimodal ``(3, 4, 4)`` cube, so a two-component mixture splits cleanly."""
    rng = np.random.default_rng(7)
    cube = 0.01 * rng.normal(size=(3, 4, 4)) + np.array([5.0, 6.0, 7.0])[:, None, None]
    cube[:, :, 2:] += 10.0
    return cube


def test_cluster_background_and_dispatch_are_reproducible() -> None:
    target = np.array([0.5, 1.0, -0.5])
    gt = toy_geotensor(_ndarray_cube())
    bg1 = rg.GMMClusterBackground(n_clusters=2, seed=4)(gt)
    bg2 = rg.GMMClusterBackground(n_clusters=2, seed=4)(gt)
    np.testing.assert_array_equal(np.asarray(bg1.labels), np.asarray(bg2.labels))
    assert np.asarray(bg1.labels).shape == (4, 4)
    out = rg.ApplyClusterMF(target=target)(gt, bg1)
    labels = np.asarray(bg1.labels).reshape(-1)
    samples = np.asarray(gt).reshape(3, -1).T
    expected = np.array(
        [
            float(
                apply_pixel(
                    jnp.asarray(x),
                    bg1.means[k],
                    bg1.cov_operators[k],
                    jnp.asarray(target),
                )
            )
            for x, k in zip(samples, labels, strict=True)
        ]
    ).reshape(4, 4)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)
    assert out.transform == gt.transform
    with pytest.raises(ValueError, match="labels shape"):
        rg.ApplyClusterMF(target=target)(_ndarray_cube()[:, :2], bg1)


def _brute_force_adaptive_mf(
    cube: np.ndarray, target: np.ndarray, window: int, ridge: float
) -> np.ndarray:
    """Per-pixel loop: reflect-padded window mean / population variance."""
    half = window // 2
    # scipy.ndimage mode="reflect" is numpy's half-sample "symmetric".
    padded = np.pad(cube, ((0, 0), (half, half), (half, half)), mode="symmetric")
    c, h, w = cube.shape
    out = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            win = padded[:, i : i + window, j : j + window].reshape(c, -1)
            mu, var = win.mean(axis=1), win.var(axis=1) + ridge
            out[i, j] = np.sum(target * (cube[:, i, j] - mu) / var) / np.sum(
                target**2 / var
            )
    return out


def test_apply_adaptive_mf_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    cube = rng.normal(loc=[[[1.0]], [[2.0]], [[3.0]]], size=(3, 9, 7))
    target = np.array([0.5, 1.0, -0.5])
    cube[:, 4, 3] += target
    gt = toy_geotensor(cube)
    bg = rg.AdaptiveWindowBackground(window=5)(gt)
    for ridge in (0.0, 0.3):
        out = rg.ApplyAdaptiveMF(target=target, ridge=ridge)(gt, bg)
        assert isinstance(out, GeoTensor) and out.shape == (9, 7)
        np.testing.assert_allclose(
            np.asarray(out),
            _brute_force_adaptive_mf(cube, target, window=5, ridge=ridge),
            rtol=1e-12,
            atol=1e-13,
        )
    with pytest.raises(ValueError, match="ridge"):
        rg.ApplyAdaptiveMF(target=target, ridge=-1.0)(gt, bg)
    # A flat band has zero variance: unscoreable without a ridge.
    flat = cube.copy()
    flat[0] = 1.0
    flat_bg = rg.AdaptiveWindowBackground(window=3)(flat)
    assert np.isnan(rg.ApplyAdaptiveMF(target=target)(flat, flat_bg)).all()
    assert np.isfinite(
        rg.ApplyAdaptiveMF(target=target, ridge=1e-3)(flat, flat_bg)
    ).all()


def test_adaptive_guard_window_excludes_the_pixel_itself() -> None:
    rng = np.random.default_rng(12)
    cube = rng.normal(size=(2, 7, 7))
    bg = rg.AdaptiveWindowBackground(window=3, guard_window=1)(cube)
    ring = np.delete(cube[:, 2:5, 2:5].reshape(2, -1), 4, axis=1)
    np.testing.assert_allclose(bg.mean[:, 3, 3], ring.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(bg.variance[:, 3, 3], ring.var(axis=1), rtol=1e-10)


# ── targets ──────────────────────────────────────────────────────────────────


def test_linear_target_from_a_callable_matches_finite_difference() -> None:
    a = np.array([0.5, -1.0, 2.0])

    def obs_model(x: np.ndarray) -> np.ndarray:
        return (x * a[:, None, None]).sum(axis=(1, 2))

    target = rg.LinearTargetFromObs(obs_model=obs_model)(
        toy_geotensor(np.zeros((3, 2, 3)))
    )
    # A linear model: the finite difference is exact up to round-off.
    np.testing.assert_allclose(target, a * 2 * 3, rtol=1e-9)


@pytest.fixture
def obs_at_background(synthetic_lut, hyperspectral_srf) -> RadianceObservationModel:
    """The var3d fixture model about a methane background of 1.9 ppm.

    A non-zero reference, so a default background of zeros would be caught
    (Beer–Lambert is nonlinear in the VMR).
    """
    return RadianceObservationModel.from_lut(
        synthetic_lut,
        srf=hyperspectral_srf,
        T_K=280.0,
        p_atm=1.0,
        path_length_cm=8.4e5,
        amf=2.0,
        vmr_reference=1.9e-6,
    )


def test_targets_from_an_observation_model_use_the_core(obs_at_background) -> None:
    """An observation model gets the core's exact JVP (and finite amplitude)."""
    obs = obs_at_background
    shape = (4, 5)
    gt = np.zeros((obs.n_bands, *shape))
    background = jnp.full(shape, 1.9e-6)
    np.testing.assert_array_equal(
        rg.LinearTargetFromObs(obs_model=obs)(gt),
        np.asarray(linear_target_from_obs(obs, background)),
    )
    np.testing.assert_array_equal(
        rg.LinearTargetFromObs(obs_model=obs, pattern="impulse", pixel=(1, 2))(gt),
        np.asarray(
            linear_target_from_obs(obs, background, pattern="impulse", pixel=(1, 2))
        ),
    )
    np.testing.assert_array_equal(
        rg.NonlinearTargetFromObs(obs_model=obs, amplitude=5e-7)(gt),
        np.asarray(nonlinear_target_from_obs(obs, background, amplitude=5e-7)),
    )
    # The band axis need not be first: the grid comes from the other axes.
    np.testing.assert_array_equal(
        rg.LinearTargetFromObs(obs_model=obs, axis=-1)(np.moveaxis(gt, 0, -1)),
        np.asarray(linear_target_from_obs(obs, background)),
    )


def test_callable_target_matches_the_jvp_at_vmr_scale(obs_at_background) -> None:
    """The central difference, with its step relative to a ~2e-6 state, agrees
    with the exact JVP. Bound, relative to the target's scale: truncation
    ~(a·h)²/6 ≈ 1e-10 for Δτ = a·h ≈ 2e-5, round-off ~1e-16/2e-5 ≈ 5e-12;
    1e-8 leaves room above both."""
    obs = obs_at_background
    shape = (4, 5)
    forward = obs.make_forward()

    def band_first(vmr: np.ndarray) -> np.ndarray:
        return np.moveaxis(np.asarray(forward(jnp.asarray(vmr))), -1, 0)

    gt = np.zeros((obs.n_bands, *shape))
    background = np.full(shape, 1.9e-6)
    target = rg.LinearTargetFromObs(obs_model=band_first, vmr_background=background)(gt)
    exact = np.asarray(linear_target_from_obs(obs, jnp.asarray(background)))
    # Entries span ~1e-198 to ~1e5 (line wings to centres): bound the error
    # against the target's scale, not entry by entry.
    np.testing.assert_allclose(target, exact, rtol=0, atol=1e-8 * np.max(np.abs(exact)))
    # A band-last response is caught, not averaged over the wrong axis.
    with pytest.raises(ValueError, match="band axis"):
        rg.LinearTargetFromObs(obs_model=forward, vmr_background=background)(gt)
    with pytest.raises(TypeError, match="callable"):
        rg.LinearTargetFromObs(obs_model="not callable")(gt)


def test_nonlinear_target_from_obs_amplitude_difference() -> None:
    def obs_model(x: np.ndarray) -> np.ndarray:
        return (x**2).mean(axis=(1, 2))

    target = rg.NonlinearTargetFromObs(obs_model=obs_model, amplitude=0.5)(
        toy_geotensor(np.full((2, 3, 3), 1.0))
    )
    # y(1) = 1, y(1.5) = 2.25: the difference is 1.25 per band, exactly.
    np.testing.assert_allclose(target, np.full(2, 1.25), rtol=1e-15)


@pytest.mark.parametrize(
    ("pattern", "pixel", "total"),
    [("impulse", None, 3.0), ("impulse", (1, 2), 3.0), ("uniform", None, 48.0)],
)
def test_target_pattern_on_geotensor_returns_plain_array(
    pattern: str, pixel: tuple[int, int] | None, total: float
) -> None:
    from radtrax.geo.matched_filter import _target_pattern

    out = _target_pattern(
        toy_geotensor(np.ones((3, 4, 4))), pattern=pattern, pixel=pixel, axis=-3
    )
    assert type(out) is np.ndarray
    assert out.shape == (3, 4, 4) and out.sum() == total
    if pattern == "impulse":
        # Every band of one pixel: the centre by default, as in the core.
        i, j = pixel or (2, 2)
        assert (out[:, i, j] == 1.0).all()
    # A 2-D state has no band axis: the impulse is one pixel.
    flat = _target_pattern(np.ones((4, 4)), pattern="impulse", pixel=(1, 2), axis=None)
    assert flat.sum() == 1.0 and flat[1, 2] == 1.0
    with pytest.raises(ValueError, match="non-band axes"):
        _target_pattern(np.ones((3, 4, 4)), pattern="impulse", pixel=(1,), axis=-3)
    with pytest.raises(ValueError, match="unknown target pattern"):
        _target_pattern(np.ones((3, 4, 4)), pattern="nope", pixel=None, axis=-3)


def test_column_enhancement_end_to_end() -> None:
    rng = np.random.default_rng(0)
    bg_mean = np.array([10.0, 11.0, 12.0, 13.0])
    gt = toy_geotensor(bg_mean[:, None, None] + rng.normal(size=(4, 5, 6)) * 0.01)
    out = rg.ColumnEnhancement()(gt)
    assert np.asarray(out).shape == (5, 6)
    # A unit target on noise of std 0.01 in 4 bands scores with std
    # σ = 1/√(tᵀΣ⁻¹t) = 0.005; centring on the median adds an offset of the
    # same order, so six standard errors σ/√30 bound the mean score.
    assert abs(float(np.asarray(out).mean())) < 6 * 0.005 / np.sqrt(30)
    assert out.transform == gt.transform
    lowrank = rg.ColumnEnhancement(cov_method="lowrank", rank=2)(gt)
    assert np.isfinite(lowrank).all()

    def obs_model(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=(1, 2))

    with_target = rg.ColumnEnhancement(obs_model=obs_model)(gt)
    assert np.isfinite(with_target).all()


# ── carriers, nodata and stacks ──────────────────────────────────────────────


def _run_matched_filter(cube):
    return rg.MatchedFilter(
        mean=np.array([5.0, 6.0, 7.0]),
        cov_op=np.eye(3),
        target=np.array([1.0, -0.5, 0.25]),
    )(cube)


def _run_apply_cluster_mf(cube):
    bg = rg.GMMClusterBackground(n_clusters=2, seed=0)(cube)
    return rg.ApplyClusterMF(target=np.array([1.0, -0.5, 0.25]))(cube, bg)


def _run_column_enhancement(cube):
    return rg.ColumnEnhancement()(cube)


@pytest.mark.parametrize(
    "run_op",
    [_run_matched_filter, _run_apply_cluster_mf, _run_column_enhancement],
    ids=["matched_filter", "apply_cluster_mf", "column_enhancement"],
)
def test_score_map_ops_support_plain_ndarray_carrier(run_op) -> None:
    cube = _ndarray_cube()
    out_arr, out_gt = run_op(cube), run_op(toy_geotensor(cube))
    assert type(out_arr) is np.ndarray
    assert isinstance(out_gt, GeoTensor)
    np.testing.assert_allclose(out_arr, np.asarray(out_gt))


def _fill_cube(fill: float) -> tuple[GeoTensor, np.ndarray, np.ndarray]:
    """``(3, 4, 4)`` cube with two fill pixels, its valid mask, the valid-only cube."""
    rng = np.random.default_rng(3)
    values = rng.normal(loc=[[[1.0]], [[2.0]], [[3.0]]], size=(3, 4, 4))
    gt = toy_geotensor(values, fill_value_default=fill, with_fill_pixels=True)
    valid = ~fill_pixel_mask(values.shape)
    return gt, valid, values[:, valid][:, :, None]


@pytest.mark.parametrize("fill", [-9999.0, np.nan], ids=["fill-9999", "fill-nan"])
@pytest.mark.parametrize(
    "case",
    [
        "estimate_mean",
        "cov_empirical",
        "cov_shrunk",
        "cov_lowrank",
        "matched_filter",
        "cluster",
        "adaptive_window",
        "streaming",
    ],
)
def test_fill_pixels_are_excluded(case: str, fill: float) -> None:
    """Fill pixels never enter a statistic; score maps hold NaN there."""
    gt, valid, clean = _fill_cube(fill)
    target = np.array([1.0, 0.5, -1.0])

    def assert_fill(out: GeoTensor) -> None:
        assert np.isnan(out.fill_value_default)
        values = np.asarray(out)
        assert np.isnan(values[~valid]).all()
        assert np.isfinite(values[valid]).all()

    if case == "estimate_mean":
        for method in ("mean", "median", "trimmed", "huber"):
            np.testing.assert_allclose(
                rg.EstimateMean(method=method)(gt),
                rg.EstimateMean(method=method)(clean),
            )
    elif case == "cov_empirical":
        np.testing.assert_allclose(
            _matrix(rg.EstimateCovEmpirical()(gt)),
            _matrix(rg.EstimateCovEmpirical()(clean)),
        )
    elif case == "cov_shrunk":
        np.testing.assert_allclose(
            _matrix(rg.EstimateCovShrunk()(gt)), _matrix(rg.EstimateCovShrunk()(clean))
        )
    elif case == "cov_lowrank":
        np.testing.assert_allclose(
            _matrix(rg.EstimateCovLowRank(rank=2)(gt)),
            _matrix(rg.EstimateCovLowRank(rank=2)(clean)),
            rtol=1e-12,
        )
    elif case == "matched_filter":
        op = rg.MatchedFilter(target=target, mean_method="mean")
        out = op(gt)
        np.testing.assert_allclose(op.mean_, clean.reshape(3, -1).mean(axis=1))
        ref = rg.MatchedFilter(target=target, mean_method="mean")(clean)
        np.testing.assert_allclose(np.asarray(out)[valid], np.asarray(ref).ravel())
        assert_fill(out)
    elif case == "cluster":
        cluster = rg.GMMClusterBackground(n_clusters=2)(gt)
        labels = np.asarray(cluster.labels)
        assert (labels[~valid] == -1).all() and (labels[valid] >= 0).all()
        assert all(np.isfinite(np.asarray(m)).all() for m in cluster.means)
        assert_fill(rg.ApplyClusterMF(target=target)(gt, cluster))
    elif case == "adaptive_window":
        bg = rg.AdaptiveWindowBackground(window=3)(gt)
        assert (
            np.isnan(bg.mean[:, ~valid]).all()
            and np.isnan(bg.variance[:, ~valid]).all()
        )
        # Pixel (1, 1): its 3×3 window holds the (0, 0) fill pixel.
        window = np.asarray(gt)[:, 0:3, 0:3].reshape(3, -1)[:, 1:]
        np.testing.assert_allclose(bg.mean[:, 1, 1], window.mean(axis=1), rtol=1e-12)
        np.testing.assert_allclose(bg.variance[:, 1, 1], window.var(axis=1), rtol=1e-10)
        assert_fill(rg.ApplyAdaptiveMF(target=target)(gt, bg))
    elif case == "streaming":
        result = rg.StreamingBackground(cov_kind="empirical")([gt, gt])
        expected = rg.StreamingBackground(cov_kind="empirical")([clean, clean])
        np.testing.assert_allclose(result.mean, expected.mean)
        np.testing.assert_allclose(_matrix(result.cov_op), _matrix(expected.cov_op))


def test_4d_time_stack() -> None:
    """One background over every frame; score maps come back ``(T, 1, H, W)``."""
    rng = np.random.default_rng(5)
    stack = toy_geotensor(rng.normal(size=(2, 3, 6, 6)))
    samples = np.moveaxis(np.asarray(stack), 1, -1).reshape(-1, 3)
    np.testing.assert_allclose(
        rg.EstimateMean(method="mean")(stack), samples.mean(axis=0)
    )
    op = rg.MatchedFilter(target=np.ones(3))
    scores = op(stack)
    assert isinstance(scores, GeoTensor) and scores.shape == (2, 1, 6, 6)
    # One background for the stack; each frame scores as on its own.
    for t in range(2):
        np.testing.assert_allclose(
            np.asarray(scores)[t, 0],
            np.asarray(op.transform(np.asarray(stack)[t])),
            rtol=1e-12,
        )
    holed = np.asarray(stack).copy()
    holed[1, :, 2, 3] = np.nan
    holed_scores = np.asarray(op.transform(holed))
    assert np.isnan(holed_scores[1, 0, 2, 3])
    assert np.isfinite(holed_scores).sum() == 71
    with pytest.raises(ValueError, match="MatchedFilter: spectral axis -3"):
        rg.MatchedFilter(target=np.ones(3))(toy_geotensor(np.ones((6, 6))))
    with pytest.raises(ValueError, match="AdaptiveWindowBackground"):
        rg.AdaptiveWindowBackground(window=3)(stack)


# ── configs and the operator contract ────────────────────────────────────────


def _configurable_ops() -> list[Operator]:
    target, cov, mean = np.array([1.0, 2.0]), np.eye(2), np.zeros(2)
    return [
        rg.MatchedFilter(target=target, cov_op=cov, mean=mean),
        rg.MatchedFilter(target=target),
        rg.MatchedFilterPixel(mean=mean, cov_op=cov, target=target),
        rg.MatchedFilterSNR(amplitude=1.0, cov_op=cov, target=target),
        rg.DetectionThreshold(false_alarm_rate=0.05, cov_op=cov, target=target),
        rg.ValidateMFInputs(cov_op=cov, target=target),
        rg.EstimateMean(method="huber"),
        rg.EstimateCovEmpirical(mean=mean, ridge=0.1),
        rg.EstimateCovShrunk(method="oas"),
        rg.EstimateCovLowRank(rank=3),
        rg.GMMClusterBackground(n_clusters=3, bayesian=True),
        rg.AdaptiveWindowBackground(window=5, guard_window=1),
        rg.ApplyClusterMF(target=target),
        rg.ApplyAdaptiveMF(target=target, ridge=0.1),
        rg.StreamingBackground(cov_kind="oas"),
        rg.ColumnEnhancement(cov_method="lowrank", rank=4),
    ]


@pytest.mark.parametrize("op", _configurable_ops(), ids=lambda op: type(op).__name__)
def test_every_operator_round_trips_through_its_config(op: Operator) -> None:
    """Strict JSON, and ``from_state`` rebuilds an operator with the same config."""
    state = json.loads(json.dumps(op.state, allow_nan=False))
    clone = Operator.from_state(state)
    assert type(clone) is type(op)
    assert clone.get_config() == op.get_config()


def test_config_reload_restores_arrays_and_operators() -> None:
    op = rg.MatchedFilter(
        mean=np.zeros(2),
        cov_op=lx.MatrixLinearOperator(jnp.eye(2)),
        target=np.ones(2),
    )
    clone = Operator.from_state(json.loads(json.dumps(op.state)))
    assert isinstance(clone.mean, np.ndarray)
    assert isinstance(clone.cov_op, np.ndarray)
    assert isinstance(clone.target, np.ndarray)
    cube = np.random.default_rng(3).normal(size=(2, 3, 3))
    np.testing.assert_allclose(np.asarray(clone(cube)), np.asarray(op(cube)))


def test_obs_model_operators_are_forbid_in_yaml() -> None:
    assert rg.LinearTargetFromObs.forbid_in_yaml is True
    assert rg.NonlinearTargetFromObs.forbid_in_yaml is True
    op = rg.NonlinearTargetFromObs(
        obs_model=np.sum, amplitude=0.5, pattern=np.ones((1, 2))
    )
    assert op.get_config()["obs_model"] == "sum"
    assert op.get_config()["pattern"] == [[1.0, 1.0]]


def test_column_enhancement_obs_model_refuses_reload() -> None:
    def obs_model(state: object) -> object:
        return state

    op = rg.ColumnEnhancement(obs_model=obs_model)
    assert op.get_config()["obs_model"] == {"callable": "obs_model"}
    with pytest.raises(RuntimeError, match="non-primitive"):
        Operator.from_state(json.loads(json.dumps(op.state)))
    with pytest.raises(TypeError, match="config summary"):
        rg.ColumnEnhancement(obs_model={"callable": "obs_model"})
    plain = rg.ColumnEnhancement()
    assert Operator.from_state(plain.state).get_config() == plain.get_config()


def test_matched_filter_keeps_the_carrier_contract() -> None:
    """geotoolz's own checker: config, graph mode, carrier, attrs, fills, stacks.

    ``ColumnEnhancement`` is left out: it fits one background over a whole
    ``(T, C, H, W)`` stack by design, which the checker's frame-by-frame rule
    rejects (geotoolz exempts it in its own contract test for that reason).
    """
    from geotoolz.testing import check_operator

    gt, _, _ = _fill_cube(-9999.0)
    gt.attrs = {"band_names": ["b1", "b2", "b3"]}
    check_operator(rg.MatchedFilter(target=np.array([1.0, 0.5, -1.0])), gt)


def test_traceable_callable_target_is_the_exact_jvp() -> None:
    """A JAX-traceable callable is differentiated, not finite-differenced: at a
    zero state with exp(1e4·x) a 1e-4 step would give ~11752, the JVP 1e4."""

    def obs_model(x: jax.Array) -> jax.Array:
        return jnp.exp(1e4 * x).mean(axis=(1, 2))

    target = rg.LinearTargetFromObs(obs_model=obs_model)(np.zeros((3, 2, 2)))
    np.testing.assert_allclose(target, np.full(3, 1e4), rtol=1e-14)


def test_callable_target_state_has_no_nodata() -> None:
    """Fill pixels become the band means, so an aggregating callable stays finite."""
    gt, _, _ = _fill_cube(-9999.0)
    target = rg.LinearTargetFromObs(obs_model=lambda x: x.sum(axis=(1, 2)))(gt)
    # d/dα Σ (x + α) over the 16 pixels of each band.
    np.testing.assert_allclose(target, np.full(3, 16.0), rtol=1e-14)


def test_cube_shaped_background_keeps_its_band_axis() -> None:
    """An impulse at a two-coordinate pixel perturbs every band there."""
    background = np.ones((3, 4, 5))
    op = rg.LinearTargetFromObs(
        obs_model=lambda x: x[:, 1, 2] * jnp.array([1.0, 2.0, 3.0]),
        vmr_background=background,
        pattern="impulse",
        pixel=(1, 2),
    )
    np.testing.assert_allclose(op(np.zeros((3, 4, 5))), [1.0, 2.0, 3.0], rtol=1e-14)


def test_numpy_callable_at_a_zero_state_takes_the_given_step() -> None:
    """A NumPy callable at zero has no state scale: fd_step sets the step.
    Central-difference error (k·h)²/6 ≈ 1.7e-7 for k = 1e4, h = 1e-7."""

    def obs_model(x: np.ndarray) -> np.ndarray:
        return np.exp(1e4 * np.asarray(x)).mean(axis=(1, 2))  # not traceable

    cube = np.zeros((3, 2, 2))
    default = rg.LinearTargetFromObs(obs_model=obs_model)(cube)
    np.testing.assert_allclose(default, np.full(3, np.sinh(1.0) / 1e-4), rtol=1e-12)
    scaled = rg.LinearTargetFromObs(obs_model=obs_model, fd_step=1e-7)(cube)
    np.testing.assert_allclose(scaled, np.full(3, 1e4), rtol=1e-6)


def test_column_enhancement_forwards_the_target_state(obs_at_background) -> None:
    """With a GSD the target's state lives on the high-resolution grid; a
    varying background there changes the (nonlinear) target at the centre."""
    from radtrax.instrument.psf import GroundSamplingDistance

    obs = eqx.tree_at(
        lambda m: m.gsd,
        obs_at_background,
        GroundSamplingDistance(downsample_factor=2),
        is_leaf=lambda x: x is None,
    )
    hr = 1.9e-6 * (1.0 + jnp.linspace(0.0, 4.0, 80).reshape(8, 10))
    rng = np.random.default_rng(13)
    cube = 1.0 + 0.01 * rng.normal(size=(obs.n_bands, 4, 5))
    op = rg.ColumnEnhancement(obs_model=obs, vmr_background=np.asarray(hr))
    expected_target = np.asarray(linear_target_from_obs(obs, hr))
    default_target = np.asarray(linear_target_from_obs(obs, jnp.full((4, 5), 1.9e-6)))
    assert not np.allclose(expected_target, default_target)
    mean = rg.EstimateMean(method="median")(cube)
    cov = rg.EstimateCovShrunk(mean=mean)(cube)
    expected = rg.MatchedFilter(mean=mean, cov_op=cov, target=expected_target)(cube)
    np.testing.assert_allclose(np.asarray(op(cube)), np.asarray(expected), rtol=1e-10)
    assert len(op.get_config()["vmr_background"]) == 8
    pixel = rg.ColumnEnhancement(target_pattern="impulse", target_pixel=(1, 2))
    assert Operator.from_state(pixel.state).get_config() == pixel.get_config()


def test_column_enhancement_forwards_fd_step() -> None:
    """The fd_step escape hatch reaches the target inside ColumnEnhancement."""

    def obs_model(x: np.ndarray) -> np.ndarray:
        return np.exp(1e4 * np.asarray(x)).mean(axis=(1, 2))  # not traceable

    rng = np.random.default_rng(14)
    cube = 0.01 * rng.normal(size=(3, 5, 6))
    state = np.zeros_like(cube)
    op = rg.ColumnEnhancement(obs_model=obs_model, vmr_background=state, fd_step=1e-7)
    # The target is 1e4 per band (to (k·h)²/6 ≈ 1.7e-7); the scores scale as 1/t.
    mean = rg.EstimateMean(method="median")(cube)
    cov = rg.EstimateCovShrunk(mean=mean)(cube)
    expected = rg.MatchedFilter(mean=mean, cov_op=cov, target=np.full(3, 1e4))(cube)
    np.testing.assert_allclose(op(cube), expected, rtol=1e-6)
    assert op.get_config()["fd_step"] == 1e-7
