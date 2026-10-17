"""The MBMP operator, geotoolz's former ``plume.SBMP`` (radtrax#30).

Ported from geotoolz's ``tests/test_plume.py`` SBMP cases. Needs the
``[geo]`` extra (in the dev group).
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest


pytestmark = pytest.mark.integration
GeoTensor = pytest.importorskip("georeader.geotensor").GeoTensor
rasterio = pytest.importorskip("rasterio")

from geoops.carrier import SENTINEL2_L2A_BANDS
from pipekit import Operator

from geo._carriers import TRANSFORM
from radtrax.geo import MBMP, MBSP, SBMP


TRUTH = np.linspace(0.0, 0.2, 25).reshape(5, 5)
# ε = 1e-10 in every log moves the signal by ~1e-10 at reflectance 1.
ATOL = 1e-9


def _gt(values: np.ndarray, *, fill: float = -9999.0) -> GeoTensor:
    return GeoTensor(
        values, transform=TRANSFORM, crs="EPSG:32629", fill_value_default=fill
    )


def test_reference_scene_recovers_the_injected_signal() -> None:
    """In float64 even with JAX in float32 (float32 is off by ~5e-8)."""
    reference = np.ones((2, 5, 5))
    scene = np.stack([np.exp(TRUTH), np.ones_like(TRUTH)])
    with jax.enable_x64(False):
        out = MBMP(swir1=0, swir2=1)(_gt(scene), _gt(reference))
    assert isinstance(out, GeoTensor) and out.transform == TRANSFORM
    np.testing.assert_allclose(np.asarray(out), TRUTH, atol=ATOL)
    assert np.isnan(out.fill_value_default)


def test_default_sentinel2_band_names_on_a_plain_l2a_array() -> None:
    """A bare 12-band array falls back to the documented S2 L2A order."""
    reference = np.ones((12, 5, 5))
    scene = reference.copy()
    scene[10] = np.exp(TRUTH)  # B11
    out = MBMP()(scene, reference)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, TRUTH, atol=ATOL)


def test_band_names_resolve_from_carrier_attrs() -> None:
    """A GeoTensor names its own bands: a 13-band L1C stack picks the real B11."""
    l1c = [*SENTINEL2_L2A_BANDS[:10], "B10", *SENTINEL2_L2A_BANDS[10:]]
    reference = np.ones((13, 5, 5))
    scene = reference.copy()
    scene[l1c.index("B11")] = np.exp(TRUTH)
    scene_gt, ref_gt = _gt(scene), _gt(reference)
    scene_gt.attrs["band_names"] = l1c
    ref_gt.attrs["band_names"] = l1c
    np.testing.assert_allclose(np.asarray(MBMP()(scene_gt, ref_gt)), TRUTH, atol=ATOL)


def test_named_bands_need_names_outside_plain_l2a_arrays() -> None:
    with pytest.raises(ValueError, match="'B11'"):
        MBMP()(_gt(np.ones((12, 2, 2))))
    with pytest.raises(TypeError, match="band-name metadata"):
        MBMP()(np.ones((13, 2, 2)))


def test_single_pass_prefilter_clips_negative_values() -> None:
    op = MBMP(swir1=0, swir2=1)
    scene = np.array([[[0.0, -1.0], [2.0, 3.0]], [[0.0, 1.0], [-2.0, 3.0]]])
    out = np.asarray(op(_gt(scene)))
    c1, c2 = np.maximum(scene[0], 0.0), np.maximum(scene[1], 0.0)
    np.testing.assert_allclose(out, (c1 - c2) / (c1 + c2 + op.eps), rtol=1e-15)


def test_nodata_in_either_scene_is_nan() -> None:
    scene = np.stack([np.exp(TRUTH), np.ones_like(TRUTH)])
    reference = np.ones((2, 5, 5))
    scene[1, 0, 0] = -9999.0
    reference[0, 4, 4] = np.nan
    out = np.asarray(MBMP(swir1=0, swir2=1)(_gt(scene), _gt(reference)))
    assert np.isnan(out[0, 0]) and np.isnan(out[4, 4])
    assert np.isfinite(out).sum() == 23


def test_non_positive_bands_are_unscoreable_with_a_reference() -> None:
    """A dark pixel or a -9999 sentinel in a plain reference would read ±22."""
    scene = np.stack([np.exp(TRUTH), np.ones_like(TRUTH)])
    reference = np.ones((2, 5, 5))
    scene[1, 1, 1] = -0.001  # noisy dark band 12
    reference[0, 2, 2] = 0.0  # dark band 11 on the reference
    reference[1, 3, 3] = -9999.0  # an unnamed reference's sentinel
    out = np.asarray(MBMP(swir1=0, swir2=1)(_gt(scene), reference))
    assert np.isnan(out[1, 1]) and np.isnan(out[2, 2]) and np.isnan(out[3, 3])
    assert np.isfinite(out).sum() == 22
    assert np.nanmax(np.abs(out)) <= 0.2 + ATOL


def test_reference_on_another_grid_is_rejected() -> None:
    shifted = GeoTensor(
        np.ones((2, 5, 5)),
        transform=TRANSFORM * rasterio.Affine.translation(1, 0),
        crs="EPSG:32629",
        fill_value_default=-9999.0,
    )
    with pytest.raises(ValueError, match="reference_scene"):
        MBMP(swir1=0, swir2=1)(_gt(np.ones((2, 5, 5))), shifted)


def test_time_stack_reads_bands_along_minus_three() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0.1, 1.0, size=(2, 3, 4, 4))
    score = MBMP(swir1=1, swir2=0, eps=0.0)(_gt(values))
    assert score.shape == (2, 1, 4, 4)
    expected = (values[:, 1] - values[:, 0]) / (values[:, 1] + values[:, 0])
    np.testing.assert_allclose(np.asarray(score)[:, 0], expected, rtol=1e-15)


def test_carrier_kwarg_is_gone_and_config_round_trips() -> None:
    with pytest.raises(TypeError, match="reference_scene"):
        MBMP(reference_scene=np.ones((2, 2, 2)))  # type: ignore[call-arg]
    op = MBMP(swir1="B11", swir2=1, eps=1e-6)
    clone = Operator.from_state(json.loads(json.dumps(op.state, allow_nan=False)))
    assert clone.get_config() == op.get_config()


def test_keeps_the_carrier_contract() -> None:
    from geoops.testing import check_operator

    values = np.random.default_rng(1).uniform(0.1, 1.0, size=(2, 4, 4))
    values[:, 0, 0] = -9999.0
    gt = _gt(values)
    gt.attrs = {"band_names": ["B11", "B12"]}
    reference = _gt(np.full((2, 4, 4), 0.5))
    reference.attrs = {"band_names": ["B11", "B12"]}
    check_operator(MBMP(), gt)
    check_operator(MBMP(), gt, reference)


# ── scale factors, SBMP and MBSP (radtrax#111) ─────────────────────────────

SURFACE = np.random.default_rng(2).uniform(0.2, 0.4, size=(5, 5))


def _passes(brighten: float) -> tuple[np.ndarray, np.ndarray]:
    """A reference pass and a brighter plume pass: B11 = A, B12 = 0.8 A e^(−s)."""
    reference = np.stack([SURFACE, 0.8 * SURFACE])
    scene = brighten * np.stack([SURFACE, 0.8 * SURFACE * np.exp(-TRUTH)])
    return scene, reference


def test_scale_factors_remove_a_scene_wide_change() -> None:
    """Fitted on every pixel (the plume included), the factor moves with the
    plume: MBMP without factors reads the 5 % brightening as nothing, with
    factors the plume less its scene mean."""
    scene, reference = _passes(1.05)
    plain = np.asarray(MBMP(swir1=0, swir2=1)(scene, reference))
    scaled = np.asarray(MBMP(swir1=0, swir2=1, scale_factors=True)(scene, reference))
    np.testing.assert_allclose(plain, TRUTH, atol=ATOL)
    # c / c′ = Σ x y / Σ x² of each pass: the scaled signal is TRUTH less ln(c/c′)
    c = (scene[1] * scene[0]).sum() / (scene[1] ** 2).sum()
    c_ref = (reference[1] * reference[0]).sum() / (reference[1] ** 2).sum()
    np.testing.assert_allclose(scaled, TRUTH - np.log(c / c_ref), atol=ATOL)


def test_sbmp_removes_brightening_and_keeps_the_plume() -> None:
    scene, reference = _passes(1.05)
    out = np.asarray(SBMP(swir2=1)(scene, reference))
    c = (scene[1] * reference[1]).sum() / (scene[1] ** 2).sum()
    expected = -np.log(c * scene[1] / reference[1])
    np.testing.assert_allclose(out, expected, atol=ATOL)
    # the factor is fitted on the plume too, so the plume reads relative to it
    np.testing.assert_allclose(out - out.min(), TRUTH - TRUTH.min(), atol=ATOL)


def test_mbsp_reads_the_band_ratio_on_one_pass() -> None:
    scene, _ = _passes(1.0)
    out = np.asarray(MBSP(swir1=0, swir2=1)(scene))
    c = (scene[1] * scene[0]).sum() / (scene[1] ** 2).sum()
    np.testing.assert_allclose(out, -np.log(c * scene[1] / scene[0]), atol=ATOL)


def test_scale_factors_are_fitted_per_time_step() -> None:
    scene, reference = _passes(1.05)
    stack = np.stack([scene, 2.0 * scene])  # the second step: another brightening
    refs = np.stack([reference, reference])
    out = np.asarray(MBMP(swir1=0, swir2=1, scale_factors=True)(stack, refs))
    single = np.asarray(MBMP(swir1=0, swir2=1, scale_factors=True)(scene, reference))
    np.testing.assert_allclose(out[0, 0], single, atol=ATOL)
    np.testing.assert_allclose(out[1, 0], single, atol=ATOL)


def test_nodata_is_nan_and_stays_out_of_the_fit() -> None:
    scene, reference = _passes(1.05)
    scene[:, 0, 0] = -9999.0
    for out in (
        np.asarray(
            MBMP(swir1=0, swir2=1, scale_factors=True)(_gt(scene), _gt(reference))
        ),
        np.asarray(SBMP(swir2=1)(_gt(scene), _gt(reference))),
        np.asarray(MBSP(swir1=0, swir2=1)(_gt(scene))),
    ):
        assert np.isnan(out[0, 0]) and np.isfinite(out[1:]).all()


def test_new_operators_round_trip_their_config() -> None:
    for op in (
        MBMP(scale_factors=True),
        SBMP(swir2="B12", eps=1e-6),
        MBSP(swir1=0, swir2=1),
    ):
        clone = Operator.from_state(json.loads(json.dumps(op.state, allow_nan=False)))
        assert clone.get_config() == op.get_config()


def test_new_operators_keep_the_carrier_contract() -> None:
    from geoops.testing import check_operator

    values = np.random.default_rng(1).uniform(0.1, 1.0, size=(2, 4, 4))
    values[:, 0, 0] = -9999.0
    gt = _gt(values)
    gt.attrs = {"band_names": ["B11", "B12"]}
    reference = _gt(np.random.default_rng(2).uniform(0.1, 1.0, size=(2, 4, 4)))
    reference.attrs = {"band_names": ["B11", "B12"]}
    check_operator(MBSP(), gt)
    check_operator(SBMP(), gt, reference)
    check_operator(MBMP(scale_factors=True), gt, reference)
