"""The GeoTensor matched filter with v0's ForwardModel target (radtrax#98).

The model is radtrax · 3's v0 slice on the synthetic tables (``tests/_v0.py``):
CH₄ and a stand-in H₂O, Beer–Lambert, EMIT-like bands. Scenes are radiance
cubes ``(B, H, W)`` from that model on a GeoTensor grid, the albedo varying
from pixel to pixel and a CH₄ plume added to a block of pixels.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr


pytestmark = pytest.mark.integration
pytest.importorskip("georeader.geotensor")
pytest.importorskip("pipekit")

from _v0 import GEOMETRY, SURFACE, v0_atmosphere, v0_model

import radtrax.geo as rg
from geo._carriers import toy_geotensor
from radtrax import ForwardModel, ShapeError, unit_absorption_target
from radtrax.atmosphere import AtmosphericState
from radtrax.retrieval.matched_filter import matched_filter_snr
from radtrax.surface import Lambertian


H = W = 12
PLUME = 0.05  # mol/m², 9 % of the CH₄ column (as in tests/test_forward_model.py)


@pytest.fixture(scope="module")
def model(voigt_table: xr.Dataset) -> ForwardModel:
    return v0_model(voigt_table)


@pytest.fixture(scope="module")
def atm() -> AtmosphericState:
    return v0_atmosphere()


def _scene(
    model: ForwardModel,
    atm: AtmosphericState,
    delta: np.ndarray,
    seed: int,
    *,
    log: bool,
    noise: float = 1e-4,
) -> np.ndarray:
    """``ln y`` (``log``) or ``y`` ``(B, H, W)``, relative noise ``noise``.

    For ``ln y`` the albedo is uniform in [0.1, 0.4], an additive ``ln A``
    the background covariance learns. For ``y`` it is 0.25 everywhere: a
    radiance-space target ``μ ⊙ t`` scales with each pixel's brightness, so
    an albedo varying across the plume would scale its score with it.
    """
    rng = np.random.default_rng(seed)
    albedo = rng.uniform(0.1, 0.4, delta.shape) if log else np.full(delta.shape, 0.25)

    def y(d: jax.Array, a: jax.Array) -> jax.Array:
        return model(atm.add_column("CH4", d), Lambertian(a), GEOMETRY)

    clean = eqx.filter_jit(eqx.filter_vmap(y))(
        jnp.asarray(delta.ravel()), jnp.asarray(albedo.ravel())
    )
    cube = np.asarray(clean).reshape(*delta.shape, -1)
    cube = cube * (1.0 + noise * rng.standard_normal(cube.shape))
    return np.moveaxis(np.log(cube) if log else cube, -1, 0)


@pytest.mark.slow  # ~9 s, compiling the model's target and a scene
def test_target_is_the_mean_times_the_unit_absorption_spectrum(
    model: ForwardModel, atm: AtmosphericState
) -> None:
    gt = toy_geotensor(_scene(model, atm, np.zeros((4, 5)), seed=1, log=False))
    t = np.asarray(unit_absorption_target(model, atm, SURFACE, GEOMETRY))
    op = rg.UnitAbsorptionTarget(
        model=model, atmosphere=atm, surface=SURFACE, geometry=GEOMETRY
    )
    mean = rg.EstimateMean(method="median")(gt)
    np.testing.assert_allclose(op(gt), mean * t, rtol=1e-10)
    log_op = rg.UnitAbsorptionTarget(
        model=model, atmosphere=atm, surface=SURFACE, geometry=GEOMETRY, log_cube=True
    )
    np.testing.assert_allclose(log_op(gt), t, rtol=1e-10)
    assert np.all(t < 0.0)  # every band sees some CH₄ absorption


@pytest.mark.slow  # ~3 s per cube kind, most of it compiling the scenes
@pytest.mark.parametrize("log", [True, False], ids=["ln_y", "y"])
def test_a_geotensor_pipeline_retrieves_the_plume_column(
    model: ForwardModel, atm: AtmosphericState, log: bool
) -> None:
    """Background fitted on a plume-free scene, the target from the
    ForwardModel, scores on a scene with a 3 × 3 plume of 0.05 mol/m².

    The plume's noiseless score is ΔX less the curve of growth, which a
    linear target cannot see (< 3 %, as for the ln y filter in
    tests/test_forward_model.py); the noisy scores scatter about it, and
    about 0 off the plume, with the standard error 1/√(tᵀΣ⁻¹t).
    """
    delta = np.zeros((H, W))
    delta[4:7, 4:7] = PLUME
    plume = delta > 0
    background = toy_geotensor(_scene(model, atm, np.zeros((H, W)), 2, log=log))
    scene = toy_geotensor(_scene(model, atm, delta, 3, log=log))
    noiseless = toy_geotensor(_scene(model, atm, delta, 3, log=log, noise=0.0))

    target = rg.UnitAbsorptionTarget(
        model=model, atmosphere=atm, surface=SURFACE, geometry=GEOMETRY, log_cube=log
    )(background)
    # The empirical covariance without a ridge: ln y's noise variance is 1e-8,
    # the ridge MatchedFilter(cov_method="empirical") would add.
    mean = rg.EstimateMean(method="mean")(background)
    cov = rg.EstimateCovEmpirical(mean=mean)(background)
    mf = rg.MatchedFilter(mean=mean, cov_op=cov, target=target)
    scores = mf(scene)
    assert isinstance(scores, type(scene)) and scores.transform == scene.transform
    values = np.asarray(scores.values).reshape(H, W)
    expected = np.asarray(mf(noiseless).values).reshape(H, W)[plume].mean()
    se = 1.0 / float(matched_filter_snr(1.0, cov, jnp.asarray(target)))
    assert se < PLUME / 20  # the plume stands well above the noise

    assert 0 < 1 - expected / PLUME < 0.03
    # The mean of 9 plume pixels within 5 of its standard errors (se / 3).
    assert abs(values[plume].mean() - expected) < 5 * se / 3
    # Background scores: mean 0 within 5 standard errors (se / √135), and
    # their sample standard deviation se within 5 of its own standard
    # errors, 1/√(2 · 134) relative.
    assert abs(values[~plume].mean()) < 5 * se / np.sqrt(135)
    assert abs(values[~plume].std(ddof=1) / se - 1) < 5 / np.sqrt(2 * 134)


def test_band_count_must_match_the_cube(
    model: ForwardModel, atm: AtmosphericState
) -> None:
    op = rg.UnitAbsorptionTarget(
        model=model, atmosphere=atm, surface=SURFACE, geometry=GEOMETRY
    )
    n_bands = model.instrument.response.shape[0]
    with pytest.raises(ShapeError, match="bands"):
        op(toy_geotensor(np.ones((n_bands + 1, 2, 2))))


def test_config_and_yaml(model: ForwardModel, atm: AtmosphericState) -> None:
    assert rg.UnitAbsorptionTarget.forbid_in_yaml is True
    op = rg.UnitAbsorptionTarget(
        model=model,
        atmosphere=atm,
        surface=SURFACE,
        geometry=GEOMETRY,
        profile=np.ones(20),
        log_cube=True,
    )
    config = op.get_config()
    assert config["gas"] == "CH4" and config["log_cube"] is True
    assert config["profile"] == [1.0] * 20
    assert isinstance(config["model"], str)
