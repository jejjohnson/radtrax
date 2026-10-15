"""GeoTensor operators for the matched filter (behind the ``[geo]`` extra).

pipekit ``Operator`` wrappers around the one matched-filter core,
:mod:`radtrax.retrieval.matched_filter` — ported from geotoolz's
``matched_filter`` operators (radtrax#29), which now delegate here. They keep
geotoolz's carrier contract: a ``GeoTensor`` in gives a ``GeoTensor`` out on
the same grid (a plain array gives a plain array), score maps are a new
quantity holding ``NaN`` at nodata pixels with ``fill_value_default=NaN``,
and estimators return non-carrier results (vectors, covariance operators,
backgrounds).

Layout: the band axis is ``-3`` (georeader's ``(C, H, W)`` /
``(T, C, H, W)``), so these operators move it to the core's band-last
``(H, W, B)`` and back. On a ``(T, C, H, W)`` stack the background is fitted
once over the pixels of every frame and score maps come back
``(T, 1, H, W)``.

Nodata: a pixel that is non-finite or equals the carrier's
``fill_value_default`` in any band is left out of every statistic.

Precision: the core is JAX; the operators run it in float64
(``jax.enable_x64`` for the call only), matching geotoolz's NumPy results
whatever the global JAX setting. Only the operators do: a covariance operator
they return (``EstimateCov*``, ``MatchedFilter.cov_op_``,
``StreamingBackgroundResult.cov_op``) holds float64 arrays, and using it
directly with the core under float32 JAX truncates it.

Conventions that differ from geotoolz's own implementation:

- covariances are the ``1/n`` (maximum-likelihood) scatter, as scikit-learn
  and the rest of radtrax; geotoolz used ``1/(n − 1)``;
- :class:`AdaptiveWindowBackground` gives the population variance, takes a
  ``guard_window`` and always reflects at the edges (no ``pad_mode``);
- :class:`StreamingBackground` names its shrinkage ``"ledoit_wolf"`` /
  ``"oas"`` (geotoolz: ``"shrunk"``);
- :class:`GMMClusterBackground` fits scikit-learn's Gaussian mixture.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Bool, Float

from radtrax.retrieval.matched_filter.background import (
    _dense_operator,
    estimate_cov_empirical,
    estimate_cov_lowrank,
    estimate_cov_shrunk,
    estimate_mean,
)
from radtrax.retrieval.matched_filter.cluster import (
    ClusterBackground,
    adaptive_window_background,
    apply_adaptive_mf,
    apply_cluster_mf,
    gmm_cluster_background,
)
from radtrax.retrieval.matched_filter.core import (
    apply_image,
    apply_pixel,
    detection_threshold,
    matched_filter_snr,
    validate_mf_inputs,
)
from radtrax.retrieval.matched_filter.streaming import streaming_background
from radtrax.retrieval.matched_filter.target import (
    linear_target_from_obs,
    nonlinear_target_from_obs,
)
from radtrax.retrieval.var3d.obs_operator import RadianceObservationModel


try:
    from geotoolz.carrier import (
        callable_name,
        cube_to_samples,
        fit_once,
        invalid_values,
        jsonable,
        keep_band_axis,
        reject_config_summary,
        require_ndim,
        restore_fill,
        wrap_like,
    )
    from pipekit import Operator
except ModuleNotFoundError as err:  # pragma: no cover - the base-install gate
    raise ImportError(
        "radtrax.geo.matched_filter needs the [geo] extra: pip install 'radtrax[geo]'"
    ) from err


# Shapes, in radtrax's dimension names: B bands, H W pixels, N samples. A cube
# has its band axis at ``axis`` (default -3), so a time stack is "*time B H W".
Spectrum = Float[np.ndarray, " B"]
CovMatrix = Float[np.ndarray, "B B"]
Cube = Float[np.ndarray, "*time B H W"]
VMRField = Float[np.ndarray, "H W"]
ScoreArray = Float[np.ndarray, "*time H W"]
ValidMap = Bool[np.ndarray, "*time H W"]
Samples = Float[np.ndarray, "N 1 B"]

if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

    #: A spectral cube: a GeoTensor or a plain array.
    Carrier = GeoTensor | Cube
    #: A score map on the input's carrier: ``(H, W)``, or ``(T, 1, H, W)``.
    ScoreMap = GeoTensor | ScoreArray
else:
    Carrier = Any
    ScoreMap = Any


__all__ = [
    "AdaptiveBackground",
    "AdaptiveWindowBackground",
    "ApplyAdaptiveMF",
    "ApplyClusterMF",
    "ColumnEnhancement",
    "DetectionThreshold",
    "EstimateCovEmpirical",
    "EstimateCovLowRank",
    "EstimateCovShrunk",
    "EstimateMean",
    "GMMClusterBackground",
    "LinearTargetFromObs",
    "MatchedFilter",
    "MatchedFilterPixel",
    "MatchedFilterSNR",
    "NonlinearTargetFromObs",
    "StreamingBackground",
    "StreamingBackgroundResult",
    "ValidateMFInputs",
]


MeanMethod = Literal["mean", "median", "trimmed", "huber"]
CovMethod = Literal["empirical", "ledoit_wolf", "oas", "lowrank"]
ShrinkMethod = Literal["ledoit_wolf", "oas"]
#: A covariance: a lineax / gaussx operator or a dense ``(B, B)`` matrix.
CovLike = lx.AbstractLinearOperator | CovMatrix

#: Step of the central-difference target for a plain callable ``obs_model``,
#: relative to the largest magnitude in the background state (absolute when the
#: state is all zero). An observation model gets an exact JVP instead.
TARGET_FINITE_DIFFERENCE_EPSILON = 1e-4

#: Defaults of the low-rank covariance (``EstimateCovLowRank`` and the
#: ``"lowrank"`` method of ``MatchedFilter`` / ``ColumnEnhancement``).
LOWRANK_RANK = 10
LOWRANK_TIKHONOV = 1e-3


class AdaptiveBackground(NamedTuple):
    """Local window statistics, in the layout of the cube they came from.

    Attributes:
        mean: Per-pixel local mean, the cube's shape.
        variance: Per-pixel per-band local (population) variance, the cube's
            shape.
    """

    mean: Cube
    variance: Cube


class StreamingBackgroundResult(NamedTuple):
    """Background from :class:`StreamingBackground`.

    Attributes:
        mean: Mean spectrum ``(B,)``.
        cov_op: Covariance operator ``(B, B)``.
    """

    mean: Spectrum
    cov_op: lx.AbstractLinearOperator


def _float64[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` with JAX in float64, for this call only."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with jax.enable_x64(True):
            return fn(*args, **kwargs)

    return wrapper


# ── matched filter ───────────────────────────────────────────────────────────


class MatchedFilter(Operator):
    """Per-pixel matched-filter score map.

    ``y = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)`` for every pixel, through
    :func:`radtrax.retrieval.matched_filter.apply_image`, as a single-band
    map on the input's carrier.

    A fitted operator: :meth:`fit` estimates whichever of ``mean`` /
    ``cov_op`` the constructor left unset (with ``mean_method`` /
    ``cov_method``) into ``mean_`` / ``cov_op_``; :meth:`transform` scores
    with that background; calling the operator fits the missing statistics
    once, on the first call, or — with ``fit_on_call=True`` — estimates them
    from every cube without storing anything. Fitted arrays never reach
    ``get_config()``.

    Args:
        mean: Background mean spectrum ``(B,)``; fitted when ``None``.
        cov_op: Background covariance, a lineax operator or a ``(B, B)``
            matrix (a nested list from a config reload is accepted); fitted
            when ``None``.
        target: Target signature ``(B,)``; required before scoring.
        fit_on_call: Estimate the background from each cube (per scene)
            instead of fitting the missing statistics once.
        mean_method: Mean estimator when fitting. Default ``"median"``.
        cov_method: ``"empirical"`` (with a ``1e-8`` ridge),
            ``"ledoit_wolf"`` (default), ``"oas"`` or ``"lowrank"``.
        axis: Position of the band axis. Default ``-3``.

    Attributes:
        mean_: Fitted mean, or ``None``.
        cov_op_: Fitted covariance operator, or ``None``.
    """

    def __init__(
        self,
        *,
        mean: Spectrum | None = None,
        cov_op: CovLike | None = None,
        target: Spectrum | None = None,
        fit_on_call: bool = False,
        mean_method: MeanMethod = "median",
        cov_method: CovMethod = "ledoit_wolf",
        axis: int = -3,
    ) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.cov_op = _as_cov(cov_op)
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.fit_on_call = fit_on_call
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.axis = axis
        self.mean_: Spectrum | None = None
        self.cov_op_: CovLike | None = None

    def _estimate(
        self, samples: Samples, *, keep_supplied: bool
    ) -> tuple[Spectrum, CovLike]:
        mean = self.mean if keep_supplied else None
        cov_op = self.cov_op if keep_supplied else None
        if mean is None:
            mean = estimate_mean(samples, method=self.mean_method)
        if cov_op is None:
            cov_op = _estimate_cov(samples, mean, self.cov_method)
        return np.asarray(mean, dtype=float), cov_op

    @_float64
    def fit(self, gt: Carrier) -> Self:
        """Estimate the unset background statistics from the valid pixels of ``gt``.

        Returns:
            This operator, for chaining.
        """
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        mean, cov_op = self._estimate(samples, keep_supplied=True)
        self.cov_op_ = cov_op
        self.mean_ = mean  # published last: the fit-once check reads mean_
        return self

    def _background(self) -> tuple[Spectrum, CovLike]:
        if self.mean_ is not None and self.cov_op_ is not None:
            return self.mean_, self.cov_op_
        if self.mean is not None and self.cov_op is not None:
            return self.mean, self.cov_op
        raise ValueError(
            "MatchedFilter requires mean and cov_op, fit(), or a call on data"
        )

    @_float64
    def transform(self, gt: Carrier) -> ScoreMap:
        """Score ``gt`` against the fitted (or supplied) background.

        Raises:
            ValueError: If no background is available or ``target`` is unset.
        """
        mean, cov_op = self._background()
        return self._score(gt, mean, cov_op)

    def _score(
        self,
        gt: Carrier,
        mean: Spectrum,
        cov_op: CovLike,
        prepared: tuple[Samples, tuple[int, ...], ValidMap | None] | None = None,
    ) -> ScoreMap:
        if self.target is None:
            raise ValueError("target must be supplied before applying MatchedFilter")
        samples, shape, valid = (
            _samples(gt, self.axis, type(self).__name__)
            if prepared is None
            else prepared
        )
        scores = apply_image(
            jnp.asarray(samples),
            jnp.asarray(mean),
            _operator(cov_op),
            jnp.asarray(self.target),
        )
        return _score_map(gt, np.asarray(scores).reshape(shape), valid)

    @_float64
    def _apply(self, gt: Carrier) -> ScoreMap:
        if self.fit_on_call:
            # Per-scene background: estimated locally, never stored, so
            # concurrent calls share no mutable state.
            prepared = _samples(gt, self.axis, type(self).__name__)
            mean, cov_op = self._estimate(prepared[0], keep_supplied=False)
            return self._score(gt, mean, cov_op, prepared)
        if self.mean is None or self.cov_op is None:
            fit_once(self, gt, lambda: self.mean_ is not None)
        return self.transform(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _array_config(self.mean),
            "cov_op": _cov_config(self.cov_op),
            "target": _array_config(self.target),
            "fit_on_call": self.fit_on_call,
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "axis": self.axis,
        }


class MatchedFilterPixel(Operator):
    """Matched-filter score of one pixel spectrum, as a ``float``.

    Args:
        mean: Background mean spectrum ``(B,)``.
        cov_op: Background covariance, a lineax operator or a ``(B, B)``
            matrix.
        target: Target signature ``(B,)``.
    """

    _terminal: ClassVar[bool] = True

    def __init__(self, *, mean: Spectrum, cov_op: CovLike, target: Spectrum) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.cov_op = _as_cov(cov_op)
        self.target = np.asarray(target, dtype=float)

    @_float64
    def _apply(self, pixel: Spectrum) -> float:
        return float(
            apply_pixel(
                jnp.asarray(pixel, dtype=float),
                jnp.asarray(self.mean),
                _operator(self.cov_op),
                jnp.asarray(self.target),
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _array_config(self.mean),
            "cov_op": _cov_config(self.cov_op),
            "target": _array_config(self.target),
        }


class MatchedFilterSNR(Operator):
    """Theoretical detection SNR ``amplitude · √(tᵀ Σ⁻¹ t)``.

    A graph source: computed from its configuration; an optional input is
    accepted and ignored, so it can close a ``Sequential``.

    Args:
        amplitude: Target amplitude.
        cov_op: Background covariance, a lineax operator or a ``(B, B)``
            matrix.
        target: Target signature ``(B,)``.
    """

    _terminal: ClassVar[bool] = True

    def __init__(self, *, amplitude: float, cov_op: CovLike, target: Spectrum) -> None:
        self.amplitude = amplitude
        self.cov_op = _as_cov(cov_op)
        self.target = np.asarray(target, dtype=float)

    @_float64
    def _apply(self, _input: Any | None = None) -> float:
        return float(
            matched_filter_snr(
                self.amplitude, _operator(self.cov_op), jnp.asarray(self.target)
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "cov_op": _cov_config(self.cov_op),
            "target": _array_config(self.target),
        }


class DetectionThreshold(Operator):
    """Score threshold with a given false-alarm rate under the Gaussian null.

    A graph source, like :class:`MatchedFilterSNR`.

    Args:
        false_alarm_rate: Exceedance probability, in ``(0, 1)``.
        cov_op: Background covariance, a lineax operator or a ``(B, B)``
            matrix.
        target: Target signature ``(B,)``.
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self, *, false_alarm_rate: float, cov_op: CovLike, target: Spectrum
    ) -> None:
        self.false_alarm_rate = false_alarm_rate
        self.cov_op = _as_cov(cov_op)
        self.target = np.asarray(target, dtype=float)

    @_float64
    def _apply(self, _input: Any | None = None) -> float:
        return float(
            detection_threshold(
                self.false_alarm_rate, _operator(self.cov_op), jnp.asarray(self.target)
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "false_alarm_rate": self.false_alarm_rate,
            "cov_op": _cov_config(self.cov_op),
            "target": _array_config(self.target),
        }


class ValidateMFInputs(Operator):
    """Pass-through guard: fail early on a zero target or a singular covariance.

    Returns its input unchanged, whatever its type.

    Args:
        cov_op: Background covariance, a lineax operator or a ``(B, B)``
            matrix.
        target: Target signature ``(B,)``.
    """

    def __init__(self, *, cov_op: CovLike, target: Spectrum) -> None:
        self.cov_op = _as_cov(cov_op)
        self.target = np.asarray(target, dtype=float)

    @_float64
    def _apply(self, value: Any = None) -> Any:
        validate_mf_inputs(_operator(self.cov_op), jnp.asarray(self.target))
        return value

    def get_config(self) -> dict[str, Any]:
        return {
            "cov_op": _cov_config(self.cov_op),
            "target": _array_config(self.target),
        }


# ── background estimators ────────────────────────────────────────────────────


class EstimateMean(Operator):
    """Background mean spectrum ``(B,)`` of a cube's valid pixels.

    Args:
        method: ``"mean"`` (default), ``"median"``, ``"trimmed"`` or
            ``"huber"``.
        trim_proportion: Fraction cut from each tail for ``"trimmed"``.
        huber_c: Huber threshold in MAD units for ``"huber"``.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self,
        *,
        method: MeanMethod = "mean",
        trim_proportion: float = 0.1,
        huber_c: float = 1.345,
        axis: int = -3,
    ) -> None:
        self.method = method
        self.trim_proportion = trim_proportion
        self.huber_c = huber_c
        self.axis = axis

    def _apply(self, gt: Carrier) -> Spectrum:
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        return np.asarray(
            estimate_mean(
                samples,
                method=self.method,
                trim_proportion=self.trim_proportion,
                huber_c=self.huber_c,
            )
        )


class EstimateCovEmpirical(Operator):
    """Empirical (``1/n``) covariance operator of a cube's valid pixels.

    Args:
        mean: Mean spectrum ``(B,)`` to centre on; the sample mean when
            ``None``.
        ridge: Diagonal ridge added to the covariance.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self, *, mean: Spectrum | None = None, ridge: float = 0.0, axis: int = -3
    ) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.ridge = ridge
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> lx.AbstractLinearOperator:
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        return estimate_cov_empirical(samples, mean=self.mean, ridge=self.ridge)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _array_config(self.mean),
            "ridge": self.ridge,
            "axis": self.axis,
        }


class EstimateCovShrunk(Operator):
    """Ledoit–Wolf or OAS covariance operator of a cube's valid pixels.

    Args:
        mean: Mean spectrum ``(B,)`` to centre on; the sample mean when
            ``None``.
        method: ``"ledoit_wolf"`` (default) or ``"oas"``.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self,
        *,
        mean: Spectrum | None = None,
        method: ShrinkMethod = "ledoit_wolf",
        axis: int = -3,
    ) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.method = method
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> lx.AbstractLinearOperator:
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        return estimate_cov_shrunk(samples, mean=self.mean, method=self.method)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _array_config(self.mean),
            "method": self.method,
            "axis": self.axis,
        }


class EstimateCovLowRank(Operator):
    """Low-rank plus Tikhonov covariance ``λI + V D Vᵀ`` of a cube's valid pixels.

    Args:
        mean: Mean spectrum ``(B,)`` to centre on; the sample mean when
            ``None``.
        rank: Leading components kept.
        tikhonov: Diagonal floor ``λ > 0``.
        seed: Seed of the randomised SVD.
        n_oversamples: Extra random directions beyond ``rank``.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self,
        *,
        mean: Spectrum | None = None,
        rank: int = LOWRANK_RANK,
        tikhonov: float = LOWRANK_TIKHONOV,
        seed: int | None = 0,
        n_oversamples: int = 10,
        axis: int = -3,
    ) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.rank = rank
        self.tikhonov = tikhonov
        self.seed = seed
        self.n_oversamples = n_oversamples
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> lx.AbstractLinearOperator:
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        return estimate_cov_lowrank(
            samples,
            mean=self.mean,
            rank=self.rank,
            tikhonov=self.tikhonov,
            random_state=self.seed,
            n_oversamples=self.n_oversamples,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _array_config(self.mean),
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "seed": self.seed,
            "n_oversamples": self.n_oversamples,
            "axis": self.axis,
        }


class GMMClusterBackground(Operator):
    """Per-cluster backgrounds from a Gaussian mixture over the valid pixels.

    Returns a :class:`~radtrax.retrieval.matched_filter.ClusterBackground`
    whose ``labels`` have the cube's spatial shape (``-1`` at nodata pixels);
    wire it into :class:`ApplyClusterMF`.

    Args:
        n_clusters: Number of mixture components.
        cov_estimator: Per-cluster covariance, ``"empirical"``,
            ``"ledoit_wolf"`` (default) or ``"oas"``.
        seed: Seed of the mixture fit.
        bayesian: Use a Bayesian mixture, which prunes unused components.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self,
        *,
        n_clusters: int,
        cov_estimator: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
        seed: int | None = 0,
        bayesian: bool = False,
        axis: int = -3,
    ) -> None:
        self.n_clusters = n_clusters
        self.cov_estimator = cov_estimator
        self.seed = seed
        self.bayesian = bayesian
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> ClusterBackground:
        samples, shape, _ = _samples(gt, self.axis, type(self).__name__)
        background = gmm_cluster_background(
            samples,
            n_clusters=self.n_clusters,
            cov_estimator=self.cov_estimator,
            random_state=self.seed,
            bayesian=self.bayesian,
        )
        return eqx.tree_at(
            lambda b: b.labels, background, background.labels.reshape(shape)
        )


class ApplyClusterMF(Operator):
    """Matched filter with each pixel's own cluster background.

    The background arrives at call time, so the config (``target``,
    ``axis``) stays YAML-safe: ``ApplyClusterMF(target=t)(gt, background)``
    with the background from :class:`GMMClusterBackground`.

    Args:
        target: Target signature ``(B,)``.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(self, *, target: Spectrum, axis: int = -3) -> None:
        self.target = np.asarray(target, dtype=float)
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier, cluster: ClusterBackground) -> ScoreMap:
        samples, shape, valid = _samples(gt, self.axis, type(self).__name__)
        labels = jnp.asarray(cluster.labels)
        if labels.shape != shape:
            raise ValueError(
                f"ApplyClusterMF: labels shape {labels.shape} does not match the "
                f"cube's spatial shape {shape}"
            )
        flat = eqx.tree_at(lambda b: b.labels, cluster, labels.reshape(-1, 1))
        scores = apply_cluster_mf(jnp.asarray(samples), flat, jnp.asarray(self.target))
        return _score_map(gt, np.asarray(scores).reshape(shape), valid)

    def get_config(self) -> dict[str, Any]:
        return {"target": _array_config(self.target), "axis": self.axis}


class AdaptiveWindowBackground(Operator):
    """Per-pixel local mean and variance over a square window.

    Returns an :class:`AdaptiveBackground` of two cubes in the input's layout
    (not a carrier); nodata pixels carry no weight and are ``NaN`` in both.

    Args:
        window: Side length of the window, an odd integer ``≥ 3``. Default
            ``7``.
        guard_window: Side length of a central guard region left out of each
            window (``0``, the default, keeps the pixel itself; ``1`` drops
            it, so a plume does not bias its own background).
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self, *, window: int = 7, guard_window: int = 0, axis: int = -3
    ) -> None:
        self.window = window
        self.guard_window = guard_window
        self.axis = axis

    def _apply(self, gt: Carrier) -> AdaptiveBackground:
        require_ndim(gt, 3, type(self).__name__)
        cube = _band_last(gt, self.axis, type(self).__name__)
        mean, var = adaptive_window_background(
            cube, window_size=self.window, guard_size=self.guard_window
        )
        return AdaptiveBackground(
            mean=np.moveaxis(mean, -1, self.axis),
            variance=np.moveaxis(var, -1, self.axis),
        )


class ApplyAdaptiveMF(Operator):
    """Matched filter against each pixel's local diagonal background.

    ``α = tᵀ Λ⁻¹ (x − μ) / (tᵀ Λ⁻¹ t)`` with ``Λ = diag(σ² + ridge)`` from
    :class:`AdaptiveWindowBackground`, passed at call time:
    ``ApplyAdaptiveMF(target=t)(gt, background)``.

    Args:
        target: Target signature ``(B,)``.
        ridge: Non-negative variance floor. Default ``0``.
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(self, *, target: Spectrum, ridge: float = 0.0, axis: int = -3) -> None:
        self.target = np.asarray(target, dtype=float)
        self.ridge = ridge
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier, background: AdaptiveBackground) -> ScoreMap:
        name = type(self).__name__
        require_ndim(gt, 3, name)
        masked, valid = _masked(gt, self.axis, name)
        cube = np.moveaxis(masked, self.axis, -1)
        mean = np.moveaxis(np.asarray(background.mean, dtype=float), self.axis, -1)
        var = np.moveaxis(np.asarray(background.variance, dtype=float), self.axis, -1)
        scores = apply_adaptive_mf(
            cube, mean, var, jnp.asarray(self.target), ridge=self.ridge
        )
        return _score_map(gt, np.asarray(scores), valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "target": _array_config(self.target),
            "ridge": self.ridge,
            "axis": self.axis,
        }


class StreamingBackground(Operator):
    """Mean and covariance accumulated over an iterable of cubes.

    Streams the valid pixels of every cube (``GeoTensor`` or array) through
    :func:`radtrax.retrieval.matched_filter.streaming_background`; the
    cubes arrive at call time, so the config stays YAML-safe.

    Args:
        cov_kind: ``"ledoit_wolf"`` (default), ``"oas"`` or ``"empirical"``;
            the shrunk forms equal scikit-learn's on all the pixels at once.
        ridge: Diagonal ridge added to the covariance. Default ``1e-8``.
        axis: Position of the band axis in each cube. Default ``-3``.
    """

    def __init__(
        self,
        *,
        cov_kind: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
        ridge: float = 1e-8,
        axis: int = -3,
    ) -> None:
        self.cov_kind = cov_kind
        self.ridge = ridge
        self.axis = axis

    @_float64
    def _apply(self, cubes: Iterable[Carrier]) -> StreamingBackgroundResult:
        name = type(self).__name__
        batches: Iterator[Samples] = (
            _samples(cube, self.axis, name)[0] for cube in cubes
        )
        first = next(batches, None)
        if first is None:
            raise ValueError(
                "StreamingBackground: cubes must contain at least one cube"
            )
        mean, cov_op = streaming_background(
            itertools.chain([first], batches),
            n_bands=first.shape[-1],
            ridge=self.ridge,
            cov_kind=self.cov_kind,
        )
        return StreamingBackgroundResult(mean=np.asarray(mean), cov_op=cov_op)


# ── targets ──────────────────────────────────────────────────────────────────


class LinearTargetFromObs(Operator):
    """Tangent-linear target signature ``(B,)`` from an observation model.

    With a
    :class:`~radtrax.retrieval.var3d.RadianceObservationModel` the target is
    the exact JVP of its forward model
    (:func:`radtrax.retrieval.matched_filter.linear_target_from_obs`) about
    ``vmr_background`` (default: the model's reference VMR on the cube's
    grid; pass it on the high-resolution grid when the model has a GSD).
    Any other callable maps a state cube to a response with its band axis at
    ``axis`` (or to a ``(B,)`` vector), about ``vmr_background`` (default:
    the cube itself, its nodata pixels set to the band means of the valid
    ones); a cube response is averaged over the valid pixels. A JAX-traceable
    callable is linearised exactly with :func:`jax.jvp`; one that is not
    (NumPy code converting its input) by a central difference, which is only
    as good as its step: ``fd_step`` when given, else ``1e-4`` times the
    state's largest magnitude (``1e-4`` itself for an all-zero state, where
    the state carries no scale — set ``fd_step`` from the model's then). (geotoolz always took a
    one-sided step of ``1e-4`` in the state's own units, far outside the
    linear regime for a VMR of ~2e-6.)

    Args:
        obs_model: An observation model, or a callable.
        vmr_background: Background state; see above.
        pattern: ``"uniform"`` (default), ``"impulse"`` or an explicit
            perturbation array.
        pixel: Impulse location (and, for an observation model, the output
            pixel read).
        axis: Position of the band axis. Default ``-3``.
        fd_step: Absolute central-difference step for a callable that JAX
            cannot trace; see above.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: VMRField | Cube | None = None,
        pattern: str | VMRField | Cube = "uniform",
        pixel: tuple[int, int] | None = None,
        axis: int = -3,
        fd_step: float | None = None,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.pattern = pattern
        self.pixel = pixel
        self.axis = axis
        self.fd_step = fd_step

    @_float64
    def _apply(self, gt: Carrier) -> Spectrum:
        if isinstance(self.obs_model, RadianceObservationModel):
            return np.asarray(
                linear_target_from_obs(
                    self.obs_model,
                    _vmr_background(self.obs_model, gt, self.vmr_background, self.axis),
                    pattern=_jax_pattern(self.pattern),
                    pixel=self.pixel,
                )
            )
        base, direction, valid = _callable_state(
            self.obs_model, gt, self.vmr_background, self.pattern, self.pixel, self.axis
        )
        try:
            _, tangent = jax.jvp(
                lambda state: jnp.asarray(self.obs_model(state), dtype=float),
                (jnp.asarray(base),),
                (jnp.asarray(direction),),
            )
            response = np.asarray(tangent)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerBoolConversionError,
            jax.errors.TracerIntegerConversionError,
        ):
            # Not JAX-traceable: a central difference instead.
            h = self.fd_step
            if h is None:
                scale = float(np.max(np.abs(base), initial=0.0))
                h = TARGET_FINITE_DIFFERENCE_EPSILON * (scale if scale > 0.0 else 1.0)
            response = (
                np.asarray(self.obs_model(base + h * direction), dtype=float)
                - np.asarray(self.obs_model(base - h * direction), dtype=float)
            ) / (2.0 * h)
        return _response_vector(response, gt, self.axis, valid)

    def get_config(self) -> dict[str, Any]:
        return _target_config(self) | {"fd_step": self.fd_step}


class NonlinearTargetFromObs(Operator):
    """Finite-amplitude target ``H(x_b + α δx) − H(x_b)``, ``(B,)``.

    Keeps the forward model's nonlinearity at amplitude ``α``. With a
    :class:`~radtrax.retrieval.var3d.RadianceObservationModel` it is
    :func:`radtrax.retrieval.matched_filter.nonlinear_target_from_obs` about
    ``vmr_background`` (default: the model's reference VMR on the cube's
    grid). Any other callable maps a state cube to a response with its band
    axis at ``axis`` (or to a ``(B,)`` vector), about ``vmr_background``
    (default: the cube, nodata pixels set to the valid band means); a cube
    response is averaged over the valid pixels.

    Args:
        obs_model: An observation model, or a callable.
        vmr_background: Background state; see above (the incoming cube for a
            callable).
        amplitude: Perturbation amplitude ``α``. Default ``1``.
        pattern: ``"uniform"`` (default), ``"impulse"`` or an explicit
            perturbation array.
        pixel: Impulse location for ``"impulse"``.
        axis: Position of the band axis. Default ``-3``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: VMRField | Cube | None = None,
        amplitude: float = 1.0,
        pattern: str | VMRField | Cube = "uniform",
        pixel: tuple[int, int] | None = None,
        axis: int = -3,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.amplitude = amplitude
        self.pattern = pattern
        self.pixel = pixel
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> Spectrum:
        if isinstance(self.obs_model, RadianceObservationModel):
            return np.asarray(
                nonlinear_target_from_obs(
                    self.obs_model,
                    _vmr_background(self.obs_model, gt, self.vmr_background, self.axis),
                    amplitude=self.amplitude,
                    pattern=_jax_pattern(self.pattern),
                    pixel=self.pixel,
                )
            )
        base, direction, valid = _callable_state(
            self.obs_model, gt, self.vmr_background, self.pattern, self.pixel, self.axis
        )
        response = np.asarray(
            self.obs_model(base + self.amplitude * direction), dtype=float
        ) - np.asarray(self.obs_model(base), dtype=float)
        return _response_vector(response, gt, self.axis, valid)

    def get_config(self) -> dict[str, Any]:
        return _target_config(self) | {"amplitude": self.amplitude}


class ColumnEnhancement(Operator):
    """Trace-gas enhancement map: mean, covariance, target and matched filter.

    Chains :class:`EstimateMean`, :class:`EstimateCovShrunk` (or
    :class:`EstimateCovLowRank`), :class:`LinearTargetFromObs` (a uniform
    unit target without ``obs_model``) and :class:`MatchedFilter`.

    Args:
        gas: Gas label (metadata only). Default ``"CH4"``.
        sensor: Sensor label (metadata only). Default ``"EMIT"``.
        obs_model: Observation model or callable for the target.
        mean_method: Mean estimator. Default ``"median"``.
        cov_method: ``"ledoit_wolf"`` (default), ``"oas"`` or ``"lowrank"``.
        target_pattern: Perturbation pattern of the target.
        vmr_background: Background state of the target, forwarded to
            :class:`LinearTargetFromObs`; pass it on the model's
            high-resolution grid when the observation model has a GSD.
        target_pixel: Impulse location / output pixel of the target.
        fd_step: Central-difference step of the target for an ``obs_model``
            JAX cannot trace, forwarded to :class:`LinearTargetFromObs`.
        rank: Rank for ``"lowrank"`` (default 10 when ``None``).
        tikhonov: Floor for ``"lowrank"`` (default ``1e-3`` when ``None``).
        axis: Position of the band axis. Default ``-3``.
    """

    def __init__(
        self,
        *,
        gas: str = "CH4",
        sensor: str = "EMIT",
        obs_model: Any | None = None,
        mean_method: MeanMethod = "median",
        cov_method: ShrinkMethod | Literal["lowrank"] = "ledoit_wolf",
        target_pattern: str = "uniform",
        vmr_background: VMRField | Cube | None = None,
        target_pixel: tuple[int, int] | None = None,
        fd_step: float | None = None,
        rank: int | None = None,
        tikhonov: float | None = None,
        axis: int = -3,
    ) -> None:
        self.gas = gas
        self.sensor = sensor
        self.obs_model = reject_config_summary(obs_model, "obs_model")
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.target_pattern = target_pattern
        self.vmr_background = (
            None if vmr_background is None else np.asarray(vmr_background, dtype=float)
        )
        self.target_pixel = None if target_pixel is None else tuple(target_pixel)
        self.fd_step = fd_step
        self.rank = rank
        self.tikhonov = tikhonov
        self.axis = axis

    @_float64
    def _apply(self, gt: Carrier) -> ScoreMap:
        mean = EstimateMean(method=self.mean_method, axis=self.axis)(gt)
        samples, _, _ = _samples(gt, self.axis, type(self).__name__)
        cov_op = _estimate_cov(
            samples,
            mean,
            self.cov_method,
            rank=LOWRANK_RANK if self.rank is None else self.rank,
            tikhonov=LOWRANK_TIKHONOV if self.tikhonov is None else self.tikhonov,
        )
        if self.obs_model is None:
            target = np.ones_like(mean)
        else:
            target = LinearTargetFromObs(
                obs_model=self.obs_model,
                vmr_background=self.vmr_background,
                pattern=self.target_pattern,
                pixel=self.target_pixel,
                axis=self.axis,
                fd_step=self.fd_step,
            )(gt)
        return MatchedFilter(mean=mean, cov_op=cov_op, target=target, axis=self.axis)(
            gt
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "gas": self.gas,
            "sensor": self.sensor,
            # A callable has no config form: the summary makes from_state
            # refuse the reload instead of silently using a uniform target.
            "obs_model": None
            if self.obs_model is None
            else {"callable": callable_name(self.obs_model)},
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "target_pattern": self.target_pattern,
            "vmr_background": _array_config(self.vmr_background),
            "target_pixel": None
            if self.target_pixel is None
            else list(self.target_pixel),
            "fd_step": self.fd_step,
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "axis": self.axis,
        }


# ── helpers ──────────────────────────────────────────────────────────────────


def _as_cov(cov_op: CovLike | Any | None) -> CovLike | None:
    """A lineax operator as given; anything else (a matrix, a nested list) as an array."""
    if cov_op is None or isinstance(cov_op, lx.AbstractLinearOperator):
        return cov_op
    return np.asarray(cov_op, dtype=float)


def _operator(cov_op: CovLike) -> lx.AbstractLinearOperator:
    """The core's covariance argument: a lineax operator."""
    if isinstance(cov_op, lx.AbstractLinearOperator):
        return cov_op
    return _dense_operator(np.asarray(cov_op, dtype=float))


def _estimate_cov(
    samples: Samples,
    mean: Spectrum,
    method: str,
    *,
    rank: int = LOWRANK_RANK,
    tikhonov: float = LOWRANK_TIKHONOV,
) -> CovLike:
    if method == "empirical":
        return estimate_cov_empirical(samples, mean=mean, ridge=1e-8)
    if method == "lowrank":
        return estimate_cov_lowrank(samples, mean=mean, rank=rank, tikhonov=tikhonov)
    if method in ("ledoit_wolf", "oas"):
        return estimate_cov_shrunk(samples, mean=mean, method=method)  # type: ignore[arg-type]
    raise ValueError(f"MatchedFilter: unknown cov_method {method!r}")


def _array_config(value: Float[np.ndarray, "*shape"] | None) -> Any:
    return None if value is None else jsonable(np.asarray(value))


def _cov_config(cov_op: CovLike | None) -> Any:
    """A covariance for ``get_config``: its dense matrix as nested lists."""
    if cov_op is None:
        return None
    if isinstance(cov_op, lx.AbstractLinearOperator):
        return jsonable(np.asarray(cov_op.as_matrix()))
    return jsonable(np.asarray(cov_op))


def _target_config(op: LinearTargetFromObs | NonlinearTargetFromObs) -> dict[str, Any]:
    return {
        "vmr_background": _array_config(op.vmr_background),
        "pattern": op.pattern
        if isinstance(op.pattern, str)
        else _array_config(op.pattern),
        "obs_model": callable_name(op.obs_model),
        "pixel": op.pixel,
        "axis": op.axis,
    }


def _check_band_axis(gt: Any, axis: int, name: str) -> None:
    """A clear error when ``axis`` is not an axis of ``gt`` (e.g. a 2-D map)."""
    ndim = np.ndim(gt)
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"{name}: spectral axis {axis} does not exist on a {ndim}-D input "
            f"of shape {np.shape(gt)}; pass a (C, H, W) cube or a (T, C, H, W) "
            "stack (or set `axis` for a sample matrix)"
        )


def _masked(gt: Carrier, axis: int, name: str) -> tuple[Cube, ValidMap | None]:
    """Float cube with nodata pixels NaN in every band, and the validity map.

    ``valid`` is ``None`` when every pixel is valid; otherwise a boolean map
    over the non-band axes.
    """
    _check_band_axis(gt, axis, name)
    cube = np.asarray(gt, dtype=float)
    invalid = np.asarray(np.asarray(invalid_values(gt)).any(axis=axis), dtype=bool)
    if not invalid.any():
        return cube, None
    cube = cube.copy()
    cube[np.broadcast_to(np.expand_dims(invalid, axis), cube.shape)] = np.nan
    return cube, ~invalid


def _samples(
    gt: Carrier, axis: int, name: str
) -> tuple[Samples, tuple[int, ...], ValidMap | None]:
    """``(N, 1, B)`` pixel cube for the core, the non-band shape, the validity map.

    Every pixel is a row, nodata ones as ``NaN``: the core leaves them out of
    statistics and scores them ``NaN``.
    """
    cube, valid = _masked(gt, axis, name)
    samples, layout = cube_to_samples(cube, band_axis=axis)
    return samples[:, None, :], tuple(layout.sample_shape), valid


def _band_last(gt: Carrier, axis: int, name: str) -> Float[np.ndarray, "H W B"]:
    """``(H, W, B)`` float cube with nodata pixels NaN."""
    cube, _ = _masked(gt, axis, name)
    return np.moveaxis(cube, axis, -1)


def _score_map(gt: Carrier, scores: ScoreArray, valid: ValidMap | None) -> ScoreMap:
    """A score map on ``gt``'s carrier, ``NaN`` at nodata pixels."""
    out = scores if valid is None else restore_fill(scores, valid, np.nan)
    return wrap_like(gt, keep_band_axis(out, gt), fill_value_default=np.nan)


def _spatial_shape(gt: Carrier, axis: int) -> tuple[int, ...]:
    """The shape of ``gt`` without its band axis."""
    shape = list(np.shape(gt))
    del shape[axis % len(shape)]
    return tuple(shape)


def _vmr_background(
    obs: RadianceObservationModel,
    gt: Carrier,
    vmr_background: VMRField | Cube | None,
    axis: int,
) -> jax.Array:
    """The VMR field the target is taken about: given, or the reference on ``gt``'s grid."""
    if vmr_background is not None:
        return jnp.asarray(vmr_background, dtype=float)
    return jnp.full(_spatial_shape(gt, axis)[-2:], obs.vmr_reference, dtype=float)


def _jax_pattern(pattern: str | VMRField) -> str | Float[jax.Array, "H W"]:
    return pattern if isinstance(pattern, str) else jnp.asarray(pattern, dtype=float)


def _callable_state(
    obs_model: Any,
    gt: Carrier,
    vmr_background: VMRField | Cube | None,
    pattern: str | VMRField | Cube,
    pixel: tuple[int, ...] | None,
    axis: int,
) -> tuple[VMRField | Cube, VMRField | Cube, ValidMap | None]:
    """``(base, direction, valid)`` for a callable.

    The state, its perturbation, and the validity map of ``gt``'s pixels
    (``None`` when all are valid, or for an explicit ``vmr_background``).
    The cube as a state has its nodata pixels set to the band means of the
    valid ones, so a callable that aggregates over pixels sees no ``NaN``.
    """
    if not callable(obs_model):
        raise TypeError("obs_model must be an observation model or a callable")
    if vmr_background is None:
        base, valid = _masked(gt, axis, "target")
        if valid is not None:
            band_means = np.nanmean(np.moveaxis(base, axis, -1)[valid], axis=0)
            base = np.where(np.isnan(base), _along(band_means, axis, base.ndim), base)
        band_axis: int | None = axis
    else:
        base, valid = np.asarray(vmr_background, dtype=float), None
        # A cube-shaped background keeps its band axis; an (H, W) one has none.
        band_axis = axis if base.ndim == np.ndim(gt) else None
    direction = _target_pattern(base, pattern=pattern, pixel=pixel, axis=band_axis)
    return base, direction, valid


def _along(vector: Spectrum, axis: int, ndim: int) -> Float[np.ndarray, "*shape"]:
    """``vector`` shaped to broadcast along ``axis`` of an ``ndim``-D array."""
    shape = [1] * ndim
    shape[axis % ndim] = vector.shape[0]
    return vector.reshape(shape)


def _response_vector(
    response: Float[np.ndarray, "*shape"],
    gt: Carrier,
    axis: int,
    valid: ValidMap | None = None,
) -> Spectrum:
    """A callable's response as the ``(B,)`` target, checked against ``gt``'s bands.

    A cube response is averaged over its finite pixels that are also valid
    in ``gt`` (when it is on ``gt``'s grid).
    """
    if response.ndim > 1:
        samples, layout = cube_to_samples(response, band_axis=axis)
        keep = np.isfinite(samples).all(axis=1)
        if valid is not None and tuple(layout.sample_shape) == valid.shape:
            keep &= valid.reshape(-1)
        response = np.mean(samples if keep.all() else samples[keep], axis=0)
    vector = response.reshape(-1)
    n_bands = np.shape(gt)[axis]
    if vector.shape[0] != n_bands:
        raise ValueError(
            f"obs_model gave {vector.shape[0]} values per pixel, the cube has "
            f"{n_bands} bands; a callable's response must keep the band axis at "
            f"axis={axis} (or be a ({n_bands},) vector)"
        )
    return vector


def _target_pattern(
    base: VMRField | Cube,
    *,
    pattern: str | VMRField | Cube,
    pixel: tuple[int, ...] | None,
    axis: int | None,
) -> VMRField | Cube:
    """The perturbation direction, the shape of ``base``.

    ``"impulse"`` perturbs one pixel (every band when ``base`` has a band
    axis): ``pixel`` indexes the non-band axes, by default their centre, as
    the core's targets do.
    """
    if not isinstance(pattern, str):
        return np.asarray(pattern, dtype=float)
    out = np.zeros(np.shape(base), dtype=float)
    if pattern == "uniform":
        out[...] = 1.0
    elif pattern == "impulse":
        band = None if axis is None else axis % out.ndim
        spatial = [n for k, n in enumerate(out.shape) if k != band]
        index: list[Any] = list((n // 2 for n in spatial) if pixel is None else pixel)
        if len(index) != len(spatial):
            raise ValueError(
                f"pixel {pixel} must index the {len(spatial)} non-band axes"
            )
        if band is not None:
            index.insert(band, slice(None))
        out[tuple(index)] = 1.0
    else:
        raise ValueError(f"unknown target pattern {pattern!r}")
    return out
