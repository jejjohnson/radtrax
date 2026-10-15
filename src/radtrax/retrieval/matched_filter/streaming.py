"""Streaming background estimation via Welford's algorithm.

For multi-scene MF workflows (e.g. aggregating background statistics across a
campaign of flights), the full ``(n_samples, n_bands)`` stack does not fit in
memory. The parallel Welford update [#welford1962]_ [#chan1979]_ maintains a
mean and (scaled) covariance in one pass, merges partial results from
separate chunks exactly, and is numerically stable:

.. math::
    n \\leftarrow n_A + n_B
    \\delta = \\mu_B - \\mu_A
    \\mu \\leftarrow \\mu_A + \\delta \\cdot n_B / n
    M_2 \\leftarrow M_{2,A} + M_{2,B} + \\delta \\delta^\\top \\cdot n_A n_B / n

Covariance is then ``M_2 / (n - ddof)``; the accumulator also merges the
third and fourth moments exactly, so a stream can be shrunk (Ledoit–Wolf,
OAS) without a second pass.

.. [#welford1962] Welford (1962), Note on a method for calculating corrected
   sums of squares and products.
.. [#chan1979] Chan, Golub & LeVeque (1979), Updating formulae and a pairwise
   algorithm for computing sample variances.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/matched_filter/streaming.py``); the
third and fourth moments, non-finite skipping and streaming shrinkage come from
geotoolz (radtrax#26).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import lineax as lx
import numpy as np
from jaxtyping import ArrayLike, Float

from radtrax.exceptions import DegenerateInputError, ParameterError, ShapeError
from radtrax.retrieval.matched_filter.background import (
    _dense_operator,
    _finite_pixels,
    shrink_covariance,
)


__all__ = [
    "WelfordAccumulator",
    "streaming_background",
]


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


@dataclass
class WelfordAccumulator:
    """Online mean, covariance and fourth moment for ``(n_samples, n_bands)`` streams.

    Use :meth:`update` to feed one batch at a time; :meth:`merge` to combine
    two accumulators computed on disjoint subsets; :meth:`mean`,
    :meth:`covariance` and :meth:`fourth_moment` to finalise. Rows with a
    non-finite band are skipped, as in the batch estimators.

    Besides ``M₂ = Σ dₖdₖᵀ`` (``dₖ = xₖ − x̄``) it keeps the third-moment
    vector ``M₃ = Σ ‖dₖ‖² dₖ`` and the scalar ``M₄ = Σ ‖dₖ‖⁴``, so a stream
    can be shrunk with Ledoit–Wolf (:func:`shrink_covariance` needs
    ``M₄ / n``). Merging parts ``X`` with offsets ``δ_X = x̄_X − x̄`` from
    the pooled mean is exact::

        M₂ = Σ_X [M₂_X + n_X δ_X δ_Xᵀ]
        M₃ = Σ_X [M₃_X + tr(M₂_X) δ_X + 2 M₂_X δ_X + n_X ‖δ_X‖² δ_X]
        M₄ = Σ_X [M₄_X + 4 δ_Xᵀ M₃_X + 4 δ_Xᵀ M₂_X δ_X
                  + 2 ‖δ_X‖² tr(M₂_X) + n_X ‖δ_X‖⁴]

    (expand ``‖d + δ‖²`` with ``d = x − x̄_X``, ``Σ d = 0``). Ported from
    geotoolz's accumulator (radtrax#26).

    Attributes
    ----------
    n_bands
        Number of spectral bands.
    count
        Running sample count.
    """

    n_bands: int
    count: int = 0
    _mean: np.ndarray = field(init=False)
    _M2: np.ndarray = field(init=False)
    _M3: np.ndarray = field(init=False)
    _M4: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.n_bands <= 0:
            raise ParameterError("WelfordAccumulator: n_bands must be > 0.")
        self._mean = np.zeros(self.n_bands, dtype=float)
        self._M2 = np.zeros((self.n_bands, self.n_bands), dtype=float)
        self._M3 = np.zeros(self.n_bands, dtype=float)
        self._M4 = 0.0

    def update(self, batch: Float[ArrayLike, "*batch B"]) -> WelfordAccumulator:
        """Ingest a batch of pixel spectra (shape ``(n_samples, n_bands)`` or ``(H, W, n_bands)``)."""
        arr = np.asarray(batch, dtype=float)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[1] != self.n_bands:
            raise ShapeError(
                f"WelfordAccumulator.update: expected shape (*, {self.n_bands}); "
                f"got {arr.shape}."
            )
        finite = _finite_pixels(arr)
        if not finite.all():
            arr = arr[finite]
        if arr.shape[0] == 0:
            return self
        part = WelfordAccumulator(n_bands=self.n_bands)
        part.count = arr.shape[0]
        part._mean = arr.mean(axis=0)
        d = arr - part._mean
        sq = np.sum(d * d, axis=1)
        part._M2 = d.T @ d
        part._M3 = sq @ d
        part._M4 = float(sq @ sq)
        return self.merge(part)

    def merge(self, other: WelfordAccumulator) -> WelfordAccumulator:
        """Combine with another accumulator on disjoint data (exact)."""
        if other.n_bands != self.n_bands:
            raise ShapeError(
                f"WelfordAccumulator.merge: n_bands mismatch "
                f"({self.n_bands} vs {other.n_bands})."
            )
        if other.count == 0:
            return self
        if self.count == 0:
            self._mean = other._mean.copy()
            self._M2 = other._M2.copy()
            self._M3 = other._M3.copy()
            self._M4 = other._M4
            self.count = other.count
            return self
        n = self.count + other.count
        mean = self._mean + (other._mean - self._mean) * (other.count / n)
        M2 = np.zeros_like(self._M2)
        M3 = np.zeros_like(self._M3)
        M4 = 0.0
        for part in (self, other):
            delta = part._mean - mean
            delta_sq = float(delta @ delta)
            M2_delta = part._M2 @ delta
            tr_M2 = float(np.trace(part._M2))
            M2 += part._M2 + part.count * np.outer(delta, delta)
            M3 += (
                part._M3
                + tr_M2 * delta
                + 2.0 * M2_delta
                + part.count * delta_sq * delta
            )
            M4 += (
                part._M4
                + 4.0 * float(part._M3 @ delta)
                + 4.0 * float(delta @ M2_delta)
                + 2.0 * delta_sq * tr_M2
                + part.count * delta_sq**2
            )
        self._mean, self._M2, self._M3, self._M4 = mean, M2, M3, M4
        self.count = n
        return self

    def mean(self) -> Float[np.ndarray, " B"]:
        """Mean spectrum of the samples seen so far."""
        if self.count == 0:
            raise DegenerateInputError("WelfordAccumulator.mean: no data yet.")
        return self._mean.copy()

    def covariance(self, ddof: int = 1) -> Float[np.ndarray, "B B"]:
        """Covariance ``M₂ / (count − ddof)``."""
        if self.count <= ddof:
            raise DegenerateInputError(
                f"WelfordAccumulator.covariance: need count > ddof; got count={self.count}."
            )
        return self._M2 / (self.count - ddof)

    def fourth_moment(self) -> float:
        """``m₄ = M₄ / count = (1/n) Σ ‖xₖ − x̄‖⁴``, the Ledoit–Wolf input."""
        if self.count == 0:
            raise DegenerateInputError("WelfordAccumulator.fourth_moment: no data yet.")
        return self._M4 / self.count


def streaming_background(
    batches: Iterable[Float[ArrayLike, "*_batch B"]],
    *,
    n_bands: int,
    ridge: float = 0.0,
    ddof: int = 0,
    cov_kind: Literal["empirical", "ledoit_wolf", "oas"] = "empirical",
) -> tuple[Float[np.ndarray, " B"], LinearOperator]:
    """Streamed ``(μ, Σ)`` from an iterable of batches.

    Convenience wrapper for the common pattern of looping over scenes/tiles.
    Returns the final mean and a dense lineax operator for the covariance —
    pass the covariance through :func:`~radtrax.retrieval.matched_filter.core.apply_image`
    exactly like any other ``cov_op``.

    Parameters
    ----------
    batches
        Iterable yielding arrays of shape ``(n_samples, n_bands)`` or
        ``(H, W, n_bands)``.
    n_bands
        Number of spectral bands (declared up front since the iterable can be
        lazy).
    ridge
        Optional diagonal ridge added to the final covariance.
    ddof
        Delta degrees of freedom for the covariance denominator:
        ``M_2 / (n - ddof)``. Defaults to ``0`` (MLE, matches sklearn's
        ``EmpiricalCovariance``, ``LedoitWolf``, ``OAS`` and our
        :func:`estimate_cov_lowrank`), so
        :func:`~radtrax.retrieval.matched_filter.core.matched_filter_snr` and
        :func:`~radtrax.retrieval.matched_filter.core.detection_threshold`
        give the same scaling whether the covariance came from a streaming
        pass or a batch estimator. Set ``ddof=1`` to match ``np.cov`` /
        sample-covariance conventions instead.
    cov_kind
        ``"empirical"`` (default), or ``"ledoit_wolf"`` / ``"oas"`` to shrink
        the streamed covariance toward a scaled identity with
        :func:`~radtrax.retrieval.matched_filter.background.shrink_covariance`.
        With ``ddof=0`` the shrunk matrix equals scikit-learn's estimator on
        the concatenated batches; a non-zero ``ddof`` rescales it by
        ``n / (n − ddof)`` (the intensity is scale-free).
    """
    acc = WelfordAccumulator(n_bands=n_bands)
    for batch in batches:
        acc.update(batch)
    mu = acc.mean()
    cov = acc.covariance(ddof=ddof)
    if cov_kind in ("ledoit_wolf", "oas"):
        shrunk = shrink_covariance(
            acc.covariance(ddof=0),
            method=cov_kind,
            n_samples=acc.count,
            fourth_moment=acc.fourth_moment(),
        )
        cov = shrunk * (acc.count / (acc.count - ddof))
    elif cov_kind != "empirical":
        raise ParameterError(
            "streaming_background: cov_kind must be 'empirical', 'ledoit_wolf' "
            f"or 'oas'; got {cov_kind!r}."
        )
    if ridge > 0.0:
        cov = cov + ridge * np.eye(n_bands)
    return mu, _dense_operator(cov)
