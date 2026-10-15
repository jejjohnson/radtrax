"""Background mean and covariance estimators for the matched filter.

Estimation is not on the gradient path — the matched filter treats the
background :math:`(\\mu, \\Sigma)` as known inputs. That lets us lean on
scikit-learn's mature estimators (which have careful numerics, shrinkage,
robust variants, etc.) without any JAX/autodiff constraints. The outputs
are wrapped as :mod:`lineax` / :mod:`gaussx` linear operators so that
:func:`radtrax.retrieval.matched_filter.core.apply_image` can use the same
``gaussx.solve`` dispatch path for all covariance flavours:

============================== ====================================================
Estimator                      Operator returned
============================== ====================================================
``estimate_cov_empirical``     :class:`lineax.MatrixLinearOperator` (dense)
``estimate_cov_shrunk``        :class:`lineax.MatrixLinearOperator` (dense, shrunk)
``estimate_cov_lowrank``       :class:`gaussx.LowRankUpdate` (``λI + UDUᵀ``)
============================== ====================================================

The low-rank path uses scikit-learn's randomised :class:`TruncatedSVD`, which
is ``O(n · k)`` on a ``(n_samples, n_bands)`` matrix and is typically 10–100×
faster than a full SVD for the small ``k`` a matched filter needs.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/matched_filter/background.py``); only import paths changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


_RobustMethod = Literal["mean", "median", "trimmed", "huber"]


# ── mean estimators ──────────────────────────────────────────────────────────


def estimate_mean(
    cube: Float[Array, "H W B"] | np.ndarray,
    *,
    method: _RobustMethod = "mean",
    trim_proportion: float = 0.1,
    huber_c: float = 1.345,
) -> np.ndarray:
    """Estimate the per-band background mean ``μ`` from a scene.

    Parameters
    ----------
    cube
        Hyperspectral scene of shape ``(H, W, n_bands)``.
    method
        - ``'mean'``    — arithmetic mean (sensitive to anomalies).
        - ``'median'``  — per-band median (very robust).
        - ``'trimmed'`` — two-sided trimmed mean (``scipy.stats.trim_mean``).
        - ``'huber'``   — Huber M-estimator (sklearn's ``HuberRegressor`` on
          a constant design).
    trim_proportion
        Fraction discarded from each tail for the trimmed mean. Ignored
        otherwise.
    huber_c
        Huber influence threshold in units of the per-band MAD. Ignored
        unless ``method='huber'``.

    Pixels with a non-finite band (NaN fill, saturated or missing data) are
    left out, as in every estimator of this module.

    Returns
    -------
    np.ndarray
        Mean spectrum of length ``n_bands``.
    """
    X = _flatten_cube_masked(cube, None, min_samples=1)
    if method == "mean":
        return X.mean(axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "trimmed":
        from scipy.stats import trim_mean

        return trim_mean(X, proportiontocut=trim_proportion, axis=0)
    if method == "huber":
        return _huber_mean(X, c=huber_c)
    raise ValueError(f"estimate_mean: unknown method {method!r}.")


def _huber_mean(
    X: np.ndarray, *, c: float, n_iter: int = 20, tol: float = 1e-8
) -> np.ndarray:
    """Per-band Huber M-estimator via iteratively-reweighted least squares."""
    mu = np.median(X, axis=0)
    mad = np.median(np.abs(X - mu), axis=0) + 1e-12
    scale = 1.4826 * mad
    for _ in range(n_iter):
        z = (X - mu) / scale
        w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z).clip(min=1e-12))
        mu_new = (w * X).sum(axis=0) / w.sum(axis=0).clip(min=1e-12)
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    return mu


# ── dense covariance estimators ──────────────────────────────────────────────


def _dense_operator(cov: np.ndarray) -> LinearOperator:
    """Wrap a dense PSD matrix as a lineax operator with the right tags."""
    return lx.MatrixLinearOperator(
        jnp.asarray(cov),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def estimate_cov_empirical(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    ridge: float = 0.0,
    mask: np.ndarray | None = None,
) -> LinearOperator:
    """Empirical covariance via :class:`sklearn.covariance.EmpiricalCovariance`.

    Returns a dense :class:`lineax.MatrixLinearOperator`. Optionally adds a
    diagonal ``ridge`` for numerical PD-ness when ``n_samples ≲ n_bands``.
    ``mask`` (boolean exclude-pixels) drops flagged pixels — e.g. a detected
    plume — so ``Σ`` is estimated from background only, consistent with a robust
    masked ``μ``.
    """
    from sklearn.covariance import EmpiricalCovariance

    X = _flatten_cube_masked(cube, mask)
    est = EmpiricalCovariance(store_precision=False).fit(X)
    cov = est.covariance_
    if mean is not None:
        # sklearn uses the empirical mean internally; re-centre if the caller
        # supplied a robust mean so (μ, Σ) are consistent.
        Xc = X - mean
        cov = (Xc.T @ Xc) / X.shape[0]
    if ridge > 0.0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return _dense_operator(cov)


def estimate_cov_shrunk(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    method: Literal["ledoit_wolf", "oas"] = "ledoit_wolf",
    mask: np.ndarray | None = None,
) -> LinearOperator:
    """Shrinkage covariance estimator.

    Uses Ledoit–Wolf (``method='ledoit_wolf'``, default) or OAS
    (``'oas'``) — both shrink the sample covariance toward a scaled identity
    and are PD by construction even when ``n_samples < n_bands``. ``mask``
    (boolean exclude-pixels) drops flagged pixels before fitting.
    """
    X = _flatten_cube_masked(cube, mask)
    if mean is not None:
        X = X - mean
        assume_centered = True
    else:
        assume_centered = False
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        est = LedoitWolf(assume_centered=assume_centered).fit(X)
    elif method == "oas":
        from sklearn.covariance import OAS

        est = OAS(assume_centered=assume_centered).fit(X)
    else:
        raise ValueError(
            f"estimate_cov_shrunk: method must be 'ledoit_wolf' or 'oas'; got {method!r}."
        )
    return _dense_operator(est.covariance_)


def shrink_covariance(
    empirical: Float[Array, "B B"] | np.ndarray,
    *,
    method: Literal["ledoit_wolf", "oas"],
    n_samples: int,
    fourth_moment: float | None = None,
) -> np.ndarray:
    """Shrink a ``1/n`` empirical covariance toward a scaled identity.

    Returns ``(1 − s)·Sₙ + s·μ·I`` with ``μ = tr(Sₙ)/p``, where ``Sₙ`` is the
    maximum-likelihood (``1/n``) scatter of ``n`` samples in ``p`` bands — the
    normalisation every estimator in this module uses — and the intensity
    ``s ∈ [0, 1]`` is estimated from the data. Both methods reproduce
    scikit-learn's :class:`~sklearn.covariance.LedoitWolf` and
    :class:`~sklearn.covariance.OAS` exactly. This is the form a streaming
    pass needs (:func:`~radtrax.retrieval.matched_filter.streaming.streaming_background`),
    where only the moments, not the samples, are kept.

    - ``"ledoit_wolf"`` (Ledoit & Wolf 2004), with ``dₖ = xₖ − x̄`` and
      ``m₄ = (1/n) Σₖ ‖dₖ‖⁴``::

          δ² = ‖Sₙ − μ I‖²_F
          β̄² = (1/n²) Σₖ ‖dₖdₖᵀ − Sₙ‖²_F = (m₄ − ‖Sₙ‖²_F) / n
          s  = min(β̄², δ²) / δ²                    (0 when δ² = 0)

      β̄² needs the fourth moment ``m₄``, which ``Sₙ`` alone does not carry.

    - ``"oas"`` (Chen et al. 2010, as scikit-learn implements it, without
      the ``2/p`` terms of their Eq. 23)::

          s = min(1, (tr(Sₙ²) + tr(Sₙ)²) / ((n + 1)·(tr(Sₙ²) − tr(Sₙ)²/p)))

    Parameters
    ----------
    empirical
        ``1/n`` empirical covariance ``Σₖ dₖdₖᵀ / n``, shape ``(B, B)``.
    method
        Shrinkage-intensity estimator.
    n_samples
        Number of samples behind ``empirical``.
    fourth_moment
        ``m₄`` over the same centred samples; required for
        ``"ledoit_wolf"``, ignored by ``"oas"``.
        :meth:`~radtrax.retrieval.matched_filter.streaming.WelfordAccumulator.fourth_moment`
        tracks it.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(30, 4))
    >>> d = x - x.mean(axis=0)
    >>> s_n = d.T @ d / 30
    >>> m4 = float(np.mean(np.sum(d * d, axis=1) ** 2))
    >>> shrink_covariance(
    ...     s_n, method="ledoit_wolf", n_samples=30, fourth_moment=m4
    ... ).shape
    (4, 4)
    """
    cov = np.asarray(empirical, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"shrink_covariance: empirical must be square, got shape {cov.shape}."
        )
    if n_samples < 1:
        raise ValueError(f"shrink_covariance: n_samples must be ≥ 1, got {n_samples}.")
    p = cov.shape[0]
    mu = float(np.trace(cov)) / p
    fro_sq = float(np.sum(cov * cov))
    if method == "oas":
        trace_sq = float(np.trace(cov)) ** 2
        denom = (n_samples + 1) * (fro_sq - trace_sq / p)
        shrinkage = 1.0 if denom <= 0 else min(1.0, (fro_sq + trace_sq) / denom)
    elif method == "ledoit_wolf":
        if fourth_moment is None:
            raise ValueError(
                "shrink_covariance: 'ledoit_wolf' needs fourth_moment = "
                "mean(‖xₖ − x̄‖⁴) over the samples (WelfordAccumulator.fourth_moment())."
            )
        diff = cov - mu * np.eye(p)
        d_sq = float(np.sum(diff * diff))
        b_bar_sq = max((float(fourth_moment) - fro_sq) / n_samples, 0.0)
        b_sq = min(b_bar_sq, d_sq)
        shrinkage = 0.0 if b_sq == 0 else b_sq / d_sq
    else:
        raise ValueError(
            f"shrink_covariance: method must be 'ledoit_wolf' or 'oas'; got {method!r}."
        )
    return (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(p)


# ── low-rank covariance (Woodbury-friendly) ──────────────────────────────────


def estimate_cov_lowrank(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    rank: int,
    tikhonov: float,
    random_state: int | jax.Array | None = 0,
    n_oversamples: int = 10,
    mask: np.ndarray | None = None,
    rank_rtol: float = 1e-10,
) -> LinearOperator:
    """Low-rank + Tikhonov covariance: ``Σ = λI + V D Vᵀ``.

    Uses :func:`gaussx.randomized_svd` (Halko–Martinsson–Tropp, JAX, 5 power
    iterations) to find the top ``rank`` spectral directions of
    the sample covariance. Returned as a :class:`gaussx.LowRankUpdate` so that
    :func:`gaussx.solve` routes through the Woodbury identity — the MF
    precompute cost is ``O(n_bands · rank + rank³)`` instead of
    ``O(n_bands³)``.

    Parameters
    ----------
    cube
        Hyperspectral scene ``(H, W, n_bands)``.
    mean
        Background mean. If ``None``, subtracts the sample mean.
    rank
        Number of leading components to keep. Clamped to
        ``min(rank, n_samples - 1, n_bands - 1)``.
    tikhonov
        Diagonal floor ``λ > 0`` — ensures strict PD.
    random_state
        Seed for the randomised SVD: an ``int`` (converted with
        ``jax.random.key``), a JAX PRNG key, or ``None`` (seed 0).
    n_oversamples
        Extra random directions for the Halko sampling — higher is slower but
        better-conditioned.
    mask
        Boolean exclude-pixels mask; flagged pixels (e.g. a detected plume) are
        dropped before the SVD so ``Σ`` reflects background only.
    rank_rtol
        Components whose covariance eigenvalue ``d_i`` is below
        ``rank_rtol · max(d)`` are dropped before building the operator. On a
        rank-deficient scene (constant regions, duplicated spectra) those
        ``d_i ≈ 0`` would make gaussx's Woodbury capacitance ``diag(1/d) +
        VᵀL⁻¹U`` blow up to inf/NaN; dropping them keeps every score finite.
    """
    if tikhonov <= 0.0:
        raise ValueError("estimate_cov_lowrank: tikhonov must be > 0.")
    X = _flatten_cube_masked(cube, mask)
    mu = X.mean(axis=0) if mean is None else np.asarray(mean)
    Xc = X - mu
    n_samples, n_bands = Xc.shape
    rank = max(1, min(int(rank), n_samples - 1, n_bands - 1))
    key = _as_key(random_state)
    # Same MLE normalization s_i² / n_samples as the empirical / LedoitWolf /
    # OAS paths in this module (using n_samples - 1 here would make
    # matched_filter_snr and detection_threshold depend on the estimator).
    V, d = lowrank_factors(
        jnp.asarray(Xc), rank, n_oversamples=n_oversamples, n_power_iter=5, key=key
    )
    V, d = np.asarray(V), np.asarray(d)
    diag = jnp.asarray(tikhonov * np.ones(n_bands, dtype=float))
    # Rank-deficiency guard: drop near-zero eigen-directions so the Woodbury
    # capacitance stays finite. If none survive (e.g. a constant scene), Σ is
    # just the strictly-PD Tikhonov floor λI.
    d_max = float(d.max()) if d.size else 0.0
    keep = d > rank_rtol * d_max if d_max > 0.0 else np.zeros(d.shape, dtype=bool)
    V, d = V[:, keep], d[keep]
    if d.size == 0:
        return lx.DiagonalLinearOperator(diag)
    # λ I + U diag(d) Uᵀ via the gaussx SVD constructor (auto-inferred
    # symmetric/PSD tags, orthonormal-factor solve path).
    U = jnp.asarray(V)  # (n_bands, n_kept) — spectral directions as columns
    return gx.svd_low_rank_plus_diag(diag, U, jnp.asarray(d), U)


# ── helpers ──────────────────────────────────────────────────────────────────


def _as_key(random_state: int | jax.Array | None) -> jax.Array:
    if random_state is None:
        random_state = 0
    if isinstance(random_state, (int, np.integer)):
        return jax.random.key(int(random_state))
    return random_state


def _lowrank_factors(
    Xc: Float[Array, "N B"],
    rank: int,
    *,
    n_oversamples: int = 10,
    n_power_iter: int = 5,
    key: jax.Array,
) -> tuple[Float[Array, "B r"], Float[Array, " r"]]:
    """Jittable randomised-SVD core: ``(V, d)`` of centred samples ``Xc``.

    ``V`` is ``(n_bands, rank)`` (right singular vectors as columns) and
    ``d = s² / n_samples`` the MLE covariance eigenvalues.
    """
    _, sv, Vt = gx.randomized_svd(
        lx.MatrixLinearOperator(Xc),
        rank,
        oversample=n_oversamples,
        n_power_iter=n_power_iter,
        key=key,
    )
    return Vt.T, sv**2 / Xc.shape[0]


lowrank_factors = jax.jit(
    _lowrank_factors, static_argnames=("rank", "n_oversamples", "n_power_iter")
)


def _flatten_cube(cube: Float[Array, "H W B"] | np.ndarray) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(
            f"background estimator: cube must be (H, W, n_bands), got shape {arr.shape}."
        )
    return arr.reshape(-1, arr.shape[-1])


def _finite_pixels(X: np.ndarray) -> np.ndarray:
    """Boolean ``(...)`` mask of pixels whose bands (last axis) are all finite.

    The one validity rule of the matched filter: a pixel with any non-finite
    band (NaN fill, missing or saturated data) is left out of every statistic.
    """
    return np.asarray(np.isfinite(X).all(axis=-1), dtype=bool)


def _flatten_cube_masked(
    cube: Float[Array, "H W B"] | np.ndarray,
    mask: np.ndarray | None,
    *,
    min_samples: int = 2,
) -> np.ndarray:
    """Flatten a cube to ``(n_kept, n_bands)``, dropping excluded pixels.

    A pixel is excluded when any of its bands is non-finite (a NaN fill,
    missing data) or when ``mask`` flags it. ``mask`` is a boolean **exclude**
    mask — ``True`` marks pixels (e.g. detected plume / target) to leave out of
    the background statistics so the filter is not attenuated by its own
    signal. Any shape with one entry per pixel is accepted (``(H, W)`` or
    flat).
    """
    X = _flatten_cube(cube)
    keep = _finite_pixels(X)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape[0] != X.shape[0]:
            raise ValueError(
                f"background estimator: `mask` must have one entry per pixel "
                f"({X.shape[0]}), got {m.size}."
            )
        keep &= ~m
    kept = X if keep.all() else X[keep]
    if kept.shape[0] < min_samples:
        raise ValueError(
            f"background estimator: {kept.shape[0]} of {X.shape[0]} pixel(s) are "
            f"finite and not masked; need ≥ {min_samples}."
        )
    return kept
