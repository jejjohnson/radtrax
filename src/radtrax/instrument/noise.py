"""The ``NoiseModel`` protocol: the measurement-noise covariance ``S_ε``.

Optimal estimation needs ``S_ε`` as a structured operator, and synthetic
scenes need draws from it. A noise model returns ``S_ε(y)`` for the band
values ``y`` it is evaluated at (signal-dependent noise needs ``y``), as a
lineax operator tagged positive semidefinite (so ``gx.solve``, ``gx.logdet``
and ``lx.CG`` all take it), draws ``ε ~ 𝒩(0, S_ε(y))`` from a
``jax.random`` key the caller passes, and scores an observation with
``log 𝒩(y_obs | y, S_ε(y))``.

Two diagonal models:

- ``DiagonalNoise``: a fixed per-band standard deviation, ``S_ε = diag(σ²)``.
- ``SNRNoise``: a signal-to-noise ratio with a noise floor,
  ``σ_b = √((y_b / SNR_b)² + σ_floor,b²)``.

Shapes: ``B`` bands.
"""

from __future__ import annotations

import abc

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, Real

from radtrax.exceptions import ParameterError, ShapeError


__all__ = ["DiagonalNoise", "NoiseModel", "SNRNoise"]


class NoiseModel(eqx.Module):
    """Protocol: the measurement-noise covariance ``S_ε(y)`` and draws from it.

    An abstract Equinox module, so ``filter_jit`` compiles each concrete
    model and a retrieval can take any of them. A subclass implements
    ``covariance``; ``sample`` draws through ``gx.sample_mvn``, which
    dispatches on the operator's structure (``√σ² · z`` for a diagonal).
    """

    @abc.abstractmethod
    def covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator:
        """``S_ε`` at the band values ``y``, a positive-definite ``(B, B)`` operator.

        Args:
            y: Band values the noise is evaluated at, ``(B,)``.
        """

    def sample(
        self, key: PRNGKeyArray, y: Float[ArrayLike, " B"]
    ) -> Float[Array, " B"]:
        """One draw ``ε ~ 𝒩(0, S_ε(y))``; add it to ``y`` for a noisy observation.

        Args:
            key: A ``jax.random`` key; the same key gives the same draw.
            y: Band values the noise is evaluated at, ``(B,)``.

        Returns:
            The noise ``ε``, ``(B,)``.
        """
        y = jnp.asarray(y, dtype=float)
        return gx.sample_mvn(jnp.zeros_like(y), self.covariance(y), key=key)[0]

    def log_prob(
        self, y_obs: Float[ArrayLike, " B"], y_model: Float[ArrayLike, " B"]
    ) -> Float[Array, ""]:
        """``log 𝒩(y_obs | y_model, S_ε(y_model))``, through ``gx.gaussian_log_prob``.

        The noise is evaluated at the modelled values, so for signal-dependent
        noise the ``log |S_ε|`` term depends on them too and is kept: dropping
        it would favour brighter solutions.

        Args:
            y_obs: Observed band values, ``(B,)``.
            y_model: Modelled band values, ``(B,)``.

        Returns:
            The log-density, a scalar.
        """
        y_model = jnp.asarray(y_model, dtype=float)
        return gx.gaussian_log_prob(
            y_model, self.covariance(y_model), jnp.asarray(y_obs, dtype=float)
        )


class DiagonalNoise(NoiseModel):
    """Independent noise with a fixed standard deviation per band, ``S_ε = diag(σ²)``.

    Attributes:
        sigma: Standard deviation per band, in the units of ``y``, ``(B,)``;
            each strictly positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> noise = DiagonalNoise(jnp.array([0.1, 0.2]))
        >>> import gaussx as gx
        >>> [round(v, 6) for v in gx.diag(noise.covariance(jnp.ones(2))).tolist()]
        [0.01, 0.04]
    """

    sigma: Float[Array, " B"]

    def __init__(self, sigma: Real[ArrayLike, " B"]) -> None:
        self.sigma = jnp.asarray(sigma, dtype=float)
        _check_bands("DiagonalNoise", sigma=self.sigma)
        _check_positive("DiagonalNoise", sigma=self.sigma)

    def covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator:
        """``diag(σ²)``, whatever ``y``; ``y`` sets only the band count."""
        _check_y("DiagonalNoise", y, self.sigma)
        return _diagonal_covariance(self.sigma**2)


class SNRNoise(NoiseModel):
    """Signal-dependent noise from a signal-to-noise ratio and a floor.

    ``σ_b = √((y_b / SNR_b)² + σ_floor,b²)``, ``S_ε = diag(σ²)``: the floor
    keeps ``S_ε`` positive definite where the signal vanishes.

    Attributes:
        snr: Signal-to-noise ratio per band (dimensionless), ``(B,)``; each
            strictly positive.
        floor: Noise floor per band, in the units of ``y``, ``(B,)``; each
            strictly positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> noise = SNRNoise(snr=jnp.array([100.0]), floor=jnp.array([0.03]))
        >>> import gaussx as gx
        >>> round(float(gx.diag(noise.covariance(jnp.array([4.0])))[0]), 6)
        0.0025
    """

    snr: Float[Array, " B"]
    floor: Float[Array, " B"]

    def __init__(
        self, snr: Real[ArrayLike, " B"], floor: Real[ArrayLike, " B"]
    ) -> None:
        self.snr = jnp.asarray(snr, dtype=float)
        self.floor = jnp.asarray(floor, dtype=float)
        _check_bands("SNRNoise", snr=self.snr, floor=self.floor)
        _check_positive("SNRNoise", snr=self.snr, floor=self.floor)

    def covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator:
        """``diag((y / SNR)² + σ_floor²)``, differentiable in ``y``."""
        y = _check_y("SNRNoise", y, self.snr)
        return _diagonal_covariance((y / self.snr) ** 2 + self.floor**2)


def _diagonal_covariance(variance: Float[Array, " B"]) -> lx.AbstractLinearOperator:
    """``diag(variance)``, tagged so lineax and gaussx know it is positive definite.

    A bare ``lx.DiagonalLinearOperator`` is not known to be positive (its
    entries could be negative), so ``lx.CG`` refuses it; gaussx still takes
    its diagonal fast paths through the tag.
    """
    return lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(variance),
        (lx.positive_semidefinite_tag, lx.symmetric_tag, lx.diagonal_tag),
    )


def _check_bands(name: str, **fields: Float[Array, "*_shape"]) -> None:
    """Each field is ``(B,)``, all with the same ``B``."""
    shapes = {key: jnp.shape(value) for key, value in fields.items()}
    if (
        any(len(shape) != 1 for shape in shapes.values())
        or len(set(shapes.values())) != 1
    ):
        raise ShapeError(
            f"{name}: {', '.join(shapes)} must be (B,) alike, got {shapes}."
        )


def _check_positive(name: str, **fields: Float[Array, " B"]) -> None:
    """Fail fast on concrete values; a traced field has nothing to inspect."""
    for key, value in fields.items():
        try:
            ok = bool(jnp.all(value > 0))
        except (
            jax.errors.TracerBoolConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            continue
        if not ok:
            raise ParameterError(f"{name}: {key} must be > 0.")


def _check_y(
    name: str, y: Float[ArrayLike, " B"], like: Float[Array, " B"]
) -> Float[Array, " B"]:
    y = jnp.asarray(y, dtype=float)
    if jnp.shape(y) != jnp.shape(like):
        raise ShapeError(f"{name}: y must be {jnp.shape(like)}, got {jnp.shape(y)}.")
    return y
