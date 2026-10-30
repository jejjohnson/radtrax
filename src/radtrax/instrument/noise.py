"""The ``NoiseModel`` protocol: the measurement-noise covariance ``S_ε``.

Optimal estimation needs ``S_ε`` as a structured operator, and synthetic
scenes need draws from it. A noise model returns ``S_ε(y)`` for the band
values ``y`` it is evaluated at (signal-dependent noise needs ``y``), as a
lineax operator tagged positive semidefinite (so ``gx.solve``, ``gx.logdet``
and ``lx.CG`` all take it), draws ``ε ~ 𝒩(0, S_ε(y))`` from a
``jax.random`` key the caller passes, and scores an observation with
``log 𝒩(y_obs | y, S_ε(y))``.

Three diagonal models:

- ``DiagonalNoise``: a fixed per-band standard deviation, ``S_ε = diag(σ²)``.
- ``SNRNoise``: a signal-to-noise ratio with a noise floor,
  ``σ_b = √((y_b / SNR_b)² + σ_floor,b²)``.
- ``NEdTNoise``: a thermal band's noise-equivalent temperature difference
  turned into radiance, ``σ_b = NEΔT_b · ∂B̄_b/∂T`` at a reference
  temperature or at the scene's band brightness temperature (radtrax#249).

Shapes: ``B`` bands.
"""

from __future__ import annotations

import abc
from typing import Literal

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, Real

from radtrax.exceptions import ParameterError, ShapeError
from radtrax.instrument.spectral import Instrument
from radtrax.radiometry.brightness import (
    band_brightness_temperature,
    band_planck_radiance,
)


__all__ = ["DiagonalNoise", "NEdTNoise", "NoiseModel", "SNRNoise"]


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


class NEdTNoise(NoiseModel):
    """Thermal-band noise from a noise-equivalent temperature difference.

    $$
    S_ε = \\mathrm{diag}(σ^2), \\qquad
    σ_b = \\mathrm{NEΔT}_b \\, \\frac{\\partial \\bar B_b}{\\partial T}\\Big|_{T},
    \\qquad \\bar B_b(T) = \\sum_g A_{bg} B(\\tilde\\nu_g, T),
    $$

    with $A$ the instrument's response matrix and $B$ the Planck radiance
    (`radtrax.radiometry.band_planck_radiance`). Two conventions for $T$:

    - ``"reference"`` (the default): $T = T_\\mathrm{ref}$, the temperature the
      specification quotes NEΔT at (300 K for TIRS-2 and ECOSTRESS). $σ$ is
      constant in ``y``, as for a detector-noise-limited instrument;
      $\\partial \\bar B / \\partial T$ at a scalar $T_\\mathrm{ref}$ is one JVP.
    - ``"scene"``: $T = T_b(y)$, each band's brightness temperature of ``y``
      (`radtrax.radiometry.band_brightness_temperature`), for "NEΔT at the
      scene temperature" taken literally. $σ$ depends on ``y``, so
      ``log_prob`` keeps $\\log |S_ε|$ (the `NoiseModel` contract), and a
      band with $y_b \\le 0$ has no brightness temperature: its $σ$ is NaN.

    $σ$ is recomputed in every call (not cached at construction), so
    gradients reach the instrument's parameters; it is in the instrument's
    radiance units (W m⁻² sr⁻¹ (cm⁻¹)⁻¹ for rows that sum to 1).

    Attributes:
        netd_k: NEΔT per band [K], ``(B,)``; each strictly positive.
        instrument: The bands $\\bar B$ is integrated over.
        t_ref_k: Reference temperature [K], a scalar > 0 (used in the
            ``"reference"`` mode).
        mode: ``"reference"`` or ``"scene"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.instrument import GaussianInstrument
        >>> nu = jnp.linspace(916.0, 919.5, 3501)  # a 1 nm band at 10.9 µm
        >>> band = GaussianInstrument(nu, jnp.array([10900.0]), jnp.array([1.0]))
        >>> noise = NEdTNoise(jnp.array([0.048]), band)
        >>> f"{float(noise.radiance_sigma(jnp.zeros(1))[0]):.4e}"
        '8.1485e-05'
    """

    netd_k: Float[Array, " B"]
    instrument: Instrument
    t_ref_k: Float[Array, ""]
    mode: Literal["reference", "scene"] = eqx.field(static=True)

    def __init__(
        self,
        netd_k: Real[ArrayLike, " B"],
        instrument: Instrument,
        *,
        t_ref_k: Real[ArrayLike, ""] = 300.0,
        mode: Literal["reference", "scene"] = "reference",
    ) -> None:
        self.netd_k = jnp.asarray(netd_k, dtype=float)
        self.instrument = instrument
        self.t_ref_k = jnp.asarray(t_ref_k, dtype=float)
        self.mode = mode
        if mode not in ("reference", "scene"):
            raise ParameterError(
                f"NEdTNoise: mode must be 'reference' or 'scene', got {mode!r}."
            )
        _check_bands("NEdTNoise", netd_k=self.netd_k)
        if self.t_ref_k.ndim != 0:
            raise ShapeError(
                f"NEdTNoise: t_ref_k must be a scalar, got {self.t_ref_k.shape}."
            )
        _check_positive("NEdTNoise", netd_k=self.netd_k)
        _check_positive("NEdTNoise", t_ref_k=self.t_ref_k[None])
        n_bands = jax.eval_shape(instrument, instrument.nu_cm).shape
        if n_bands != self.netd_k.shape:
            raise ShapeError(
                f"NEdTNoise: netd_k is {self.netd_k.shape} for an instrument "
                f"with {n_bands} bands."
            )

    def _slope(self, t_k: Float[Array, "*B"]) -> Float[Array, " B"]:
        """∂B̄_b/∂T at ``t_k``, a scalar or one per band (B̄_b depends on T_b only)."""
        _, slope = jax.jvp(
            lambda t: band_planck_radiance(self.instrument, t),
            (t_k,),
            (jnp.ones_like(t_k),),
        )
        return slope

    def _temperature(self, y: Float[Array, " B"]) -> Float[Array, "*B"]:
        if self.mode == "reference":
            return self.t_ref_k
        return band_brightness_temperature(self.instrument, y)

    def radiance_sigma(self, y: Float[ArrayLike, " B"]) -> Float[Array, " B"]:
        """$σ_b = \\mathrm{NEΔT}_b \\, \\partial \\bar B_b / \\partial T$, in the instrument's units.

        Args:
            y: Band values the noise is evaluated at, ``(B,)``; read only in
                the ``"scene"`` mode.

        Returns:
            $σ$, ``(B,)``.
        """
        y = _check_y("NEdTNoise", y, self.netd_k)
        return self.netd_k * self._slope(self._temperature(y))

    def netd_at(self, y: Float[ArrayLike, " B"]) -> Float[Array, " B"]:
        """The NEΔT this noise amounts to at the scene: $σ_b / \\partial \\bar B_b / \\partial T|_{T_b(y)}$ [K].

        The same radiance noise is a different NEΔT at another scene
        temperature: in the ``"reference"`` mode it equals ``netd_k`` only at
        $y = \\bar B(T_\\mathrm{ref})$; in the ``"scene"`` mode it is
        ``netd_k`` wherever $y > 0$.

        Args:
            y: Band values, ``(B,)``, each > 0.

        Returns:
            NEΔT per band [K], ``(B,)``.
        """
        y = _check_y("NEdTNoise", y, self.netd_k)
        t_b = band_brightness_temperature(self.instrument, y)
        return self.radiance_sigma(y) / self._slope(t_b)

    def covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator:
        """``diag(σ²)`` with $σ$ from `radiance_sigma`, differentiable in the instrument."""
        return _diagonal_covariance(self.radiance_sigma(y) ** 2)


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
