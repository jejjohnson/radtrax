"""The ``Instrument`` protocol and a band instrument with a spectral shift.

An instrument turns radiance on the model's wavenumber grid into what the
sensor records per band (*radtrax · 3*). v0's instrument integrates over
each band's spectral response and lets the wavenumber scale shift:

.. math::
    y_b = \\sum_g f_{bg}\\, L(ν_g + δ),

with ``f`` the row-normalised response matrix ``(B, G)``, ``L`` the
radiance ``(G,)`` and ``δ`` the spectral shift [cm⁻¹]: a positive ``δ``
moves every band's response to higher wavenumber (shorter wavelength). A
uniform shift in wavenumber is not one in wavelength (δλ = −λ² δν / 10⁷):
across 1600–2500 nm the same δν moves the bands by amounts 2.4 times apart,
so a calibration given per nm converts band by band.

``L(ν + δ)`` is linear interpolation on the grid, so ``y`` is piecewise
linear in ``δ`` with kinks at multiples of the grid step; on a node the
derivative is the slope of the segment above it. On a 0.01 cm⁻¹ grid that
derivative is within 0.2 % of an analytically shifted SRF's and a
Gauss–Newton step from ``δ = 0`` converges; ``∂²y/∂δ²`` is zero, so a
method that needs the curvature in ``δ`` needs the analytic shift. Beyond
the grid's ends the radiance is held at its edge value, so keep each band's
response well inside the grid. A changing FWHM and the PSF and GSD of
:mod:`radtrax.instrument.psf` join it in v1 of the roadmap.

The response matrix comes from the NumPy
:class:`~radtrax.instrument.SpectralResponseFunction` (exact Gaussian FWHM,
trapezoidal weights) through :meth:`BandInstrument.from_srf`, so both
integrate a band the same way; this module only moves it onto the gradient
path. Ported from the radtrax · 3 design notebook's ``Instrument``.

Shapes: ``G`` wavenumbers, ``B`` bands.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from radtrax.instrument.srf import SpectralResponseFunction


__all__ = ["BandInstrument", "Instrument"]


class Instrument(eqx.Module):
    """Protocol: radiance on a wavenumber grid in, one value per band out.

    An abstract Equinox module, so ``filter_jit`` compiles each concrete
    instrument and a forward model can hold any of them.

    Attributes:
        nu_cm: The wavenumber grid the instrument reads [cm⁻¹], ``(G,)``.
    """

    nu_cm: eqx.AbstractVar[Float[Array, " G"]]

    @abc.abstractmethod
    def __call__(self, radiance: Float[Array, " G"]) -> Float[Array, " B"]:
        """Band values ``(B,)`` from radiance on :attr:`nu_cm`, ``(G,)``."""


class BandInstrument(Instrument):
    """Band integration with a spectral shift, ``y_b = Σ_g f_bg L(ν_g + δ)``.

    Attributes:
        nu_cm: Wavenumber grid [cm⁻¹], strictly increasing, ``(G,)``.
        response: Spectral response per band on ``nu_cm``, each row summing
            to 1, ``(B, G)``.
        shift_cm: Spectral shift ``δ`` [cm⁻¹], a differentiable leaf.
        band_names: One label per band.

    Examples:
        >>> import jax.numpy as jnp
        >>> instrument = BandInstrument(
        ...     jnp.array([4300.0, 4300.1, 4300.2]),
        ...     jnp.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
        ... )
        >>> [float(y) for y in instrument(jnp.array([1.0, 2.0, 4.0]))]
        [1.5, 3.0]
    """

    nu_cm: Float[Array, " G"]
    response: Float[Array, "B G"]
    shift_cm: Float[Array, ""]
    band_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        nu_cm: Float[ArrayLike, " G"],
        response: Float[ArrayLike, "B G"],
        shift_cm: Float[ArrayLike, ""] = 0.0,
        band_names: tuple[str, ...] | None = None,
    ) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        self.response = jnp.asarray(response, dtype=float)
        self.shift_cm = jnp.asarray(shift_cm, dtype=float)
        n_bands = self.response.shape[0] if self.response.ndim == 2 else 0
        self.band_names = (
            tuple(f"B{b}" for b in range(n_bands))
            if band_names is None
            else tuple(band_names)
        )

    def __check_init__(self) -> None:
        if self.nu_cm.ndim != 1:
            raise ValueError(
                f"BandInstrument: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        _check_increasing(self.nu_cm)
        if self.response.ndim != 2 or self.response.shape[1] != self.nu_cm.shape[0]:
            raise ValueError(
                f"BandInstrument: response must be (B, G) with G = "
                f"{self.nu_cm.shape[0]}, got {self.response.shape}."
            )
        if len(self.band_names) != self.response.shape[0]:
            raise ValueError(
                f"BandInstrument: {len(self.band_names)} band names for "
                f"{self.response.shape[0]} bands."
            )
        if self.shift_cm.ndim != 0:
            raise ValueError(
                f"BandInstrument: shift_cm must be a scalar, got {self.shift_cm.shape}."
            )

    @property
    def n_bands(self) -> int:
        """Number of bands ``B``."""
        return int(self.response.shape[0])

    def __call__(self, radiance: Float[Array, " G"]) -> Float[Array, " B"]:
        # L(ν_g + δ), linear on the grid, edge values beyond it            (G,)
        # On ν − ν₀, so δ is resolved to its own precision, not to ν's ulp
        # (4.9e-4 cm⁻¹ at 4300 cm⁻¹ in float32).
        rel = self.nu_cm - self.nu_cm[0]
        shifted = jnp.interp(rel + self.shift_cm, rel, radiance)
        # y_b = Σ_g f_bg L(ν_g + δ)                                         (B,)
        return self.response @ shifted

    @classmethod
    def from_srf(
        cls,
        srf: SpectralResponseFunction,
        nu_cm: Float[ArrayLike, " G"],
        *,
        shift_cm: Float[ArrayLike, ""] = 0.0,
    ) -> BandInstrument:
        """From a :class:`~radtrax.instrument.SpectralResponseFunction` built
        on the wavelengths of ``nu_cm``.

        ``srf.wavelengths_hr_nm`` must be ``1e7 / nu_cm`` in ascending
        order (``nu_cm`` reversed), so the response is the SRF's own matrix,
        its columns reversed onto the wavenumber grid.

        Raises:
            ValueError: If the SRF is on a different grid.
        """
        nu = np.asarray(nu_cm, dtype=float)
        expected_nm = 1e7 / nu[::-1]
        if srf.wavelengths_hr_nm.shape != expected_nm.shape or not np.allclose(
            srf.wavelengths_hr_nm, expected_nm, rtol=1e-12, atol=0.0
        ):
            raise ValueError(
                "BandInstrument.from_srf: build the SRF on wavelengths_hr_nm = "
                "1e7 / nu_cm[::-1] (the wavenumber grid in ascending wavelength)."
            )
        return cls(nu, srf.matrix[:, ::-1], shift_cm, srf.band_names)


def _check_increasing(nu_cm: Float[Array, " G"]) -> None:
    """Fail fast on a concrete grid that is not strictly increasing."""
    try:
        ok = bool(jnp.all(jnp.diff(nu_cm) > 0))
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return
    if not ok:
        raise ValueError("BandInstrument: nu_cm must be strictly increasing.")
