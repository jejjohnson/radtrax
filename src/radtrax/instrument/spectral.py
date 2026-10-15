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

:class:`GaussianInstrument` (radtrax#96) builds the same Gaussian bands in
JAX from their centres and FWHM in nm, and shifts the response rather than
the spectrum:

.. math::
    y_b = \\frac{\\sum_g w_g\\, f_b(ν_g)\\, L(ν_g)}{\\sum_g w_g\\, f_b(ν_g)},\\qquad
    f_b(ν) = \\exp\\Big(-\\frac{(10^7/(ν - δ) - δ_λ - c_b)^2}{2σ_b^2}\\Big),

with ``w`` the trapezoidal weights on the wavelengths ``10⁷/ν``, ``c_b`` the
centre, ``σ_b`` the FWHM over ``2√(2 ln 2)``, and a shift ``δ`` in cm⁻¹ or
``δ_λ`` in nm. At zero shift it is ``BandInstrument.from_srf`` of the same
bands; ``y`` is smooth in both shifts and in the FWHM, so ``∂²y/∂δ²`` is
the band's curvature, not zero, and a calibration per nm needs no
conversion.

Shapes: ``G`` wavenumbers, ``B`` bands.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from radtrax.exceptions import GridError, ParameterError, ShapeError
from radtrax.instrument.srf import FWHM_PER_SIGMA, SpectralResponseFunction


__all__ = ["BandInstrument", "GaussianInstrument", "Instrument"]


class Instrument(eqx.Module):
    """Protocol: radiance on a wavenumber grid in, one value per band out.

    An abstract Equinox module, so ``filter_jit`` compiles each concrete
    instrument and a forward model can hold any of them.

    Attributes:
        nu_cm: The wavenumber grid the instrument reads [cm⁻¹], ``(G,)``.
    """

    nu_cm: eqx.AbstractVar[Float[Array, " G"]]

    @abc.abstractmethod
    def __call__(self, radiance: Float[ArrayLike, " G"]) -> Float[Array, " B"]:
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
            raise ShapeError(
                f"BandInstrument: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        _check_increasing(self.nu_cm, "BandInstrument")
        if self.response.ndim != 2 or self.response.shape[1] != self.nu_cm.shape[0]:
            raise ShapeError(
                f"BandInstrument: response must be (B, G) with G = "
                f"{self.nu_cm.shape[0]}, got {self.response.shape}."
            )
        if len(self.band_names) != self.response.shape[0]:
            raise ShapeError(
                f"BandInstrument: {len(self.band_names)} band names for "
                f"{self.response.shape[0]} bands."
            )
        if self.shift_cm.ndim != 0:
            raise ShapeError(
                f"BandInstrument: shift_cm must be a scalar, got {self.shift_cm.shape}."
            )

    @property
    def n_bands(self) -> int:
        """Number of bands ``B``."""
        return int(self.response.shape[0])

    def __call__(self, radiance: Float[ArrayLike, " G"]) -> Float[Array, " B"]:
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
            raise GridError(
                "BandInstrument.from_srf: build the SRF on wavelengths_hr_nm = "
                "1e7 / nu_cm[::-1] (the wavenumber grid in ascending wavelength)."
            )
        return cls(nu, srf.matrix[:, ::-1], shift_cm, srf.band_names)


class GaussianInstrument(Instrument):
    """Gaussian bands built in JAX, shifted analytically in cm⁻¹ or in nm.

    ``y_b`` is the trapezoidal average of the radiance over band ``b``'s
    response ``f_b(ν) = exp(−(10⁷/(ν − δ) − δ_λ − c_b)² / 2σ_b²)``; see the
    module docstring. Keep each band's response inside the grid: a band
    shifted off it raises ``equinox.EquinoxRuntimeError`` (a ``RuntimeError``).

    Attributes:
        nu_cm: Wavenumber grid [cm⁻¹], strictly increasing, ``(G,)``.
        center_nm: Band centres [nm], ``(B,)``.
        fwhm_nm: Band FWHM [nm], ``(B,)``, a differentiable leaf.
        shift_cm: Spectral shift ``δ`` [cm⁻¹]; positive moves the bands to
            higher wavenumber, as in :class:`BandInstrument`.
        shift_nm: Spectral shift ``δ_λ`` [nm]; positive moves the bands to
            longer wavelength.
        band_names: One label per band.

    Examples:
        >>> import jax.numpy as jnp
        >>> nu = jnp.linspace(4250.0, 4350.0, 1001)
        >>> instrument = GaussianInstrument(
        ...     nu, jnp.array([2325.0]), jnp.array([8.5])
        ... )
        >>> round(float(instrument(jnp.ones(1001))[0]), 12)  # a flat spectrum
        1.0
    """

    nu_cm: Float[Array, " G"]
    center_nm: Float[Array, " B"]
    fwhm_nm: Float[Array, " B"]
    shift_cm: Float[Array, ""]
    shift_nm: Float[Array, ""]
    band_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        nu_cm: Float[ArrayLike, " G"],
        center_nm: Float[ArrayLike, " B"],
        fwhm_nm: Float[ArrayLike, " B"],
        *,
        shift_cm: Float[ArrayLike, ""] = 0.0,
        shift_nm: Float[ArrayLike, ""] = 0.0,
        band_names: tuple[str, ...] | None = None,
    ) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        self.center_nm = jnp.asarray(center_nm, dtype=float)
        self.fwhm_nm = jnp.asarray(fwhm_nm, dtype=float)
        self.shift_cm = jnp.asarray(shift_cm, dtype=float)
        self.shift_nm = jnp.asarray(shift_nm, dtype=float)
        n_bands = self.center_nm.shape[0] if self.center_nm.ndim == 1 else 0
        self.band_names = (
            tuple(f"B{b}" for b in range(n_bands))
            if band_names is None
            else tuple(band_names)
        )

    def __check_init__(self) -> None:
        if self.nu_cm.ndim != 1:
            raise ShapeError(
                f"GaussianInstrument: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        _check_increasing(self.nu_cm, "GaussianInstrument")
        if self.center_nm.ndim != 1 or self.fwhm_nm.shape != self.center_nm.shape:
            raise ShapeError(
                "GaussianInstrument: center_nm and fwhm_nm must be (B,) alike, got "
                f"{self.center_nm.shape} and {self.fwhm_nm.shape}."
            )
        if len(self.band_names) != self.center_nm.shape[0]:
            raise ShapeError(
                f"GaussianInstrument: {len(self.band_names)} band names for "
                f"{self.center_nm.shape[0]} bands."
            )
        if self.shift_cm.ndim != 0 or self.shift_nm.ndim != 0:
            raise ShapeError(
                "GaussianInstrument: shift_cm and shift_nm must be scalars."
            )

    @property
    def n_bands(self) -> int:
        """Number of bands ``B``."""
        return int(self.center_nm.shape[0])

    def response(self) -> Float[Array, "B G"]:
        """The shifted, trapezoid-weighted response, each row summing to 1, ``(B, G)``."""
        wavelength_nm = 1e7 / (self.nu_cm - self.shift_cm) - self.shift_nm  # (G,)
        sigma = self.fwhm_nm[:, None] / FWHM_PER_SIGMA  # (B, 1)
        f = jnp.exp(-((wavelength_nm - self.center_nm[:, None]) ** 2) / (2 * sigma**2))
        # Trapezoidal weights on the (unshifted) wavelengths, as the NumPy SRF.
        half = jnp.abs(jnp.diff(1e7 / self.nu_cm)) / 2  # (G − 1,)
        weights = jnp.pad(half, (0, 1)) + jnp.pad(half, (1, 0))  # (G,)
        weighted = f * weights
        total = jnp.sum(weighted, axis=1, keepdims=True)  # (B, 1)
        # A band shifted off the grid has no response left: fail loudly (also
        # under jit), as the NumPy SRF does, rather than return 0/0.
        total = eqx.error_if(
            total,
            jnp.any(total <= 0),
            "GaussianInstrument: a band has no response on the grid.",
        )
        return weighted / total

    def __call__(self, radiance: Float[ArrayLike, " G"]) -> Float[Array, " B"]:
        return self.response() @ jnp.asarray(radiance)

    @classmethod
    def from_srf(
        cls,
        srf: SpectralResponseFunction,
        nu_cm: Float[ArrayLike, " G"],
        *,
        shift_cm: Float[ArrayLike, ""] = 0.0,
        shift_nm: Float[ArrayLike, ""] = 0.0,
    ) -> GaussianInstrument:
        """The bands of a Gaussian :class:`~radtrax.instrument.SpectralResponseFunction`.

        Raises:
            ParameterError: If the SRF's bands are not Gaussian.
        """
        if srf.srf_type != "gaussian":
            raise ParameterError(
                f"GaussianInstrument.from_srf: the SRF must be Gaussian, got {srf.srf_type!r}."
            )
        return cls(
            nu_cm,
            srf.band_centers_nm,
            srf.band_widths_nm,
            shift_cm=shift_cm,
            shift_nm=shift_nm,
            band_names=srf.band_names,
        )


def _check_increasing(nu_cm: Float[Array, " G"], name: str) -> None:
    """Fail fast on a concrete grid that is not strictly increasing."""
    try:
        ok = bool(jnp.all(jnp.diff(nu_cm) > 0))
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return
    if not ok:
        raise GridError(f"{name}: nu_cm must be strictly increasing.")
