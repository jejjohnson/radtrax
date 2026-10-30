"""JAX observation operator for the methane VMR retrieval.

The full forward map factors as

    x  ──┐
         │ ΔVMR = x − x_ref
         ▼
        Δτ(ν, i, j) = a(ν) · ΔVMR(i, j)         a(ν) = σ(ν, T, P) · N · ΔX · AMF
         │
         ▼
        L_hr(i, j, ν) = exp(−Δτ)                 (or 1 − Δτ for linear)
         │
         ▼ PSF (per channel)
         ▼ GSD (per channel)
         ▼ BandInstrument (band integration, per pixel)
        y_obs(I, J, b)

Everything below the LUT lookup is JAX, so

- ``forward`` runs end-to-end under :func:`jax.jit`,
- ``jax.grad(loss∘forward)`` produces the **discrete adjoint** for free,
- ``jax.jvp(forward, …)`` produces the tangent-linear,

without us writing or maintaining any explicit adjoint. The reason
``radtran/instrument.py`` *does* expose hand-rolled ``adjoint`` methods is to
let unit tests verify the inner-product identity independently — JAX's autodiff
inherits the correctness, not the other way around.

The factory :meth:`RadianceObservationModel.from_lut` precomputes ``a(ν)`` once
from the HAPI cross-section LUT so the per-iteration cost in 3D-Var is dominated
by the cube-shaped operations (PSF, GSD, SRF).

The band integration is :class:`~radtrax.instrument.BandInstrument`, the one
JAX band product in radtrax, mapped over pixels with ``eqx.filter_vmap``; its
spectral shift is a leaf, so a 3D-Var retrieval can estimate it too
(radtrax#95). The high-resolution axis is the instrument's wavenumber grid,
ascending.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/assimilation/obs_operator.py``); its frozen
dataclasses became Equinox modules in radtrax#58.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from radtrax.exceptions import ArgumentTypeError, MissingKeyError, ShapeError
from radtrax.instrument.psf import (
    GroundSamplingDistance,
    PointSpreadFunction,
)
from radtrax.instrument.spectral import BandInstrument
from radtrax.slab.config import number_density_cm3


__all__ = [
    "RadianceObservationModel",
]


if TYPE_CHECKING:
    import xarray as xr

    from radtrax.instrument.srf import SpectralResponseFunction


class RadianceObservationModel(eqx.Module):
    """Bundle of precomputed pieces that defines the forward operator H.

    Attributes
    ----------
    absorption_coeff_hr : jnp.ndarray
        Per-wavenumber absorption coefficient ``a(ν) = σ(ν,T,P) · N · ΔX · AMF``
        on ``instrument.nu_cm`` (ascending wavenumber), shape ``(n_lambda,)``.
        Captures everything *except* the VMR field, so the forward map is just
        ``L_hr(i,j,ν) = exp(-a(ν) · ΔVMR(i,j))``.
    instrument : BandInstrument
        The band integration on the same wavenumber grid, ``(n_lambda,) →
        (n_bands,)``, applied per pixel; its ``shift_cm`` (0 by default) is a
        differentiable leaf.
    vmr_reference : Array
        Scalar background VMR, stored as an array so it is traced, not static,
        under ``eqx.filter_jit`` and differentiable with ``eqx.filter_grad``. Background VMR ``x_ref`` around which radiances are normalised. Setting
        ``vmr_reference = 0`` recovers absolute Beer-Lambert; setting it to the
        scene's true background produces the differential signal that matched
        filter retrievals consume.
    psf, gsd : optional
        JAX-side instrument operators from :mod:`radtrax.instrument.psf`.
        ``None`` skips that stage — convenient for hyperspectral tests where
        the spectral cube is already at sensor resolution.
    """

    absorption_coeff_hr: Float[Array, " Ghr"]
    instrument: BandInstrument
    vmr_reference: Float[Array, ""]
    psf: PointSpreadFunction | None = None
    gsd: GroundSamplingDistance | None = None

    def __init__(
        self,
        absorption_coeff_hr: Float[ArrayLike, " Ghr"],
        instrument: BandInstrument | Float[ArrayLike, "B Ghr"],
        vmr_reference: Float[ArrayLike, ""] = 0.0,
        psf: PointSpreadFunction | None = None,
        gsd: GroundSamplingDistance | None = None,
    ) -> None:
        a = jnp.asarray(absorption_coeff_hr)
        if a.ndim != 1:
            raise ShapeError(
                f"RadianceObservationModel: absorption_coeff_hr must be 1-D, "
                f"got shape {a.shape}."
            )
        if not isinstance(instrument, BandInstrument):
            instrument, a = _from_srf_matrix(instrument, a)
        if instrument.response.shape[1] != a.size:
            raise ShapeError(
                f"RadianceObservationModel: the instrument's response "
                f"{instrument.response.shape} does not align with "
                f"absorption_coeff_hr (n_lambda={a.size})."
            )
        self.absorption_coeff_hr = a
        self.instrument = instrument
        # An array, not a Python float: a float field is static under
        # eqx.filter_jit, so a new value would recompile and get no gradient.
        self.vmr_reference = jnp.asarray(vmr_reference, dtype=a.dtype)
        self.psf = psf
        self.gsd = gsd

    @property
    def n_lambda(self) -> int:
        """Number of high-resolution spectral samples."""
        return int(self.absorption_coeff_hr.shape[-1])

    @property
    def n_bands(self) -> int:
        """Number of sensor bands."""
        return int(self.instrument.response.shape[-2])

    @property
    def srf_matrix(self) -> Float[Array, "B Ghr"]:
        """Deprecated: the response in ascending wavelength, ``instrument.response[:, ::-1]``."""
        warnings.warn(
            "RadianceObservationModel.srf_matrix is deprecated; use "
            "instrument.response (on instrument.nu_cm, ascending wavenumber).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.instrument.response[:, ::-1]

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_lut(
        cls,
        lut: xr.Dataset,
        *,
        srf: SpectralResponseFunction,
        T_K: float,
        p_atm: float,
        path_length_cm: float,
        amf: float,
        vmr_reference: float = 0.0,
        psf: PointSpreadFunction | None = None,
        gsd: GroundSamplingDistance | None = None,
        var: str = "absorption_cross_section",
        shift_cm: float = 0.0,
    ) -> RadianceObservationModel:
        """Build the operator from a HAPI σ(ν,T,P) LUT and an SRF.

        Pre-collapses ``σ · N · ΔX · AMF`` into a single ``a(ν)`` vector so the
        forward call only multiplies by ``ΔVMR`` per pixel.

        The HR spectral grid is **taken from the SRF**: its wavelengths
        ``srf.wavelengths_hr_nm``, reversed into ascending wavenumber
        ``ν = 10⁷ / λ``, are the grid of both ``a(ν)`` and the
        :class:`~radtrax.instrument.BandInstrument` built by
        :meth:`BandInstrument.from_srf
        <radtrax.instrument.BandInstrument.from_srf>`, so the band
        integration contracts aligned spectral bins by construction. Earlier
        revisions accepted a separate ``nu_obs`` argument; when callers
        passed a wavenumber-ordered grid while the SRF was wavelength-ordered
        (common whenever ``wl = 1e7/ν`` was sorted), the product silently
        mixed non-corresponding bins — self-consistent for twin retrievals
        but physically wrong for anything cross-validated.

        ``shift_cm`` is the instrument's spectral shift ``δ`` [cm⁻¹], 0 by
        default (the band integration of earlier revisions).
        """
        if var not in lut:
            raise MissingKeyError(
                f"RadianceObservationModel.from_lut: variable {var!r} not in "
                f"dataset (have {list(lut.data_vars)})."
            )
        import xarray as xr_mod

        # Source-of-truth HR grid: the SRF's own wavelength axis, reversed into
        # ascending wavenumber, the grid BandInstrument.from_srf puts the
        # response on.
        wl_hr_nm = np.asarray(srf.wavelengths_hr_nm, dtype=float)
        nu_hr = 1e7 / wl_hr_nm[::-1]
        nu_da = xr_mod.DataArray(nu_hr, dims=["obs_nu"])
        sigma = lut[var].interp(
            wavenumber=nu_da, temperature=T_K, pressure=p_atm, method="linear"
        )
        sigma_np = np.asarray(sigma.values, dtype=float)
        N_total = number_density_cm3(p_atm, T_K)
        a = sigma_np * N_total * path_length_cm * amf
        return cls(
            absorption_coeff_hr=jnp.asarray(a),
            instrument=BandInstrument.from_srf(srf, nu_hr, shift_cm=shift_cm),
            vmr_reference=vmr_reference,
            psf=psf,
            gsd=gsd,
        )

    # ── forward operator ────────────────────────────────────────────────────

    def forward(
        self, vmr_field: Float[Array, "H W"], *, linear: bool = False
    ) -> Float[Array, "h w B"]:
        """Map a 2-D VMR field to a band-integrated obs cube.

        Parameters
        ----------
        vmr_field : Array
            Per-pixel VMR, shape ``(ny_hr, nx_hr)``.
        linear : bool
            If True, replace ``exp(-Δτ)`` by its Maclaurin-1 approximation
            ``1 - Δτ``. The tangent-linear of the nonlinear forward at
            ``vmr_field == vmr_reference`` coincides with this; the matched
            filter is exactly this case with ``δx`` propagated through.

        Returns
        -------
        Array
            Sensor-space radiance cube, shape ``(ny_lr, nx_lr, n_bands)``
            (``ny_lr = ny_hr // f``, ``nx_lr = nx_hr // f`` if a GSD is set).
        """
        if vmr_field.ndim != 2:
            raise ShapeError(
                f"RadianceObservationModel.forward: vmr_field must be 2-D, got shape {vmr_field.shape}."
            )
        delta_vmr = vmr_field - self.vmr_reference
        # Outer product (ny, nx) ⊗ (n_lambda,) → (ny, nx, n_lambda).
        delta_tau = delta_vmr[..., None] * self.absorption_coeff_hr
        # `linear` is a Python bool (static); branch on it directly so JAX
        # doesn't trace the unused branch — `exp(huge δτ)` would otherwise
        # overflow on stress-tests like the adjoint inner-product check.
        L_hr = (1.0 - delta_tau) if linear else jnp.exp(-delta_tau)
        if self.psf is not None:
            L_hr = self.psf.apply(L_hr)
        if self.gsd is not None:
            L_hr = self.gsd.apply(L_hr)
        # Band integration per pixel: (h, w, n_lambda) → (h, w, n_bands).
        return eqx.filter_vmap(eqx.filter_vmap(self.instrument))(L_hr)

    def make_forward(
        self, *, linear: bool = False
    ) -> Callable[[Float[Array, "H W"]], Float[Array, "h w B"]]:
        """Return a closure ``vmr_field → obs`` with the model captured.

        The closure composes directly with ``jax.value_and_grad`` and the
        optimisers, which differentiate with respect to the state only. To
        differentiate with respect to the model itself (its SRF, its PSF
        kernel), pass the model as an argument instead:
        ``eqx.filter_grad(lambda m, x: loss(m.forward(x)))``.

        ``jax.jit(model.forward)`` no longer works: the model holds arrays, so
        it is not hashable (it was a hashable frozen dataclass in plumax). Use
        ``eqx.filter_jit(model.forward)`` or ``jax.jit(model.make_forward())``.
        """
        linear_flag = bool(linear)

        def _fwd(vmr_field: Float[Array, "H W"]) -> Float[Array, "h w B"]:
            return self.forward(vmr_field, linear=linear_flag)

        return _fwd


def _from_srf_matrix(
    srf_matrix: Float[ArrayLike, "B Ghr"], absorption_coeff_hr: Float[Array, " Ghr"]
) -> tuple[BandInstrument, Float[Array, " Ghr"]]:
    """The deprecated ``(absorption_coeff_hr, srf_matrix)`` form, in ascending wavelength.

    Both are reversed into ascending wavenumber. No wavenumber grid comes with
    them, so the instrument's grid is the sample index: exact at zero shift,
    the only shift the old form had.
    """
    if isinstance(srf_matrix, eqx.Module):
        raise ArgumentTypeError(
            "RadianceObservationModel: `instrument` must be a BandInstrument, got "
            f"{type(srf_matrix).__name__}."
        )
    warnings.warn(
        "RadianceObservationModel(absorption_coeff_hr, srf_matrix) is deprecated; "
        "pass a BandInstrument and absorption_coeff_hr on its wavenumber grid, or "
        "use RadianceObservationModel.from_lut.",
        DeprecationWarning,
        stacklevel=3,
    )
    s = jnp.asarray(srf_matrix)
    if s.ndim != 2 or s.shape[1] != absorption_coeff_hr.size:
        raise ShapeError(
            f"RadianceObservationModel: srf_matrix shape {s.shape} does not "
            f"align with absorption_coeff_hr (n_lambda={absorption_coeff_hr.size})."
        )
    index = jnp.arange(s.shape[1], dtype=float)
    return BandInstrument(index, s[:, ::-1]), absorption_coeff_hr[::-1]
