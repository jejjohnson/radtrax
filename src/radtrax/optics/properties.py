"""``OpticalProperties``: what a solver needs from the atmosphere.

The stage between opacity and the solver in *radtrax · 3*: per-layer
optical depth on a wavenumber grid. v0's solvers need nothing else; the
single-scattering albedo and phase-function moments of v3 join as fields
with defaults (a non-scattering atmosphere), so a solver written against
this module keeps working.

Shapes: ``L`` layers, ``G`` wavenumbers.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.exceptions import ShapeError


__all__ = ["OpticalProperties"]


class OpticalProperties(eqx.Module):
    """Per-layer optical depth on a wavenumber grid.

    Build it from any ``OpacityProvider``:
    ``OpticalProperties(opacity.nu_cm, opacity(state))``.

    Attributes:
        nu_cm: Wavenumber grid [cm⁻¹], ``(G,)``.
        tau: Optical depth of each layer at each wavenumber
            (dimensionless), ``(L, G)``, layers from the surface up.
        z_m: Height of each layer's middle above the surface [m], ``(L,)``, or ``None``;
            with ``dz_m``, what a spherical air mass needs.
        dz_m: Thickness of each layer [m], ``(L,)``, or ``None``.

    Examples:
        >>> import jax.numpy as jnp
        >>> optics = OpticalProperties(
        ...     jnp.array([4300.0, 4300.1]), jnp.zeros((3, 2))
        ... )
        >>> optics.n_layers
        3
    """

    nu_cm: Float[Array, " G"]
    tau: Float[Array, "L G"]
    z_m: Float[Array, " L"] | None
    dz_m: Float[Array, " L"] | None

    def __init__(
        self,
        nu_cm: Real[ArrayLike, " G"],
        tau: Real[ArrayLike, "L G"],
        *,
        z_m: Real[ArrayLike, " L"] | None = None,
        dz_m: Real[ArrayLike, " L"] | None = None,
    ) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        self.tau = jnp.asarray(tau, dtype=float)
        self.z_m = None if z_m is None else jnp.asarray(z_m, dtype=float)
        self.dz_m = None if dz_m is None else jnp.asarray(dz_m, dtype=float)

    def __check_init__(self) -> None:
        if self.nu_cm.ndim != 1:
            raise ShapeError(
                f"OpticalProperties: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        if self.tau.ndim != 2 or self.tau.shape[1] != self.nu_cm.shape[0]:
            raise ShapeError(
                f"OpticalProperties: tau must be (L, G) with G = "
                f"{self.nu_cm.shape[0]}, got {self.tau.shape}."
            )
        if (self.z_m is None) != (self.dz_m is None):
            raise ShapeError("OpticalProperties: give both z_m and dz_m, or neither.")
        for name, value in (("z_m", self.z_m), ("dz_m", self.dz_m)):
            if value is not None and value.shape != self.tau.shape[:1]:
                raise ShapeError(
                    f"OpticalProperties: {name} must be (L,) = {self.tau.shape[:1]}, "
                    f"got {value.shape}."
                )

    @property
    def n_layers(self) -> int:
        """Number of layers ``L``."""
        return int(self.tau.shape[0])
