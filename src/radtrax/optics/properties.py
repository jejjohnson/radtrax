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
from jaxtyping import Array, ArrayLike, Float


__all__ = ["OpticalProperties"]


class OpticalProperties(eqx.Module):
    """Per-layer optical depth on a wavenumber grid.

    Build it from any ``OpacityProvider``:
    ``OpticalProperties(opacity.nu_cm, opacity(state))``.

    Attributes:
        nu_cm: Wavenumber grid [cm⁻¹], ``(G,)``.
        tau: Optical depth of each layer at each wavenumber
            (dimensionless), ``(L, G)``, layers from the surface up.

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

    def __init__(
        self, nu_cm: Float[ArrayLike, " G"], tau: Float[ArrayLike, "L G"]
    ) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        self.tau = jnp.asarray(tau, dtype=float)

    def __check_init__(self) -> None:
        if self.nu_cm.ndim != 1:
            raise ValueError(
                f"OpticalProperties: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        if self.tau.ndim != 2 or self.tau.shape[1] != self.nu_cm.shape[0]:
            raise ValueError(
                f"OpticalProperties: tau must be (L, G) with G = "
                f"{self.nu_cm.shape[0]}, got {self.tau.shape}."
            )

    @property
    def n_layers(self) -> int:
        """Number of layers ``L``."""
        return int(self.tau.shape[0])
