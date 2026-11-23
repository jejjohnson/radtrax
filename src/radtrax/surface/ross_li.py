"""The Ross–Li kernel BRDF: RossThick and LiSparse-Reciprocal.

The MODIS kernel-driven model (Lucht et al. 2000) writes the bidirectional
reflectance factor as a linear combination of an isotropic term and two
kernels of the geometry:

$$
R(\\theta_0, \\theta, \\phi) = f_{\\mathrm{iso}} + f_{\\mathrm{vol}} K_{\\mathrm{vol}}
    + f_{\\mathrm{geo}} K_{\\mathrm{geo}}.
$$

- **RossThick** (volume scattering by a dense canopy), with the phase angle
  ``ξ``, ``cos ξ = cos θ₀ cos θ + sin θ₀ sin θ cos φ``:
  ``K_vol = ((π/2 − ξ) cos ξ + sin ξ) / (cos θ₀ + cos θ) − π/4``.
- **LiSparse-Reciprocal** (geometric-optical shadowing by sparse crowns)
  with MODIS's crown shape ``h/b = 2``, ``b/r = 1`` (so ``θ' = θ``):
  ``K_geo = O − sec θ₀ − sec θ + ½ (1 + cos ξ) sec θ₀ sec θ``, with the
  overlap ``O = (t − sin t cos t)(sec θ₀ + sec θ) / π`` and
  ``cos t = (h/b) √(D² + (tan θ₀ tan θ sin φ)²) / (sec θ₀ + sec θ)``,
  clipped to ``[−1, 1]``, ``D² = tan² θ₀ + tan² θ − 2 tan θ₀ tan θ cos φ``.

``φ`` is :class:`~radtrax.atmosphere.ViewingGeometry`'s relative azimuth,
sensor minus sun, so ``φ = 0`` with ``θ₀ = θ`` is the hotspot (backscatter).
Both kernels are 0 at nadir/nadir and symmetric in ``θ₀ ↔ θ``.

The kernels are radiomax's, of the cosines that
:class:`~radtrax.atmosphere.ViewingGeometry` holds
(`radiomax.retrievals.ross_thick_from_cosines` and
`li_sparse_reciprocal_from_cosines`). First-order gradients are finite
everywhere, nadir and the hotspot included, where the kernels have a kink and
the gradient is the symmetric (sub)gradient 0 that a central difference
measures; second derivatives are not (the Hessian at exact nadir is NaN).

Shapes: ``G`` wavenumbers; ``*G`` means a scalar or ``(G,)``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real
from radiomax.retrievals import (
    li_sparse_reciprocal_from_cosines,
    ross_thick_from_cosines,
)

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.surface.lambertian import Surface


__all__ = ["RossLi"]


class RossLi(Surface):
    """The Ross–Li kernel BRDF: ``R = f_iso + f_vol K_vol + f_geo K_geo``.

    With ``f_vol = f_geo = 0`` it is ``Lambertian(f_iso)``.

    Attributes:
        f_iso: Isotropic weight (dimensionless), a scalar or ``(G,)``.
        f_vol: RossThick (volume) weight, a scalar or ``(G,)``.
        f_geo: LiSparse-Reciprocal (geometric) weight, a scalar or ``(G,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> surface = RossLi(0.3, 0.1, 0.05)
        >>> nadir = ViewingGeometry(1.0, 1.0)
        >>> float(surface.reflectance(nadir, jnp.array([4300.0])))
        0.3
    """

    f_iso: Float[Array, "*G"]
    f_vol: Float[Array, "*G"]
    f_geo: Float[Array, "*G"]

    def __init__(
        self,
        f_iso: Real[ArrayLike, "*G"],
        f_vol: Real[ArrayLike, "*G"],
        f_geo: Real[ArrayLike, "*G"],
    ) -> None:
        self.f_iso = jnp.asarray(f_iso, dtype=float)
        self.f_vol = jnp.asarray(f_vol, dtype=float)
        self.f_geo = jnp.asarray(f_geo, dtype=float)

    def reflectance(
        self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]
    ) -> Float[Array, "*G"]:
        """``R = f_iso + f_vol K_vol + f_geo K_geo``; the kernels do not depend on ``ν``."""
        del nu_cm
        angles = (geometry.mu0, geometry.mu, geometry.relative_azimuth_rad)
        # R = f_iso + f_vol K_vol + f_geo K_geo
        return (
            self.f_iso
            + self.f_vol * ross_thick_from_cosines(*angles, degrees=False)
            + self.f_geo * li_sparse_reciprocal_from_cosines(*angles, degrees=False)
        )
