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

First-order gradients are finite everywhere, nadir and the hotspot
included (second derivatives are not: the Hessian at exact nadir is NaN):

- ``(π/2 − ξ) cos ξ + sin ξ`` is ``c · arcsin c + √(1 − c²)`` in
  ``c = cos ξ``, whose derivative is ``arcsin c``; and ``t − sin t cos t``
  is ``arccos u − u √(1 − u²)`` in ``u = cos t``, whose derivative is
  ``−2 √(1 − u²)``. Both are written with that derivative (``custom_jvp``),
  where ``arccos`` differentiated naively gives ``0 · ∞`` at ``±1``.
- ``√(D² + …)`` vanishes at nadir and at the hotspot, where the kernel has
  a kink (a distance); the double-where there gives the symmetric
  (sub)gradient 0, which is what a central difference measures.

Shapes: ``G`` wavenumbers; ``*G`` means a scalar or ``(G,)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.surface.lambertian import Surface


__all__ = ["RossLi", "li_sparse_reciprocal", "ross_thick"]

#: MODIS's LiSparse crown shape: relative height h/b (b/r = 1).
_H_OVER_B = 2.0


def ross_thick(geometry: ViewingGeometry) -> Float[Array, ""]:
    """The RossThick volume-scattering kernel ``K_vol`` (dimensionless).

    Args:
        geometry: Sun and sensor directions.

    Returns:
        ``K_vol``, 0 at nadir/nadir.

    Examples:
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> round(float(ross_thick(ViewingGeometry(1.0, 1.0))), 12)
        0.0
    """
    mu0, mu = geometry.mu0, geometry.mu
    cos_xi = _cos_phase(geometry)
    return _ross_numerator(cos_xi) / (mu0 + mu) - jnp.pi / 4


def li_sparse_reciprocal(geometry: ViewingGeometry) -> Float[Array, ""]:
    """The LiSparse-Reciprocal geometric kernel ``K_geo`` (dimensionless), MODIS's shape.

    Args:
        geometry: Sun and sensor directions.

    Returns:
        ``K_geo``, 0 at nadir/nadir.

    Examples:
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> round(float(li_sparse_reciprocal(ViewingGeometry(1.0, 1.0))), 12)
        0.0
    """
    mu0, mu, phi = geometry.mu0, geometry.mu, geometry.relative_azimuth_rad
    sec0, sec1 = 1.0 / mu0, 1.0 / mu
    tan0, tan1 = _safe_sqrt(1.0 - mu0**2) / mu0, _safe_sqrt(1.0 - mu**2) / mu
    tan01 = tan0 * tan1
    # D² = tan²θ₀ + tan²θ − 2 tanθ₀ tanθ cos φ, rearranged with
    # 1 − cos φ = 2 sin²(φ/2) so it does not cancel near the hotspot.
    d_sq = (tan0 - tan1) ** 2 + 4.0 * tan01 * jnp.sin(phi / 2) ** 2
    root = _safe_sqrt(d_sq + (tan01 * jnp.sin(phi)) ** 2)
    cos_t = _H_OVER_B * root / (sec0 + sec1)
    overlap = _overlap_shape(cos_t) * (sec0 + sec1) / jnp.pi
    cos_xi = _cos_phase(geometry)
    return overlap - sec0 - sec1 + 0.5 * (1.0 + cos_xi) * sec0 * sec1


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
        return (
            self.f_iso
            + self.f_vol * ross_thick(geometry)
            + self.f_geo * li_sparse_reciprocal(geometry)
        )


def _cos_phase(geometry: ViewingGeometry) -> Float[Array, ""]:
    """``cos ξ = μ₀ μ + sin θ₀ sin θ cos φ``."""
    mu0, mu = geometry.mu0, geometry.mu
    sines = _safe_sqrt(1.0 - mu0**2) * _safe_sqrt(1.0 - mu**2)
    return mu0 * mu + sines * jnp.cos(geometry.relative_azimuth_rad)


def _safe_sqrt(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """``√x`` for ``x ≥ 0`` with gradient 0 at ``x = 0`` (double-where), 0 below."""
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


@jax.custom_jvp
def _ross_numerator(c: Float[Array, ""]) -> Float[Array, ""]:
    """``(π/2 − ξ) cos ξ + sin ξ`` as ``c · arcsin c + √(1 − c²)``, ``c = cos ξ``."""
    c = jnp.clip(c, -1.0, 1.0)
    return c * jnp.arcsin(c) + jnp.sqrt(jnp.maximum(1.0 - c**2, 0.0))


@_ross_numerator.defjvp
def _ross_numerator_jvp(
    primals: tuple[Float[Array, ""]], tangents: tuple[Float[Array, ""]]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    (c,), (dc,) = primals, tangents
    return _ross_numerator(c), jnp.arcsin(jnp.clip(c, -1.0, 1.0)) * dc


@jax.custom_jvp
def _overlap_shape(u: Float[Array, ""]) -> Float[Array, ""]:
    """``t − sin t cos t`` as ``arccos u − u √(1 − u²)``, ``u = cos t`` clipped to ``[−1, 1]``."""
    u = jnp.clip(u, -1.0, 1.0)
    return jnp.arccos(u) - u * jnp.sqrt(jnp.maximum(1.0 - u**2, 0.0))


@_overlap_shape.defjvp
def _overlap_shape_jvp(
    primals: tuple[Float[Array, ""]], tangents: tuple[Float[Array, ""]]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    (u,), (du,) = primals, tangents
    # Beyond ±1 the clip holds the value constant, and √(1 − 1) = 0 there too.
    return _overlap_shape(u), -2.0 * jnp.sqrt(jnp.maximum(1.0 - u**2, 0.0)) * du
