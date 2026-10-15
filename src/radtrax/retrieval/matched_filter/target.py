"""Target signature construction from the radiance forward model.

The matched-filter target ``t(ν)`` is the spectral signature that a unit
plume enhancement would leave in the observed radiance. Because we already
have a JAX :class:`radtrax.retrieval.var3d.RadianceObservationModel`,
we do *not* reimplement the Beer–Lambert derivation from scratch (1379 lines
in the original snippet). Instead:

- **Linear target** is a directional derivative ``∂L/∂VMR`` at the background
  state, obtained by a single :func:`jax.jvp` call — robust, cheap, and
  exactly consistent with the forward model used by the 3D-Var retrieval.
- **Nonlinear target** is the finite-amplitude response
  ``L(x_b + α·dx) − L(x_b)``, which captures Beer–Lambert saturation for
  strong plumes. For small ``α`` this reduces to ``α · t_linear`` and the MF
  score becomes scale-invariant to that choice.

- **Unit-absorption target** :func:`unit_absorption_target` is
  ``∂ ln y / ∂ΔX`` per pixel on the layered :class:`radtrax.ForwardModel`
  (radtrax#37): the scene's own target, replacing a MODTRAN table.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/matched_filter/target.py``); the read-out
pixel follows a GSD since radtrax#80.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


if TYPE_CHECKING:
    from jaxtyping import ArrayLike

    from radtrax.atmosphere.geometry import ViewingGeometry
    from radtrax.atmosphere.state import AtmosphericState
    from radtrax.forward_model import ForwardModel
    from radtrax.retrieval.var3d.obs_operator import RadianceObservationModel
    from radtrax.surface.lambertian import Surface


Pattern = Literal["uniform", "impulse"]


def _build_pattern(
    vmr_field: Float[Array, "H W"],
    pattern: Pattern | Float[Array, "H W"],
    pixel: tuple[int, int] | None,
) -> Float[Array, "H W"]:
    if isinstance(pattern, str):
        if pattern == "uniform":
            return jnp.ones_like(vmr_field)
        if pattern == "impulse":
            h, w = vmr_field.shape
            i, j = pixel if pixel is not None else (h // 2, w // 2)
            if not (0 <= i < h and 0 <= j < w):
                raise ValueError(
                    f"target: pixel {(i, j)} is out of bounds for vmr_field "
                    f"shape {vmr_field.shape}."
                )
            return jnp.zeros_like(vmr_field).at[i, j].set(1.0)
        raise ValueError(
            f"target: pattern must be 'uniform', 'impulse', or an array; got {pattern!r}."
        )
    arr = jnp.asarray(pattern)
    if arr.shape != vmr_field.shape:
        raise ValueError(
            f"target: pattern array shape {arr.shape} must match vmr_field shape "
            f"{vmr_field.shape}."
        )
    return arr


def _vmr_pixel(
    vmr_field: Float[Array, "H W"],
    pattern: Pattern | Float[Array, "H W"],
    pixel: tuple[int, int] | None,
) -> tuple[int, int]:
    """The pixel the target is read at, on the **VMR** grid.

    Resolution rules (explicit first):

    1. If ``pixel`` is given, use it (after bounds checking).
    2. Otherwise, for ``'uniform'`` and ``'impulse'``, the scene centre — the
       impulse's default location.
    3. For a custom array pattern, the pixel with the largest absolute weight
       (``argmax(|pattern|)``), its natural "perturbation location".
    """
    h, w = vmr_field.shape
    if pixel is not None:
        i, j = pixel
    elif isinstance(pattern, str):
        i, j = h // 2, w // 2
    else:
        idx = int(jnp.argmax(jnp.abs(jnp.asarray(pattern))))
        i, j = idx // w, idx % w
    if not (0 <= i < h and 0 <= j < w):
        raise ValueError(
            f"target: pixel {(i, j)} is out of bounds for vmr_field shape ({h}, {w})."
        )
    return i, j


def _read_out(
    cube: Float[Array, "h w B"],
    vmr_pixel: tuple[int, int],
    obs: RadianceObservationModel,
) -> Float[Array, " B"]:
    """The response at the output pixel that contains ``vmr_pixel``.

    A GSD block-averages ``f × f`` VMR pixels into one output pixel, so VMR
    pixel ``(i, j)`` lands in output pixel ``(i // f, j // f)``; without a
    GSD ``f = 1`` and the grids coincide (radtrax#80).
    """
    factor = 1 if obs.gsd is None else int(obs.gsd.downsample_factor)
    i, j = vmr_pixel[0] // factor, vmr_pixel[1] // factor
    h, w, _ = cube.shape
    if not (0 <= i < h and 0 <= j < w):
        raise ValueError(
            f"target: VMR pixel {vmr_pixel} maps to output pixel {(i, j)}, out of "
            f"bounds for the output shape ({h}, {w}): the GSD crops the VMR rows "
            "and columns past the last whole block."
        )
    return cube[i, j]


def linear_target_from_obs(
    obs: RadianceObservationModel,
    vmr_background: Float[Array, "H W"],
    *,
    pattern: Pattern | Float[Array, "H W"] = "uniform",
    pixel: tuple[int, int] | None = None,
    linear_forward: bool = False,
) -> Float[Array, B]:
    """Matched-filter target via the tangent-linear of the forward operator.

    Computes

    .. math::
        t = \\frac{\\partial}{\\partial \\alpha}
            H(x_b + \\alpha \\, \\delta x) \\Big|_{\\alpha=0}

    at a single output pixel. For the default ``pattern='uniform'`` this
    captures the spectral signature of a uniform VMR enhancement across the
    scene (appropriate when the plume fills several PSF footprints). For
    ``pattern='impulse'`` it captures the instrument's point response to a
    single-pixel enhancement (appropriate when the plume is subpixel or
    compact compared to the PSF).

    Parameters
    ----------
    obs
        The JAX observation model.
    vmr_background
        Background VMR field, shape ``(H, W)``. Typically
        ``jnp.full((H, W), x_ref)`` where ``x_ref == obs.vmr_reference``.
    pattern
        ``'uniform'`` (default), ``'impulse'``, or an explicit 2-D perturbation
        pattern of the same shape as ``vmr_background``.
    pixel
        Pixel ``(i, j)`` **of the VMR grid**: the impulse location, and the
        place the signature is read — at the output pixel containing it,
        ``(i // f, j // f)`` for a GSD of factor ``f``. Defaults to the scene
        centre (or a custom pattern's peak). ``pattern`` and ``pixel`` must be
        concrete (static, or closed over under ``jax.jit``).
    linear_forward
        If True, use the linearised forward ``1 - Δτ`` instead of
        ``exp(-Δτ)``. For small ``vmr_background`` the two agree to within
        Beer–Lambert curvature.

    Returns
    -------
    Float[Array, "B"]
        Target signature of length ``obs.n_bands``.
    """
    dx = _build_pattern(vmr_background, pattern, pixel)
    fwd = obs.make_forward(linear=linear_forward)
    _, dL = jax.jvp(fwd, (vmr_background,), (dx,))
    return _read_out(dL, _vmr_pixel(vmr_background, pattern, pixel), obs)


def nonlinear_target_from_obs(
    obs: RadianceObservationModel,
    vmr_background: Float[Array, "H W"],
    *,
    amplitude: float = 1.0,
    pattern: Pattern | Float[Array, "H W"] = "uniform",
    pixel: tuple[int, int] | None = None,
) -> Float[Array, B]:
    """Matched-filter target via a finite-amplitude difference.

    Computes ``t = H(x_b + α·δx) − H(x_b)`` without linearising Beer–Lambert.
    This is the "nonlinear" variant from the original snippets; it captures
    the saturation curvature that matters for strong plumes
    (``α · σ · N · ΔX ≳ 0.1`` optical depth).

    For small ``amplitude`` this reduces to ``amplitude × linear_target``.
    """
    dx = _build_pattern(vmr_background, pattern, pixel)
    fwd = obs.make_forward(linear=False)
    L_perturbed = fwd(vmr_background + amplitude * dx)
    L_base = fwd(vmr_background)
    return _read_out(
        L_perturbed - L_base, _vmr_pixel(vmr_background, pattern, pixel), obs
    )


def unit_absorption_target(
    model: ForwardModel,
    state: AtmosphericState,
    surface: Surface,
    geometry: ViewingGeometry,
    *,
    gas: str = "CH4",
    profile: Float[ArrayLike, " L"] | None = None,
) -> Float[Array, " B"]:
    """Matched-filter target ``t = ∂ ln y / ∂ΔX`` at ``ΔX = 0`` [per mol/m²].

    One forward-mode derivative of the band log-radiance with respect to a
    column enhancement ``ΔX`` of ``gas`` spread over the layers by
    ``profile`` (default: the bottom layer, a plume at the surface; see
    :meth:`~radtrax.atmosphere.AtmosphericState.add_column`). The scene
    enters through ``state`` (water vapour, surface pressure, temperature),
    ``geometry`` and ``surface``, so ``t`` is the scene's own unit
    absorption spectrum. It is differentiable in all of them.

    Feed it to :func:`apply_image` with a cube of ``ln y``, or as ``μ ⊙ t``
    with a cube of ``y`` and background mean ``μ`` (``∂y/∂ΔX = y t``).
    :func:`linear_target_from_obs` and :func:`nonlinear_target_from_obs`
    are the targets of the single-slab
    :class:`~radtrax.retrieval.var3d.RadianceObservationModel` (2-D fields,
    PSF and GSD); this one is per pixel on the layered
    :class:`~radtrax.ForwardModel`.

    Args:
        model: The forward model.
        state: The background atmosphere.
        surface: The surface.
        geometry: Sun and sensor directions.
        gas: The plume's gas.
        profile: Relative weight of the plume per layer, ``(L,)``.

    Returns:
        ``t``, one value per band [1/(mol/m²)]; negative where the gas
        absorbs.
    """

    def log_bands(delta: Float[Array, ""]) -> Float[Array, " B"]:
        plume = state.add_column(gas, delta, profile)
        return jnp.log(model(plume, surface, geometry))

    # ΔX = 0 in the state's precision (float32 when x64 is off).
    return jax.jacfwd(log_bands)(jnp.zeros((), dtype=state.vmr[gas].dtype))
