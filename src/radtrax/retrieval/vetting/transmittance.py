"""The vetting transmittance: a scene's band transmittance on a grid of columns.

``E_b(ΔX) = y_b(x + ΔX e) / y_b(x)`` from the same :class:`radtrax.ForwardModel`
the matched filter's target comes from (:func:`radtrax.unit_absorption_target`),
so the fit and the filter see the same gas and the curve of growth is built
in. It replaces the reference code's MODTRAN tables. Theory: the
plume-vetting page, section 2.

Shapes: ``D`` column enhancements, ``B`` bands, ``L`` layers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.atmosphere.state import AtmosphericState
from radtrax.exceptions import GridError, MissingKeyError, ShapeError
from radtrax.forward_model import ForwardModel
from radtrax.retrieval.matched_filter.target import unit_absorption_target
from radtrax.surface.lambertian import Lambertian


__all__ = ["transmittance_table"]


def _check_delta_grid(delta_grid: Float[Array, " D"]) -> None:
    """Raise unless ``delta_grid`` is 1-D, starts at 0 and strictly increases.

    The values are checked when concrete; a traced grid (under ``jit``) has
    nothing to inspect.
    """
    if delta_grid.ndim != 1 or delta_grid.shape[0] < 1:
        raise ShapeError(
            f"transmittance_table: delta_grid must be (D,) with D ≥ 1, "
            f"got {delta_grid.shape}."
        )
    try:
        values = np.asarray(delta_grid)
    except jax.errors.TracerArrayConversionError:
        return
    if not (values[0] == 0 and np.all(np.diff(values) > 0)):
        raise GridError("transmittance_table: delta_grid must start at 0 and increase.")


def transmittance_table(
    model: ForwardModel,
    state: AtmosphericState,
    geometry: ViewingGeometry,
    delta_grid: Float[ArrayLike, " D"],
    *,
    gas: str = "CH4",
    profile: Float[ArrayLike, " L"] | None = None,
    linear: bool = False,
) -> Float[Array, "D B"]:
    """The band transmittance ``E_b(ΔX)`` of a column enhancement, on a grid.

    .. math::
        E_b(\\Delta X) = \\frac{y_b(x + \\Delta X\\, e)}{y_b(x)},
        \\qquad
        E^{\\text{lin}}_b(\\Delta X) = e^{-\\varepsilon_b \\Delta X},
        \\quad
        \\varepsilon_b = -\\left.\\frac{\\partial \\ln y_b}{\\partial \\Delta X}
        \\right|_0 ,

    with ``y`` the model's band values for a unit Lambertian albedo, ``ΔX``
    added by :meth:`~radtrax.atmosphere.AtmosphericState.add_column` with
    weights ``e`` from ``profile``, and ``ε_b = −t_b`` from
    :func:`radtrax.unit_absorption_target`. Beer–Lambert radiance is linear
    in a Lambertian albedo, so the albedo cancels in the ratio exactly per
    wavenumber, and after band integration to the extent the albedo is flat
    across a band's response. ``E(0) = 1`` exactly; ``E`` is decreasing and
    convex in ``ΔX``, and ``E^lin ≤ E`` (Jensen): a constant-coefficient fit
    of a strong plume reads its column low.

    One forward model per grid point, ``jax.vmap`` over the grid. It is
    differentiable in the model's leaves (the spectral shift, say), the
    state and the geometry.

    Args:
        model: The forward model.
        state: The background atmosphere.
        geometry: Sun and sensor directions.
        delta_grid: Column enhancements ``ΔX`` [mol/m²], starting at 0 and
            strictly increasing, ``(D,)``; checked when concrete.
        gas: The enhanced gas.
        profile: Relative weight of the enhancement per layer, ``(L,)``; the
            default puts it all in the bottom layer.
        linear: Return the constant-coefficient model ``E^lin`` instead.

    Returns:
        ``E``, one row per enhancement, ``(D, B)``.

    Raises:
        MissingKeyError: If the state carries no ``gas``.
        ShapeError: If ``delta_grid`` is not ``(D,)``.
        GridError: If ``delta_grid`` does not start at 0 or does not
            strictly increase.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax import ForwardModel
        >>> from radtrax.atmosphere import ViewingGeometry, afgl_us_standard
        >>> from radtrax.instrument import BandInstrument
        >>> from radtrax.solvers import BeerLambert
        >>> from radtrax.spectroscopy import CrossSectionTable, TableOpacity
        >>> nu = jnp.array([4300.0, 4300.1, 4300.2])
        >>> sigma = jnp.array([1e-21, 4e-21, 1e-21])  # cm²/molecule
        >>> table = CrossSectionTable(
        ...     jnp.log(jnp.array([1e3, 1.2e5])),
        ...     jnp.array([200.0, 310.0]),
        ...     jnp.broadcast_to(sigma, (2, 2, 3)),
        ... )
        >>> model = ForwardModel(
        ...     TableOpacity(nu, {"CH4": table}),
        ...     BeerLambert(),
        ...     BandInstrument(nu, jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])),
        ... )
        >>> state = afgl_us_standard(z_top_km=3.0, gases=["CH4"])
        >>> geometry = ViewingGeometry(1.0, 1.0)
        >>> grid = jnp.array([0.0, 0.1, 0.2])  # mol/m²
        >>> exact = transmittance_table(model, state, geometry, grid)
        >>> lin = transmittance_table(model, state, geometry, grid, linear=True)
        >>> exact.shape, bool(jnp.all(exact[0] == 1.0))
        ((3, 2), True)
        >>> # One wavenumber: E is exactly exponential; two: the curve bends.
        >>> float(jnp.max(jnp.abs(exact[:, 0] - lin[:, 0]))) < 1e-12
        True
        >>> bool(jnp.all(exact[1:, 1] > lin[1:, 1]))
        True
    """
    if gas not in state.vmr:
        raise MissingKeyError(
            f"transmittance_table: the state has no mixing ratio for {gas!r} "
            f"(have {sorted(state.vmr)})."
        )
    delta = jnp.asarray(delta_grid, dtype=state.vmr[gas].dtype)
    _check_delta_grid(delta)
    surface = Lambertian(1.0)
    if linear:
        # t = ∂ ln y / ∂ΔX |₀ = −ε;  E^lin = exp(ΔX t)                     (D, B)
        t = unit_absorption_target(
            model, state, surface, geometry, gas=gas, profile=profile
        )
        return jnp.exp(jnp.outer(delta, t))

    def bands(d: Float[Array, ""]) -> Float[Array, " B"]:
        return model(state.add_column(gas, d, profile), surface, geometry)

    # y(ΔX_d) for every d; ΔX_0 = 0, so row 0 is the background     (D, B)
    y = jax.vmap(bands)(delta)
    # E_d = y_d / y_0, with row 0 set to 1 exactly (XLA may divide by a
    # reciprocal, which leaves y_0 / y_0 one ulp off)                  (D, B)
    return jnp.concatenate([jnp.ones_like(y[:1]), y[1:] / y[0]])
