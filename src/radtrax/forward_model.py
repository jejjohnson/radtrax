"""The layered forward model.

``ForwardModel`` composes the stages of *radtrax · 3*: an atmospheric state
through an ``OpacityProvider`` into per-layer optical depth, through an
``RTSolver`` (with a ``Surface`` and a ``ViewingGeometry``) into radiance at
the top of the atmosphere, through an ``Instrument`` into one value per
band:

.. math::
    y = I\\bigl(S(O(x), R, g)\\bigr) \\in \\mathbb{R}^B .

Every leaf is differentiable, so a retrieval's Jacobian is ``jax.jacrev``
of the model, and a swath is ``eqx.filter_vmap`` of that per-pixel Jacobian
(never a Jacobian of a batch).

:func:`radtrax.unit_absorption_target` (in
:mod:`radtrax.retrieval.matched_filter`) is v0's deliverable built on it:
the matched-filter target ``t = ∂ ln y / ∂ΔX`` for a scene.

Shapes: ``L`` layers, ``G`` wavenumbers, ``B`` bands.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.atmosphere.state import AtmosphericState
from radtrax.exceptions import GridError
from radtrax.instrument.spectral import Instrument
from radtrax.optics.properties import OpticalProperties
from radtrax.solvers.beer_lambert import RTSolver
from radtrax.spectroscopy.opacity import OpacityProvider
from radtrax.surface.lambertian import Surface


__all__ = ["ForwardModel"]


class ForwardModel(eqx.Module):
    """State → opacity → solver → instrument: band values for one pixel.

    The surface and the geometry are arguments, not fields, because they
    change from pixel to pixel; the opacity, solver and instrument are
    shared by a scene. To differentiate a module leaf (the spectral shift,
    say) and not the tables, ``eqx.tree_at`` it in from a parameter, or
    split with ``eqx.partition``. The opacity's and the instrument's grids
    are checked equal when they are concrete (not when the model is built
    inside ``jit``). It is an observation operator ``H(x)``, not a
    time-stepping ``pipekit_cycle.ForwardModel`` (no ``step``).

    Attributes:
        opacity: Per-layer optical depth from the state.
        solver: Radiance from optical depth, surface and geometry.
        instrument: Band values from radiance; its wavenumber grid is the
            opacity's.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import ViewingGeometry, layers_from_levels
        >>> from radtrax.instrument import BandInstrument
        >>> from radtrax.solvers import BeerLambert
        >>> from radtrax.spectroscopy import CrossSectionTable, TableOpacity
        >>> from radtrax.surface import Lambertian
        >>> nu = jnp.array([4300.0, 4300.1, 4300.2])
        >>> table = CrossSectionTable(
        ...     jnp.log(jnp.array([1e4, 1e5])),
        ...     jnp.array([200.0, 300.0]),
        ...     jnp.full((2, 2, 3), 1e-21),
        ... )
        >>> model = ForwardModel(
        ...     TableOpacity(nu, {"CH4": table}),
        ...     BeerLambert(),
        ...     BandInstrument(nu, jnp.full((1, 3), 1 / 3)),
        ... )
        >>> atm = layers_from_levels(
        ...     jnp.array([1e5, 5e4]),
        ...     jnp.array([288.0, 255.0]),
        ...     {"CH4": jnp.full(2, 1.9e-6)},
        ... )
        >>> model(atm, Lambertian(0.3), ViewingGeometry(1.0, 1.0)).shape
        (1,)
    """

    opacity: OpacityProvider
    solver: RTSolver
    instrument: Instrument

    def __check_init__(self) -> None:
        nu_o, nu_i = self.opacity.nu_cm, self.instrument.nu_cm
        if jnp.shape(nu_o) != jnp.shape(nu_i):
            raise GridError(
                f"ForwardModel: the opacity has {jnp.shape(nu_o)} wavenumbers, "
                f"the instrument {jnp.shape(nu_i)}."
            )
        try:
            same = bool(jnp.all(nu_o == nu_i))
        except (
            jax.errors.TracerBoolConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return
        if not same:
            raise GridError(
                "ForwardModel: the opacity and the instrument are on different "
                "wavenumber grids."
            )

    def optics(self, state: AtmosphericState) -> OpticalProperties:
        """Per-layer optical depth on the model's grid."""
        return OpticalProperties(self.opacity.nu_cm, self.opacity(state))

    def radiance(
        self, state: AtmosphericState, surface: Surface, geometry: ViewingGeometry
    ) -> Float[Array, " G"]:
        """Radiance at the top of the atmosphere, before the instrument."""
        return self.solver(self.optics(state), surface, geometry)

    def __call__(
        self, state: AtmosphericState, surface: Surface, geometry: ViewingGeometry
    ) -> Float[Array, " B"]:
        """Band values ``y = I(S(O(x), R, g))``."""
        return self.instrument(self.radiance(state, surface, geometry))
