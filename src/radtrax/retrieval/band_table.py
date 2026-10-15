"""Band-table inversion: a band observable ``g`` tabulated against the enhancement.

*radtrax · 1* (stage 4) and *radtrax · 2* invert a band observable through a
table rather than a linear slope, which the curve of growth biases low:

- one band, ``g = τ₁₂(ΔΩ) = −ln(T₁₂(Ω_bg + ΔΩ) / T₁₂(Ω_bg))`` (SBMP);
- two bands, ``g = τ₁₂ − τ₁₁`` (MBSP and MBMP).

:class:`BandTable` holds ``g`` on a grid of two-way air-mass factors ``m``
(and, optionally, water-vapour scalings ``h``) against the enhancement
``ΔΩ``. It interpolates linearly in ``m`` and ``h`` and piecewise-linearly in
``ΔΩ``, so the inverse is exact on the same nodes:
``invert(table(ΔΩ, m), m) = ΔΩ`` to rounding. Noise makes ``g`` negative off a
plume, so ``g`` is extended as an odd function of ``ΔΩ`` (as both notes do).
Everything is JAX and differentiable in ``g``, ``m``, ``h`` and the table.

Build a table from any observable with :meth:`BandTable.from_function` (the
notes' monochromatic band transmittances) or from the layered forward model
with :func:`band_table_from_model` (``ΔΩ`` then in mol/m², ``h`` scales the
H₂O profile).

Shapes: ``M`` air masses, ``H`` water-vapour scalings, ``D`` enhancements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.atmosphere.state import AtmosphericState
from radtrax.exceptions import GridError, ParameterError, ShapeError
from radtrax.forward_model import ForwardModel
from radtrax.surface.lambertian import Surface


__all__ = ["BandTable", "band_table_from_model"]


class BandTable(eqx.Module):
    """A band observable ``g(m[, h], ΔΩ)``, monotone in ``ΔΩ``, and its inverse.

    Attributes:
        m_grid: Two-way air-mass factors, increasing, ``(M,)``; one value
            makes a table for a single geometry.
        delta_grid: Enhancements, increasing from 0, ``(D,)``, in the units
            the table was built in (kg/m² in the notes, mol/m² from the
            forward model).
        g: The observable, strictly increasing in ``ΔΩ``, ``(M, D)``, or
            ``(M, H, D)`` with a water-vapour axis.
        h2o_grid: Water-vapour scalings, increasing, ``(H,)``, or ``None``.

    Examples:
        >>> import jax.numpy as jnp
        >>> delta = jnp.linspace(0.0, 0.4, 41)
        >>> table = BandTable(
        ...     jnp.array([2.0, 3.0]), delta, jnp.stack([0.3 * delta, 0.45 * delta])
        ... )
        >>> round(float(table(0.2, 2.5)), 6)  # g at m = 2.5, halfway
        0.075
        >>> round(float(table.invert(0.075, 2.5)), 6)
        0.2
    """

    m_grid: Float[Array, " M"]
    delta_grid: Float[Array, " D"]
    g: Float[Array, "M *H D"]
    h2o_grid: Float[Array, " H"] | None

    def __init__(
        self,
        m_grid: Float[ArrayLike, " M"],
        delta_grid: Float[ArrayLike, " D"],
        g: Float[ArrayLike, "M *H D"],
        h2o_grid: Float[ArrayLike, " H"] | None = None,
    ) -> None:
        self.m_grid = jnp.asarray(m_grid, dtype=float)
        self.delta_grid = jnp.asarray(delta_grid, dtype=float)
        self.g = jnp.asarray(g, dtype=float)
        self.h2o_grid = None if h2o_grid is None else jnp.asarray(h2o_grid, dtype=float)

    def __check_init__(self) -> None:
        n_m, n_d = jnp.shape(self.m_grid), jnp.shape(self.delta_grid)
        h = () if self.h2o_grid is None else jnp.shape(self.h2o_grid)
        if len(n_m) != 1 or len(n_d) != 1 or len(h) > 1:
            raise ShapeError("BandTable: m_grid, delta_grid and h2o_grid must be 1-D.")
        if jnp.shape(self.g) != (*n_m, *h, *n_d):
            raise ShapeError(
                f"BandTable: g must be (M, [H,] D) = {(*n_m, *h, *n_d)}, got {jnp.shape(self.g)}."
            )
        # Each check runs in NumPy on its own arrays, so a concrete grid is
        # checked even under jit (where jnp operations on it would be traced).
        delta = _concrete(self.delta_grid)
        if delta is not None and not (delta[0] == 0 and np.all(np.diff(delta) > 0)):
            raise GridError("BandTable: delta_grid must start at 0 and increase.")
        for grid in (self.m_grid, self.h2o_grid):
            values = None if grid is None else _concrete(grid)
            if values is not None and not np.all(np.diff(values) > 0):
                raise GridError("BandTable: m_grid and h2o_grid must increase.")
        g = _concrete(self.g)
        if g is not None and not np.all(np.diff(g, axis=-1) > 0):
            raise GridError(
                "BandTable: g must increase strictly with ΔΩ to be inverted."
            )

    def row(
        self, m: Float[ArrayLike, ""], h2o: Float[ArrayLike, ""] | None = None
    ) -> Float[Array, " D"]:
        """``g`` against ``ΔΩ`` at ``m`` (and ``h``), linear between the nodes, ``(D,)``."""
        g = _lerp(self.g, self.m_grid, m)
        if self.h2o_grid is not None:
            if h2o is None:
                raise ParameterError(
                    "BandTable: this table has a water-vapour axis; pass h2o."
                )
            g = _lerp(g, self.h2o_grid, h2o)
        return g

    def __call__(
        self,
        delta: Float[ArrayLike, "*shape"],
        m: Float[ArrayLike, ""],
        h2o: Float[ArrayLike, ""] | None = None,
    ) -> Float[Array, "*shape"]:
        """``g(ΔΩ)`` at ``m`` (and ``h``), odd in ``ΔΩ``; held at the table's end beyond it."""
        row = self.row(m, h2o)
        return _odd(
            lambda x: jnp.interp(x, self.delta_grid, row),
            jnp.asarray(delta, dtype=float),
        )

    def invert(
        self,
        g: Float[ArrayLike, "*shape"],
        m: Float[ArrayLike, ""],
        h2o: Float[ArrayLike, ""] | None = None,
    ) -> Float[Array, "*shape"]:
        """``ΔΩ = g⁻¹(g)`` at ``m`` (and ``h``), odd in ``g``; held at the table's end beyond it."""
        row = self.row(m, h2o)
        return _odd(
            lambda x: jnp.interp(x, row, self.delta_grid), jnp.asarray(g, dtype=float)
        )

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Float[ArrayLike, " D"]],
        m_grid: Float[ArrayLike, " M"],
        delta_grid: Float[ArrayLike, " D"],
        h2o_grid: Float[ArrayLike, " H"] | None = None,
    ) -> BandTable:
        """Tabulate ``fn(m, delta_grid)`` (or ``fn(m, delta_grid, h)``), each ``(D,)``."""
        m_grid, delta_grid = (
            np.asarray(m_grid, dtype=float),
            np.asarray(delta_grid, dtype=float),
        )
        if h2o_grid is None:
            g = np.stack([np.asarray(fn(m, delta_grid), dtype=float) for m in m_grid])
        else:
            h2o_grid = np.asarray(h2o_grid, dtype=float)
            g = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(fn(m, delta_grid, h), dtype=float)
                            for h in h2o_grid
                        ]
                    )
                    for m in m_grid
                ]
            )
        return cls(m_grid, delta_grid, g, h2o_grid)


def band_table_from_model(
    model: ForwardModel,
    state: AtmosphericState,
    surface: Surface,
    *,
    bands: Sequence[int],
    m_grid: Float[ArrayLike, " M"],
    delta_grid: Float[ArrayLike, " D"],
    h2o_grid: Float[ArrayLike, " H"] | None = None,
    gas: str = "CH4",
    h2o_gas: str = "H2O",
    mu: float = 1.0,
) -> BandTable:
    """A :class:`BandTable` from the layered forward model.

    ``τ_b(ΔΩ) = −ln(y_b(ΔΩ) / y_b(0))`` with ``ΔΩ`` [mol/m²] of ``gas`` added
    by :meth:`~radtrax.atmosphere.AtmosphericState.add_column`;
    ``g = τ_{b₀}`` for one band, ``g = τ_{b₀} − τ_{b₁}`` for two. Each
    ``m`` is a geometry with the sensor at ``mu`` and the sun at
    ``μ₀ = 1 / (m − 1/μ)``; each ``h`` scales the state's ``h2o_gas``
    profile.

    Args:
        model: The forward model.
        state: The background state.
        surface: The surface.
        bands: One band index (SBMP) or two, strong then weak (MBSP, MBMP).
        m_grid: Two-way air-mass factors, each at least ``1 + 1/mu``, ``(M,)``.
        delta_grid: Enhancements [mol/m²], increasing from 0, ``(D,)``.
        h2o_grid: Water-vapour scalings, ``(H,)``, or ``None``.
        gas: The enhanced gas.
        h2o_gas: The water-vapour gas the scalings apply to.
        mu: Cosine of the viewing zenith angle.

    Returns:
        The table.
    """
    if len(bands) not in (1, 2):
        raise ParameterError("band_table_from_model: give one band or two.")
    m_grid = np.asarray(m_grid, dtype=float)
    if np.any(m_grid < 1.0 + 1.0 / mu):
        raise ParameterError(
            f"band_table_from_model: every m must be at least 1 + 1/mu = {1 + 1 / mu}."
        )
    delta_grid = jnp.asarray(delta_grid, dtype=float)

    @eqx.filter_jit
    def observables(
        base: AtmosphericState, geometry: ViewingGeometry, deltas: Float[Array, " D"]
    ) -> Float[Array, " D"]:
        # One compilation for every (m, h): the state and geometry are arguments.
        y0 = model(base, surface, geometry)

        def tau(delta: Float[Array, ""]) -> Float[Array, ""]:
            t = -jnp.log(model(base.add_column(gas, delta), surface, geometry) / y0)
            return t[bands[0]] - t[bands[1]] if len(bands) == 2 else t[bands[0]]

        return jax.vmap(tau)(deltas)

    def observable(m: float, h: float | None) -> np.ndarray:
        base = (
            state
            if h is None
            else eqx.tree_at(lambda s: s.vmr[h2o_gas], state, h * state.vmr[h2o_gas])
        )
        geometry = ViewingGeometry(1.0 / (m - 1.0 / mu), mu)
        return np.asarray(observables(base, geometry, delta_grid))

    if h2o_grid is None:
        return BandTable.from_function(
            lambda m, _: observable(m, None), m_grid, delta_grid
        )
    return BandTable.from_function(
        lambda m, _, h: observable(m, h), m_grid, delta_grid, h2o_grid
    )


def _concrete(values: Float[Array, "*shape"]) -> np.ndarray | None:
    """``values`` as NumPy when concrete; ``None`` when traced (nothing to inspect)."""
    try:
        return np.asarray(values)
    except jax.errors.TracerArrayConversionError:
        return None


def _lerp(
    values: Float[Array, " K *rest"], grid: Float[Array, " K"], x: Float[ArrayLike, ""]
) -> Float[Array, "*rest"]:
    """Linear interpolation along the first axis, held at the ends; one node is that node."""
    if values.shape[0] == 1:
        return values[0]
    position = jnp.interp(x, grid, jnp.arange(grid.shape[0], dtype=float))
    lo = jnp.clip(jnp.floor(position).astype(int), 0, grid.shape[0] - 2)
    w = position - lo
    return (1.0 - w) * values[lo] + w * values[lo + 1]


def _odd(f: Callable[[Array], Array], x: Array) -> Array:
    """``f`` extended as an odd function, with ``f′(0)`` as the gradient at 0."""
    return jnp.where(x >= 0, f(x), -f(-x))
