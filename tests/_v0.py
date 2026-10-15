"""radtrax · 3's v0 slice on the synthetic tables, shared by the forward
model's tests and the v0 gates (tests/validation).

AFGL US Standard to 20 km (20 one-kilometre layers, inside the tables'
0.03–1.1 atm), CH₄ from the 150-line Voigt table and a second synthetic
table (another seed) standing in for H₂O, Beer–Lambert, EMIT-like bands.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import xarray as xr
from _bands import emit_like_srf
from _synthetic_table import NU, synthetic_table

from radtrax import ForwardModel
from radtrax.atmosphere import AtmosphericState, ViewingGeometry, afgl_us_standard
from radtrax.instrument import BandInstrument
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.surface import Lambertian


GEOMETRY = ViewingGeometry.from_angles(35.0, 10.0)
SURFACE = Lambertian(0.25)


def v0_model(voigt_table: xr.Dataset) -> ForwardModel:
    """TableOpacity (CH₄, "H₂O") → BeerLambert → BandInstrument."""
    opacity = TableOpacity.from_datasets(
        {"CH4": voigt_table, "H2O": synthetic_table(seed=1)}
    )
    instrument = BandInstrument.from_srf(emit_like_srf(NU), NU)
    return ForwardModel(opacity, BeerLambert(), instrument)


def v0_atmosphere() -> AtmosphericState:
    """AFGL US Standard to 20 km with CH₄ and H₂O."""
    return afgl_us_standard(z_top_km=20.0, gases=["CH4", "H2O"])


def theta_to_bands(
    model: ForwardModel, atm: AtmosphericState
) -> Callable[[jax.Array], jax.Array]:
    """y(θ), θ = (ΔX [mol/m²], albedo, shift [cm⁻¹]) (3,) → (B,): the K of
    radtrax · 3's swath check."""

    def y_of(theta: jax.Array) -> jax.Array:
        fm = eqx.tree_at(lambda m: m.instrument.shift_cm, model, theta[2])
        return fm(atm.add_column("CH4", theta[0]), Lambertian(theta[1]), GEOMETRY)

    return y_of
