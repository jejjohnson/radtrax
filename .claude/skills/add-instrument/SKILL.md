---
name: add-instrument
description: Add a new Instrument (radiance on a wavenumber grid in, one value per band out, linear in radiance) to radtrax — an abstract Equinox module with an eqx.AbstractVar wavenumber grid, a response_matrix that matches the Jacobian of the call, jaxtyping shapes, a gradient check in the radiance and the instrument parameters, tests beside test_spectral.py, the API page and the capability index. Use when asked to add a new spectral response model, a band-integration scheme or a sensor's instrument (and its geo constructor).
---

# Add an instrument

Read "The contracts" in `AGENTS.md` first. The protocol is `Instrument` in `src/radtrax/instrument/spectral.py`; the implementations to copy from are `BandInstrument` (a response matrix on the grid with a spectral shift) and `GaussianInstrument` (Gaussian bands built in JAX, shifted analytically in cm⁻¹ or nm), both in the same file. `radtrax.instrument.srf` (`SpectralResponseFunction`) is the one SRF band integration; do not write a second.

## 1. Check it does not exist

Search `docs/api/capabilities.md`. A sensor with tabulated responses is `BandInstrument.from_srf(srf, nu_cm)`, with Gaussian bands `GaussianInstrument.from_srf`; neither needs a new class. A new class is for a new way of forming the response (a different line shape, a parametrised slit). A sensor-specific constructor (`sentinel2_instrument`, `emit_instrument`, `landsat_instrument`) belongs in `src/radtrax/geo/` and returns one of these classes.

## 2. Write the class

The contract: an instrument is linear in radiance, `y = A L` with `A` the response matrix `(B, G)`, so `I(a L₁ + b L₂) = a I(L₁) + b I(L₂)`. Its parameters (a shift, a FWHM) may enter `A` non-linearly. Band brightness temperature and NEΔT (`radtrax.radiometry.brightness`) rely on the linearity.

- Subclass `Instrument`. The base declares `nu_cm: eqx.AbstractVar[Float[Array, " G"]]` and an `abc.abstractmethod` `__call__(self, radiance: Float[ArrayLike, " G"]) -> Float[Array, " B"]`; the subclass declares `nu_cm` as a field (strictly increasing, [cm⁻¹]).
- `response_matrix()` returns `A = ∂y/∂L`, `(B, G)`. The base default is `jax.jacrev` of the call at zero radiance, exact for a linear instrument and `B` reverse passes; override it with the closed form when you hold `A` (`GaussianInstrument` returns `self.response()`).
- Parameters (`shift_cm`, `fwhm_nm`) are differentiable leaves, scalars or `(B,)`. Band labels are the one plain-Python field and the only `eqx.field(static=True)` (a `tuple[str, ...]`); arrays are never static.
- Float64, converted in `__init__`. Interpolate relative to `ν − ν₀` so a shift is resolved to its own precision rather than the ulp of ν (`BandInstrument.__call__`).
- Validate in `__check_init__` (`ShapeError`, `GridError`, `ParameterError` from `radtrax/exceptions.py`), checking a grid is increasing only when it is concrete.
- A band shifted off the grid has no response left: fail loudly, also under `jit`, with `eqx.error_if` (it raises `equinox.EquinoxRuntimeError`, a `RuntimeError`; say so in the docstring) rather than return 0/0. `GaussianInstrument.response` is the pattern.
- Radiance is per cm⁻¹ by default; a per-µm band value (`per_um=True` in `from_srf`) folds `ν²/10⁴` into the response so the call stays linear. Say in the docstring which unit goes in and out.

A minimal instrument, run against the current code:

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from radtrax.instrument import Instrument


class BoxcarInstrument(Instrument):
    """Mean radiance over ``[edge_b, edge_{b+1})`` [cm⁻¹], one band per pair of edges."""

    nu_cm: Float[Array, " G"]
    response: Float[Array, "B G"]

    def __init__(self, nu_cm, edges_cm) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        edges = jnp.asarray(edges_cm, dtype=float)
        inside = (self.nu_cm >= edges[:-1, None]) & (self.nu_cm < edges[1:, None])
        self.response = inside / jnp.sum(inside, axis=1, keepdims=True)  # (B, G)

    def __call__(self, radiance: Float[ArrayLike, " G"]) -> Float[Array, " B"]:
        return self.response @ jnp.asarray(radiance)

    def response_matrix(self) -> Float[Array, "B G"]:
        return self.response


nu = jnp.linspace(4250.0, 4350.0, 101)
instrument = BoxcarInstrument(nu, jnp.array([4250.0, 4300.0, 4351.0]))
print(instrument(jnp.ones(101)))  # a flat spectrum reads 1 in every band
print(
    bool(jnp.allclose(instrument.response_matrix(), Instrument.response_matrix(instrument)))
)
```

## 3. Compose it

`ForwardModel(opacity, solver, instrument)` requires the instrument's `nu_cm` to equal the opacity's (`GridError`), so build it on `opacity.nu_cm`. A swath maps `eqx.filter_vmap` over pixels outside `jax.jacrev`. A noise model (`add-noise-model`) takes the band values this returns.

## 4. Tests

In `tests/instrument/test_<name>.py`, following `tests/instrument/test_spectral.py` and `test_gaussian_instrument.py`.

- The protocol cannot be built (`pytest.raises(TypeError)`).
- Parity with the one SRF integration (`test_unshifted_bands_equal_the_numpy_srf`) at zero shift, and a flat spectrum reads flat in every band.
- Linearity (`test_band_instrument_is_linear_in_radiance`) and `response_matrix` equal to the Jacobian of the call (`test_response_matrix_is_the_jacobian_of_the_call`), at a nonzero parameter.
- The gradient check, for every public differentiable function: `from _gradcheck import assert_gradients_match`, in the radiance (the gradient is the response) and in each parameter, between grid nodes and on a node where a kink makes the derivative one-sided (`test_gradient_in_the_shift_between_grid_nodes`, `test_on_a_node_the_derivative_is_the_slope_above_it`).
- Shape and grid errors with the `shapes_unchecked` fixture (`tests/conftest.py`), since jaxtyping's run-time check would fire first; a band off the grid raises under `eqx.filter_jit`.
- Under `eqx.filter_jit`, and `eqx.filter_vmap` over shifts matching a loop.
- Parity against NumPy is near rounding; never compare independently computed floats across machines tighter than about `rtol=1e-12`, and state a reason for each tolerance.

## 5. Make it public

The class goes in the module's sorted `__all__`, then `src/radtrax/instrument/__init__.py` (import it, add it and, for a new module, the module name to `__all__`, sorted). Add a `## <name>` block to `docs/api/instrument.md` for a new module and mention the class in its intro; update the Instrument row of `docs/api/index.md`, the `instrument/` row of the `AGENTS.md` module table and `docs/llms.txt`.

## Checklist (every extension point)

- [ ] A gradient check (`assert_gradients_match`) for every public differentiable function, finite at the edges of the valid domain.
- [ ] A missing optional dependency raises `MissingExtraError` naming the extra (the `[geo]` extra for a sensor constructor in `radtrax/geo/`), with the import inside the function and `raise … from err`.
- [ ] The module declares a sorted `__all__`, and the name is also in the `__all__` of `instrument/__init__.py`. `tests/test_public_api.py` fails otherwise (`SUBMODULES` lists only top-level modules and `SUBPACKAGES` the subpackages, so a new file in `instrument/` needs no edit there).
- [ ] Units in names or docstrings (`_cm`, `_nm`, per cm⁻¹ or per µm); jaxtyping shapes on every public argument and return, dimension strings quoted with a leading space.
- [ ] Arrays are Equinox leaves, never `eqx.field(static=True)`.
- [ ] Iterative solves stay in their valid domain (`band_brightness_temperature` solves in `ln T` so no Newton step leaves `T_b > 0`); anything that can leave the grid fails loudly.
- [ ] No float comparison between independently computed values tighter than about 1e-12, and every tolerance has a stated reason.
- [ ] `make capabilities`.
- [ ] The docs API page (`docs/api/instrument.md`, the `docs/api/index.md` row), `docs/llms.txt` and the `AGENTS.md` module table.
- [ ] `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `make typecheck`, `uv run pytest --no-cov tests/instrument`, `make docs-api`.
- [ ] Run the `reuse-reviewer` and `numerics-reviewer` subagents (`radtrax-review`) on the diff before committing.
