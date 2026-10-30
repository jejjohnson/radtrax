---
name: add-opacity-provider
description: Add a new OpacityProvider (a source of per-layer optical depth τ (L, G) from an AtmosphericState) to radtrax — an abstract Equinox module with an eqx.AbstractVar wavenumber grid, jaxtyping shapes, float64, validation through the RadtraxError hierarchy, a gradient check, tests beside test_opacity.py, the API page and the capability index. Use when asked to add a continuum, collision-induced absorption, a line-by-line or correlated-k opacity, or any other way of computing optical depth from the state.
---

# Add an opacity provider

Read "The contracts" in `AGENTS.md` first. The protocol is `OpacityProvider` in `src/radtrax/spectroscopy/opacity.py`; the implementations to copy from are `TableOpacity` (σ tables interpolated in (ln p, T)) and `SumOpacity` (providers that add on one grid) in the same file.

## 1. Check it does not exist

Search `docs/api/capabilities.md` for the name and the physics. A second source of optical depth on the same grid (a continuum, CIA, Rayleigh) is usually its own provider added with `SumOpacity`, not a new field on `TableOpacity`. If it is new, it goes in `spectroscopy/opacity.py` when it is a small variation on the table opacity, or in `spectroscopy/<name>.py` when it has its own tables or maths.

## 2. Write the class

The contract: an atmospheric state in, the optical depth of each layer at each wavenumber out, `τ (L, G)` dimensionless, layers from the surface up.

- Subclass `OpacityProvider`. The base declares `nu_cm: eqx.AbstractVar[Float[Array, " G"]]` and an `abc.abstractmethod` `__call__(self, state: AtmosphericState) -> Float[Array, "L G"]`; the subclass declares `nu_cm` as an ordinary field and implements `__call__`.
- Take the column from the state, not from your own density: `state.air_column()` is `n Δz` in molecules/cm² (`(L,)`), so a cross section in cm²/molecule times `air_column() * state.vmr[gas]` is dimensionless. Gases are a `dict` keyed by name in `state.vmr`; per-gas tables are a `dict` mapped with `jax.tree.map` (see `TableOpacity.__call__`), so the gas list lives in the data.
- Arrays are leaves. Never `eqx.field(static=True)` on an array; static is for plain Python metadata only (a gas name, a band label). Choose what is not differentiated with `eqx.partition` or `jax.lax.stop_gradient`.
- Float64: convert in `__init__` (`jnp.asarray(x, dtype=float)`) and warn on a float32 grid as `TableOpacity` does.
- Validate in `__init__` or `__check_init__` with the classes from `radtrax/exceptions.py` (`ShapeError`, `GridError`, `ParameterError`, `MissingKeyError`), never a built-in. A check on a value must skip itself under tracing (the `try`/`except jax.errors.TracerBoolConversionError` in `CrossSectionTable`). A gas the state lacks is a `MissingKeyError` naming it.
- Keep lookups in their valid domain. `CrossSectionTable` clamps with `jnp.where` (not `jnp.clip`, which halves the gradient on a bound) and says in its docstring that outside the table the edge is used with zero gradient; do the same for anything you interpolate or extrapolate.
- No Python control flow on traced values inside `__call__`; no NumPy on the gradient path (NumPy and xarray are for building tables offline, behind an extra, imported inside the function).

A minimal provider, run against the current code:

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from radtrax import MissingKeyError
from radtrax.atmosphere import AtmosphericState, layers_from_levels
from radtrax.spectroscopy import OpacityProvider


class GrayOpacity(OpacityProvider):
    """A wavenumber-independent cross section κ [cm²/molecule] of one gas."""

    nu_cm: Float[Array, " G"]
    kappa_cm2: Float[Array, ""]
    gas: str = eqx.field(static=True)

    def __init__(self, nu_cm, kappa_cm2, gas: str) -> None:
        self.nu_cm = jnp.asarray(nu_cm, dtype=float)
        self.kappa_cm2 = jnp.asarray(kappa_cm2, dtype=float)
        self.gas = gas

    def __call__(self, state: AtmosphericState) -> Float[Array, "L G"]:
        if self.gas not in state.vmr:
            raise MissingKeyError(
                f"GrayOpacity: the state has no mixing ratio for {self.gas!r}."
            )
        column = state.air_column() * state.vmr[self.gas]  # molecules/cm²   (L,)
        return self.kappa_cm2 * column[:, None] * jnp.ones_like(self.nu_cm)


nu = jnp.linspace(4250.0, 4350.0, 101)
atm = layers_from_levels(
    jnp.array([1e5, 5e4, 2e4]),
    jnp.array([288.0, 255.0, 220.0]),
    {"CH4": jnp.full(3, 1.9e-6)},
)
print(GrayOpacity(nu, 1e-21, "CH4")(atm).shape)  # (L, G) = (2, 101)
```


## 3. Compose it

- `ForwardModel(opacity, solver, instrument)` checks that `opacity.nu_cm` equals `instrument.nu_cm` (a `GridError` otherwise); `SumOpacity` checks every provider shares the first one's grid. Build the instrument on `opacity.nu_cm`.
- Per pixel, a swath is `eqx.filter_vmap` over pixels of a per-pixel Jacobian, never a Jacobian of a batch (`eqx.filter_vmap(jax.jacrev(f))`).

## 4. Tests

In `tests/spectroscopy/test_opacity.py`, or `tests/spectroscopy/test_<name>.py` for a new module. Follow the structure of the existing file.

- The protocol cannot be built (`pytest.raises(TypeError)`), as `test_opacity_provider_is_an_abstract_protocol` does.
- A closed form: choose a state and a σ where `τ = σ n x Δz` is known by hand, and compare.
- The gradient check, for every public differentiable function: `from _gradcheck import assert_gradients_match` (`tests/_gradcheck.py`) compares `jax.jacrev` with central differences, rtol 1e-6, on a reduced float64 state. Check a physically small variable in units that make it O(1) (ppm for a mixing ratio), as `test_gradients_in_the_mixing_ratios` does:

  ```python
  assert_gradients_match(
      lambda ppm: GrayOpacity(nu, 1e-21, "CH4")(
          eqx.tree_at(lambda s: s.vmr["CH4"], atm, ppm * 1e-6)
      ),
      jnp.full(2, 1.9),
  )
  ```

- Validation: a bad shape, a missing gas and a mismatched grid raise the specific class. Tests of radtrax's own shape validation use the `shapes_unchecked` fixture (`tests/conftest.py`), since jaxtyping's run-time check would fire first.
- Works under `eqx.filter_jit` and `eqx.filter_vmap`.
- Parity against ported code is near rounding. Never compare independently computed floats across implementations or machines tighter than about `rtol=1e-12`, and give any tolerance a one-line reason in a comment.

## 5. Make it public

Run through the checklist below. `docs/api/spectroscopy.md` already has a `## opacity` block, so a new class in that module is picked up; a new module gets its own `## <name>` block there.

## Checklist (every extension point)

- [ ] A gradient check (`assert_gradients_match`) for every public differentiable function, with the `jax.jacrev` call finite at the edges of the valid domain.
- [ ] A missing optional dependency raises `MissingExtraError` naming the extra (`pip install 'radtrax[hitran]'`), with the import inside the function and `raise … from err`.
- [ ] The module declares a sorted `__all__`, and the name is also in the `__all__` of its subpackage `__init__.py` (import it there too, and the module name if it is new). `tests/test_public_api.py` fails otherwise; `SUBMODULES` there lists only top-level modules, `SUBPACKAGES` the subpackages, so a new file inside `spectroscopy/` needs no edit there.
- [ ] Units in names (`_cm`, `_k`, `_pa`, `_m`) or in the docstring; shapes in jaxtyping on every public argument and return, dimension strings quoted with a leading space (`" G"`).
- [ ] Arrays are Equinox leaves, never `eqx.field(static=True)`.
- [ ] Iterative solves (an `optimistix` or `lineax` solve, a Newton step) stay in their valid domain: parametrise so no step can leave it (`band_brightness_temperature` solves in `ln T`) and say what happens outside.
- [ ] No float comparison between independently computed values tighter than about 1e-12, and every tolerance has a stated reason.
- [ ] `make capabilities` (regenerates `docs/api/capabilities.md`; `tests/test_capabilities.py` checks it is current and that no name is bound to two objects).
- [ ] The docs API page: the module's `:::` block in `docs/api/<subpackage>.md` and the row in `docs/api/index.md`; `docs/llms.txt` and the module table in `AGENTS.md` when the headline usage changes.
- [ ] `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `make typecheck`, `uv run pytest --no-cov tests/<your tests>`, `make docs-api`.
- [ ] Run the `reuse-reviewer` and `numerics-reviewer` subagents (`radtrax-review`) on the diff before committing.
