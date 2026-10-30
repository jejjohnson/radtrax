---
name: add-solver
description: Add a new RTSolver (optical properties, a surface and a viewing geometry in, top-of-atmosphere radiance (G,) out) to radtrax — an abstract Equinox module with an abc-abstract __call__, jaxtyping shapes, the double-where for 0/0 branches, a gradient check in τ, the surface and the geometry, tests beside test_beer_lambert.py, the API page and the capability index. Use when asked to add single scattering, two-stream, discrete ordinates, thermal emission or any other way of turning optical depth into radiance.
---

# Add a solver

Read "The contracts" in `AGENTS.md` first. The protocol is `RTSolver` in `src/radtrax/solvers/beer_lambert.py`; the implementation to copy from is `BeerLambert` in the same file (plane-parallel, and `spherical=True` for each layer's spherical-shell air mass).

## 1. Check it does not exist

Search `docs/api/capabilities.md`. A change to the air mass or the path is a flag or a variation on `BeerLambert`, not a second solver; a different physics (scattering, emission) is a new `RTSolver` in `src/radtrax/solvers/<name>.py`. Do not hand-write a tridiagonal or fixed-point solve inside it: use `lineax`, `gaussx.BlockTriDiag` or `optimistix` (the "Reuse before you write" table in `AGENTS.md`).

## 2. Write the class

The contract: `OpticalProperties`, a `Surface` and a `ViewingGeometry` in, radiance `(G,)` out, in the units of the solar irradiance per steradian.

- Subclass `RTSolver` and implement `__call__(self, optics: OpticalProperties, surface: Surface, geometry: ViewingGeometry) -> Float[Array, " G"]`. The base has no fields, so a stateless solver needs none; configuration (a solar irradiance `E₀`, `(*G)`) is an ordinary field converted to float64 in `__init__`.
- Inputs you use: `optics.nu_cm` `(G,)` [cm⁻¹], `optics.tau` `(L, G)` dimensionless with layers from the surface up, and `optics.z_m`, `optics.dz_m` `(L,)` [m], or `None` when the optics were not built by `ForwardModel.optics`. If you need the altitudes and they are `None`, raise `ParameterError` that says how to build the optics, as `BeerLambert(spherical=True)` does.
- Geometry: `geometry.mu0` and `geometry.mu` (cosines in `(0, 1]`, checked on construction), `geometry.relative_azimuth_rad`, `geometry.air_mass()` and `geometry.layer_air_mass(z_m, dz_m)`. Do not recompute an air mass; `air_mass_factor` and these are the one home.
- Surface: call `surface.reflectance(geometry, optics.nu_cm)`, which is the bidirectional reflectance factor `R` (a scalar or `(G,)`), and broadcast the result to `(G,)` with `jnp.broadcast_to`, as `BeerLambert` does. The radiance leaving a surface is `E₀ μ₀ R / π`; the slab models of `radtrax.slab` write `F₀ R / π`, so their `solar_irradiance` is `E₀ μ₀` here.
- A branch that is 0/0 at a reachable point (`μ = μ₀`, a vanishing denominator in a two-stream) uses the double-where: a safe denominator in the branch not taken, then `jnp.where`; a bare `jnp.where` still propagates the NaN into the gradient. Run `jax.grad` at the edge and look.
- Python flags that select code (`spherical`) are `eqx.field(static=True)` only because they are plain `bool`s; arrays never are. Branch on them in Python, never on traced values.
- Iterative parts (a layer-by-layer recursion is a `jax.lax.scan`; a solve is `optimistix`/`lineax`) stay in their valid domain, e.g. solve for `ln` of a positive quantity, and document the error under `jit` (`eqx.error_if` raises `equinox.EquinoxRuntimeError`).
- Raise `ShapeError`, `GridError`, `ParameterError` from `radtrax/exceptions.py`, never a built-in.

A minimal solver, run against the current code:

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

from radtrax.atmosphere import ViewingGeometry
from radtrax.optics import OpticalProperties
from radtrax.solvers import RTSolver
from radtrax.surface import Lambertian, Surface


class SunPathOnly(RTSolver):
    """Transmittance along the solar path only: ``L = (E₀ μ₀ / π) R exp(−Σ τ / μ₀)``."""

    def __call__(
        self, optics: OpticalProperties, surface: Surface, geometry: ViewingGeometry
    ) -> Float[Array, " G"]:
        transmittance = jnp.exp(-jnp.sum(optics.tau, axis=0) / geometry.mu0)  # (G,)
        reflectance = surface.reflectance(geometry, optics.nu_cm)  # (*G)
        reflected = geometry.mu0 / jnp.pi * reflectance
        return jnp.broadcast_to(reflected * transmittance, transmittance.shape)


optics = OpticalProperties(
    jnp.array([4300.0, 4300.1]), jnp.array([[0.0, 0.1], [0.0, 0.2]])
)
radiance = SunPathOnly()(optics, Lambertian(1.0), ViewingGeometry(1.0, 1.0))
print([round(float(x), 6) for x in radiance * jnp.pi])  # exp(-Σ τ) at μ₀ = 1
```

## 3. Compose it

`ForwardModel(opacity, solver, instrument)` calls `solver(self.optics(state), surface, geometry)` (`ForwardModel.radiance`); the optics carry `z_m` and `dz_m` from the state. Any `Surface` must work with your solver; test it with `Lambertian` and with a surface that depends on `geometry` (`RossLi`) and on `nu_cm` (`PolynomialAlbedo`).

## 4. Tests

In `tests/solvers/test_<name>.py`, following `tests/solvers/test_beer_lambert.py`.

- The protocol cannot be built (`pytest.raises(TypeError)`).
- The closed form: a case solvable by hand (one layer, nadir) compared with the solver; parity against another solver in the limit where they agree (a scattering solver with no scatterers should equal `BeerLambert`).
- The gradient check, for every public differentiable function: `from _gradcheck import assert_gradients_match`, on a reduced state of two layers and three wavenumbers, in `τ`, the surface parameters, `μ₀`, `μ` and `E₀`, as `test_beer_lambert.py` does. Include the edge points (nadir, `μ = μ₀`) and assert `jnp.isfinite(jax.jacrev(f)(x))` there.
- Tolerances: a closed form against NumPy is a few ulp (`test_beer_lambert.py` uses 1e-14 with its provenance in a comment); never compare independently computed floats across machines tighter than about `rtol=1e-12`, and state why for any tolerance.
- A swath: `eqx.filter_vmap(jax.jacrev(f))` over pixels equals the per-pixel loop.

## 5. Make it public

The class goes in the module's sorted `__all__`, then `src/radtrax/solvers/__init__.py` (import and `__all__`, with the module name if new). Add a `## <name>` block to `docs/api/solvers.md` for a new module and mention the solver in its intro paragraph; update the Solvers row of `docs/api/index.md`, the module table in `AGENTS.md` and `docs/llms.txt`.

## Checklist (every extension point)

- [ ] A gradient check (`assert_gradients_match`) for every public differentiable function, finite at the edges of the valid domain.
- [ ] A missing optional dependency raises `MissingExtraError` naming the extra, with the import inside the function and `raise … from err`.
- [ ] The module declares a sorted `__all__`, and the name is also in the `__all__` of `solvers/__init__.py`. `tests/test_public_api.py` fails otherwise (`SUBMODULES` lists only top-level modules and `SUBPACKAGES` the subpackages, so a new file in `solvers/` needs no edit there).
- [ ] Units in names or docstrings (`E₀` per cm⁻¹ or per µm, cm⁻¹, m); jaxtyping shapes on every public argument and return, dimension strings quoted with a leading space.
- [ ] Arrays are Equinox leaves, never `eqx.field(static=True)`.
- [ ] Iterative solves stay in their valid domain; document what happens outside it.
- [ ] No float comparison between independently computed values tighter than about 1e-12, and every tolerance has a stated reason.
- [ ] `make capabilities`.
- [ ] The docs API page (`docs/api/solvers.md`, the `docs/api/index.md` row), `docs/llms.txt` and the `AGENTS.md` module table where the headline usage changes.
- [ ] `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `make typecheck`, `uv run pytest --no-cov tests/solvers`, `make docs-api`.
- [ ] Run the `reuse-reviewer` and `numerics-reviewer` subagents (`radtrax-review`) on the diff before committing.
