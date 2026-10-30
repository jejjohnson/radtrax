---
name: add-surface
description: Add a new Surface (the bidirectional reflectance factor R from a ViewingGeometry and the wavenumber grid) to radtrax — an abstract Equinox module with an abc-abstract reflectance method, jaxtyping shapes, the double-where at kinks such as nadir and the hotspot, a gradient check in the angles and the parameters, tests beside test_ross_li.py, the API page and the capability index. Use when asked to add a kernel BRDF (RPV, Cox–Munk), a spectral albedo model or any other surface reflectance.
---

# Add a surface

Read "The contracts" in `AGENTS.md` first. The protocol is `Surface` in `src/radtrax/surface/lambertian.py`; the implementations to copy from are `Lambertian` (same file), `PolynomialAlbedo` (`polynomial.py`, a spectral albedo) and `RossLi` (`ross_li.py`, a geometry-dependent BRDF with guarded gradients).

## 1. Check it does not exist

Search `docs/api/capabilities.md`. A kernel is a function and a surface that weights kernels is a module (the AGENTS.md rule: modules carry state, functions carry transforms): `ross_thick` and `li_sparse_reciprocal` are public functions and `RossLi` holds the weights. A new surface goes in `src/radtrax/surface/<name>.py`.

## 2. Write the class

The contract: `reflectance(geometry, nu_cm)` returns the bidirectional reflectance factor `R` (dimensionless), a scalar or `(G,)`. It is the reflected radiance relative to a white Lambertian surface under the same illumination, so the solver forms `L = E₀ μ₀ R / π`; do not include `μ₀` or `π` in `R`.

- Subclass `Surface` and implement `reflectance(self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]) -> Float[Array, "*G"]`. Parameters are ordinary fields, converted with `jnp.asarray(x, dtype=float)` in `__init__`, scalar or `(G,)` where a per-wavenumber value makes sense (`Lambertian.albedo`, `RossLi.f_iso`).
- Delete the arguments you do not use (`del geometry, nu_cm`) as `Lambertian` and `RossLi` do; a spectral model uses `nu_cm`, a BRDF uses the geometry.
- Geometry: `geometry.mu0`, `geometry.mu` (cosines in `(0, 1]`) and `geometry.relative_azimuth_rad` (sensor minus sun, `φ = 0` with `θ₀ = θ` is the hotspot). `jnp.arccos` differentiated naively is `0 · ∞` at `±1`, so a model that recovers an angle from a cosine guards it; `ross_li.py` writes the affected terms with a `custom_jvp` and its module docstring says which first and second derivatives are finite.
- A point where an expression is 0/0 or a distance vanishes (nadir, the hotspot) uses the double-where: a safe argument in the branch not taken, then `jnp.where`. A bare `jnp.where` still propagates the NaN. State in the docstring what the gradient is at a kink (`RossLi`: the symmetric subgradient, 0).
- A window or domain that the model is only valid in (`PolynomialAlbedo`'s Legendre window) is documented, checked in `__check_init__` when concrete (skipping under tracing) and raises `GridError` or `ParameterError`; the model grid is kept inside it.
- Arrays are leaves, never `eqx.field(static=True)`.
- Raise classes from `radtrax/exceptions.py`, never a built-in.

A minimal surface, run against the current code:

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

from radtrax.atmosphere import ViewingGeometry
from radtrax.surface import Surface


class Minnaert(Surface):
    """Minnaert's law: ``R = A (μ₀ μ)^(k − 1)``, with ``k = 1`` Lambertian."""

    albedo: Float[Array, ""]
    k: Float[Array, ""]

    def __init__(self, albedo, k) -> None:
        self.albedo = jnp.asarray(albedo, dtype=float)
        self.k = jnp.asarray(k, dtype=float)

    def reflectance(
        self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]
    ) -> Float[Array, "*G"]:
        del nu_cm
        return self.albedo * (geometry.mu0 * geometry.mu) ** (self.k - 1.0)


geometry = ViewingGeometry.from_angles(35.0, 10.0)
print(float(Minnaert(0.3, 1.0).reflectance(geometry, jnp.array([4300.0]))))  # 0.3
```

`μ₀ μ` is in `(0, 1]` by the geometry's own check, so the power has no singular input.

## 3. Compose it

A solver calls `surface.reflectance(geometry, optics.nu_cm)`, so the surface works with every `RTSolver`; `ForwardModel` takes it per pixel, as an argument, not a field. A retrieval's nuisance state (albedo coefficients, BRDF weights) is differentiated through it, so keep every parameter a leaf. Per-pixel geometry is a `ViewingGeometry` whose leaves are `(H, W)`, mapped with `eqx.filter_vmap`.

## 4. Tests

In `tests/surface/test_<name>.py`, following `tests/surface/test_ross_li.py` and `test_polynomial.py`.

- The protocol cannot be built (`pytest.raises(TypeError)`), as `test_lambertian.py` does.
- Limits: a parameter setting where it reduces to `Lambertian` gives the same bands through the model (`test_one_coefficient_gives_lambertian_bands`, using `tests/_v0.py`'s `v0_model`, `v0_atmosphere` and `GEOMETRY`); reciprocity (`θ₀ ↔ θ`) and the value at nadir if the model has them.
- The gradient check, for every public differentiable function: `from _gradcheck import assert_gradients_match`, in the angles and in the weights, at a general geometry and at each kink (nadir, hotspot) with `indices=[...]` selecting the entries that have a two-sided derivative, and `assert jnp.all(jnp.isfinite(jax.jacrev(f)(x)))` there. `tests/surface/test_ross_li.py::test_gradients_in_angles_and_weights` is the pattern.
- A known value with a tolerance whose provenance is a comment; never compare independently computed floats across machines tighter than about `rtol=1e-12`.
- Under `eqx.filter_jit` and `eqx.filter_vmap` over pixels.

## 5. Make it public

The class goes in the module's sorted `__all__`, then `src/radtrax/surface/__init__.py` (import the class, add it and, for a new module, the module name to `__all__`, sorted). Add a `## <name>` block to `docs/api/surface.md` and mention the model in its intro; update the Surface row of `docs/api/index.md`, the `surface/` row of the `AGENTS.md` module table and `docs/llms.txt`.

## Checklist (every extension point)

- [ ] A gradient check (`assert_gradients_match`) for every public differentiable function, finite at every kink and edge.
- [ ] A missing optional dependency raises `MissingExtraError` naming the extra, with the import inside the function and `raise … from err`.
- [ ] The module declares a sorted `__all__`, and the name is also in the `__all__` of `surface/__init__.py`. `tests/test_public_api.py` fails otherwise (`SUBMODULES` lists only top-level modules and `SUBPACKAGES` the subpackages, so a new file in `surface/` needs no edit there).
- [ ] Units in names or docstrings (angles in radians or degrees stated, wavenumbers in cm⁻¹); jaxtyping shapes on every public argument and return, dimension strings quoted with a leading space.
- [ ] Arrays are Equinox leaves, never `eqx.field(static=True)`.
- [ ] Iterative solves stay in their valid domain; a series or kernel states the window it is valid on.
- [ ] No float comparison between independently computed values tighter than about 1e-12, and every tolerance has a stated reason.
- [ ] `make capabilities`.
- [ ] The docs API page (`docs/api/surface.md`, the `docs/api/index.md` row), `docs/llms.txt` and the `AGENTS.md` module table.
- [ ] `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `make typecheck`, `uv run pytest --no-cov tests/surface`, `make docs-api`.
- [ ] Run the `reuse-reviewer` and `numerics-reviewer` subagents (`radtrax-review`) on the diff before committing.
