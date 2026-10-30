---
name: add-noise-model
description: Add a new NoiseModel (the measurement-noise covariance S_ε(y) as a positive-definite lineax operator, with sample and log_prob inherited) to radtrax — an abstract Equinox module with an abc-abstract covariance method, jaxtyping shapes, validation through the RadtraxError hierarchy, a gradient check of the covariance in the signal, tests beside test_noise.py, the API page and the capability index. Use when asked to add a signal-dependent, correlated or sensor-specific noise model (NEΔT, SNR, read noise plus shot noise).
---

# Add a noise model

Read "The contracts" in `AGENTS.md` first. The protocol is `NoiseModel` in `src/radtrax/instrument/noise.py`; the implementations to copy from are `DiagonalNoise` (fixed σ), `SNRNoise` (signal-dependent, with a floor) and `NEdTNoise` (NEΔT through `∂B̄/∂T` and an `Instrument`), in the same file. The covariance is a `lineax` operator so `gaussx` dispatches on its structure; a dense inverse of it is never written by hand.

## 1. Check it does not exist

Search `docs/api/capabilities.md`. A per-band σ is `DiagonalNoise`, a signal-to-noise ratio with a floor is `SNRNoise`, a thermal band's NEΔT is `NEdTNoise` (Landsat's is `radtrax.geo.landsat_noise`). A new class is for a new dependence on the signal or a correlated covariance (a structured `gaussx` operator, a low-rank update). It goes in `noise.py`.

## 2. Write the class

The contract: `covariance(y)` returns `S_ε` at the band values `y` `(B,)` as a positive-definite `(B, B)` `lineax` operator. `sample(key, y)` and `log_prob(y_obs, y_model)` are inherited (`gx.sample_mvn` and `gx.gaussian_log_prob`), and `log_prob` keeps `log |S_ε|`, which depends on `y_model` for signal-dependent noise.

- Subclass `NoiseModel` and implement `covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator`. Parameters are ordinary fields, float64, `(B,)` or scalars, in the units of `y` (or dimensionless, say which).
- Tag the operator so `lineax` and `gaussx` know it is positive definite: a bare `lx.DiagonalLinearOperator` could have negative entries, so `lx.CG` refuses it. Use `lx.TaggedLinearOperator(op, (lx.positive_semidefinite_tag, lx.symmetric_tag, lx.diagonal_tag))` as the private `_diagonal_covariance` in `noise.py` does.
- Keep `S_ε` positive definite where the signal vanishes: a floor term (`SNRNoise`) or a signal-independent part. A variance that is zero at `y = 0` makes `log_prob` infinite.
- Validate in `__init__` with `ShapeError` (`(B,)` alike) and `ParameterError` (strictly positive), skipping the value check under tracing, and check `y` against the parameters' shape in `covariance` (`ShapeError`).
- Randomness comes from the `jax.random` key the caller passes; nothing seeds a global state.
- If you go through an iterative solve (`NEdTNoise` inverts a band Planck function by Newton in `ln T`), keep it in its valid domain so no step leaves it.
- Arrays are leaves, never `eqx.field(static=True)`.

A minimal noise model, run against the current code:

```python
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from radtrax.instrument import NoiseModel


class RelativeNoise(NoiseModel):
    """``σ² = (r y)² + σ_floor²``: a relative error ``r`` and a floor, the units of ``y``."""

    rel: Float[Array, ""]
    floor: Float[Array, ""]

    def __init__(self, rel, floor) -> None:
        self.rel = jnp.asarray(rel, dtype=float)
        self.floor = jnp.asarray(floor, dtype=float)

    def covariance(self, y: Float[ArrayLike, " B"]) -> lx.AbstractLinearOperator:
        y = jnp.asarray(y, dtype=float)
        variance = (self.rel * y) ** 2 + self.floor**2
        return lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(variance),
            (lx.positive_semidefinite_tag, lx.symmetric_tag, lx.diagonal_tag),
        )


noise = RelativeNoise(0.01, 1e-3)
y = jnp.array([0.5, 2.0])
print(gx.diag(noise.covariance(y)))  # [2.6e-05, 4.01e-04]
print(noise.sample(jax.random.key(0), y), noise.log_prob(y, y))
```

## 3. Compose it

A retrieval takes the model's `covariance(y_model)` as `S_ε`; `Instrument` outputs are its `y`. Test it beside an instrument (`BandInstrument` or `GaussianInstrument`) so the band count is right.

## 4. Tests

In `tests/instrument/test_<name>.py` for a new module, or `tests/instrument/test_noise.py` for a small class, following `tests/instrument/test_noise.py` and `test_nedt_noise.py`.

- The covariance is a positive diagonal operator (`test_covariance_is_a_positive_diagonal_operator`): `gx.diag` equals the closed-form variance.
- `log_prob` equals the Gaussian log density against `scipy.stats` (`test_log_prob_is_the_gaussian_log_density`), including the `log |S_ε|` term for signal-dependent noise (`test_snr_log_prob_gradient_keeps_the_log_determinant`).
- `sample` matches the covariance in variance with a bound from its own sampling distribution (a number of standard errors, with a comment saying where it comes from, per the AGENTS.md rule on random draws), and the same key gives the same draw (`test_sample_is_seeded_by_the_callers_key`).
- The gradient check, for every public differentiable function: `from _gradcheck import assert_gradients_match`, of the covariance diagonal in the signal and in each parameter (`test_snr_covariance_gradient_in_the_signal`).
- The floor holds where the signal vanishes (`test_snr_noise_keeps_the_floor_where_the_signal_vanishes`).
- Works under `eqx.filter_jit` and `eqx.filter_vmap` over pixels (`test_noise_models_jit_and_vmap_over_pixels`).
- Shape and positivity errors with the `shapes_unchecked` fixture (`tests/conftest.py`) for shapes (`test_shapes_are_checked`), `ParameterError` for values.
- Never compare independently computed floats across machines tighter than about `rtol=1e-12`, and give every tolerance a reason.

## 5. Make it public

The class goes in `noise.py`'s sorted `__all__`, then `src/radtrax/instrument/__init__.py` (import and `__all__`, sorted). The `docs/api/instrument.md` page already has a `## noise` block that picks it up; update the Instrument row of `docs/api/index.md` (it lists the noise models), the `instrument/` row of the `AGENTS.md` module table and `docs/llms.txt`.

## Checklist (every extension point)

- [ ] A gradient check (`assert_gradients_match`) for every public differentiable function, finite at the edges of the valid domain (at `y = 0`).
- [ ] A missing optional dependency raises `MissingExtraError` naming the extra, with the import inside the function and `raise … from err`.
- [ ] The module declares a sorted `__all__`, and the name is also in the `__all__` of `instrument/__init__.py`. `tests/test_public_api.py` fails otherwise (`SUBMODULES` lists only top-level modules and `SUBPACKAGES` the subpackages, so a new file in `instrument/` needs no edit there).
- [ ] Units in names or docstrings (the units of `y`, or dimensionless, or K for NEΔT); jaxtyping shapes on every public argument and return, dimension strings quoted with a leading space.
- [ ] Arrays are Equinox leaves, never `eqx.field(static=True)`.
- [ ] Iterative solves stay in their valid domain.
- [ ] No float comparison between independently computed values tighter than about 1e-12, and every tolerance has a stated reason.
- [ ] `make capabilities`.
- [ ] The docs API page (`docs/api/instrument.md`, the `docs/api/index.md` row), `docs/llms.txt` and the `AGENTS.md` module table.
- [ ] `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `make typecheck`, `uv run pytest --no-cov tests/instrument`, `make docs-api`.
- [ ] Run the `reuse-reviewer` and `numerics-reviewer` subagents (`radtrax-review`) on the diff before committing.
