# radtrax

[![Tests](https://github.com/jejjohnson/radtrax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/radtrax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/radtrax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/radtrax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/radtrax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/radtrax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/radtrax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/radtrax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/radtrax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/radtrax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Differentiable radiative transfer in JAX and Equinox, for methane and
multi-gas remote sensing. Formerly **rtmx**.

radtrax computes a forward model $y = F(x)$ from an atmospheric state to what
a sensor measures, and gets every Jacobian a retrieval needs,
$K = \partial F / \partial x$, from automatic differentiation: matched-filter
targets, optimal-estimation retrievals, multispectral retrievals and
emission-rate fits all use the same model and its derivatives.

## Status

radtrax is being assembled. Its first releases move code that already exists
in the GeoStack, then rebuild it as one layered forward model:

| Release | What lands |
|---|---|
| 0.1 | plumax's band-integrated Beer–Lambert models, HITRAN look-up tables, matched filter and 3D-Var retrieval, ported with their tests |
| 0.2 | one matched filter, one SRF integration, one air-mass factor; GeoTensor operators (from geotoolz) behind `radtrax[geo]` |
| 0.3 | the layered v0 forward model (atmosphere → opacity → Beer–Lambert → instrument) and matched-filter targets from autodiff |

The plan lives in the issues: start at the
[Wave 0 epic](https://github.com/jejjohnson/radtrax/issues/1). The design is
*radtrax · 3* and the package moves are *package boundaries*, both in the
[GeoStack docs](https://github.com/jejjohnson/geostack/tree/main/docs/geomodels).

## Install

```bash
uv add "radtrax @ git+https://github.com/jejjohnson/radtrax.git"
```

The base install is the JAX numerics (JAX, Equinox, jaxtyping, lineax,
optimistix, gaussx) plus NumPy, xarray and scikit-learn for the matched
filter's background estimators. Extras add the rest as they land: `[hitran]`
(HAPI, to build cross-section tables), `[geo]` (geotoolz and georeader, for
GeoTensor operators), `[bayes]` (NumPyro and optax retrievals).

## Where it sits

```
  gaussx · lineax · optimistix      Numerics
            │
  radtrax                           GeoModels: radiative transfer
     ┊ [geo] → geotoolz, georeader  GeoStack
            │
  plumax                            GeoModels: plume physics
```

radtrax never imports plumax: plume physics uses radiative transfer, not the
other way round.

## Develop

```bash
make install     # uv sync --all-groups + pre-commit hooks
make test        # tests + doctests (float64 on)
make lint typecheck
make docs        # MyST prose + MkDocs API reference into public/
```

The rules for contributors and coding agents are in [`AGENTS.md`](AGENTS.md);
code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).

## License

MIT.
