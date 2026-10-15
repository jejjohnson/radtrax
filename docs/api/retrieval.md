# Matched filter

The one matched-filter core: background estimators (empirical, shrunk, low-rank, cluster, streaming), the kernel on gaussx operators, targets from the observation model, and xarray I/O. Theory: [matched filter](../../matched-filter/).

::: radtrax.retrieval.matched_filter
    options:
      members: false

The package resolves its public names lazily (PEP 562), so they are documented here from the submodules that define them.

## `core`

::: radtrax.retrieval.matched_filter.core
    options:
      show_root_heading: false
      heading_level: 3

## `background`

::: radtrax.retrieval.matched_filter.background
    options:
      show_root_heading: false
      heading_level: 3

## `cluster`

::: radtrax.retrieval.matched_filter.cluster
    options:
      show_root_heading: false
      heading_level: 3

## `streaming`

::: radtrax.retrieval.matched_filter.streaming
    options:
      show_root_heading: false
      heading_level: 3

## `target`

::: radtrax.retrieval.matched_filter.target
    options:
      show_root_heading: false
      heading_level: 3

## `io`

::: radtrax.retrieval.matched_filter.io
    options:
      show_root_heading: false
      heading_level: 3

# 3D-Var

Variational methane retrieval: the radiance observation operator, background covariances, control transforms, solvers and diagnostics. Theory: [3D-Var](../../var3d/).

::: radtrax.retrieval.var3d
    options:
      members: false

## `obs_operator`

::: radtrax.retrieval.var3d.obs_operator
    options:
      show_root_heading: false
      heading_level: 3

## `background`

::: radtrax.retrieval.var3d.background
    options:
      show_root_heading: false
      heading_level: 3

## `control`

::: radtrax.retrieval.var3d.control
    options:
      show_root_heading: false
      heading_level: 3

## `cost`

::: radtrax.retrieval.var3d.cost
    options:
      show_root_heading: false
      heading_level: 3

## `solve`

::: radtrax.retrieval.var3d.solve
    options:
      show_root_heading: false
      heading_level: 3

## `diagnostics`

::: radtrax.retrieval.var3d.diagnostics
    options:
      show_root_heading: false
      heading_level: 3

# MBMP

The Sentinel-2 multi-band multi-pass methane signal and the single-pass SWIR
contrast, as differentiable JAX functions; the GeoTensor operator is
[`MBMP`](geo.md) in `radtrax.geo`.

::: radtrax.retrieval.mbmp
    options:
      show_root_heading: false
      heading_level: 2

# Band table

A band observable `g(m[, h], ΔΩ)` (one band's `τ₁₂`, or `τ₁₂ − τ₁₁`) tabulated against the enhancement and inverted by interpolation, as *radtrax · 1* and *· 2* do, from any observable or from the layered forward model.

::: radtrax.retrieval.band_table
    options:
      show_root_heading: false
      heading_level: 2
