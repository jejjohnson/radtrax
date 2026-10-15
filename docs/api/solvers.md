# Solvers

Radiative transfer solvers turn `OpticalProperties` (per-layer optical depth
`τ (L, G)`), a surface and a `ViewingGeometry` into the radiance at the top
of the atmosphere, behind the
`RTSolver` protocol (*radtrax · 3*). v0 has `BeerLambert`, the two-way
direct path with no scattering; single scattering, two-stream and discrete
ordinates arrive with v3 and v4 of the roadmap behind the same protocol.

::: radtrax.solvers
    options:
      members: false

## `beer_lambert`

::: radtrax.solvers.beer_lambert
    options:
      show_root_heading: false
      heading_level: 3
