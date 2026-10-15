# Surface

Surfaces give the bidirectional reflectance factor `R(μ₀, μ)` a solver
reflects, behind the `Surface` protocol. v0 has `Lambertian`, `R = A`;
`PolynomialAlbedo` makes `A` a Legendre series in wavenumber; `RossLi` is the
MODIS kernel BRDF (RossThick and LiSparse-Reciprocal). RPV and Cox–Munk come
later in the roadmap.

::: radtrax.surface
    options:
      members: false

## `lambertian`

::: radtrax.surface.lambertian
    options:
      show_root_heading: false
      heading_level: 3

## `polynomial`

::: radtrax.surface.polynomial
    options:
      show_root_heading: false
      heading_level: 3

## `ross_li`

::: radtrax.surface.ross_li
    options:
      show_root_heading: false
      heading_level: 3
