# Forward model

v0's layered forward model, `ForwardModel`: an `AtmosphericState` through an
`OpacityProvider`, an `RTSolver` (with a `Surface` and a `ViewingGeometry`)
and an `Instrument`, to one value per band. Built on it,
`unit_absorption_target` (in `radtrax.retrieval.matched_filter`) is the
matched-filter target `t = ∂ ln y / ∂ΔX` for a scene, by one forward-mode
derivative. Both are importable from the top level.

A swath's Jacobians are `eqx.filter_vmap` of a per-pixel `jax.jacrev`:

```python
def y_of(theta):  # (ΔX [mol/m²], albedo, shift [cm⁻¹]) → y (B,)
    fm = eqx.tree_at(lambda m: m.instrument.shift_cm, model, theta[2])
    return fm(atm.add_column("CH4", theta[0]), Lambertian(theta[1]), geometry)

K = eqx.filter_jit(eqx.filter_vmap(jax.jacrev(y_of)))(thetas)  # (N, B, 3)
```

::: radtrax.forward_model
