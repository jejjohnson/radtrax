"""Shared pytest configuration.

radtrax computes in float64: line-by-line opacity and finite-difference
gradient checks are not meaningful in float32 (radtrax · 3, differentiability,
item 1). JAX defaults to float32, so the suite switches x64 on before any
test imports JAX arrays.
"""

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)
