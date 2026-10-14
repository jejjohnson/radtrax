"""Variational data assimilation (3D-Var) for methane VMR retrieval.

Modules
-------
- :mod:`.obs_operator` — JAX forward operator
  ``H = SRF ∘ GSD ∘ PSF ∘ exp(−a·ΔVMR)`` built from a HAPI cross-section LUT.
- :mod:`.background`  — covariance ``B`` constructors: diagonal, Kronecker
  (separable AR(1) spatial), low-rank update from a sample stack.
- :mod:`.control`     — control-variable transforms: identity vs whitening
  (Cholesky / CVT) — the latter dramatically improves Hessian conditioning.
- :mod:`.cost`        — cost+grad+HVP bundles in either model space (``δx``)
  or whitened space (``ξ``). Reverse-mode AD handles the obs-term adjoint.
- :mod:`.solve`       — three solver paths: primal LBFGS, Gauss-Newton with
  a custom :mod:`lineax` linear solver, and the dual / PSAS reformulation.
- :mod:`.diagnostics` — reduced χ², degrees-of-freedom-for-signal, posterior
  covariance proxy via CG on the Hessian.

The math derivation lives in
``notebooks/assimilation/00_3dvar_derivation.md`` and the end-to-end demo in
``notebooks/assimilation/06_3dvar_methane_retrieval.ipynb``.

Notes
-----
Submodules are imported lazily (PEP 562), only on first attribute access.

Where this lives (decision D2 of the radtrax backlog): the retrieval moves
here with its radiance observation operator, which is radiative transfer.
Its solvers are generic variational data assimilation, which is vardax's
job; whether they move there waits for the rule on whether GeoModels may
import GeoML packages (package-boundaries open questions 1 and 5).

Ported from plumax 0.1.2 at 28f257e (``plumax.assimilation``); only import
paths changed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # obs_operator
    "RadianceObservationModel": ("obs_operator", "RadianceObservationModel"),
    # background
    "build_diagonal_background": ("background", "build_diagonal_background"),
    "build_kronecker_background": ("background", "build_kronecker_background"),
    "build_lowrank_background": ("background", "build_lowrank_background"),
    # control
    "IdentityTransform": ("control", "IdentityTransform"),
    "WhiteningTransform": ("control", "WhiteningTransform"),
    # cost
    "Cost": ("cost", "Cost"),
    "build_cost_x": ("cost", "build_cost_x"),
    "build_cost_xi": ("cost", "build_cost_xi"),
    "finite_difference_grad": ("cost", "finite_difference_grad"),
    # solve
    "SolveResult": ("solve", "SolveResult"),
    "run_lbfgs": ("solve", "run_lbfgs"),
    "run_gauss_newton": ("solve", "run_gauss_newton"),
    "run_dual_psas": ("solve", "run_dual_psas"),
    # diagnostics
    "reduced_chi_squared": ("diagnostics", "reduced_chi_squared"),
    "degrees_of_freedom_for_signal": ("diagnostics", "degrees_of_freedom_for_signal"),
    "posterior_covariance_proxy": ("diagnostics", "posterior_covariance_proxy"),
}


def __getattr__(name: str):  # PEP 562 — lazy module attribute access
    try:
        submodule, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'radtrax.retrieval.var3d' has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"radtrax.retrieval.var3d.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # static type checkers see eagerly-imported symbols
    from radtrax.retrieval.var3d.background import (
        build_diagonal_background,
        build_kronecker_background,
        build_lowrank_background,
    )
    from radtrax.retrieval.var3d.control import (
        IdentityTransform,
        WhiteningTransform,
    )
    from radtrax.retrieval.var3d.cost import (
        Cost,
        build_cost_x,
        build_cost_xi,
        finite_difference_grad,
    )
    from radtrax.retrieval.var3d.diagnostics import (
        degrees_of_freedom_for_signal,
        posterior_covariance_proxy,
        reduced_chi_squared,
    )
    from radtrax.retrieval.var3d.obs_operator import (
        RadianceObservationModel,
    )
    from radtrax.retrieval.var3d.solve import (
        SolveResult,
        run_dual_psas,
        run_gauss_newton,
        run_lbfgs,
    )


__all__ = sorted(_LAZY_EXPORTS)
