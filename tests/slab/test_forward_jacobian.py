"""The slab forward models' hand-written Jacobians against finite differences.

The ported models are NumPy, so ``jax.jacrev`` cannot trace them; this checks
their ``dL/dVMR`` against a central difference of their own radiance instead,
with the step of ``tests/_gradcheck.py``. The layered v0 model replaces the
hand Jacobian with autodiff and is checked against these (gate V1).
"""

from __future__ import annotations

import numpy as np
import pytest

from radtrax.slab import (
    forward_maclaurin,
    forward_maclaurin_normalized,
    forward_nonlinear,
    forward_nonlinear_normalized,
    forward_taylor,
    forward_taylor_normalized,
)


# (model, the state it differentiates in, that state's value, fixed extras)
CASES = [
    (forward_nonlinear, "vmr", 1.9e-6, {}),
    (forward_nonlinear_normalized, "delta_vmr", 0.5e-6, {"vmr_background": 1.9e-6}),
    (forward_maclaurin, "vmr", 1.9e-6, {}),
    (forward_maclaurin_normalized, "delta_vmr", 0.5e-6, {}),
    (forward_taylor, "vmr", 2.4e-6, {"vmr_background": 1.9e-6}),
    (forward_taylor_normalized, "delta_vmr", 0.5e-6, {}),
]


@pytest.mark.parametrize(
    ("model", "state", "value", "extra"), CASES, ids=[c[0].__name__ for c in CASES]
)
def test_hand_jacobian_matches_central_differences(
    model, state, value, extra, synthetic_lut
) -> None:
    nu = np.linspace(4150.0, 4350.0, 81)
    kw = dict(T_K=280.0, p_atm=0.8, path_length_cm=8.0e5, amf=2.2, **extra)
    h = np.finfo(float).eps ** (1 / 3) * value  # gradcheck's step, scaled

    def run(x: float):
        return model(synthetic_lut, nu, **{state: x}, **kw)

    jac = run(value).jacobian
    fd = (run(value + h).radiance - run(value - h).radiance) / (2 * h)

    # Relative to the Jacobian's scale; 1e-6 as in tests/_gradcheck.py.
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-6 * np.max(np.abs(fd)))
