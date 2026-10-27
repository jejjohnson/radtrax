"""One-to-one spectral pairs and the ratio of their mean radiances.

Steps 3–4 of Xiang et al.'s vetting, as the reference code runs them
(``pv/src/pv/pv.py``): each seed is paired with one background pixel of the
pool, the assignment minimising the total Euclidean distance between their
radiances on bands the gas does not absorb, and the best-matched half of the
pairs is kept; the ratio spectrum divides the kept pairs' mean in-plume
radiance by their mean background radiance.

Pairing is NumPy and SciPy, off the gradient path (an assignment is
discrete); the distance it minimises is the one private step
``_pairing_distance``. The ratio is JAX and differentiable in the cube.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from radtrax.exceptions import DegenerateInputError, ParameterError, ShapeError


__all__ = ["pair_pixels", "ratio_spectrum"]


def pair_pixels(
    cube: Float[ArrayLike, "H W B"],
    seeds: Bool[ArrayLike, "H W"],
    pool: Bool[ArrayLike, "H W"],
    pairing_bands: Bool[ArrayLike, " B"],
    *,
    keep: float = 0.5,
) -> tuple[Int[np.ndarray, " N"], Int[np.ndarray, " N"]]:
    """One-to-one pairs of seeds and pool pixels, the best-matched kept.

    With ``s(p)`` pixel ``p``'s radiance on the pairing bands, the seeds
    ``I`` and the pool ``O``, the pairing is the injective assignment

    ``π* = argmin_{π: I → O injective} ∑_{i ∈ I} ‖s(i) − s(π(i))‖₂``,

    solved exactly by ``scipy.optimize.linear_sum_assignment``; the
    ``N = int(keep · |I|)`` pairs of smallest distance are kept. One to one,
    so no background pixel's noise or surface mismatch enters two pairs.

    Args:
        cube: Radiance, ``(H, W, B)``.
        seeds: The in-plume set ``I``, ``(H, W)``, from
            [`select_seeds`][radtrax.retrieval.vetting.select_seeds].
        pool: The background pool ``O``, ``(H, W)``, disjoint from
            ``seeds``, from
            [`background_pool`][radtrax.retrieval.vetting.background_pool].
        pairing_bands: The bands compared, ``(B,)``
            (``VettingWindow.pairing_bands``: those outside every absorbing
            range).
        keep: Fraction of the pairs kept, in (0, 1]; the reference code's
            ``SPECTRAL_SIMILARITY_FRACTION_RETAINED``, 0.5.

    Returns:
        ``(in_idx, out_idx)``, flat pixel indices into ``(H, W)``, ``(N,)``
        each: pair ``n`` is seed ``in_idx[n]`` and background pixel
        ``out_idx[n]``, ordered from the best-matched pair to the worst
        kept. No index repeats in either.

    Raises:
        ParameterError: If ``keep`` is outside (0, 1].
        ShapeError: If ``cube`` is not 3-D, ``seeds`` or ``pool`` is not
            ``(H, W)``, or ``pairing_bands`` is not ``(B,)``.
        DegenerateInputError: If there are no seeds, the pool is smaller
            than the seeds, no band is a pairing band, ``keep`` keeps no
            pair, or a seed or pool pixel has a non-finite radiance on a
            pairing band.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.vetting import pair_pixels
        >>> cube = np.array([[[1.0, 9.0], [3.0, 9.0], [3.1, 0.0], [0.8, 0.0]]])
        >>> seeds = np.array([[True, True, False, False]])
        >>> pool = ~seeds
        >>> pair_pixels(cube, seeds, pool, np.array([True, False]), keep=1.0)
        (array([1, 0]), array([2, 3]))
        >>> pair_pixels(cube, seeds, pool, np.array([True, False]))
        (array([1]), array([2]))
    """
    if not 0.0 < keep <= 1.0:
        raise ParameterError(f"pair_pixels: keep must be in (0, 1], got {keep}")
    cube = np.asarray(cube, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=bool)
    pool = np.asarray(pool, dtype=bool)
    pairing_bands = np.asarray(pairing_bands, dtype=bool)
    if cube.ndim != 3:
        raise ShapeError(f"pair_pixels: cube must be (H, W, B), got {cube.shape}")
    image = cube.shape[:2]
    for name, mask in (("seeds", seeds), ("pool", pool)):
        if mask.shape != image:
            raise ShapeError(
                f"pair_pixels: {name} has shape {mask.shape}, the cube's image "
                f"is {image}"
            )
    if pairing_bands.shape != cube.shape[2:]:
        raise ShapeError(
            f"pair_pixels: pairing_bands has shape {pairing_bands.shape}, the "
            f"cube has {cube.shape[2]} bands"
        )
    i_in = np.flatnonzero(seeds)
    i_out = np.flatnonzero(pool)
    n_kept = int(keep * i_in.size)
    if i_in.size == 0:
        raise DegenerateInputError("pair_pixels: there are no seeds")
    if i_out.size < i_in.size:
        raise DegenerateInputError(
            f"pair_pixels: the pool has {i_out.size} pixels, fewer than the "
            f"{i_in.size} seeds; one-to-one pairing needs at least as many"
        )
    if not pairing_bands.any():
        raise DegenerateInputError("pair_pixels: no band is a pairing band")
    if n_kept == 0:
        raise DegenerateInputError(
            f"pair_pixels: keep={keep} of {i_in.size} seeds keeps no pair"
        )
    flat = cube.reshape(-1, cube.shape[-1])
    s_in = flat[i_in][:, pairing_bands]  # (n_in, B_o)
    s_out = flat[i_out][:, pairing_bands]  # (n_out, B_o)
    if not (np.isfinite(s_in).all() and np.isfinite(s_out).all()):
        raise DegenerateInputError(
            "pair_pixels: a seed or pool pixel has a non-finite radiance on a "
            "pairing band"
        )

    from scipy.optimize import linear_sum_assignment

    distance = _pairing_distance(s_in, s_out)  # (n_in, n_out)
    rows, cols = linear_sum_assignment(distance)  # rows = 0..n_in-1
    best = np.argsort(distance[rows, cols], kind="stable")[:n_kept]
    return i_in[rows[best]], i_out[cols[best]]


def ratio_spectrum(
    cube: Float[ArrayLike, "H W B"],
    in_idx: Int[ArrayLike, " N"],
    out_idx: Int[ArrayLike, " N"],
) -> Float[Array, " B"]:
    """The ratio of the pairs' mean in-plume and mean background radiances.

    ``R(λ) = ∑_n w_n(λ) L_in,n(λ) / ∑_n w_n(λ) L_out,n(λ)``, with
    ``w_n(λ) = 1[L_out,n(λ) > 0]``: a pair whose background radiance is not
    positive in a band is left out of that band. When every background
    radiance is positive, ``R`` is the ratio of the means,
    ``(1/N) ∑_n L_in,n / (1/N) ∑_n L_out,n``. Averaging before dividing
    keeps the ratio's noise bias at the size of the mean's, not of one
    pixel's.

    Differentiable in ``cube`` (the weights are piecewise constant) and
    traceable under ``jax.jit`` in ``cube``; the indices are NumPy, checked
    eagerly, and enter as constants.

    Args:
        cube: Radiance, ``(H, W, B)``.
        in_idx: Flat indices of the pairs' in-plume pixels, ``(N,)``, from
            [`pair_pixels`][radtrax.retrieval.vetting.pair_pixels].
        out_idx: Flat indices of their background pixels, ``(N,)``.

    Returns:
        ``R``, ``(B,)``; NaN in a band where no pair's background radiance
        is positive (its gradient there is zero, not NaN).

    Raises:
        ShapeError: If ``cube`` is not 3-D or the indices are not two
            ``(N,)`` arrays of one length.
        DegenerateInputError: If there are no pairs.
        ParameterError: If an index is outside ``[0, H · W)``.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.vetting import ratio_spectrum
        >>> cube = np.array([[[1.0, 2.0], [3.0, 4.0], [2.0, 8.0], [2.0, 0.0]]])
        >>> print(ratio_spectrum(cube, np.array([0, 1]), np.array([2, 3])))
        [1.   0.25]
    """
    cube = jnp.asarray(cube)
    in_idx = np.asarray(in_idx)
    out_idx = np.asarray(out_idx)
    if cube.ndim != 3:
        raise ShapeError(f"ratio_spectrum: cube must be (H, W, B), got {cube.shape}")
    if in_idx.ndim != 1 or in_idx.shape != out_idx.shape:
        raise ShapeError(
            f"ratio_spectrum: in_idx and out_idx must be (N,) of one length, got "
            f"{in_idx.shape} and {out_idx.shape}"
        )
    if in_idx.size == 0:
        raise DegenerateInputError("ratio_spectrum: there are no pairs")
    n_pixels = cube.shape[0] * cube.shape[1]
    for name, idx in (("in_idx", in_idx), ("out_idx", out_idx)):
        if idx.min() < 0 or idx.max() >= n_pixels:
            raise ParameterError(
                f"ratio_spectrum: {name} must lie in [0, {n_pixels}), got "
                f"[{idx.min()}, {idx.max()}]"
            )
    flat = cube.reshape(-1, cube.shape[-1])
    l_in = flat[in_idx]  # (N, B)
    l_out = flat[out_idx]  # (N, B)
    w = l_out > 0  # (N, B)
    numerator = jnp.sum(jnp.where(w, l_in, 0.0), axis=0)  # (B,)
    denominator = jnp.sum(jnp.where(w, l_out, 0.0), axis=0)  # (B,)
    usable = jnp.any(w, axis=0)  # (B,)
    # Double where: a safe denominator where no pair is usable, so the NaN
    # returned there does not reach the gradient.
    safe = jnp.where(usable, denominator, 1.0)
    return jnp.where(usable, numerator / safe, jnp.nan)


def _pairing_distance(
    s_in: Float[np.ndarray, " n_in Bo"], s_out: Float[np.ndarray, " n_out Bo"]
) -> Float[np.ndarray, " n_in n_out"]:
    """The cost the assignment minimises: Euclidean distance on the pairing bands.

    ``d_ij = ‖s_in,i − s_out,j‖₂``, the reference code's choice; ``cdist``
    holds only the ``(n_in, n_out)`` matrix, never the
    ``(n_in, n_out, Bo)`` differences.
    """
    from scipy.spatial.distance import cdist

    return cdist(s_in, s_out, metric="euclidean")
