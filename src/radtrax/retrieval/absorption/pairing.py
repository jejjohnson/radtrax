"""One-to-one spectral pairs and the ratio of their mean radiances.

The pairing and ratio steps of the absorption-shape test (steps 3–4 of
Xiang et al.'s method, as the reference code runs them,
``pv/src/pv/pv.py``): given two pixel sets, each in-pixel is paired with one
out-pixel, the assignment minimising the total Euclidean distance between
their radiances on bands the gas does not absorb, and the best-matched half
of the pairs is kept; the ratio spectrum divides the kept pairs' mean
in-pixel radiance by their mean out-pixel radiance. Which pixels form the
two sets (a plume candidate's seeds and its background pool, say) is the
caller's choice: ``plumax.vetting`` makes it.

Pairing is NumPy and SciPy, off the gradient path (an assignment is
discrete); the distance it minimises is the one private step
``_pairing_distance``. The ratio is JAX and differentiable in the cube.

Shapes: ``n_in`` in-pixels, ``n_out`` out-pixels, ``N`` kept pairs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from radtrax.exceptions import DegenerateInputError, ParameterError, ShapeError


__all__ = ["pair_pixels", "ratio_spectrum"]


def _check_pixel_set(name: str, idx: np.ndarray, n_pixels: int) -> None:
    """Raise unless ``idx`` is a 1-D set of distinct flat indices into ``H · W``."""
    if idx.ndim != 1:
        raise ShapeError(
            f"pair_pixels: {name} must be (n,) flat pixel indices, got {idx.shape}"
        )
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise ParameterError(
            f"pair_pixels: {name} must hold integer flat indices, got {idx.dtype}"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= n_pixels):
        raise ParameterError(
            f"pair_pixels: {name} must lie in [0, {n_pixels}), got "
            f"[{idx.min()}, {idx.max()}]"
        )
    if np.unique(idx).size != idx.size:
        raise ParameterError(f"pair_pixels: {name} repeats a pixel")


def pair_pixels(
    cube: Float[ArrayLike, "H W B"],
    in_pixels: Int[ArrayLike, " n_in"],
    out_pixels: Int[ArrayLike, " n_out"],
    pairing_bands: Bool[ArrayLike, " B"],
    *,
    keep: float = 0.5,
) -> tuple[Int[np.ndarray, " N"], Int[np.ndarray, " N"]]:
    """One-to-one pairs of in-pixels and out-pixels, the best-matched kept.

    With ``s(p)`` pixel ``p``'s radiance on the pairing bands, the in-pixels
    ``I`` and the out-pixels ``O``, the pairing is the injective assignment

    ``π* = argmin_{π: I → O injective} ∑_{i ∈ I} ‖s(i) − s(π(i))‖₂``,

    solved exactly by ``scipy.optimize.linear_sum_assignment``; the
    ``N = int(keep · |I|)`` pairs of smallest distance are kept. One to one,
    so no out-pixel's noise or surface mismatch enters two pairs.

    The two sets are flat indices into the image, ``r · W + c`` for row
    ``r`` and column ``c``; a boolean ``(H, W)`` mask converts with
    ``np.flatnonzero(mask)``. Nothing here knows what the sets are.

    Args:
        cube: Radiance, ``(H, W, B)``.
        in_pixels: The set ``I`` (the pixels whose absorption is tested),
            distinct flat indices into ``H · W``, ``(n_in,)``.
        out_pixels: The set ``O`` (the reference pixels they are paired
            from), distinct flat indices into ``H · W`` and disjoint from
            ``in_pixels``, ``(n_out,)`` with ``n_out ≥ n_in``.
        pairing_bands: The bands compared, ``(B,)``
            (``AbsorptionWindow.pairing_bands``: those outside every
            absorbing range).
        keep: Fraction of the pairs kept, in (0, 1]; the reference code's
            ``SPECTRAL_SIMILARITY_FRACTION_RETAINED``, 0.5.

    Returns:
        ``(in_idx, out_idx)``, flat pixel indices into ``(H, W)``, ``(N,)``
        each: pair ``n`` is in-pixel ``in_idx[n]`` and out-pixel
        ``out_idx[n]``, ordered from the best-matched pair to the worst
        kept. No index repeats in either.

    Raises:
        ParameterError: If ``keep`` is outside (0, 1], or an index set is
            not integer, has an index outside ``[0, H · W)`` or repeats a
            pixel, or the two sets share a pixel.
        ShapeError: If ``cube`` is not 3-D, an index set is not 1-D, or
            ``pairing_bands`` is not ``(B,)``.
        DegenerateInputError: If there are no in-pixels, there are fewer
            out-pixels than in-pixels, no band is a pairing band, ``keep``
            keeps no pair, or an in- or out-pixel has a non-finite radiance
            on a pairing band.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.absorption import pair_pixels
        >>> cube = np.array([[[1.0, 9.0], [3.0, 9.0], [3.1, 0.0], [0.8, 0.0]]])
        >>> in_pixels, out_pixels = np.array([0, 1]), np.array([2, 3])
        >>> bands = np.array([True, False])
        >>> pair_pixels(cube, in_pixels, out_pixels, bands, keep=1.0)
        (array([1, 0]), array([2, 3]))
        >>> pair_pixels(cube, in_pixels, out_pixels, bands)
        (array([1]), array([2]))
    """
    if not 0.0 < keep <= 1.0:
        raise ParameterError(f"pair_pixels: keep must be in (0, 1], got {keep}")
    cube = np.asarray(cube, dtype=np.float64)
    i_in = np.asarray(in_pixels)
    i_out = np.asarray(out_pixels)
    pairing_bands = np.asarray(pairing_bands, dtype=bool)
    if cube.ndim != 3:
        raise ShapeError(f"pair_pixels: cube must be (H, W, B), got {cube.shape}")
    n_pixels = cube.shape[0] * cube.shape[1]
    _check_pixel_set("in_pixels", i_in, n_pixels)
    _check_pixel_set("out_pixels", i_out, n_pixels)
    if pairing_bands.shape != cube.shape[2:]:
        raise ShapeError(
            f"pair_pixels: pairing_bands has shape {pairing_bands.shape}, the "
            f"cube has {cube.shape[2]} bands"
        )
    if np.intersect1d(i_in, i_out).size:
        raise ParameterError(
            "pair_pixels: in_pixels and out_pixels share a pixel; they must be disjoint"
        )
    n_kept = int(keep * i_in.size)
    if i_in.size == 0:
        raise DegenerateInputError("pair_pixels: there are no in-pixels")
    if i_out.size < i_in.size:
        raise DegenerateInputError(
            f"pair_pixels: there are {i_out.size} out-pixels, fewer than the "
            f"{i_in.size} in-pixels; one-to-one pairing needs at least as many"
        )
    if not pairing_bands.any():
        raise DegenerateInputError("pair_pixels: no band is a pairing band")
    if n_kept == 0:
        raise DegenerateInputError(
            f"pair_pixels: keep={keep} of {i_in.size} in-pixels keeps no pair"
        )
    flat = cube.reshape(-1, cube.shape[-1])
    s_in = flat[i_in][:, pairing_bands]  # (n_in, B_o)
    s_out = flat[i_out][:, pairing_bands]  # (n_out, B_o)
    if not (np.isfinite(s_in).all() and np.isfinite(s_out).all()):
        raise DegenerateInputError(
            "pair_pixels: an in- or out-pixel has a non-finite radiance on a "
            "pairing band"
        )

    from scipy.optimize import linear_sum_assignment

    distance = _pairing_distance(s_in, s_out)  # (n_in, n_out)
    rows, cols = linear_sum_assignment(distance)  # rows = 0..n_in-1
    best = np.argsort(distance[rows, cols], kind="stable")[:n_kept]
    return i_in[rows[best]].astype(np.intp), i_out[cols[best]].astype(np.intp)


def ratio_spectrum(
    cube: Float[ArrayLike, "H W B"],
    in_idx: Int[ArrayLike, " N"],
    out_idx: Int[ArrayLike, " N"],
) -> Float[Array, " B"]:
    """The ratio of the pairs' mean in-pixel and mean out-pixel radiances.

    ``R(λ) = ∑_n w_n(λ) L_in,n(λ) / ∑_n w_n(λ) L_out,n(λ)``, with
    ``w_n(λ) = 1[L_in,n(λ) and L_out,n(λ) finite, L_out,n(λ) > 0]``: a pair
    is left out of a band where either of its radiances is unusable (a NaN
    or infinite one, or an out-pixel radiance that is not positive). When
    every pair is usable, ``R`` is the ratio of the means,
    ``(1/N) ∑_n L_in,n / (1/N) ∑_n L_out,n``. Averaging before dividing
    keeps the ratio's noise bias at the size of the mean's, not of one
    pixel's.

    Differentiable in ``cube`` (the weights are piecewise constant; a
    radiance left out has a zero gradient, and does not make its partner's
    NaN) and
    traceable under ``jax.jit`` in ``cube``; the indices are NumPy, checked
    eagerly, and enter as constants.

    Args:
        cube: Radiance, ``(H, W, B)``.
        in_idx: Flat indices of the pairs' in-pixels, ``(N,)``, from
            [`pair_pixels`][radtrax.retrieval.absorption.pair_pixels].
        out_idx: Flat indices of their out-pixels, ``(N,)``.

    Returns:
        ``R``, ``(B,)``; NaN in a band where no pair is usable (its gradient
        there is zero, not NaN).

    Raises:
        ShapeError: If ``cube`` is not 3-D or the indices are not two
            ``(N,)`` arrays of one length.
        DegenerateInputError: If there are no pairs.
        ParameterError: If an index is outside ``[0, H · W)``.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.absorption import ratio_spectrum
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
    # A pair counts in a band only where both radiances are finite and the
    # out-pixel's is positive: a NaN in-pixel would otherwise reach the sum,
    # and through it every out-pixel's gradient                      (N, B)
    w = jnp.isfinite(l_in) & jnp.isfinite(l_out) & (l_out > 0)
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
