"""Exception hierarchy for `radtrax`.

Every error raised by this package derives from
[`RadtraxError`][radtrax.RadtraxError], so a caller can trap the whole
package with a single ``except`` clause. Concrete errors also inherit from the
closest standard-library exception, so code written against the standard
library keeps working: a [`ShapeError`][radtrax.ShapeError] is still caught by
``except ValueError``.

The classes name what went wrong, not the module that raised it. Checks that
run under ``jit`` (``eqx.error_if``) cannot carry a radtrax class: they raise
a ``RuntimeError`` at run time, ``equinox.EquinoxRuntimeError`` under
``eqx.filter_jit`` and ``jax.errors.JaxRuntimeError`` under ``jax.jit``.
"""

from __future__ import annotations


__all__ = [
    "ArgumentTypeError",
    "DegenerateInputError",
    "GridError",
    "MissingExtraError",
    "MissingFileError",
    "MissingKeyError",
    "MissingTableError",
    "ParameterError",
    "RadtraxError",
    "ShapeError",
    "TableBuildError",
]


class RadtraxError(Exception):
    """Base class for every exception raised by ``radtrax``.

    Examples:
        >>> from radtrax import RadtraxError
        >>> issubclass(RadtraxError, Exception)
        True
    """


class ShapeError(RadtraxError, ValueError):
    """An array has the wrong shape or rank, or two arrays disagree in shape.

    Examples:
        >>> from radtrax import ShapeError
        >>> issubclass(ShapeError, ValueError)
        True
    """


class GridError(RadtraxError, ValueError):
    """A grid is not strictly monotone, has too few points, or does not match another grid.

    Also raised for a value that falls outside a table's grid.

    Examples:
        >>> from radtrax import GridError
        >>> issubclass(GridError, ValueError)
        True
    """


class ParameterError(RadtraxError, ValueError):
    """A scalar or option is outside its allowed range or set, or a required one is missing.

    Examples:
        >>> from radtrax import ParameterError
        >>> issubclass(ParameterError, ValueError)
        True
    """


class DegenerateInputError(RadtraxError, ValueError):
    """Well-shaped input that cannot be used: too few samples, no data yet, or a zero norm.

    Examples:
        >>> from radtrax import DegenerateInputError
        >>> issubclass(DegenerateInputError, ValueError)
        True
    """


class ArgumentTypeError(RadtraxError, TypeError):
    """An argument is of a type the function does not accept.

    Examples:
        >>> from radtrax import ArgumentTypeError
        >>> issubclass(ArgumentTypeError, TypeError)
        True
    """


class MissingKeyError(RadtraxError, KeyError):
    """A gas, band or variable is missing from a state, table or dataset.

    Examples:
        >>> from radtrax import MissingKeyError
        >>> issubclass(MissingKeyError, KeyError)
        True
    """


class MissingExtraError(RadtraxError, ImportError):
    """An optional dependency is not installed; the message names the extra.

    Examples:
        >>> from radtrax import MissingExtraError
        >>> issubclass(MissingExtraError, ImportError)
        True
    """


class MissingFileError(RadtraxError, FileNotFoundError):
    """An input file or glob pattern matched nothing.

    Examples:
        >>> from radtrax import MissingFileError
        >>> issubclass(MissingFileError, FileNotFoundError)
        True
    """


class MissingTableError(MissingFileError):
    """A table (cross sections, a solar spectrum) is not in the cache; its message names the command that builds or fetches it.

    Examples:
        >>> from radtrax import MissingTableError
        >>> issubclass(MissingTableError, FileNotFoundError)
        True
    """


class TableBuildError(RadtraxError, RuntimeError):
    """Building a cross-section table failed: a HITRAN fetch or a line-by-line computation.

    Examples:
        >>> from radtrax import TableBuildError
        >>> issubclass(TableBuildError, RuntimeError)
        True
    """
