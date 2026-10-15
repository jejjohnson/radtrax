"""Exception hierarchy for `radtrax`.

Every error raised by this package derives from
[`RadtraxError`][radtrax.RadtraxError], so a caller can trap the whole
package with a single ``except`` clause. Concrete errors also inherit from the
closest standard-library exception, so code written against the standard
library keeps working.
"""

from __future__ import annotations


__all__ = ["MissingTableError", "RadtraxError"]


class RadtraxError(Exception):
    """Base class for every exception raised by ``radtrax``.

    Examples:
        >>> from radtrax import RadtraxError
        >>> issubclass(RadtraxError, Exception)
        True
    """


class MissingTableError(RadtraxError, FileNotFoundError):
    """A cross-section table is not in the cache; its message names the build command.

    Examples:
        >>> from radtrax import MissingTableError
        >>> issubclass(MissingTableError, FileNotFoundError)
        True
    """
