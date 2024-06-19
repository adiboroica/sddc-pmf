from typing import Literal
import warnings

import numpy as np
from scipy.special import loggamma, gammainc
from mpmath import gammainc as mpmath_gammainc, log as mpmath_log


def gamma_log_lower(a, z, version: Literal["scipy", "mpmath", "auto"] = "auto"):
    """
    Compute the log of the unregularized lower incomplete gamma function.
    """
    # Ensure version is one of the allowed values
    if version not in {"scipy", "mpmath", "auto"}:
        raise ValueError("Invalid version specified. Use 'scipy', 'mpmath', or 'auto'.")

    if version == "scipy":
        return _scipy_version(a, z)

    if version == "mpmath":
        return _mpmath_version(a, z)

    a = np.asarray(a)
    z = np.asarray(z)

    # Output array initialized with NaNs to identify failures
    result = np.full(a.shape, np.nan)

    # Attempt to calculate using scipy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            result = _scipy_version(a, z)
    except Exception:
        pass

    # Check for non-finite results and apply fallback
    failed_indices = ~np.isfinite(result)
    if np.any(failed_indices):
        result[failed_indices] = _mpmath_version(a[failed_indices], z[failed_indices])

    return result


def _scipy_version(a, z):
    return loggamma(a) + np.log(gammainc(a, z))


def _mpmath_version_unvectorized(a, z):
    return float(mpmath_log(mpmath_gammainc(a, a=0, b=z)))


_mpmath_version = np.vectorize(_mpmath_version_unvectorized)
