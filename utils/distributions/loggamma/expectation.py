from typing import Literal
import warnings

import numpy as np
from scipy.special import psi
from mpmath import digamma as mpmath_digamma, log as mpmath_log


def expectation(shape, rate, version: Literal["scipy", "mpmath", "auto"] = "auto"):
    """
    Compute the expectation of the log-Gamma distribution.
    """
    # Ensure version is one of the allowed values
    if version not in {"scipy", "mpmath", "auto"}:
        raise ValueError("Invalid version specified. Use 'scipy', 'mpmath', or 'auto'.")

    if version == "scipy":
        return _scipy_version(shape, rate)

    if version == "mpmath":
        return _mpmath_version(shape, rate)

    shape = np.asarray(shape)
    rate = np.asarray(rate)

    # Output array initialized with NaNs to identify failures
    result = np.full(shape.shape, np.nan)

    # Attempt to calculate using scipy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            result = _scipy_version(shape, rate)
    except Exception:
        pass

    # Check for non-finite results and apply fallback
    failed_indices = ~np.isfinite(result)
    if np.any(failed_indices):
        result[failed_indices] = _mpmath_version(
            shape[failed_indices], rate[failed_indices]
        )

    return result


def _scipy_version(shape, rate):
    return psi(shape) - np.log(rate)


def _mpmath_version_unvectorized(shape, rate):
    return float(mpmath_digamma(shape) - mpmath_log(rate))


_mpmath_version = np.vectorize(_mpmath_version_unvectorized)
