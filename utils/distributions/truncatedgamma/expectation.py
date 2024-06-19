from typing import Literal
import warnings

import numpy as np
from scipy.special import gammainc
from mpmath import gammainc as mpmath_gammainc, log as mpmath_log, exp as mpmath_exp


def expectation(shape, rate, version: Literal["scipy", "mpmath", "auto"] = "auto"):
    """
    Compute the expectation of the Truncated Gamma distribution over [0, 1].
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
    log_numerator = np.log(shape) + np.log(gammainc(shape + 1, rate))
    log_denominator = np.log(rate) + np.log(gammainc(shape, rate))

    return np.exp(log_numerator - log_denominator)


def _mpmath_version_unvectorized(shape, rate):
    log_numerator = mpmath_log(mpmath_gammainc(shape + 1, a=0, b=rate))
    log_denominator = mpmath_log(rate) + mpmath_log(mpmath_gammainc(shape, a=0, b=rate))

    return float(mpmath_exp(log_numerator - log_denominator))


_mpmath_version = np.vectorize(_mpmath_version_unvectorized)
