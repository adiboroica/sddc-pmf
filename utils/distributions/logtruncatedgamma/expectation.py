from typing import Literal
import warnings

import numpy as np
from mpmath import (
    hyper as mpmath_hyper,
    log as mpmath_log,
    gammainc as mpmath_gammainc,
    exp as mpmath_exp,
    extradps,
)

from utils.logger import Logger
from utils.special import gamma_log_lower


def expectation(
    shape, rate, version: Literal["scipy", "mpmath", "auto"] = "auto"
) -> np.ndarray:
    """
    Compute the expectation of the log of the Truncated Gamma distribution over (0, 1).
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
    log_numerator = (
        +shape * np.log(rate) + float(_log_hyp2f2(shape, rate)) - 2 * np.log(shape)
    )
    log_denominator = gamma_log_lower(shape, rate)

    return -np.exp(log_numerator - log_denominator)


def _mpmath_version_unvectorized(shape, rate):
    log_numerator = (
        +shape * mpmath_log(rate) + _log_hyp2f2(shape, rate) - 2 * mpmath_log(shape)
    )
    log_denominator = mpmath_log(mpmath_gammainc(shape, a=0, b=rate))

    return float(-mpmath_exp(log_numerator - log_denominator))


_mpmath_version = np.vectorize(_mpmath_version_unvectorized)


_hyp2f2_logger = Logger(
    "logtruncatedgamma.expectation", log_filename="hyp2f2_errors.log", log_level="ERROR"
)


def _log_hyp2f2_unvectorized(
    alpha,
    beta,
    additional_precision=0,
    max_additional_precision=1000,
    precision_step=20,
):
    while additional_precision <= max_additional_precision:
        try:
            # Temporarily increase precision
            with extradps(additional_precision):
                result = mpmath_log(
                    mpmath_hyper([alpha, alpha], [alpha + 1, alpha + 1], -beta)
                )
                return result
        except Exception as e:
            # Log the error
            if additional_precision != 0:
                _hyp2f2_logger.error(
                    f"Error in _log_hyp2f2_unvectorized with alpha={alpha}, beta={beta}, additional_precision={additional_precision}: {e}",
                )

            # Increase additional precision
            additional_precision += precision_step

    raise ValueError(
        f"Computation failed for alpha={alpha}, beta={beta} after reaching max additional precision {max_additional_precision}"
    )


# Create a vectorized version of the hyp2f2 function
_log_hyp2f2 = np.vectorize(_log_hyp2f2_unvectorized)
