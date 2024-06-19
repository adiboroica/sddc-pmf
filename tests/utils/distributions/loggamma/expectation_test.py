from typing import Literal
import pytest

import numpy as np
from scipy.stats import loggamma as loggamma_dist

from utils.distributions import loggamma

from ._params import (
    params,
    params_low,
    params_normal,
    params_high,
    params_low_high_combinations,
)


@pytest.fixture
def tolerance():
    return 1e-10


"""
Test the implementation.
"""


@pytest.mark.parametrize(params, params_low)
def test_low_values(shape, rate, tolerance):
    _test(shape, rate, tolerance)


@pytest.mark.parametrize(params, params_normal)
def test_normal_values(shape, rate, tolerance):
    _test(shape, rate, tolerance)


@pytest.mark.parametrize(params, params_high)
def test_high_values(shape, rate, tolerance):
    _test(shape, rate, tolerance)


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_low_high_combinations(shape, rate, tolerance):
    _test(shape, rate, tolerance)


"""
Test the fallback when the implementation fails.
"""


@pytest.mark.parametrize(params, params_low)
def test_fallback_low_values(shape, rate, tolerance):
    _test(shape, rate, tolerance, version="mpmath")


@pytest.mark.parametrize(params, params_normal)
def test_fallback_normal_values(shape, rate, tolerance):
    _test(shape, rate, tolerance, version="mpmath")


@pytest.mark.parametrize(params, params_high)
def test_fallback_high_values(shape, rate, tolerance):
    _test(shape, rate, tolerance, version="mpmath")


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_fallback_low_high_combinations(shape, rate, tolerance):
    _test(shape, rate, tolerance, version="mpmath")


"""
Helper functions.
"""


def _test(shape, rate, tolerance, version: Literal["auto", "mpmath"] = "auto"):
    """
    Compare the expectation of the log-gamma distribution with the one from the scipy library.

    Note that the scipy library uses only the shape parameter for the log-gamma distribution.
    If X ~ Gamma(shape, rate), then rate*X ~ Gamma(shape, 1).
    For log(rate*X) ~ LogGamma(shape), we use the scipy library to compute its expectation.
    Then E[log(X)] = E[log(rate*X)] - log(rate) gives us the scipy expectation for the log-gamma distribution.
    """

    result = loggamma.expectation(shape, rate, version=version)
    result_scipy = loggamma_dist.mean(shape) - np.log(rate)

    assert np.isclose(
        result, result_scipy, atol=tolerance
    ), f"Failed for shape={shape}, rate={rate}; Result: {result}; [Scipy] Expected: {result_scipy}"
