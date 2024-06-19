from typing import Literal
import pytest

import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma

from utils.distributions import truncatedgamma, logtruncatedgamma

from ._params import (
    params,
    params_low,
    params_normal,
    params_high,
    params_low_high_combinations,
)


@pytest.fixture
def tolerance():
    return 5e-2


@pytest.fixture
def rng():
    return np.random.default_rng(43)


@pytest.fixture
def num_samples_mc():
    return 10**5


"""
Test the implementation.
"""


@pytest.mark.parametrize(params, params_low)
def test_low_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng)


@pytest.mark.parametrize(params, params_normal)
def test_normal_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng)


@pytest.mark.parametrize(params, params_high)
def test_high_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng)


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_low_high_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng)


"""
Test the fallback when the implementation fails.
"""


@pytest.mark.parametrize(params, params_low)
def test_fallback_low_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng, version="mpmath")


@pytest.mark.parametrize(params, params_normal)
def test_fallback_normal_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng, version="mpmath")


@pytest.mark.parametrize(params, params_high)
def test_fallback_high_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng, version="mpmath")


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_fallback_low_high_values(shape, rate, tolerance, num_samples_mc, rng):
    _test(shape, rate, tolerance, num_samples_mc, rng, version="mpmath")


"""
Helper functions.
"""


def _test(
    shape,
    rate,
    tolerance,
    num_samples_mc,
    rng,
    version: Literal["auto", "mpmath"] = "auto",
):
    """
    Compare the expectation of the log truncated gamma distribution computed using different methods.
    The test passes if our implementation is close to at least one of the other methods.

    In some cases, one of the methods may not converge,
    so it is preferable to verify that at least one method aligns closely with our estimate.
    """

    result = logtruncatedgamma.expectation(shape, rate, version=version)
    result_mc = _expectation_mc(shape, rate, num_samples=num_samples_mc, rng=rng)
    result_quad = _expectation_quad(shape, rate)

    close_to_mc = np.isclose(result, result_mc, atol=tolerance)
    close_to_quad = np.isclose(result, result_quad, atol=tolerance)

    assert (
        close_to_mc or close_to_quad
    ), f"Failed for shape={shape}, rate={rate}; Result: {result}; [MC] Expected: {result_mc}; [Quad] Expected: {result_quad}"


def _expectation_quad_unvectorized(shape, rate) -> np.ndarray:
    """
    Compute the expectation of the log-truncated gamma distribution over (0, 1) 
    using quadrature.
    """

    # Integrand: log(x) * truncated gamma PDF (x)
    integrand = lambda x: np.log(x) * np.exp(
        gamma.logpdf(x, shape, scale=1 / rate) - gamma.logcdf(1, shape, scale=1 / rate)
    )

    expectation, _ = quad(integrand, 0, 1)

    return expectation


_expectation_quad = np.vectorize(_expectation_quad_unvectorized)


def _expectation_mc_unvectorized(
    shape,
    rate,
    num_samples=10**4,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """
    Compute the expectation of the log-truncated gamma distribution over (0, 1) 
    using Monte Carlo.
    """

    # Sample from the truncated gamma distribution
    samples = truncatedgamma.sample(shape, rate, size=num_samples, rng=rng)

    # Compute the estimate of the expectation
    estimate = np.mean(np.log(samples))

    return estimate


_expectation_mc = np.vectorize(_expectation_mc_unvectorized)
