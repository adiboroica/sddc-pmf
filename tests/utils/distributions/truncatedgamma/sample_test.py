import numpy as np
import pytest
from scipy.stats import gamma, kstest, cramervonmises

from utils.distributions import truncatedgamma

from ._params import (
    params,
    params_low,
    params_normal,
    params_high,
    params_low_high_combinations,
)


@pytest.fixture
def rng():
    return np.random.default_rng(43)


@pytest.fixture
def size():
    return 10**5


"""
Test the implementation.
"""


@pytest.mark.parametrize(params, params_low)
def test_low_values(shape, rate, size, rng):
    _test(shape, rate, size, rng)


@pytest.mark.parametrize(params, params_normal)
def test_normal_values(shape, rate, size, rng):
    _test(shape, rate, size, rng)


@pytest.mark.parametrize(params, params_high)
def test_high_values(shape, rate, size, rng):
    _test(shape, rate, size, rng)


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_low_high_values(shape, rate, size, rng):
    _test(shape, rate, size, rng)


"""
Helper functions.
"""


def _test(shape, rate, size: int, rng: np.random.Generator = np.random.default_rng()):
    """
    Check that the samples are within the interval (0, 1) and perform a KS test.
    """

    samples = truncatedgamma.sample(shape, rate, size=size, rng=rng)

    # Check that all samples are within the interval (0, 1)
    assert np.all(
        (samples > 0) & (samples < 1)
    ), "Not all samples are within the interval (0, 1)"

    # Perform a KS test
    _, ks_test_p_value = _ks_test(samples, shape, rate)
    # Perform a Cramer-von Mises test
    cramer_von_mises_result = cramer_von_mises_test(samples, shape, rate)

    # Check that at least one of the tests passes
    ks_test_passed = ks_test_p_value > 0.05
    cramer_von_mises_test_passed = cramer_von_mises_result.pvalue > 0.05

    assert ks_test_passed or cramer_von_mises_test_passed, (
        f"KS test failed with p-value {ks_test_p_value}; "
        f"Cramer-von Mises test failed with p-value {cramer_von_mises_result.pvalue}"
    )


def _ks_test(samples, shape, rate):
    """
    Perform a Kolmogorov-Smirnov test to compare the empirical and theoretical distributions.
    """
    cdf_truncated_gamma = lambda x: np.exp(
        gamma.logcdf(x, a=shape, scale=1 / rate)
        - gamma.logcdf(1, a=shape, scale=1 / rate)
    )
    return kstest(samples, cdf_truncated_gamma)


def cramer_von_mises_test(samples, shape, rate):
    cdf_truncated_gamma = lambda x: np.exp(
        gamma.logcdf(x, a=shape, scale=1 / rate)
        - gamma.logcdf(1, a=shape, scale=1 / rate)
    )
    result = cramervonmises(samples, cdf_truncated_gamma)
    return result
