import numpy as np
import pytest
from mpmath import gammainc as mpmath_gammainc, log as mpmath_log

from utils.special import gamma_log_lower

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
def test_low_values(a, z, tolerance):
    _test(a, z, tolerance)


@pytest.mark.parametrize(params, params_normal)
def test_normal_values(a, z, tolerance):
    _test(a, z, tolerance)


@pytest.mark.parametrize(params, params_high)
def test_high_values(a, z, tolerance):
    _test(a, z, tolerance)


@pytest.mark.parametrize(params, params_low_high_combinations)
def test_low_high_values(a, z, tolerance):
    _test(a, z, tolerance)


"""
Helper functions.
"""


def _test(a, z, tolerance):
    # Compute the result using the mpmath library
    expected = float(mpmath_log(mpmath_gammainc(a, a=0, b=z)))

    # Compute the result using our implementation
    result = gamma_log_lower(a, z)

    assert np.isclose(
        result, expected, atol=tolerance
    ), f"Failed for a={a}, z={z}; Result:{result}; Expected:{expected}"
