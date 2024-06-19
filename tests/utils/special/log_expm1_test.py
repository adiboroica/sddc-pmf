import numpy as np
import pytest

from utils.special import log_expm1


@pytest.fixture
def tolerance():
    return 1e-10


"""
Test the implementation.
"""


@pytest.mark.parametrize(
    "x",
    [
        1e-10,
        5e-10,
        1e-9,
        5e-9,
        1e-8,
        5e-8,
        1e-7,
        5e-7,
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
    ],
)
def test_small_values(x, tolerance):
    _test_formula(x, tolerance)


@pytest.mark.parametrize("x", [0.1, 0.5, 1, 2, 3, 4, 5])
def test(x, tolerance):
    _test_formula(x, tolerance)


@pytest.mark.parametrize(
    "x", [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
)
def test_approximation(x, tolerance):
    """
    For large values, the result should be close to x.
    """

    expected = x
    # Compute the result using the function
    result = log_expm1(x)

    assert np.isclose(
        result, expected, atol=tolerance
    ), f"Failed for x={x}; Result: {result}; Expected: {expected}"


"""
Helper functions.
"""


def _test_formula(x, tolerance):
    """
    For small values, the formula should be used.
    """

    # Compute the result using the function
    result = log_expm1(x)
    # Compute the result using the formula
    expected = _formula(x)

    assert np.isclose(
        result, expected, atol=tolerance
    ), f"Failed for x={x}; Result:{result}; Expected:{expected}"


def _formula(x):
    return np.log(np.expm1(x))
