import numpy as np
import pytest

from utils.distributions import poissonplus


@pytest.fixture
def tolerance():
    return 5e-3


@pytest.fixture
def rng():
    return np.random.default_rng(43)


@pytest.fixture
def num_samples_mc():
    return 10**5


"""
Test the implementation.
"""


@pytest.mark.parametrize("poisson_rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
def test_small_values(poisson_rate, tolerance, num_samples_mc, rng):
    _test(poisson_rate, tolerance, num_samples_mc, rng)


@pytest.mark.parametrize("poisson_rate", [0.1, 0.5, 1, 2, 3, 4, 5])
def test(poisson_rate, tolerance, num_samples_mc, rng):
    _test(poisson_rate, tolerance, num_samples_mc, rng)


@pytest.mark.parametrize("poisson_rate", [10, 20, 30, 40, 50])
def test_large_values(poisson_rate, tolerance, num_samples_mc, rng):
    _test(poisson_rate, tolerance, num_samples_mc, rng)


"""
Helper functions.
"""


def _test(poisson_rate, tolerance, num_samples_mc, rng):
    """
    Compare the expectation of the Poisson Plus distribution with the Monte Carlo estimate.
    """

    result = poissonplus.rate(poisson_rate)
    result_mc = _expectation_mc(poisson_rate, num_samples=num_samples_mc, rng=rng)

    close_to_mc = np.isclose(result, result_mc, atol=tolerance)

    assert (
        close_to_mc
    ), f"Failed for poisson_rate={poisson_rate}; Result: {result}; Expected: {result_mc} "


def _expectation_mc(
    poisson_rate,
    num_samples: int,
    rng: np.random.Generator,
):
    """
    Compute the expectation of the Poisson Plus distribution using Monte Carlo simulation.
    """

    def _sample_from_poisson_plus(poisson_rate: np.number, no_samples: int):
        count = 0
        samples = np.empty(no_samples)
        while count < no_samples:
            # Sample from the Poisson distribution
            new_samples = rng.poisson(poisson_rate, size=no_samples - count)
            # Keep only the positive samples
            positive_samples = new_samples[new_samples > 0]
            # Add the new samples
            samples[count : count + len(positive_samples)] = positive_samples
            # Update the count
            count += len(positive_samples)

        return samples

    # Sample from the Poisson Plus distribution
    samples = _sample_from_poisson_plus(poisson_rate, num_samples)

    # Compute the estimate of the expectation
    estimate = np.mean(samples)

    return estimate
