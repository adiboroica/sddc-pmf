import numpy as np
import sparse


def rate(poisson_rate):
    """
    Compute the Poisson plus rate given a sparse matrix of Poisson rates.

    When close to zero, we use the limit of the function, which is 1.
    When far from zero, we use the formula.
    """

    if np.isscalar(poisson_rate):
        return _formula(poisson_rate) if np.abs(poisson_rate) >= THRESHOLD else 1

    # Extract the data from the Poisson rate
    data = poisson_rate.data

    # Mask for the values that are greater than the threshold
    mask = np.abs(data) >= THRESHOLD

    # Apply the formula to the values that are greater than the threshold
    poisson_plus_rate_data = np.ones_like(data)
    poisson_plus_rate_data[mask] = _formula(data[mask])

    # Create the new Poisson plus rate sparse matrix
    poisson_plus_rate = sparse.COO(
        poisson_rate.coords, poisson_plus_rate_data, shape=poisson_rate.shape
    )

    return poisson_plus_rate


# Define a threshold to determine when to use the formula
THRESHOLD = 1e-5


def _formula(rate):
    return rate / -np.expm1(-rate)
