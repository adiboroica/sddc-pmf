import numpy as np


# Use a threshold to determine when to approximate
THRESHOLD = 10


def log_expm1(x):
    """
    Compute log(exp(x) - 1) in a numerically stable way.

    For small x, we use the formula.
    For large x, we approximate the result as x.
    """

    x = np.asarray(x, dtype=float)

    # Create masks for large and small x
    small_mask = x < THRESHOLD
    large_mask = ~small_mask

    # Compute the result
    result = np.empty_like(x)
    result[small_mask] = _formula(x[small_mask])
    result[large_mask] = x[large_mask]

    return result


def _formula(x):
    return np.log(np.expm1(x))
