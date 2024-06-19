import sparse
import numpy as np

from pmf.params.estimates import ParamsEstimates
from utils.special import log_expm1


def log_likelihood(A: sparse.COO, params_estimates: ParamsEstimates):
    """
    Compute the likelihood of the data in the model.
    """

    # Estimate x, y, rho-x, and rho-y
    x = params_estimates.x
    y = params_estimates.y
    rhox = params_estimates.rhox
    rhoy = params_estimates.rhoy

    # Comput the first sum in the log likelihood
    prod_xy = {}
    first_sum = 0
    for _, (t, i, j) in enumerate(A.coords.T):
        # If the product of x and y is not already computed, compute it
        if (i, j) not in prod_xy:
            prod_xy[i, j] = np.dot(x[i], y[j])

        first_sum += log_expm1(rhox[t, i] * rhoy[t, j] * prod_xy[i, j])

    # Compute the second sum in the log likelihood
    second_sum = np.sum(
        np.einsum("ti, ir -> tr", rhox, x) * np.einsum("tj, jr -> tr", rhoy, y)
    )

    return first_sum - second_sum
