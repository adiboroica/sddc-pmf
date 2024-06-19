import numpy as np
import sparse

from pmf.params.params import Params
from pmf.params.quantities import InferenceQuantities
from pmf.params.updates import Updates


def init_rho(
    A: sparse.COO,
    params: Params,
    inference_quantities: InferenceQuantities,
    updates: Updates,
):
    """
    Initialize the parameters for the rho components.
    """

    # Estimate the expectation of the rho-y component.
    inference_quantities._etg_rhoy = np.mean(A, axis=1).todense()

    # Initialize the parameters for the rho-x component.
    rhox_shape, rhox_rate = updates.rhox()
    params.set_rhox(rhox_shape, rhox_rate)

    # Initialize the parameters for the rho-y component.
    rhoy_shape, rhoy_rate = updates.rhoy()
    params.set_rhoy(rhoy_shape, rhoy_rate)
