import numpy as np
import sparse

from pmf.hyperparams import ModelHyperParams
from pmf.params.params import Params


def init_zn(A: sparse.COO, hyperparams: ModelHyperParams, params: Params):
    """
    Initialize the parameters for the Z, N components.
    """

    # Extract the necessary variables.
    T = params.time
    N1 = params.N1
    N2 = params.N2
    d = hyperparams.num_features
    A = A
    num_data_points = A.data.size

    # Initialize the values for the N component.
    n_values = np.ones(num_data_points)

    # Initialize the values for the Z component.
    z_values = np.full(num_data_points * d, 1 / d)

    # Tile the original coords to repeat for each feature
    repeated_coords = np.repeat(A.coords, d, axis=1)
    # Create feature indices
    feature_indices = np.tile(np.arange(d), num_data_points)
    # Create the new coords for the Z component
    z_coords = np.concatenate((repeated_coords, feature_indices[None, :]), axis=0)

    # Create the sparse COO matrix for the N component.
    zn_poisson_rate = sparse.COO(
        A.coords,
        n_values,
        shape=(T, N1, N2),
    )

    # Create the sparse COO matrix for the Z component.
    zn_multinomial_prob = sparse.COO(
        z_coords,
        z_values,
        shape=(T, N1, N2, d),
    )

    # Set the Z, N components in the parameters.
    params.set_zn(zn_poisson_rate, zn_multinomial_prob)
