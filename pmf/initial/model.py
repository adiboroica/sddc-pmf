import numpy as np
import sparse

from pmf.hyperparams import ModelHyperParams
from pmf.initial.params import InitialModelParams


class InitialModel:
    """
    Initial model.
    """

    def __init__(self, hyperparams: ModelHyperParams, params: InitialModelParams):
        self.hyperparams = hyperparams
        self.params = params

    def simulate_data(self, rng: np.random.Generator = np.random.default_rng()):
        """
        Simulate some data from this model.

        Returns
        -------
        :return sparse.COO: The simulated data.
        """

        # Simulate the N values.
        N = rng.poisson(
            self.params.poisson_rate,
            size=(
                self.params.time,
                self.params.N1,
                self.params.N2,
            ),
        )

        # Compute the A values (which is just the indicator function of N)
        # and transform it into a sparse matrix.
        A = np.where(N > 0, 1, 0)
        A = sparse.COO.from_numpy(A)

        return A
