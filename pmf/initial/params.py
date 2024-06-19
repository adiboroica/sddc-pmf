import numpy as np

from pmf.hyperparams import ModelHyperParams
from utils.distributions import truncatedgamma


class InitialModelParams:
    """
    Parameters for the initial model.
    """

    def __init__(
        self,
        time: int,
        N1: int,
        N2: int,
        x: np.ndarray,
        y: np.ndarray,
        zetax: np.ndarray,
        zetay: np.ndarray,
        rhox: np.ndarray,
        rhoy: np.ndarray,
    ):
        """
        Store the model parameters.
        """

        # Store the parameters.
        self.time = time
        self.N1 = N1
        self.N2 = N2
        self.x = x
        self.y = y
        self.zetax = zetax
        self.zetay = zetay
        self.rhox = rhox
        self.rhoy = rhoy

        # Compute and store the Poisson rate.
        self.poisson_rate = np.einsum(
            "ti, tj, ir, jr -> tij",
            rhox,
            rhoy,
            x,
            y,
        )


def simulate_initial_model_params(
    hyperparams: ModelHyperParams,
    time: int,
    N1: int,
    N2: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> InitialModelParams:
    """
    Simulate the initial model parameters.

    Parameters
    ----------
    hyperparams : InitialModelHyperParams
      The hyperparameters for the initial model.

    Returns
    -------
    The simulated model parameters.
    """

    # Simulate the rho values.
    rhox = truncatedgamma.sample(
        shape=hyperparams.alpha_x,
        rate=hyperparams.beta_x,
        size=(time, N1),
        rng=rng,
    )
    rhoy = truncatedgamma.sample(
        shape=hyperparams.alpha_y,
        rate=hyperparams.beta_y,
        size=(time, N2),
        rng=rng,
    )

    # Simulate the zheta values.
    zetax = rng.gamma(hyperparams.b_x, scale=1 / hyperparams.c_x, size=N1)
    zetay = rng.gamma(hyperparams.b_y, scale=1 / hyperparams.c_y, size=N2)

    # Simulate the x and y values.
    x = np.zeros((N1, hyperparams.num_features))
    for i in range(N1):
        x[i, :] = rng.gamma(
            hyperparams.a_x,
            scale=1 / zetax[i],
            size=hyperparams.num_features,
        )
    y = np.zeros((N2, hyperparams.num_features))
    for j in range(N2):
        y[j, :] = rng.gamma(
            hyperparams.a_y,
            scale=1 / zetay[j],
            size=hyperparams.num_features,
        )

    return InitialModelParams(
        time=time,
        N1=N1,
        N2=N2,
        x=x,
        y=y,
        zetax=zetax,
        zetay=zetay,
        rhox=rhox,
        rhoy=rhoy,
    )
