from abc import ABC, abstractmethod
import numpy as np
import sparse
from scipy.sparse.linalg import svds

from pmf.hyperparams import ModelHyperParams
from pmf.params.params import Params


class InitXYStrategy(ABC):

    @abstractmethod
    def init(self, *args, **kwargs):
        pass


class InitXYsvd(InitXYStrategy):
    """
    Initialize the parameters for the x and y components
    using the Singular Value Decomposition (SVD) method.
    """

    def init(self, A: sparse.COO, hyperparams: ModelHyperParams, params: Params):
        # Extract the necessary variables.
        num_features = hyperparams.num_features
        N1 = params.N1
        N2 = params.N2

        # Compute the average of the data across the time dimension.
        data_avg = A.mean(axis=0)
        # Perform partial SVD on the average data with the number of features as the rank.
        U, D, V = svds(data_avg, k=num_features)
        V = V.T
        # Using the SVD, we initialize the shape parameters for the x and y components.

        # Initialize the parameters for the x component.
        x_shape = np.abs(U @ np.diag(np.sqrt(D)))
        x_rate = np.ones((N1, num_features))
        params.set_x(x_shape, x_rate)

        # Initialize the parameters for the y component.
        y_shape = np.abs(V @ np.diag(np.sqrt(D)))
        y_rate = np.ones((N2, num_features))
        params.set_y(y_shape, y_rate)


class InitXYconstant(InitXYStrategy):
    """
    Initialize the parameters for the x and y components
    with constant values.

    Parameters
    ----------
    :param x_constant: float
        The constant value for the x component.
    :param y_constant: float
        The constant value for the y component.
    """

    def __init__(
        self,
        x_constant=1.0,
        y_constant=1.0,
    ):
        self.x_constant = x_constant
        self.y_constant = y_constant

    def init(self, hyperparams: ModelHyperParams, params: Params):
        # Extract the necessary variables.
        num_features = hyperparams.num_features
        N1 = params.N1
        N2 = params.N2

        # Initialize the parameters for the x component.
        x_shape = np.full((N1, num_features), self.x_constant)
        x_rate = np.ones((N1, num_features))
        params.set_x(x_shape, x_rate)

        # Initialize the parameters for the y component.
        y_shape = np.full((N2, num_features), self.y_constant)
        y_rate = np.ones((N2, num_features))
        params.set_y(y_shape, y_rate)


class InitXYrandomUniform(InitXYStrategy):
    """
    Initialize the parameters for the x and y components
    using the random uniform method.

    Parameters
    ----------
    :param x_min: float
        The minimum value for the x's uniform distribution.
    :param x_max: float
        The maximum value for the x's uniform distribution.

    :param y_min: float
        The minimum value for the y's uniform distribution.
    :param y_max: float
        The maximum value for the y's uniform distribution.

    :param rng: np.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        x_min=1.0,
        x_max=1.0,
        y_min=1.0,
        y_max=1.0,
        rng=np.random.default_rng(),
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.rng = rng

    def init(self, hyperparams: ModelHyperParams, params: Params):
        # Extract the necessary variables.
        num_features = hyperparams.num_features
        N1 = params.N1
        N2 = params.N2

        # Initialize the parameters for the x component.
        x_shape = self.rng.uniform(self.x_min, self.x_max, size=(N1, num_features))
        x_rate = np.ones((N1, num_features))
        params.set_x(x_shape, x_rate)

        # Initialize the parameters for the y component.
        y_shape = self.rng.uniform(self.y_min, self.y_max, size=(N2, num_features))
        y_rate = np.ones((N2, num_features))
        params.set_y(y_shape, y_rate)


class InitXYrandomBeta(InitXYStrategy):
    """
    Initialize the parameters for the x and y components
    using the random beta method.

    Parameters
    ----------
    :param x_alpha: float
        The alpha parameter for the x's beta distribution.
    :param x_beta: float
        The beta parameter for the x's beta distribution.

    :param y_alpha: float
        The alpha parameter for the y's beta distribution.
    :param y_beta: float
        The beta parameter for the y's beta distribution.

    :param rng: np.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        x_alpha=1.0,
        x_beta=1.0,
        y_alpha=1.0,
        y_beta=1.0,
        rng=np.random.default_rng(),
    ):
        self.x_alpha = x_alpha
        self.x_beta = x_beta
        self.y_alpha = y_alpha
        self.y_beta = y_beta
        self.rng = rng

    def init(self, hyperparams: ModelHyperParams, params: Params):
        # Extract the necessary variables.
        num_features = hyperparams.num_features
        N1 = params.N1
        N2 = params.N2

        # Initialize the parameters for the x component.
        x_shape = self.rng.beta(self.x_alpha, self.x_beta, size=(N1, num_features))
        x_rate = np.ones((N1, num_features))
        params.set_x(x_shape, x_rate)

        # Initialize the parameters for the y component.
        y_shape = self.rng.beta(self.y_alpha, self.y_beta, size=(N2, num_features))
        y_rate = np.ones((N2, num_features))
        params.set_y(y_shape, y_rate)


class InitXYrandomGamma(InitXYStrategy):
    """
    Initialize the parameters for the x and y components
    using the random gamma method.

    Parameters
    ----------
    :param x_shape_shape: float
        The shape parameter for the x's gamma distribution.
    :param x_rate: float
        The rate parameter for the x's gamma distribution.

    :param y_shape: float
        The shape parameter for the y's gamma distribution.
    :param y_rate: float
        The rate parameter for the y's gamma distribution.

    :param rng: np.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        x_shape=1.0,
        x_rate=1.0,
        y_shape=1.0,
        y_rate=1.0,
        rng=np.random.default_rng(),
    ):
        self.x_shape = x_shape
        self.x_rate = x_rate
        self.y_shape = y_shape
        self.y_rate = y_rate
        self.rng = rng

    def init(self, hyperparams: ModelHyperParams, params: Params):
        # Extract the necessary variables.
        num_features = hyperparams.num_features
        N1 = params.N1
        N2 = params.N2

        # Initialize the parameters for the x component.
        x_shape = self.rng.gamma(self.x_shape, 1 / self.x_rate, size=(N1, num_features))
        x_rate = np.ones((N1, num_features))
        params.set_x(x_shape, x_rate)

        # Initialize the parameters for the y component.
        y_shape = self.rng.gamma(self.y_shape, 1 / self.y_rate, size=(N2, num_features))
        y_rate = np.ones((N2, num_features))
        params.set_y(y_shape, y_rate)
