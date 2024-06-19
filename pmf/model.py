import pickle
import numpy as np
from pmf.params.estimates import ParamsEstimates
from pmf.params.init import initialize_params
import sparse

from pmf.vi import cavi, CaviParams
from pmf.hyperparams import ModelHyperParams
from pmf.params import InitParams, PredictionParams
from utils.logger import Logger


class PMFModel:
    """
    Variational Inference Family to estimate the parameters of the given distributions.

    The model assumes the following distributions:
      - Poisson distribution for the Z, N components.
      - Gamma distribution for the x, y, zeta-x, and zeta-y components.
      - Truncated Gamma on (0, 1) for the rho-x and rho-y component

    The updates are performed using the Coordinate Ascent Variational Inference (CAVI) method.
    """

    def __init__(
        self, hyperparams: ModelHyperParams, logger: Logger = Logger("pmf_model")
    ):
        """
        Initialize the model with the given hyperparameters.
        """

        self.hyperparams = hyperparams
        self.logger = logger

        self.params = None
        self.quantities = None
        self.prediction_params = None

        self.eval_indices = None
        self.elbo_values = None
        self.elbo_has_converged = None

        self.trained = False

    def fit(
        self,
        data: sparse.COO,
        init_params: InitParams = InitParams(),
        cavi_params: CaviParams = CaviParams(),
        compute_prediction_params: bool = True,
    ):
        """
        Fit the model to the given data using the CAVI method.

        Parameters
        ----------
        :param data: sparse.COO
            The data matrix to fit the model to.
        :param elbo_interval: int
            The interval to print the ELBO values at.
        :param elbo_tol: float
            The tolerance to check for convergence of the ELBO values.
        :param max_iterations: int
            The maximum number of iterations to perform the CAVI method.
        """

        self.init_params = init_params
        self.cavi_params = cavi_params

        self.logger.info("Initializing Parameters...")

        # Initialize the parameters
        self.params, self.inference_quantities, self.evaluation_quantities, updates = (
            initialize_params(
                self.init_params, data, self.hyperparams, logger=self.logger
            )
        )

        # Create an object to store the estimates
        self.estimates = ParamsEstimates(self.inference_quantities)

        self.logger.info("All Parameters initialized.\n")
        self.logger.info("Training the model...")

        # Perform the CAVI method
        self.cavi_results = cavi(
            cavi_params=cavi_params,
            data=data,
            hyperparams=self.hyperparams,
            params=self.params,
            inference_quantities=self.inference_quantities,
            evaluation_quantities=self.evaluation_quantities,
            params_estimates=self.estimates,
            updates=updates,
            logger=self.logger,
        )

        self.logger.info("Training completed!\n")

        # Compute the prediction parameters if needed
        if compute_prediction_params:
            self.compute_prediction_params()

        self.trained = True

    def compute_prediction_params(self):
        """
        Compute the prediction parameters for the model.
        """
        self.logger.info("Computing the prediction parameters...")
        self.prediction_params = PredictionParams(self.estimates)
        self.logger.info("Prediction parameters computed!\n")

    def predict(
        self, indices_axis0: np.ndarray, indices_axis1, n_periods: int = 1
    ) -> np.ndarray:
        """
        Predict the probabilities of interaction between the x and y components at time T + 1
        for arrays of indices.

        Parameters
        ----------
        :param indices: np.ndarray of shape (2, n)
            The indices to predict the probabilities for.
        """

        # Check if the model has been trained
        if not self.trained:
            raise ValueError("Model has not been trained yet!")

        # Compute the prediction parameters if needed
        if self.prediction_params is None:
            self.compute_prediction_params()

        # Keep in mind that we must have the same number of indices for both axes
        if len(indices_axis0) != len(indices_axis1):
            raise ValueError("Indices must have the same length!")

        # We are going to denote num_samples = len(indices_axis0) = len(indices_axis1)

        # Extract the relevant estimates and future estimates for the given indices
        estimate_x = self.estimates.x[indices_axis0]  # shape: (num_samples, d)
        estimate_y = self.estimates.y[indices_axis1]  # shape: (num_samples, d)
        future_estimate_rhox = self.prediction_params.predict_rhox(n_periods)[
            :, indices_axis0
        ]  # shape: (n_periods, num_samples)
        future_estimate_rhoy = self.prediction_params.predict_rhoy(n_periods)[
            :, indices_axis1
        ]  # shape: (n_periods, num_samples)

        # Compute the Poisson rates for each pair
        poisson_rates = np.einsum(
            "ti, ti, ir, ir -> ti",
            future_estimate_rhox,
            future_estimate_rhoy,
            estimate_x,
            estimate_y,
        )  # shape: (n_periods, num_samples)

        # Compute the probabilities using the formula -expm1(-poisson_rate)
        probabilities = -np.expm1(-poisson_rates)

        # Squeeze the probabilities if needed
        probabilities = np.squeeze(probabilities)

        return probabilities

    def save(self, filename):
        """
        Save the model to a file.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

        message = f"Model saved to {filename}!"
        self.logger.info(message)
        if not self.logger.log_to_console:
            print(message)

    @staticmethod
    def load(filename):
        """
        Load the model from a file.
        """
        with open(filename, "rb") as file:
            model = pickle.load(file)

        message = f"Model loaded from {filename}!"
        model.logger.info(message)
        if not model.logger.log_to_console:
            print(message)

        return model
