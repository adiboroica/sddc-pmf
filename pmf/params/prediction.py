import numpy as np
from scipy.special import logit, expit
from pmdarima import auto_arima

from pmf.params.estimates import ParamsEstimates


class PredictionParams:

    def __init__(self, params_estimates: ParamsEstimates):
        # Fit the ARIMA models to the rhos
        self._rhox_arima_models = self._fit_arima_models(params_estimates.rhox)
        self._rhoy_arima_models = self._fit_arima_models(params_estimates.rhoy)

    def predict_rhox(self, n_periods=1):
        """
        Predict the future rhox values using the ARIMA models.
        """
        return self._predict_rhos(self._rhox_arima_models, n_periods)

    def predict_rhoy(self, n_periods=1):
        """
        Predict the future rhoy values using the ARIMA models.
        """
        return self._predict_rhos(self._rhoy_arima_models, n_periods)

    @staticmethod
    def _predict_rhos(rho_models, n_periods=1):
        """
        Predict the future rho values using the ARIMA model.
        """

        # Extract the necessary variables
        N = len(rho_models)

        predictions = np.zeros((n_periods, N))
        for i, rho_model in enumerate(rho_models):
            # Compute the future estimate
            predictions_i = rho_model.predict(n_periods=n_periods)

            # Transform the future estimate back to the (0, 1) interval
            predictions_i = expit(predictions_i)

            # Store the prediction
            predictions[:, i] = predictions_i

        return predictions

    @staticmethod
    def _fit_arima_models(rhos: np.ndarray) -> np.ndarray:
        """
        Fit ARIMA models to the given rhos.

        We're assuming that rhos is of shape (T, N),
        where T is the time dimension and N is the number of components.
        """

        # Extract the necessary variables
        N = rhos.shape[1]

        arima_models = []
        for i in range(N):
            # Extract the rho time series for the i-th index
            time_series = rhos[:, i]

            # Transform the time series to R (the real numbers)
            time_series = logit(time_series)

            # Fit the ARIMA model
            model = auto_arima(
                time_series,
                max_p=7,
                max_q=7,
            )

            # Store the model
            arima_models.append(model)

        return arima_models
