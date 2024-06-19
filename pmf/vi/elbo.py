import numpy as np
from scipy.special import loggamma

from pmf.hyperparams import ModelHyperParams
from pmf.params import Params, EvaluationQuantities, InferenceQuantities
from utils.special import log_expm1, gamma_log_lower


class Elbo:
    """
    Class to compute the Evidence Lower Bound (ELBO) for the variational model.
    """

    def __init__(
        self,
        hyperparams: ModelHyperParams,
        params: Params,
        inference_quantities: InferenceQuantities,
        evaluation_quantities: EvaluationQuantities,
    ):
        self.hyperparams = hyperparams
        self.params = params

        self.inference_quantities = inference_quantities
        self.evaluation_quantities = evaluation_quantities

    def compute(self):
        """
        Compute the ELBO.
        """

        likelihood_component = self._likelihood_component_after_zn_update()
        time_invariant_latent_feature_component = (
            self._time_invariant_latent_feature_component()
        )
        time_dependent_correction_factor_component = (
            self._time_dependent_correction_factor_component()
        )
        hyperparameter_component = self._hyperparameter_component()

        return (
            likelihood_component
            + time_invariant_latent_feature_component
            + time_dependent_correction_factor_component
            + hyperparameter_component
        )

    def _likelihood_component_after_zn_update(self):
        """
        Compute the likelihood component of the ELBO.
        """

        # Extract the necessary variables
        zn_poisson_rate = self.params.zn_poisson_rate

        # Extract the necessary variables
        eg_x = self.inference_quantities.eg_x
        eg_y = self.inference_quantities.eg_y
        etg_rhox = self.inference_quantities.etg_rhox
        etg_rhoy = self.inference_quantities.etg_rhoy

        first_sum = np.sum(log_expm1(zn_poisson_rate.data))

        second_sum = np.sum(
            np.einsum("ti, ir -> tr", etg_rhox, eg_x)
            * np.einsum("tj, jr -> tr", etg_rhoy, eg_y)
        )

        return first_sum - second_sum

    def _likelihood_component(self):
        """
        Compute the likelihood component of the ELBO
        in the case that
        """

        # Extract the necessary variables
        zn_poisson_rate = self.params.zn_poisson_rate
        zn_multinomial_prob = self.params.zn_multinomial_prob

        # Extract the necessary variables
        zn_poisson_plus_rate = self.inference_quantities.zn_poisson_plus_rate
        zn_prod_poisson_plus_rate_multinomial_prob = (
            self.inference_quantities.zn_prod_poisson_plus_rate_multinomial_prob
        )
        eg_x = self.inference_quantities.eg_x
        eg_y = self.inference_quantities.eg_y
        elg_x = self.inference_quantities.elg_x
        elg_y = self.inference_quantities.elg_y
        etg_rhox = self.inference_quantities.etg_rhox
        etg_rhoy = self.inference_quantities.etg_rhoy
        eltg_rhox = self.inference_quantities.eltg_rhox
        eltg_rhoy = self.inference_quantities.eltg_rhoy

        expectation_log_p_zn = np.sum(
            zn_prod_poisson_plus_rate_multinomial_prob
            * np.einsum("ir, jr, ti, tj -> tij", elg_x, elg_y, eltg_rhox, eltg_rhoy)
        ) - np.sum(
            np.einsum("ti, ir -> tr", etg_rhox, eg_x)
            * np.einsum("tj, jr -> tj", etg_rhoy, eg_y)
        )

        expectation_log_q_zn = (
            np.sum(zn_poisson_plus_rate.data * np.log(zn_poisson_rate.data))
            - np.sum(log_expm1(zn_poisson_rate.data))
            + np.sum(
                zn_prod_poisson_plus_rate_multinomial_prob.data
                * np.log(zn_multinomial_prob.data)
            )
        )

        return expectation_log_p_zn - expectation_log_q_zn

    def _time_invariant_latent_feature_component(self):
        """
        Compute the time-invariant latent feature component of the ELBO.
        """

        # Extract the necessary variables
        N1 = self.params.N1
        N2 = self.params.N2
        d = self.hyperparams.num_features
        alpha_x = self.hyperparams.alpha_x
        alpha_y = self.hyperparams.alpha_y

        # Extract the necessary variables
        x_shape = self.params.x_shape
        x_rate = self.params.x_rate
        y_shape = self.params.y_shape
        y_rate = self.params.y_rate

        # Extract the necessary variables
        eg_x = self.inference_quantities.eg_x
        eg_y = self.inference_quantities.eg_y
        elg_x = self.inference_quantities.elg_x
        elg_y = self.inference_quantities.elg_y
        eg_zetax = self.inference_quantities.eg_zetax
        eg_zetay = self.inference_quantities.eg_zetay
        elg_zetax = self.evaluation_quantities.elg_zetax
        elg_zetay = self.evaluation_quantities.elg_zetay

        expectation_log_p_x = (
            -N1 * d * loggamma(alpha_x)
            + d * alpha_x * np.sum(elg_zetax)
            + (alpha_x - 1) * np.sum(elg_x)
            - np.einsum("i, ir ->", eg_zetax, eg_x)
        )

        expectation_log_p_y = (
            -N2 * d * loggamma(alpha_y)
            + d * alpha_y * np.sum(elg_zetay)
            + (alpha_y - 1) * np.sum(elg_y)
            - np.einsum("j, jr ->", eg_zetay, eg_y)
        )

        expectation_log_q_x = np.sum(
            x_shape * np.log(x_rate)
            - loggamma(x_shape)
            + (x_shape - 1) * elg_x
            - x_shape
        )

        expectation_log_q_y = np.sum(
            y_shape * np.log(y_rate)
            - loggamma(y_shape)
            + (y_shape - 1) * elg_y
            - y_shape
        )

        return (
            expectation_log_p_x
            + expectation_log_p_y
            - expectation_log_q_x
            - expectation_log_q_y
        )

    def _hyperparameter_component(self):
        """
        Compute the hyperparameter component of the ELBO.
        """

        # Extract the necessary variables
        N1 = self.params.N1
        N2 = self.params.N2
        b_x = self.hyperparams.b_x
        b_y = self.hyperparams.b_y
        c_x = self.hyperparams.c_x
        c_y = self.hyperparams.c_y

        # Extract the necessary variables
        zetax_shape = self.params.zetax_shape
        zetax_rate = self.params.zetax_rate
        zetay_shape = self.params.zetay_shape
        zetay_rate = self.params.zetay_rate

        # Extract the necessary variables
        eg_zetax = self.inference_quantities.eg_zetax
        eg_zetay = self.inference_quantities.eg_zetay
        elg_zetax = self.evaluation_quantities.elg_zetax
        elg_zetay = self.evaluation_quantities.elg_zetay

        expectation_log_p_zetax = (
            N1 * b_x * np.log(c_x)
            - N1 * loggamma(b_x)
            + (b_x - 1) * np.sum(elg_zetax)
            - c_x * np.sum(eg_zetax)
        )

        expectation_log_p_zetay = (
            N2 * b_y * np.log(c_y)
            - N2 * loggamma(b_y)
            + (b_y - 1) * np.sum(elg_zetay)
            - c_y * np.sum(eg_zetay)
        )

        expectation_log_q_zetax = np.sum(
            zetax_shape * np.log(zetax_rate)
            - loggamma(zetax_shape)
            + (zetax_shape - 1) * elg_zetax
            - zetax_shape
        )

        expectation_log_q_zetay = np.sum(
            zetay_shape * np.log(zetay_rate)
            - loggamma(zetay_shape)
            + (zetay_shape - 1) * elg_zetay
            - zetay_shape
        )

        return (
            expectation_log_p_zetax
            + expectation_log_p_zetay
            - expectation_log_q_zetax
            - expectation_log_q_zetay
        )

    def _time_dependent_correction_factor_component(self):
        """
        Compute the time-dependent correction factor component of the ELBO.
        """

        # Extract the necessary variables
        T = self.params.time
        N1 = self.params.N1
        N2 = self.params.N2
        alpha_x = self.hyperparams.alpha_x
        alpha_y = self.hyperparams.alpha_y
        beta_x = self.hyperparams.beta_x
        beta_y = self.hyperparams.beta_y

        # Extract the necessary variables
        rhox_shape = self.params.rhox_shape
        rhox_rate = self.params.rhox_rate
        rhoy_shape = self.params.rhoy_shape
        rhoy_rate = self.params.rhoy_rate

        # Extract the necessary variables
        eltg_rhox = self.inference_quantities.eltg_rhox
        eltg_rhoy = self.inference_quantities.eltg_rhoy
        etg_rhox = self.inference_quantities.etg_rhox
        etg_rhoy = self.inference_quantities.etg_rhoy

        expectation_log_p_rhox = (
            T * N1 * alpha_x * np.log(beta_x)
            - T * N1 * gamma_log_lower(alpha_x, beta_x)
            + (alpha_x - 1) * np.sum(eltg_rhox)
            - beta_x * np.sum(etg_rhox)
        )

        expectation_log_p_rhoy = (
            T * N2 * alpha_y * np.log(beta_y)
            - T * N2 * gamma_log_lower(alpha_y, beta_y)
            + (alpha_y - 1) * np.sum(eltg_rhoy)
            - beta_y * np.sum(etg_rhoy)
        )

        expectation_log_q_rhox = np.sum(
            rhox_shape * np.log(rhox_rate)
            - gamma_log_lower(rhox_shape, rhox_rate)
            + (rhox_shape - 1) * eltg_rhox
            - rhox_rate * etg_rhox
        )

        expectation_log_q_rhoy = np.sum(
            rhoy_shape * np.log(rhoy_rate)
            - gamma_log_lower(rhoy_shape, rhoy_rate)
            + (rhoy_shape - 1) * eltg_rhoy
            - rhoy_rate * etg_rhoy
        )

        return (
            expectation_log_p_rhox
            + expectation_log_p_rhoy
            - expectation_log_q_rhox
            - expectation_log_q_rhoy
        )
