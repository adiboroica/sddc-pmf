import numpy as np
import sparse
from scipy.special import logsumexp

from pmf.hyperparams import ModelHyperParams
from pmf.params.params import Params
from pmf.params.quantities import InferenceQuantities


class Updates:
    """
    Class to compute the updates for the variational model.
    """

    def __init__(
        self,
        hyperparams: ModelHyperParams,
        params: Params,
        inference_quantities: InferenceQuantities,
    ):
        self.hyperparams = hyperparams
        self.params = params
        self.inference_quantities = inference_quantities

    def zn(self):
        """
        Returns the new Poisson rate and Multinomial probability for the Z, N components.
        """

        # Extract the necessary variables
        T = self.params.time
        N1 = self.params.N1
        N2 = self.params.N2
        d = self.hyperparams.num_features

        # Extract the necessary variables
        zn_poisson_rate = self.params.zn_poisson_rate
        zn_multinomial_prob = self.params.zn_multinomial_prob

        # Extract the necessary variables
        elg_x = self.inference_quantities.elg_x
        elg_y = self.inference_quantities.elg_y
        eltg_rhox = self.inference_quantities.eltg_rhox
        eltg_rhoy = self.inference_quantities.eltg_rhoy

        # Number of non-zero values
        num_non_zero_values = zn_poisson_rate.data.size

        n_values = np.zeros(num_non_zero_values)
        z_values = np.zeros(num_non_zero_values * d)

        log_unnormalized_multinomial_prob = {}
        log_normalization_constant = {}

        # Compute the new N and Z values
        for index, (t, i, j) in enumerate(zn_poisson_rate.coords.T):
            # If the log unnormalized multinomial probability is not already computed, compute it
            if (i, j) not in log_unnormalized_multinomial_prob:
                log_unnormalized_multinomial_prob[i, j] = elg_x[i] + elg_y[j]

            # If the log normalization constant is not already computed, compute it
            if (i, j) not in log_normalization_constant:
                log_normalization_constant[i, j] = logsumexp(
                    log_unnormalized_multinomial_prob[i, j]
                )

            n_values[index] = np.exp(
                eltg_rhox[t, i] + eltg_rhoy[t, j] + log_normalization_constant[i, j]
            )
            z_values[index * d : (index + 1) * d] = np.exp(
                log_unnormalized_multinomial_prob[i, j]
                - log_normalization_constant[i, j]
            )

        # Generate the new Z, N components
        new_zn_poisson_rate = sparse.COO(
            zn_poisson_rate.coords,
            n_values,
            shape=(T, N1, N2),
        )
        new_zn_multinomial_prob = sparse.COO(
            zn_multinomial_prob.coords,
            z_values,
            shape=(T, N1, N2, d),
        )

        return new_zn_poisson_rate, new_zn_multinomial_prob

    def x(self):
        """
        Returns the new shape and rate for the x distribution.
        """

        # Extract the necessary variables
        a_x = self.hyperparams.a_x
        zn_prod_poisson_plus_rate_multinomial_prob = (
            self.inference_quantities.zn_prod_poisson_plus_rate_multinomial_prob
        )
        eg_y = self.inference_quantities.eg_y
        eg_zetax = self.inference_quantities.eg_zetax
        etg_rhox = self.inference_quantities.etg_rhox
        etg_rhoy = self.inference_quantities.etg_rhoy

        # Compute the new shape and rate
        new_x_shape = (
            a_x
            + np.sum(
                zn_prod_poisson_plus_rate_multinomial_prob,
                axis=(0, 2),
            ).todense()
        )
        new_x_rate = eg_zetax[:, None] + np.einsum(
            "ti, tj, jr -> ir",
            etg_rhox,
            etg_rhoy,
            eg_y,
        )

        return new_x_shape, new_x_rate

    def y(self):
        """
        Returns the new shape and rate for the y distribution.
        """

        # Extract the necessary variables
        a_y = self.hyperparams.a_y
        zn_prod_poisson_plus_rate_multinomial_prob = (
            self.inference_quantities.zn_prod_poisson_plus_rate_multinomial_prob
        )
        eg_x = self.inference_quantities.eg_x
        eg_zetay = self.inference_quantities.eg_zetay
        etg_rhox = self.inference_quantities.etg_rhox
        etg_rhoy = self.inference_quantities.etg_rhoy

        # Compute the new shape and rate
        new_y_shape = (
            a_y
            + np.sum(
                zn_prod_poisson_plus_rate_multinomial_prob,
                axis=(0, 1),
            ).todense()
        )
        new_y_rate = eg_zetay[:, None] + np.einsum(
            "ti, tj, ir -> jr",
            etg_rhox,
            etg_rhoy,
            eg_x,
        )

        return new_y_shape, new_y_rate

    def zetax(self):
        """
        Returns the new rate for the zetax distribution.
        """

        # Extract the necessary variables
        c_x = self.hyperparams.c_x
        eg_x = self.inference_quantities.eg_x

        # Compute the new rate
        new_zetax_rate = c_x + np.sum(eg_x, axis=1)

        return new_zetax_rate

    def zetay(self):
        """
        Returns the new rate for the zetay distribution.
        """

        # Extract the necessary variables
        c_y = self.hyperparams.c_y
        eg_y = self.inference_quantities.eg_y

        # Compute the new rate
        new_zetay_rate = c_y + np.sum(eg_y, axis=1)

        return new_zetay_rate

    def rhox(self):
        """
        Returns the new shape and rate for the rhox distribution.
        """

        # Extract the necessary variables
        alpha_x = self.hyperparams.alpha_x
        beta_x = self.hyperparams.beta_x
        zn_poisson_plus_rate = self.inference_quantities.zn_poisson_plus_rate
        eg_x = self.inference_quantities.eg_x
        eg_y = self.inference_quantities.eg_y
        etg_rhoy = self.inference_quantities.etg_rhoy

        # Compute the new shape and rate
        new_rhox_shape = (
            alpha_x
            + np.sum(
                zn_poisson_plus_rate,
                axis=2,
            ).todense()
        )
        new_rhox_rate = beta_x + np.einsum(
            "tj, ir, jr -> ti",
            etg_rhoy,
            eg_x,
            eg_y,
        )

        return new_rhox_shape, new_rhox_rate

    def rhoy(self):
        """
        Returns the new shape and rate for the rhoy distribution.
        """

        # Extract the necessary variables
        alpha_y = self.hyperparams.alpha_y
        beta_y = self.hyperparams.beta_y
        zn_poisson_plus_rate = self.inference_quantities.zn_poisson_plus_rate
        eg_x = self.inference_quantities.eg_x
        eg_y = self.inference_quantities.eg_y
        etg_rhox = self.inference_quantities.etg_rhox

        # Compute the new shape and rate
        new_rhoy_shape = (
            alpha_y
            + np.sum(
                zn_poisson_plus_rate,
                axis=1,
            ).todense()
        )
        new_rhoy_rate = beta_y + np.einsum(
            "ti, ir, jr -> tj",
            etg_rhox,
            eg_x,
            eg_y,
        )

        return new_rhoy_shape, new_rhoy_rate
