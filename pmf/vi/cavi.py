import sparse

from pmf.hyperparams import ModelHyperParams
from pmf.params import (
    Params,
    Updates,
    log_likelihood,
)
from pmf.params.estimates import ParamsEstimates
from pmf.params.quantities import EvaluationQuantities, InferenceQuantities
from pmf.vi.elbo import Elbo
from utils.logger import Logger


class CaviParams:
    """
    Coordinate Ascent Variational Inference (CAVI) algorithm's parameters.

    Parameters
    ----------
    :param max_iterations: int
        The maximum number of iterations to perform the CAVI method.
    :param eval_interval: int
        The interval to print the ELBO values at.
    :param elbo_tol: float
        The tolerance to check for convergence of the ELBO values.
    """

    def __init__(
        self, max_iterations: int = 10**5, eval_interval: int = 1, elbo_tol=1e-6
    ):
        self.max_iterations = max_iterations
        self.eval_interval = eval_interval
        self.elbo_tol = elbo_tol


class CaviResults:
    """
    Results of the CAVI method.

    Parameters
    ----------
    :param eval_indices: list
        The indices of the evaluations.
    :param elbo_values: list
        The ELBO values at each evaluation.
    :param log_likehood_values: list
        The log likelihood values at each evaluation.
    :param converged: bool
        Whether the ELBO has converged or not.
    """

    def __init__(self, eval_indices, elbo_values, log_likelihood_values, converged):
        self.eval_indices = eval_indices
        self.elbo_values = elbo_values
        self.log_likelihood_values = log_likelihood_values
        self.converged = converged


def cavi(
    cavi_params: CaviParams,
    data: sparse.COO,
    hyperparams: ModelHyperParams,
    params: Params,
    inference_quantities: InferenceQuantities,
    evaluation_quantities: EvaluationQuantities,
    params_estimates: ParamsEstimates,
    updates: Updates,
    logger: Logger,
) -> CaviResults:

    # Extract necessary variables
    max_iterations = cavi_params.max_iterations
    elbo_interval = cavi_params.eval_interval
    elbo_tol = cavi_params.elbo_tol

    # Initialize the ELBO
    elbo = Elbo(
        hyperparams=hyperparams,
        params=params,
        inference_quantities=inference_quantities,
        evaluation_quantities=evaluation_quantities,
    )

    # Perform the CAVI update method
    eval_indices = []
    elbo_values = []
    log_likelihood_values = []
    current_iter = 1
    converged = False
    while current_iter <= max_iterations and not converged:
        eval_model = current_iter % elbo_interval == 0 or current_iter == max_iterations

        # Set the flag for evaluation of the model
        if eval_model:
            params.set_eval_needed(True)

        # Perform the updates
        _perform_updates(params, updates)

        if eval_model:
            # Add the current iteration to the list
            eval_indices.append(current_iter)

            # Compute the ELBO and add it to the list
            curr_elbo = elbo.compute()
            elbo_values.append(curr_elbo)

            # Compute the log likelihood and add it to the list
            log_likelihood_values.append(log_likelihood(data, params_estimates))

            # Check for convergence
            if len(elbo_values) > 1:
                prev_elbo = elbo_values[-2]
                if _elbo_has_converged(curr_elbo, prev_elbo, elbo_tol):
                    converged = True

            # Print the ELBO value
            logger.info(f"Iteration {current_iter} - ELBO: {curr_elbo}")

            # Reset the flag for evaluation of the model
            params.set_eval_needed(False)

        current_iter += 1

    if converged:
        message = "ELBO has converged."
    else:
        message = "Maximum number of iterations reached. ELBO has not converged."

    logger.info(message)
    if not logger.log_to_console:
        print(message)

    return CaviResults(
        eval_indices=eval_indices,
        elbo_values=elbo_values,
        log_likelihood_values=log_likelihood_values,
        converged=converged,
    )


def _perform_updates(params: Params, updates: Updates):
    # Update the x distribution
    x_shape, x_rate = updates.x()
    params.set_x(x_shape, x_rate)

    # Update the y distribution
    y_shape, y_rate = updates.y()
    params.set_y(y_shape, y_rate)

    # Update the zetax distribution
    # (Notice that the shape is constant at each iteration)
    new_zetax_rate = updates.zetax()
    params.set_zetax_rate(new_zetax_rate)

    # Update the zetay distribution
    # (Notice that the shape is constant at each iteration)
    new_zetay_rate = updates.zetay()
    params.set_zetay_rate(new_zetay_rate)

    # Update the rhox distribution
    new_rhox_shape, new_rhox_rate = updates.rhox()
    params.set_rhox(new_rhox_shape, new_rhox_rate)

    # Update the rhoy distribution
    new_rhoy_shape, new_rhoy_rate = updates.rhoy()
    params.set_rhoy(new_rhoy_shape, new_rhoy_rate)

    # Update the Z, N components
    # NOTE: This MUST BE DONE just before the ELBO computation,
    # in order to have the correct value for the ELBO
    zn_poisson_rate, zn_multinomial_prob = updates.zn()
    params.set_zn(zn_poisson_rate, zn_multinomial_prob)


def _elbo_has_converged(curr_elbo, prev_elbo, elbo_tol):
    return abs((curr_elbo - prev_elbo) / prev_elbo) < elbo_tol
