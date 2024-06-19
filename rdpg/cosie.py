import numpy as np

from rdpg.mase import mase


def cosie(
    adjacency_matrices,
    num_eigenvalues,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Compute the COSIE forecast of the future adjacency matrix.

    Parameters
    ----------
    :param adjacency_matrices: np.ndarray or sparse.COO
        The observed adjacency matrices A_t.
    :param num_eigenvalues: int
        The number of eigenvalues to use for the embeddings.
    :param rng: np.random.Generator, optional
        The random number generator to use.

    Returns
    -------
    :return: np.ndarray
        The forecasted adjacency matrix A_{T+1}.
    """

    # Compute MASE embeddings and R_t matrices
    X, Y, Rt_list = mase(adjacency_matrices, num_eigenvalues, rng=rng)

    # Compute the average scaling matrix R_avg
    R_avg = np.mean(Rt_list, axis=0)

    # Compute the forecasted adjacency matrix
    prediction = X @ R_avg @ Y.T

    return prediction
