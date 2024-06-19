import numpy as np

from rdpg.dase import dase


def aip(adjacency_matrices, num_eigenvalues, weights=None, rng=np.random.default_rng()):
    """
    Compute the AIP forecast of the future adjacency matrix.

    Parameters
    ----------
    :param adjacency_matrices: np.ndarray or sparse.COO
        The observed adjacency matrices A_t.
    :param num_eigenvalues: int
        The number of eigenvalues to use for the DASE embeddings.
    :param weights: np.ndarray, optional
        The weights to use for averaging. If None, equal weights are used.
    :param rng: np.random.Generator, optional
        The random number generator to use.

    Returns
    -------
    :return: np.ndarray
        The AIP forecast of the future adjacency matrix.
    """

    # Extract the dimensions
    if isinstance(adjacency_matrices, np.ndarray):
        T, N1, N2 = adjacency_matrices.shape
    elif isinstance(adjacency_matrices, list):
        T = len(adjacency_matrices)
        N1, N2 = adjacency_matrices[0].shape

    # If no weights are provided, use equal weights
    if weights is None:
        weights = np.full(T, fill_value=1 / T)

    prediction = np.zeros((N1, N2))
    for t in range(T):
        A_t = adjacency_matrices[t]
        X_t, Y_t = dase(A_t, num_eigenvalues, rng=rng)
        prediction += weights[t] * (X_t @ Y_t.T)

    return prediction
