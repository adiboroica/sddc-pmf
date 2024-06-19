import numpy as np
from scipy.sparse.linalg import svds

from rdpg.dase import dase


def mase(
    adjacency_matrices,
    num_eigenvalues,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Compute the MASE embedding for multiple adjacency matrices.

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
    :return: tuple (np.ndarray, np.ndarray, list of np.ndarray)
        U_X, U_Y embeddings and list of R_t matrices.
    """

    T = len(adjacency_matrices)

    # Compute DASE embeddings for each adjacency matrix
    dase_embeddings = [dase(A, num_eigenvalues, rng=rng) for A in adjacency_matrices]

    # Construct joint matrices U and V
    U = np.hstack([X for X, _ in dase_embeddings])
    V = np.hstack([Y for _, Y in dase_embeddings])

    # Perform SVD on the joint matrices
    U_X, D_X, _ = svds(U, k=num_eigenvalues, random_state=rng)
    U_Y, D_Y, _ = svds(V, k=num_eigenvalues, random_state=rng)

    # Sort singular values and corresponding vectors in descending order
    idx_X = np.argsort(D_X)[::-1]
    idx_Y = np.argsort(D_Y)[::-1]
    U_X = U_X[:, idx_X]
    U_Y = U_Y[:, idx_Y]

    # Compute the interactions matrices R_t
    R_t_list = []
    for t in range(T):
        A_t = adjacency_matrices[t]
        R_t = U_X.T @ A_t @ U_Y
        R_t_list.append(R_t)

    return U_X, U_Y, R_t_list
