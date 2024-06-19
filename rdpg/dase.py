import numpy as np
from scipy.sparse.linalg import svds


def dase(A, num_eigenvalues, rng: np.random.Generator = np.random.default_rng()):
    """
    Compute the DASE embedding of the adjacency matrix A.

    Parameters
    ----------
    :param A: np.ndarray or sparse.COO
        The matrix to embed.
    :param num_eigenvalues: int
        The number of eigenvalues to use.
    :param rng: np.random.Generator, optional
        The random number generator to use.

    Returns
    -------
    :return: (np.ndarray, np.ndarray)
        The DASE embedding of the matrix A.
    """

    # If the matrix has int type, convert it to float
    if A.dtype == int:
        A = A.astype(float)

    # Compute the SVD of the matrix
    U, D, V = svds(A, k=num_eigenvalues, random_state=rng)
    V = V.T

    # Sort singular values and corresponding vectors in descending order
    idx = np.argsort(D)[::-1]
    U = U[:, idx]
    V = V[:, idx]
    D = D[idx]

    # Compute the DASE embedding
    sqrt_D = np.diag(np.sqrt(D))
    X = U @ sqrt_D
    Y = V @ sqrt_D

    return X, Y
