import numpy as np
import sparse


def convert_data(data: dict):
    """
    Convert the data from a dictionary of CSR matrices to a single COO matrix.
    """

    if not data:
        return sparse.COO(np.empty((3, 0)), np.empty(0), shape=(0, 0, 0))

    # Sort the keys
    keys = sorted(data.keys())

    # Read the dimensions of the data
    T = len(keys)
    N1, N2 = data[keys[0]].shape

    # Initialize lists to store COO format data
    coords = []
    data_values = []

    # Process each matrix and store its COO components along with the time index
    for t, key in enumerate(keys):
        # Ensure all matrices have the same shape
        if data[key].shape != (N1, N2):
            raise ValueError("All matrices must have the same shape")

        matrix = data[key].tocoo()

        # Ensure the data values are either 0 or 1
        if not np.all(np.isin(matrix.data, [0, 1])):
            raise ValueError("Data values must be either 0 or 1")

        coords.append([t * np.ones_like(matrix.row), matrix.row, matrix.col])
        data_values.extend(matrix.data.astype(int))

    # Stack the coordinate arrays
    coords = np.vstack([np.concatenate(c) for c in zip(*coords)])

    # Create the COO matrix
    sparse_data = sparse.COO(coords, data_values, shape=(T, N1, N2))

    return sparse_data
