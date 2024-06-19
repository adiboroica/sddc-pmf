from typing import List, Tuple
import numpy as np
import sparse

from utils import evaluation


def create_sparse_matrix(shape, positive_coords):
    """
    Helper function to create a sparse matrix with given positive coordinates.

    Note: The positive coordinates are assumed to be valid, pairwise distinct, and within the shape of the matrix.
    """
    data = np.ones(len(positive_coords), dtype=int)
    coords = np.array(positive_coords).T
    return sparse.COO(coords, data, shape=shape)


def generate_unique_coords(shape, num_coords, rng=np.random.default_rng(42)):
    """Generate a specified number of unique coordinates within the given shape."""
    coords = set()
    while len(coords) < num_coords:
        coord = tuple(rng.integers(0, dim) for dim in shape)
        coords.add(coord)
    return list(coords)


def sample_all_labels_test(
    shape: Tuple[int, int], positive_coords: List[Tuple[int, int]]
):
    """Test sampling with 100% of both positive and negative labels."""
    labels = create_sparse_matrix(shape, positive_coords)

    # Set the seed to ensure reproducibility
    rng = np.random.default_rng(42)

    # Sample all indices
    sampled_indices = evaluation.sample_percentages(
        labels, 1, 1, shuffle_result=False, rng=rng
    )

    # Verify the number of sampled indices
    assert sampled_indices.shape[1] == np.prod(shape)

    # Verify that all positive labels are sampled
    for coord in positive_coords:
        assert tuple(coord) in map(tuple, sampled_indices.T)

    # Verify that all negative labels are sampled
    negative_coords = np.argwhere(labels.todense() == 0)
    for coord in negative_coords:
        assert tuple(coord) in map(tuple, sampled_indices.T)


def expected_number_of_samples_test(
    shape: Tuple[int, int],
    positive_coords: np.ndarray,
    negative_percentage: float,
    positive_percentage: float,
    rng=np.random.default_rng(42),
):
    # Sample indices
    labels = create_sparse_matrix(shape, positive_coords)
    sampled_indices = evaluation.sample_percentages(
        labels, negative_percentage, positive_percentage, rng=rng
    )

    # Compute the expected number of positive and negative samples
    positive_samples = np.where(labels.todense() == 1)
    negative_samples = np.where(labels.todense() == 0)
    expected_positive_samples = int(len(positive_samples[0]) * positive_percentage)
    expected_negative_samples = int(len(negative_samples[0]) * negative_percentage)

    # Compute the actual number of positive and negative samples
    num_positive_samples = np.sum(labels.todense()[tuple(sampled_indices)])
    num_negative_samples = len(sampled_indices[0]) - num_positive_samples

    # Verify the number of positive and negative samples
    assert num_positive_samples == expected_positive_samples
    assert num_negative_samples == expected_negative_samples
