import numpy as np
import pytest

from utils import evaluation

from .sample_indices import (
    create_sparse_matrix,
    generate_unique_coords,
    sample_all_labels_test,
    expected_number_of_samples_test,
)


@pytest.mark.parametrize(
    "shape, negative_percentage, positive_percentage",
    [
        ((9, 9), 0.5, 0.5),
        ((9, 9), 0.2, 0.8),
        ((10, 10), 0.5, 0.5),
        ((10, 10), 0.2, 0.8),
    ],
)
def test_no_positive_labels(shape, negative_percentage, positive_percentage):
    """Test the function with a matrix that has no positive labels."""

    positive_coords = []

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, negative_percentage, positive_percentage",
    [
        ((9, 9), 0.5, 0.5),
        ((9, 9), 0.2, 0.8),
        ((10, 10), 0.5, 0.5),
        ((10, 10), 0.2, 0.8),
    ],
)
def test_only_positive_labels(shape, negative_percentage, positive_percentage):
    """Test the function with a matrix that has only positive labels."""

    positive_coords = np.indices(shape).reshape(len(shape), -1).T

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords",
    [
        ((9, 9), [(0, 0)]),
        ((9, 9), [(0, 0), (1, 1)]),
        ((10, 10), [(0, 0), (1, 1), (2, 2)]),
        ((10, 10), [(0, 0), (1, 1), (2, 2), (3, 3)]),
    ],
)
def test_select_all_labels(shape, positive_coords):
    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(shape, positive_coords, 1, 1)


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((9, 9), [(0, 0)], 0.5, 0.5),
        ((9, 9), [(0, 0)], 0.2, 0.8),
        ((10, 10), [(0, 0), (1, 1), (2, 2)], 0.5, 0.5),
        ((10, 10), [(0, 0), (1, 1), (2, 2)], 0.2, 0.8),
    ],
)
def test_select_partial_labels(
    shape,
    positive_coords,
    negative_percentage,
    positive_percentage,
):
    """Test sampling with partial percentage of positive and negative labels."""

    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((9, 9), [(0, 0)], 0, 0),
        ((9, 9), [(0, 0)], 0, 1),
        ((9, 9), [(0, 0)], 1, 0),
        ((9, 9), [(0, 0)], 1, 1),
        ((10, 10), [(0, 0), (1, 1)], 0, 0),
        ((10, 10), [(0, 0), (1, 1)], 0, 1),
        ((10, 10), [(0, 0), (1, 1)], 1, 0),
        ((10, 10), [(0, 0), (1, 1)], 1, 1),
    ],
)
def test_select_extreme_percentages(
    shape, positive_coords, negative_percentage, positive_percentage
):
    """Test sampling with 0% or 100% of positive and negative labels."""

    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((9, 9), [(0, 0)], 0.5, 0.5),
        ((9, 9), [(0, 0)], 0.2, 0.8),
        ((10, 10), [(0, 0), (1, 1), (2, 2)], 0.5, 0.5),
        ((10, 10), [(0, 0), (1, 1), (2, 2)], 0.2, 0.8),
    ],
)
def test_no_shuffling(shape, positive_coords, negative_percentage, positive_percentage):
    """Test sampling with shuffling disabled."""

    labels = create_sparse_matrix(shape, positive_coords)
    sampled_indices_1 = evaluation.sample_percentages(
        labels,
        negative_percentage,
        positive_percentage,
        rng=np.random.default_rng(42),
    )
    sampled_indices_2 = evaluation.sample_percentages(
        labels,
        negative_percentage,
        positive_percentage,
        rng=np.random.default_rng(42),
    )

    # Verify that the samples are not shuffled (same order)
    assert np.array_equal(sampled_indices_1, sampled_indices_2)


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((0, 0), [], 1, 1),
    ],
)
def test_empty_matrix(shape, positive_coords, negative_percentage, positive_percentage):
    """Test the function with an empty matrix."""

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((1, 1), [(0, 0)], 1, 1),
        ((1, 1), [(0, 0)], 0, 0),
    ],
)
def test_single_element_matrix(
    shape, positive_coords, negative_percentage, positive_percentage
):
    """Test the function with a single element matrix."""

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords, negative_percentage, positive_percentage",
    [
        ((5, 10), [(0, 0), (1, 1), (2, 2)], 0.5, 0.5),
        ((5, 10), [(0, 0), (1, 1), (2, 2)], 0.2, 0.8),
    ],
)
def test_non_square_matrices(
    shape, positive_coords, negative_percentage, positive_percentage
):
    """Test the function with non-square matrices."""

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, positive_coords",
    [
        ((5, 5, 5), [(0, 0, 0), (1, 1, 1), (2, 2, 2)]),
        ((3, 3, 3, 3), [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]),
    ],
)
def test_high_dimensional_matrices(shape, positive_coords):
    """Test the function with higher-dimensional matrices."""

    sample_all_labels_test(shape, positive_coords)
    expected_number_of_samples_test(shape, positive_coords, 1, 1)


@pytest.mark.parametrize(
    "shape, negative_percentage, positive_percentage",
    [
        ((1000, 1000), 0.05, 0.5),
        ((1000, 1000), 0.02, 0.8),
        ((1000, 1000), 0.08, 0.2),
    ],
)
def test_large_matrices(shape, negative_percentage, positive_percentage):
    """Test the function with large matrices."""

    positive_coords = generate_unique_coords(shape, 100)

    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )


@pytest.mark.parametrize(
    "shape, negative_percentage, positive_percentage",
    [
        ((1000, 1000), 0.9999, 0.0001),
    ],
)
def test_very_high_sparsity(shape, negative_percentage, positive_percentage):
    """Test the function with very high sparsity."""

    positive_coords = generate_unique_coords(shape, 1)

    expected_number_of_samples_test(
        shape, positive_coords, negative_percentage, positive_percentage
    )
