from typing import Union
import numpy as np
import sparse


def sample(
    labels: Union[np.ndarray, sparse.COO],
    num_positive_labels=None,
    num_negative_labels=None,
    return_num_every_class: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
):
    dense_labels = labels
    if isinstance(labels, sparse.COO):
        # Convert to dense format
        dense_labels = labels.todense()

    # Get indices of positive and negative labels
    # This will be a tuple of arrays with (num_dimensions) elements,
    # where each array is of shape (num_labels,)
    positive_indices = np.where(dense_labels == 1)  # tuple of arrays
    negative_indices = np.where(dense_labels == 0)  # tuple of arrays

    num_positive_indices = len(positive_indices[0])
    num_negative_indices = len(negative_indices[0])

    assert num_positive_labels is None or num_positive_labels <= num_positive_indices
    assert num_negative_labels is None or num_negative_labels <= num_negative_indices

    # Sample all the positive labels is the number of positive labels is not specified
    num_positive_class = (
        num_positive_indices if num_positive_labels is None else num_positive_labels
    )
    num_negative_class = (
        num_negative_indices if num_negative_labels is None else num_negative_labels
    )

    # Transpose the indices such that every row contains the indices of a positive/negative label
    positive_indices = np.transpose(
        positive_indices
    )  # shape: (num_positive_labels, num_dimensions)
    negative_indices = np.transpose(
        negative_indices
    )  # shape: (num_negative_labels, num_dimensions)

    # Sample positive labels
    if num_positive_class < num_positive_indices:
        indices = np.arange(num_positive_indices)
        sampled_indices = rng.choice(indices, num_positive_class, replace=False)
        positive_indices = positive_indices[sampled_indices]

    # Sample negative labels
    if num_negative_class < num_negative_indices:
        indices = np.arange(num_negative_indices)
        sampled_indices = rng.choice(indices, num_negative_class, replace=False)
        negative_indices = negative_indices[sampled_indices]

    if num_positive_class != 0 and num_negative_class != 0:
        samples = np.concatenate([positive_indices, negative_indices], axis=0)
    elif num_positive_class == 0:
        samples = negative_indices
    else:
        samples = positive_indices

    samples = samples.T

    if return_num_every_class:
        return samples[0], samples[1], num_positive_class, num_negative_class

    return samples[0], samples[1]


def sample_percentages(
    labels: Union[np.ndarray, sparse.COO],
    percentage_of_negative_labels: float = 1,
    percentage_of_positive_labels: float = 1,
    return_num_every_class: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:

    assert 0 <= percentage_of_negative_labels <= 1
    assert 0 <= percentage_of_positive_labels <= 1

    # Compute the number of positive and negative samples to keep
    num_positive_samples_to_keep = int(
        np.sum(labels == 1) * percentage_of_positive_labels
    )
    num_negative_samples_to_keep = int(
        np.sum(labels == 0) * percentage_of_negative_labels
    )

    return sample(
        labels,
        num_positive_labels=num_positive_samples_to_keep,
        num_negative_labels=num_negative_samples_to_keep,
        return_num_every_class=return_num_every_class,
        rng=rng,
    )


def sample_equally(
    labels: Union[np.ndarray, sparse.COO],
    return_num_every_class: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    # Determine the number of samples to keep
    num_positive_samples = np.sum(labels == 1)
    num_negative_samples = np.sum(labels == 0)
    num_samples = min(num_positive_samples, num_negative_samples)

    return sample(
        labels,
        num_positive_labels=num_samples,
        num_negative_labels=num_samples,
        return_num_every_class=return_num_every_class,
        rng=rng,
    )
