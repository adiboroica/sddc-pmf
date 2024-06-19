from typing import Union, Tuple
import numpy as np
from scipy.stats import gamma


def sample(
    shape,
    rate,
    size: Union[int, Tuple[int]] = 1,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Sample from a Truncated Gamma distribution over (0, 1).
    """

    shape = np.asarray(shape)
    rate = np.asarray(rate)

    # Check that shape and rate have the same shape.
    if shape.shape != rate.shape:
        raise ValueError("Shape and rate must have the same shape.")

    # Normalize the size parameter to always be a tuple
    if isinstance(size, int):
        size = (size,)

    # Determine the shape of the output samples.
    if shape.shape == () and rate.shape == ():
        output_shape = size
    elif size == (1,):
        output_shape = shape.shape
    else:
        raise ValueError(
            "Support for broadcasting in this case is not implemented.\n"
            f"shape: {shape.shape}; rate: {rate.shape}; size: {size}"
        )

    # Compute the value of the cumulative distribution function at 1.
    cdf_1 = gamma.cdf(1, a=shape, scale=1 / rate)

    # Sample from the uniform distribution.
    uniform_samples = rng.uniform(size=output_shape)

    # Inverse transform sampling.
    truncated_gamma_samples = gamma.ppf(
        cdf_1 * uniform_samples, a=shape, scale=1 / rate
    )

    # Ensure samples are within the range (0, 1).
    eps = np.finfo(float).eps
    truncated_gamma_samples = np.clip(truncated_gamma_samples, eps, 1 - eps)

    return truncated_gamma_samples
