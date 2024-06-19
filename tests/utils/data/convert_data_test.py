import pytest
import numpy as np
from scipy.sparse import csr_matrix

from utils.data import convert_data


def test_single_matrix():
    data = {0: csr_matrix([[1, 0], [0, 2]])}
    result = convert_data(data)
    expected_coords = np.array([[0, 0], [0, 1], [0, 1]])
    expected_data = np.array([1, 2])
    expected_shape = (1, 2, 2)
    assert np.array_equal(result.coords, expected_coords)
    assert np.array_equal(result.data, expected_data)
    assert result.shape == expected_shape


def test_multiple_matrices():
    data = {0: csr_matrix([[1, 0], [0, 2]]), 1: csr_matrix([[0, 3], [4, 0]])}
    result = convert_data(data)
    expected_coords = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    expected_data = np.array([1, 2, 3, 4])
    expected_shape = (2, 2, 2)
    assert np.array_equal(result.coords, expected_coords)
    assert np.array_equal(result.data, expected_data)
    assert result.shape == expected_shape


def test_empty_matrix():
    data = {0: csr_matrix((2, 2))}
    result = convert_data(data)
    expected_coords = np.empty((3, 0))
    expected_data = np.empty(0)
    expected_shape = (1, 2, 2)
    assert np.array_equal(result.coords, expected_coords)
    assert np.array_equal(result.data, expected_data)
    assert result.shape == expected_shape


def test_inconsistent_shapes():
    data = {0: csr_matrix([[1, 0, 0], [0, 2, 0]]), 1: csr_matrix([[0, 3], [4, 0]])}
    with pytest.raises(ValueError):
        convert_data(data)


def test_no_matrices():
    data = {}
    result = convert_data(data)
    expected_coords = np.empty((3, 0))
    expected_data = np.empty(0)
    expected_shape = (0, 0, 0)
    assert np.array_equal(result.coords, expected_coords)
    assert np.array_equal(result.data, expected_data)
    assert result.shape == expected_shape
