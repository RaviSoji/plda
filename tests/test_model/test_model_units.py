# Copyright 2017 Ravi Sojitra. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import pytest

from numpy.testing import assert_allclose
from plda import plda
from plda.plda.model import (
    get_space_walk,
    transform_D_to_X,
    transform_X_to_U,
    transform_U_to_U_model,
    transform_U_model_to_U,
    transform_U_to_X,
    transform_X_to_D
)
from sklearn.decomposition import PCA


def gen_invertible_matrix(dim, scale):
    arr = np.random.random((dim, dim)) * scale

    return np.matmul(arr, arr.T)


def test_get_space_walk():
    spaces = ['U_model', 'U', 'X', 'D']

    actual = list(get_space_walk('U_model', 'D'))
    expected = list(zip(spaces[:-1], spaces[1:]))
    assert actual == expected

    actual = list(get_space_walk('U_model', 'X'))
    expected = list(zip(spaces[:-2], spaces[1:-1]))
    assert actual == expected

    actual = list(get_space_walk('U_model', 'U'))
    expected = [(spaces[0], spaces[1])]
    assert actual == expected

    actual = list(get_space_walk('U', 'X'))
    expected = [(spaces[1], spaces[2])]
    assert actual == expected

    actual = list(get_space_walk('U', 'D'))
    expected = [(spaces[1], spaces[2]), (spaces[2], spaces[3])]
    assert actual == expected

    actual = list(get_space_walk('X', 'D'))
    expected = [(spaces[2], spaces[3])]
    assert actual == expected

    actual = list(get_space_walk('D', 'X'))
    expected = [(spaces[3], spaces[2])]
    assert actual == expected

    actual = list(get_space_walk('D', 'U'))
    expected = [(spaces[3], spaces[2]), (spaces[2], spaces[1])]
    assert actual == expected

    actual = list(get_space_walk('D', 'U_model'))
    expected = list(zip(spaces[::-1][:-1], spaces[::-1][1:]))
    assert actual == expected

    actual = list(get_space_walk('X', 'U'))
    expected = [(spaces[2], spaces[1])]
    assert actual == expected

    actual = list(get_space_walk('X', 'U_model'))
    expected = [(spaces[2], spaces[1]), (spaces[1], spaces[0])]
    assert actual == expected

    actual = list(get_space_walk('U', 'U_model'))
    expected = [(spaces[1], spaces[0])]
    assert actual == expected


def test_transform_D_to_X():
    np.random.seed(1234)

    n = 100
    dim = 5
    data = np.random.random((n, dim))

    expected = data
    actual = transform_D_to_X(expected, None)

    assert_allclose(actual, expected)

    pca = PCA(n_components=2)
    pca.fit(data)

    expected = pca.transform(data)
    actual = transform_D_to_X(data, pca)

    assert_allclose(actual, expected)


def test_transform_X_to_U():
    np.random.seed(1234)

    n = 100
    dim = 5
    expected = np.random.random((n, dim))

    A = gen_invertible_matrix(dim, 10)
    m = np.random.random(dim)

    data = np.matmul(expected, A.T) + m
    actual = transform_X_to_U(data, np.linalg.inv(A), m)

    assert_allclose(actual, expected)


def test_transform_U_to_U_model():
    np.random.seed(1234)

    n = 100
    target_dim = 50
    subspace_dim = 10

    dims = np.arange(target_dim)
    np.random.shuffle(dims)
    dims = dims[:subspace_dim]
    data = np.random.random((n, target_dim))

    expected = data[:, dims]
    actual = transform_U_to_U_model(data, dims)

    assert_allclose(actual, expected)


def test_transform_U_model_to_U():
    np.random.seed(1234)

    target_dim = 100
    subspace_dim = 50
    n = 100

    relevant_U_dims = np.arange(target_dim)
    np.random.shuffle(relevant_U_dims)
    relevant_U_dims = relevant_U_dims[:subspace_dim]

    data = np.random.random((n, subspace_dim))
    expected = np.zeros((n, target_dim))
    expected[:, relevant_U_dims] = data

    actual = transform_U_model_to_U(data, relevant_U_dims, target_dim)

    assert_allclose(actual, expected)


def test_transform_U_to_X():
    np.random.seed(1234)

    n = 100
    dim = 5
    A = gen_invertible_matrix(dim, 10)
    m = np.random.random(dim)

    expected = np.random.random((n, dim))

    data = np.matmul(expected - m, np.linalg.inv(A).T)
    actual = transform_U_to_X(data, A, m)

    assert_allclose(actual, expected)


def test_transform_X_to_D():
    np.random.seed(1234)

    n = 100
    target_dim = 5
    subspace_dim = 2

    data = np.random.random((n, subspace_dim))
    expected = np.zeros((n, target_dim))
    expected[:, :subspace_dim] = data

    actual = transform_X_to_D(expected, None)

    assert_allclose(actual, expected)

    pca = PCA(n_components=subspace_dim)
    pca.fit(expected)

    actual = transform_X_to_D(pca.transform(expected), pca)

    assert_allclose(actual, expected)
