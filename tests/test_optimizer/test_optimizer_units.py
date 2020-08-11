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
from numpy.testing import assert_array_equal
from plda.plda.optimizer import (
    calc_A,
    calc_W,
    calc_Lambda_b,
    calc_Lambda_w,
    calc_m,
    calc_n_avg,
    calc_Psi,
    calc_scatter_matrices,
    as_dictionary_of_dictionaries,
    get_prior_params,
    get_posterior_params,
    get_posterior_predictive_params,
    get_relevant_U_dims
)
from plda.tests.utils import assert_diagonal
from scipy.linalg import eigh


@pytest.fixture(scope='module')
def expected_scatter_matrices():
    S_b = np.asarray([[235,  51, 137, 110,  88],
                      [ 51,  15,  26,  25,  14],
                      [137,  26, 115,  43,  58],
                      [110,  25,  43, 149, 111],
                      [ 88,  14,  58, 111, 103]])

    S_w = np.asarray([[ 78,  84,  68,  69,  67],
                      [ 84, 170,  93, 124, 153],
                      [ 68,  93, 150, 128, 128],
                      [ 69, 124, 128, 148, 143],
                      [ 67, 153, 128, 143, 207]])

    return {'S_b': S_b, 'S_w': S_w}


@pytest.fixture(scope='module')
def expected_W():
    W = np.asarray(
        [[ 0.00576829, -0.05484751,  0.03000066,  0.1206027 , -0.15255294],
         [ 0.06440085,  0.03732069,  0.07961182, -0.02237978,  0.17361407],
         [-0.03146603,  0.07052693, -0.04102432,  0.0770393 ,  0.16262058],
         [-0.04646247,  0.06805837,  0.05224777, -0.11874347, -0.16252693],
         [ 0.05452516, -0.06765041, -0.10805371,  0.01516557, -0.08011583]]
        )

    return W


@pytest.fixture(scope='module')
def expected_Lambda_w():
    Lambda_w = np.asarray(
          [[ 1.00000008e+00, -8.82918863e-10,  2.13538057e-08,
            -5.66401194e-08,  1.05187273e-07],
           [-8.82918597e-10,  9.99999975e-01,  2.36287054e-08,
            -2.92714815e-08,  1.22269490e-07],
           [ 2.13538057e-08,  2.36287052e-08,  9.99999913e-01,
             1.96704686e-08, -3.06876352e-08],
           [-5.66401194e-08, -2.92714810e-08,  1.96704685e-08,
             1.00000001e+00,  2.85858767e-08],
           [ 1.05187273e-07,  1.22269490e-07, -3.06876349e-08,
             2.85858762e-08,  9.99999962e-01]]
    )

    return Lambda_w


@pytest.fixture(scope='module')
def expected_Lambda_b():
    Lambda_b = np.asarray(
          [[ 3.95670528e-03,  8.00670672e-10,  2.23527057e-09,
            -1.22408034e-07, -4.51638159e-08],
           [ 8.00670684e-10,  5.49159152e-02,  1.04292283e-08,
            -7.51254074e-08,  2.23604450e-07],
           [ 2.23527054e-09,  1.04292284e-08,  6.63988995e-01,
            -8.40352486e-08, -4.61011650e-08],
           [-1.22408034e-07, -7.51254072e-08, -8.40352485e-08,
             4.65725826e+00,  9.72794822e-08],
           [-4.51638159e-08,  2.23604449e-07, -4.61011648e-08,
             9.72794823e-08,  1.04402521e+01]]
    )

    return Lambda_b


def test_optimize_maximum_likelihood():
    """ Imeplemented in test_optimizer_integration(). """
    pass


def test_as_dictionary_of_dictionaries():
    np.random.seed(1234)
    dim = 100

    labels = ['a', 'b', 'c', 'd']
    means = [np.random.random(dim) for label in labels]
    cov_diags = [np.random.random(dim) for label in labels]

    expected = {
        'a': {'mean': means[0], 'cov_diag': cov_diags[0]},
        'b': {'mean': means[1], 'cov_diag': cov_diags[1]},
        'c': {'mean': means[2], 'cov_diag': cov_diags[2]},
        'd': {'mean': means[3], 'cov_diag': cov_diags[3]}
    }

    actual = as_dictionary_of_dictionaries(labels, means, cov_diags)

    assert expected == actual


def test_calc_A(expected_W):
    N = 1234
    K = 23
    n_avg = N / K

    W = expected_W 
    dim = W.shape[0]
    Lambda_w = np.eye(dim)

    actual = calc_A(n_avg, Lambda_w, W)
    expected = np.linalg.inv(W.T) * (n_avg / (n_avg - 1)) ** .5

    assert_allclose(actual, expected)


def test_calc_Lambda_b(expected_scatter_matrices, expected_W,
                       expected_Lambda_b):
    S_b = expected_scatter_matrices['S_b']
    W = expected_W

    actual_Lambda_b = calc_Lambda_b(S_b, W)

    assert_allclose(actual_Lambda_b, expected_Lambda_b, rtol=1e-8)
    assert_diagonal(actual_Lambda_b, atol=1e-6)


def test_calc_Lambda_w(expected_scatter_matrices, expected_W,
                       expected_Lambda_w):
    S_w = expected_scatter_matrices['S_w']
    W = expected_W
    dim = W.shape[0]

    actual_Lambda_w = calc_Lambda_b(S_w, W)

    assert_allclose(actual_Lambda_w, expected_Lambda_w, rtol=1e-8)
    assert_diagonal(actual_Lambda_w, atol=1e-6)
    assert_allclose(np.eye(dim), actual_Lambda_w, atol=1e-6)


def test_calc_m():
    np.random.seed(1234)

    n = 1000
    dim = 100
    scale = 10

    expected_m = np.random.random(dim) * scale
    random_data = np.random.random((n, dim)) * scale
    random_data += expected_m - random_data.mean(axis=0)

    actual_m = calc_m(random_data)

    assert_allclose(actual_m, expected_m, rtol=1e-13)
    

def test_calc_n_avg():
    np.random.seed(1234)

    min_int_label = 1
    max_int_label = 10
    N = 1000

    Y = np.random.randint(min_int_label, max_int_label + 1, N)
    expected_n_avg = N / (max_int_label - min_int_label + 1)

    actual_n_avg = calc_n_avg(Y)

    assert_allclose(actual_n_avg, expected_n_avg, rtol=1e-100)


def test_calc_Psi():
    np.random.seed(1234)

    N = 1234
    K = 34
    n_avg = N / K

    dim = 5
    Lambda_b = np.diag(np.random.randint(1, 100, dim))
    Lambda_w = np.eye(dim)

    expected_Psi = (n_avg - 1) / n_avg * Lambda_b.diagonal() - (1 / n_avg)
    expected_Psi[expected_Psi < 0] = 0
    expected_Psi = np.diag(expected_Psi)

    actual_Psi = calc_Psi(Lambda_w, Lambda_b, n_avg)

    assert_diagonal(actual_Psi)
    assert_allclose(actual_Psi, expected_Psi)
    assert_allclose(actual_Psi.diagonal(), expected_Psi.diagonal())


def test_calc_scatter_matrices():
    X = np.asarray([[ 0,  1], [ 2,  3], [ 4,  5], [ 6,  7],
                    [ 8,  9], [10, 11], [12, 13], [14, 15]])

    Y = [0, 0, 0, 0, 1, 1, 1, 1]

    S_w_expected = np.asarray([[ 5,  5], [ 5,  5]])
    S_b_expected = np.asarray([[16, 16], [16, 16]])

    S_b, S_w = calc_scatter_matrices(X, Y)

    assert_allclose(S_w, S_w_expected, rtol=1e-20)
    assert_allclose(S_b, S_b_expected, rtol=1e-20)


def test_calc_W(expected_scatter_matrices, expected_W):
    S_b = expected_scatter_matrices['S_b']
    S_w = expected_scatter_matrices['S_w']

    actual_W = calc_W(S_b, S_w)

    assert_allclose(actual_W, expected_W, rtol=1e-6)


def test_get_posterior_params():
    np.random.seed(1234)

    dim = 5
    n = 4

    prior_params = {'mean': np.random.random(dim),
                    'cov_diag': np.random.random(dim)}

    U_model = np.random.random((2 * n, dim))
    Y = [0] * n + [1] * n

    actual = get_posterior_params(U_model, Y, prior_params)

    assert len(list(actual.keys())) == 2
    assert 0 in list(actual.keys()) and 1 in list(actual.keys())

    for i, key in enumerate([0, 1]):
        start = i * n
        end = i * n + n

        diag = prior_params['cov_diag']

        expected_cov_diag = diag / (1 + n * diag)
        assert_allclose(actual[key]['cov_diag'], expected_cov_diag)

        expected_mean = U_model[start: end].sum(axis=0) * expected_cov_diag
        assert_allclose(actual[key]['mean'], expected_mean)


def test_get_posterior_predictive_params():
    np.random.seed(1234)
    dim = 5

    expected = {
        'a': {'mean': np.random.random(dim),
              'cov_diag': np.random.random(dim)},
        'b': {'mean': np.random.random(dim),
              'cov_diag': np.random.random(dim)},
        'c': {'mean': np.random.random(dim),
              'cov_diag': np.random.random(dim)},
        'd': {'mean': np.random.random(dim),
              'cov_diag': np.random.random(dim)}
    }

    posterior_params = expected.copy()
    for key in posterior_params.keys():
        posterior_params[key]['cov_diag'] -= 1

    actual = get_posterior_predictive_params(posterior_params)

    assert type(actual) == dict
    for key in actual.keys():
        assert_allclose(expected[key]['mean'], actual[key]['mean'])
        assert_allclose(expected[key]['cov_diag'], actual[key]['cov_diag'])


def test_get_prior_params():
    np.random.seed(1234)
    dim = 50
    subspace_dim = 5

    dims = np.arange(dim)
    np.random.shuffle(dims)
    dims = dims[:subspace_dim]

    actual_mean = np.zeros(subspace_dim)
    actual_cov_diag = np.arange(1, 1 + subspace_dim)

    Psi = np.zeros(dim)
    Psi[dims] = actual_cov_diag
    Psi = np.diag(Psi)

    expected = {'mean': actual_mean, 'cov_diag': actual_cov_diag}
    actual = get_prior_params(Psi, dims)

    assert type(actual) == dict
    assert_allclose(actual['mean'], expected['mean'])
    assert_allclose(actual['cov_diag'], expected['cov_diag'])


def test_get_relevant_U_dims():
    np.random.seed(1234)

    dim = 100
    subspace_dim = 50

    expected = np.arange(dim)
    np.random.shuffle(expected)
    expected = expected[:subspace_dim]
    expected = np.sort(expected)

    Psi = np.zeros(dim)
    Psi[expected] = np.random.random(subspace_dim)
    Psi = np.diag(Psi)

    actual = get_relevant_U_dims(Psi)

    assert_array_equal(actual, expected)
