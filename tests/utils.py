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
import plda

from numpy.testing import assert_allclose
from scipy.stats import linregress


def assert_diagonal(matrix, rtol=1e-7, atol=0, equal_nan=True):
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]

    diagonal = matrix.diagonal()

    assert_allclose(matrix, np.diag(diagonal),
                    rtol=rtol, atol=atol, equal_nan=equal_nan)


def assert_error_falls_as_K_increases(calc_error_function,
                                      n_k, D, k_list,
                                      verbose=False):
    for k_1, k_2 in zip(k_list[:-1], k_list[1:]):
        assert k_1 > 1
        assert k_2 > 1
        assert k_2 > k_1

    errors = []
    for k in k_list:
        truth_dict = generate_data(n_k, k, D)
        error = calc_error_function(truth_dict)

        errors.append(error)

    errors = np.asarray(errors)
    X = np.arange(errors.shape[0])
    slope_of_error_vs_K = linregress(X, errors)[0]

    assert slope_of_error_vs_K < 0


def assert_error_falls_as_n_increases(calc_error_function,
                                      K, D, n_k_list,
                                      verbose=False):
    for n_k_1, n_k_2 in zip(n_k_list[:-1], n_k_list[1:]):
        assert n_k_1 > 1
        assert n_k_2 > 1
        assert n_k_2 > n_k_1

    errors = []
    for n_k in n_k_list:
        truth_dict = generate_data(n_k, K, D)
        error = calc_error_function(truth_dict)

        errors.append(error)

    errors = np.asarray(errors)
    X = np.arange(errors.shape[0])
    slope_of_error_vs_number_of_data = linregress(X, errors)[0]

    assert slope_of_error_vs_number_of_data < 0


def calc_log_mean_squared_error(expcted, predicted):
    return np.log(calc_mean_squared_error(expected, predicted))


def calc_mean_squared_error(expected, predicted, as_log=False):
    assert type(as_log) == bool

    mse = np.mean((expected - predicted) ** 2)

    if not as_log:
        return mse

    else:
        return np.log(mse)


def get_verbose_print_function(is_verbose):
    if is_verbose:
        return print

    else:
        def print_function(*args, **kwargs):
            return None

        return print_function


def generate_data(n_k, K, dimensionality):
    noise_scale = 1e-7

    Phi_w, prior_mean, Phi_b = generate_model_parameters(dimensionality)

    means = np.random.multivariate_normal(prior_mean, Phi_b, K)
    data = []
    labels = []

    for i, mean in enumerate(means):
        data_k = np.random.multivariate_normal(mean, Phi_w, n_k)

        data.append(data_k)
        labels += [i] * n_k

    truth = {
        'data': np.vstack(data),
        'labels': labels,
        'means': means,
        'Phi_w': Phi_w,
        'prior_mean': prior_mean,
        'Phi_b': Phi_b,
        'n_k': n_k
    }

    return truth


def generate_model_parameters(dimensionality):
    m_scale = np.random.randint(0, 10, 1)
    Phi_w_scale = 4
    Phi_b_scale = 8
    noise_scale = 1e-7

    prior_mean = np.random.random(dimensionality) * m_scale

    arr = np.random.random((dimensionality, dimensionality)) * Phi_w_scale
    Phi_w = np.matmul(arr, arr.T)

    arr = np.random.random((dimensionality, dimensionality)) * Phi_b_scale
    Phi_b = np.matmul(arr, arr.T)

    while np.linalg.matrix_rank(Phi_w) != dimensionality:
        diagonal_noise = np.diag(np.random.random(Phi_w.shape[0]))
        Phi_w += diagonal_noise * noise_scale

    while np.linalg.matrix_rank(Phi_b) != dimensionality:
        diagonal_noise = np.diag(np.random.random(Phi_b.shape[0]))
        Phi_b += diagonal_noise * noise_scale

    return Phi_w, prior_mean, Phi_b


def get_model(data, labels):
    return plda.Model(data, labels)
