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
from plda.plda.optimizer import (
    calc_scatter_matrices,
    optimize_maximum_likelihood
)
from plda.tests.utils import (
    assert_error_falls_as_K_increases,
    assert_error_falls_as_n_increases,
    calc_mean_squared_error,
    generate_data
)


def test_S_w_and_n_recover_Phi_w():
    def calc_error(truth_dict):
        data = truth_dict['data']
        labels = truth_dict['labels']

        S_b, S_w = calc_scatter_matrices(data, labels)
        n = truth_dict['n_k']

        expected = truth_dict['Phi_w']
        predicted = n / (n - 1) * S_w

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ns = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234) 
    assert_error_falls_as_n_increases(calc_error,
                                      K=2, D=2, n_k_list=ns) 
    assert_error_falls_as_n_increases(calc_error,
                                      K=100, D=100, n_k_list=ns)


def test_S_b_and_S_w_and_n_recover_Phi_b():
    def calc_error(truth_dict):
        data = truth_dict['data']
        labels = truth_dict['labels']

        S_b, S_w = calc_scatter_matrices(data, labels)
        n = truth_dict['n_k']

        expected = truth_dict['Phi_b']
        predicted = S_b - S_w / (n - 1)

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ns = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234) 
    assert_error_falls_as_n_increases(calc_error,
                                      K=2, D=2, n_k_list=ns) 
    assert_error_falls_as_n_increases(calc_error,
                                      K=100, D=100, n_k_list=ns)

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234) 
    assert_error_falls_as_K_increases(calc_error,
                                      n_k=2, D=2, k_list=ks) 
    assert_error_falls_as_K_increases(calc_error,
                                      n_k=100, D=100, k_list=ks)


@pytest.fixture(scope='module')
def truth_dict():
    np.random.seed(1234)
    return generate_data(n_k=500, K=2000, dimensionality=5)


@pytest.fixture(scope='module')
def fitted_parameters(truth_dict):
    X = truth_dict['data']
    Y = truth_dict['labels']

    return optimize_maximum_likelihood(X, Y)


def test_optimize_maximum_likelihood_m(truth_dict, fitted_parameters):
    expected = truth_dict['prior_mean']
    actual = fitted_parameters[0]

    assert_allclose(expected, actual, atol=.6)

    def calc_error(truth_dict):
        X = truth_dict['data']
        Y = truth_dict['labels']

        expected = truth_dict['prior_mean']
        predicted = optimize_maximum_likelihood(X, Y)[0]

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234)
    assert_error_falls_as_K_increases(calc_error, n_k=50, D=2, k_list=ks)


def test_optimize_maximum_likelihood_A():
    """ Implemented in tests/test_model/test_model_inference.py. """
    pass

def test_optimize_maximum_likelihood_Psi():
    """ Implemented in tests/test_model/test_model_inference.py. """
    pass

def test_optimize_maximum_likelihood_relevant_U_dims():
    """ Implemented in tests/test_model/test_model_inference.py. """
    pass

def test_optimize_maximum_likelihood_inv_A():
    """ Implemented in tests/test_model/test_model_inference.py. """
    pass

