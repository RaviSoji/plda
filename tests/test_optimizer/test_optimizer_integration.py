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
from plda.plda.optimizer import optimize_maximum_likelihood
from plda.tests.utils import generate_data 

# Tests parameters returned by plda.optimizer.optimize_maximum_likelihood().

@pytest.fixture(scope='module')
def data_dict():
    np.random.seed(1234)

    return generate_data(n_k=1000, K=10, dimensionality=50)


@pytest.fixture(scope='module')
def maximum_likelihood_parameters(data_dict):
    X = data_dict['data']
    Y = data_dict['labels']

    params = optimize_maximum_likelihood(X, Y)

    params_dict = dict()
    params_dict['m'] = params[0]
    params_dict['A'] = params[1]
    params_dict['Psi'] = params[2]
    params_dict['relevant_U_dims'] = params[3]
    params_dict['inv_A'] = params[4]

    return params_dict


def test_m(maximum_likelihood_parameters, data_dict):
    dim = data_dict['data'].shape[-1]
    m = maximum_likelihood_parameters['m']

    assert m.shape[0] == dim
    assert len(m.shape) == 1
    assert_allclose(m, data_dict['data'].mean(axis=0))


def test_A(maximum_likelihood_parameters, data_dict):
    dim = data_dict['data'].shape[-1]
    A = maximum_likelihood_parameters['A']

    assert len(A.shape) == 2
    assert A.shape[0] == dim
    assert A.shape[0] == A.shape[1]

    actual_Phi_w = np.matmul(A, A.T)
    assert_allclose(actual_Phi_w, data_dict['Phi_w'], rtol=1)


def test_Psi(maximum_likelihood_parameters, data_dict):
    dim = data_dict['data'].shape[-1]
    K = len(data_dict['means'])  

    relevant_U_dims = maximum_likelihood_parameters['relevant_U_dims']
    Psi = maximum_likelihood_parameters['Psi']

    assert len(Psi.shape) == 2
    assert Psi.shape[0] == dim
    assert Psi.shape[0] == Psi.shape[1]
    assert (Psi.diagonal() != 0).sum() == K - 1

    inv_A = maximum_likelihood_parameters['inv_A']
    Phi_b = data_dict['Phi_b']

    actual = np.matmul(np.matmul(inv_A, Phi_b), inv_A.T)
    actual = actual.diagonal()[relevant_U_dims]
    expected = Psi.diagonal()[relevant_U_dims]
    assert_allclose(actual, expected, rtol=4.5)


def test_relevant_U_dims(maximum_likelihood_parameters, data_dict):
    K = len(data_dict['means'])  # Number of categories in the training data.
    relevant_U_dims = maximum_likelihood_parameters['relevant_U_dims']

    assert relevant_U_dims.shape == (K - 1,)


def test_inv_A(maximum_likelihood_parameters, data_dict):
    dim = data_dict['data'].shape[-1]
    inv_A = maximum_likelihood_parameters['inv_A']

    assert len(maximum_likelihood_parameters['inv_A'].shape) == 2
    assert inv_A.shape[0] == dim
    assert inv_A.shape[0] == inv_A.shape[1]

    expected = np.linalg.inv(maximum_likelihood_parameters['A'])
    assert_allclose(inv_A, expected)
