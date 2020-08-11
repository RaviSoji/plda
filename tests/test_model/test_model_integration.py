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
    transform_D_to_X,
    transform_X_to_U,
    transform_U_to_U_model,
    transform_U_model_to_U,
    transform_U_to_X,
    transform_X_to_D
)
from plda.plda.optimizer import optimize_maximum_likelihood
from sklearn.decomposition import PCA
from plda.tests.utils import generate_data


@pytest.fixture('module')
def data_dict():
    np.random.seed(1234)

    return generate_data(n_k=1000, K=10, dimensionality=50)


@pytest.fixture('module')
def model(data_dict):
    return plda.Model(data_dict['data'], data_dict['labels'])


@pytest.fixture('module')
def expected_parameters(data_dict):
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


def test_maximum_likelihood_optimized_parameters(model, expected_parameters):
    def test_m():
        assert_allclose(model.m, expected_parameters['m'])

    def test_Psi():
        assert_allclose(model.Psi, expected_parameters['Psi'])

    def test_A():
        assert_allclose(model.A, expected_parameters['A'])

    def test_inv_A():
        assert_allclose(model.inv_A, expected_parameters['inv_A'])

    def test_relevant_U_dims():
        assert_allclose(model.relevant_U_dims,
                        expected_parameters['relevant_U_dims'])


def test_pca(data_dict, model):
    # When data has a full rank covariance matrix.
    assert model.pca is None

    # When the data does NOT have a full rank covariance matrix.
    shape = data_dict['data'].shape

    data = np.zeros((shape[0], shape[1] + 10))
    data[:, :shape[1]] = data_dict['data']

    actual = plda.Model(data, data_dict['labels'])
    assert actual.pca is not None
    assert isinstance(actual.pca, PCA)

    assert actual.pca.n_features_ == data.shape[1]
    assert actual.pca.n_components == shape[1]


def test_get_dimensionality(data_dict, model):
    # When data has a full rank covariance matrix.
    dim = data_dict['data'].shape[1]
    K = len(data_dict['means'])

    assert model.get_dimensionality('D') == dim
    assert model.get_dimensionality('X') == dim
    assert model.get_dimensionality('U') == dim
    assert model.get_dimensionality('U_model') == K - 1

    # When the data does NOT have a full rank covariance matrix.
    shape = data_dict['data'].shape

    data = np.zeros((shape[0], shape[1] + 10))
    data[:, :shape[1]] = data_dict['data']

    actual = plda.Model(data, data_dict['labels'])

    assert actual.get_dimensionality('D') == data.shape[-1]
    assert actual.get_dimensionality('X') == dim
    assert actual.get_dimensionality('U') == dim
    assert actual.get_dimensionality('U_model') == K - 1


def test_transform(data_dict, model):
    # When training data does have a full rank covariance matrix.
    # D to U_model.
    X = data_dict['data']

    expected = transform_D_to_X(X, model.pca)
    expected = transform_X_to_U(expected, model.inv_A, model.m)
    expected = transform_U_to_U_model(expected, model.relevant_U_dims)

    actual = model.transform(X, from_space='D', to_space='U_model')

    assert_allclose(actual, expected)

    # U_model to D.
    dim = model.get_dimensionality('U')
    expected = transform_U_model_to_U(actual, model.relevant_U_dims, dim)
    expected = transform_U_to_X(expected, model.A, model.m)
    expected = transform_X_to_D(expected, model.pca)

    actual = model.transform(actual, from_space='U_model', to_space='D')

    # When training data does not have a full rank covariance matrix.
    # D to U_model.
    shape = data_dict['data'].shape
    data = np.zeros((shape[0], shape[1] + 10))
    data[:, :shape[1]] = data_dict['data']

    tmp_model = plda.Model(data, data_dict['labels'])
    expected = transform_D_to_X(data, tmp_model.pca)
    expected = transform_X_to_U(expected, tmp_model.inv_A, tmp_model.m)
    expected = transform_U_to_U_model(expected, tmp_model.relevant_U_dims)

    actual = tmp_model.transform(data, from_space='D', to_space='U_model')
    assert_allclose(actual, expected)

    # U_model to D.
    dim = tmp_model.get_dimensionality('U')
    expected = transform_U_model_to_U(actual, tmp_model.relevant_U_dims, dim)
    expected = transform_U_to_X(expected, tmp_model.A, tmp_model.m)
    expected = transform_X_to_D(expected, tmp_model.pca)

    actual = model.transform(actual, from_space='U_model', to_space='D')


def test_prior_params(model):
    """
    Implemented in `tests/test_optimizer/test_optimizer_units.py`.
    Also implicitly tested in `tests/test_model/test_model_inference.py`.
    """
    pass


def test_posterior_params(model):
    """
    Implemented in `tests/test_optimizer/test_optimizer_units.py`.
    Also implicitly tested in `tests/test_model/test_model_inference.py`.
    """
    pass


def test_posterior_predictive_params(model):
    """
    Implemented in `tests/test_optimizer/test_optimizer_units.py`.
    Also implicitly tested in `tests/test_model/test_model_inference.py`.
    """
    pass


def test_calc_logp_marginal_likelihood():
    """ Implemented in `tests/test_model/test_model_inference.py`. """
    pass


def test_calc_logp_prior():
    """ Implemented in `tests/test_model/test_model_inference.py`. """
    pass


def test_calc_logp_posterior():
    """ Implemented in `tests/test_model/test_model_inference.py`. """
    pass


def test_calc_logp_posterior_predictive():
    """ Implemented in `tests/test_model/test_model_inference.py`. """
    pass
