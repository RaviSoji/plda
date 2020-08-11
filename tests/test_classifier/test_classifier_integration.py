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

from numpy.testing import (
    assert_array_equal,
    assert_allclose
)
from plda import plda
from plda.tests.utils import generate_data


@pytest.fixture(scope='module')
def data_dictionary():
    np.random.seed(1234)

    n_k = 1000
    K = 5
    dim = 10

    return generate_data(n_k, K, dim)


@pytest.fixture(scope='module')
def fitted_classifier(data_dictionary):
    X = data_dictionary['data']
    Y = data_dictionary['labels']

    classifier = plda.Classifier()
    classifier.fit_model(X, Y)

    return classifier


def test_fit_model(fitted_classifier):
    # Before fitting.
    classifier = plda.Classifier()
    assert classifier.model is None

    with pytest.raises(Exception):
        classifier.get_categories()

    # After fitting.
    assert fitted_classifier.model is not None
    assert isinstance(fitted_classifier.model, plda.Model)


def test_get_categories(fitted_classifier):
    # Before fitting.
    classifier = plda.Classifier()

    with pytest.raises(Exception):
        classifier.get_categories()

    # After fitting.
    expected = np.arange(5)
    actual = np.sort(fitted_classifier.get_categories())

    assert_array_equal(actual, expected)


def test_calc_logp_pp_categories(data_dictionary, fitted_classifier):
    means = data_dictionary['means']
    means = fitted_classifier.model.transform(means, from_space='D',
                                              to_space='U_model')
    labels = np.arange(len(means))

    # Without normalization.
    logps, k_list = fitted_classifier.calc_logp_pp_categories(means, False)
    for logp_row, k in zip(logps, k_list):
        assert labels[np.argmax(logp_row)] == k

    max_logps = np.max(logps, axis=-1)
    assert_allclose(max_logps[:-1], max_logps[1:], rtol=1e-2)

    # With normalization.
    logps, k_list = fitted_classifier.calc_logp_pp_categories(means, True)
    for logp_row, k in zip(logps, k_list):
        assert labels[np.argmax(logp_row)] == k

    assert_allclose(np.exp(logps).sum(axis=0), np.ones(logps.shape[0]))

    max_logps = np.max(logps, axis=-1)
    assert_allclose(max_logps[:-1], max_logps[1:])


def test_predict(data_dictionary, fitted_classifier):
    means_D = data_dictionary['means']
    means_X = fitted_classifier.model.transform(means_D, 'D', 'X')
    means_U = fitted_classifier.model.transform(means_X, 'X', 'U')
    labels = np.arange(len(means_D))

    # Unnormalized probabilities.
    predictions_D, logpps_D = fitted_classifier.predict(means_D, space='D')
    assert_array_equal(labels, predictions_D)

    predictions_X, logpps_X = fitted_classifier.predict(means_X, space='X')
    assert_array_equal(predictions_X, predictions_D)
    assert_allclose(logpps_X, logpps_D)

    predictions_U, logpps_U = fitted_classifier.predict(means_U, space='U')
    assert_array_equal(predictions_U, predictions_D)
    assert_allclose(logpps_U, logpps_X)

    # Normalized probabilities.
    predictions_D, logpps_D = fitted_classifier.predict(means_D, space='D',
                                                        normalize_logps=True)
    assert_array_equal(labels, predictions_D)

    predictions_X, logpps_X = fitted_classifier.predict(means_X, space='X',
                                                        normalize_logps=True)
    assert_array_equal(predictions_X, predictions_D)
    assert_allclose(logpps_X, logpps_D)

    predictions_U, logpps_U = fitted_classifier.predict(means_U, space='U',
                                                        normalize_logps=True)
    assert_array_equal(predictions_U, predictions_D)
    assert_allclose(logpps_U, logpps_X)
