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

from numpy.testing import (
    assert_array_equal,
    assert_allclose,
    assert_almost_equal
)
from scipy.stats import multivariate_normal as gaussian
from scipy.special import logsumexp
from plda import plda
from plda.tests.utils import (
    assert_error_falls_as_K_increases,
    assert_error_falls_as_n_increases,
    calc_mean_squared_error,
    generate_data
)


def test_A_recovers_Phi_w():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        expected = truth_dict['Phi_w']
        predicted = np.matmul(model.A, model.A.T)

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    n_ks = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234)
    assert_error_falls_as_n_increases(calc_error, K=2, D=2, n_k_list=n_ks)
    assert_error_falls_as_n_increases(calc_error, K=100, D=100, n_k_list=n_ks)


def test_inv_A_recovers_I_given_Phi_w():
    def calc_diagonal_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        dim = truth_dict['data'].shape[-1]
        Phi_w = truth_dict['Phi_w']

        expected = np.ones(dim)
        predicted = np.matmul(model.inv_A, Phi_w)
        predicted = np.matmul(predicted, model.inv_A.T)
        predicted = np.diag(predicted)

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    def calc_off_diagonal_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        dim = truth_dict['data'].shape[-1]
        Phi_w = truth_dict['Phi_w']

        expected = np.zeros((dim, dim))
        predicted = np.matmul(model.inv_A, Phi_w)
        predicted = np.matmul(predicted, model.inv_A.T)
        predicted[range(dim), range(dim)] = 0

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    n_ks = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234)
    assert_error_falls_as_n_increases(calc_diagonal_error,
                                      K=2, D=2, n_k_list=n_ks)
    assert_error_falls_as_n_increases(calc_diagonal_error,
                                      K=100, D=100, n_k_list=n_ks)

    assert_error_falls_as_n_increases(calc_off_diagonal_error,
                                      K=2, D=2, n_k_list=n_ks)
    assert_error_falls_as_n_increases(calc_off_diagonal_error,
                                      K=100, D=100, n_k_list=n_ks)


def test_A_and_Psi_recover_Phi_b():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        dim = truth_dict['data'].shape[-1]
        Phi_b = truth_dict['Phi_b']

        expected = Phi_b
        predicted = np.matmul(model.A, model.Psi)
        predicted = np.matmul(predicted, model.A.T)

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234)
    assert_error_falls_as_K_increases(calc_error, n_k=50, D=2, k_list=ks)
    assert_error_falls_as_K_increases(calc_error, n_k=50, D=50, k_list=ks)


def test_Psi_recovers_Phi_b_given_Phi_w():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        Phi_w = truth_dict['Phi_w']
        Phi_b = truth_dict['Phi_b']

        expected = Phi_b
        predicted = np.matmul(Phi_w, model.Psi)

        error = calc_mean_squared_error(expected, predicted, as_log=True)
        print(error)

        return error

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234)
    assert_error_falls_as_K_increases(calc_error, n_k=1000, D=2, k_list=ks)
    assert_error_falls_as_K_increases(calc_error, n_k=1000, D=50, k_list=ks)


def test_calc_logp_mariginal_likelihood():
    np.random.seed(1234)

    n_k = 100
    K = 5
    dim = 10

    data_dictionary = generate_data(n_k, K, dim)
    X = data_dictionary['data']
    Y = data_dictionary['labels']
    model = plda.Model(X, Y)

    prior_mean = model.prior_params['mean']
    prior_cov_diag = model.prior_params['cov_diag']

    logpdf = gaussian(prior_mean, np.diag(prior_cov_diag + 1)).logpdf

    data = np.random.random((n_k, prior_mean.shape[-1]))
    expected_logps = logpdf(data)
    actual_logps = model.calc_logp_marginal_likelihood(data[:, None])

    assert_allclose(actual_logps, expected_logps)


def test_calc_logp_posterior():
    """ Implicitly tested in test_calc_logp_posterior_predictive(). """
    pass


def test_calc_logp_posterior_predictive():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        Phi_w = truth_dict['Phi_w']
        likelihood_means = truth_dict['means']

        dim = Phi_w.shape[0]
        test_data = np.random.randint(-100, 100, (10, dim))

        expected = []
        predicted = []
        for mean, label in zip(truth_dict['means'], truth_dict['labels']):
            true_logps = gaussian(mean, Phi_w).logpdf(test_data)
            true_logps -= logsumexp(true_logps)

            test_U = model.transform(test_data, 'D', 'U_model')
            predicted_logps = model.calc_logp_posterior_predictive(test_U,
                                                                   label)
            predicted_logps -= logsumexp(predicted_logps)

            expected.append(true_logps)
            predicted.append(predicted_logps)

        expected = np.asarray(expected)
        predicted = np.asarray(predicted)

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ns = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234)
    assert_error_falls_as_n_increases(calc_error, K=2, D=2, n_k_list=ns)
    assert_error_falls_as_n_increases(calc_error, K=100, D=5, n_k_list=ns)


def test_calc_logp_prior_():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        Phi_b = truth_dict['Phi_b']
        prior_mean = truth_dict['prior_mean']
        dim = prior_mean.shape[0]

        random_vectors = np.random.randint(-100, 100, (10, dim))
        expected = gaussian(prior_mean, Phi_b).logpdf(random_vectors)

        latent_vectors = model.transform(random_vectors, 'D', 'U_model')
        predicted = model.calc_logp_prior(latent_vectors)

        error = calc_mean_squared_error(expected, predicted, as_log=True)
        print(error)

        return error

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234)
    assert_error_falls_as_K_increases(calc_error, n_k=1000, D=2, k_list=ks)
    assert_error_falls_as_K_increases(calc_error, n_k=1000, D=50, k_list=ks)


def test_A_m_and_posterior_means_recover_true_means():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        expected = truth_dict['means']

        predicted = []
        for k, params in model.posterior_params.items():
            predicted.append(params['mean'])

        predicted = np.asarray(predicted)
        predicted = model.transform(predicted, from_space='U_model',
                                    to_space='D')

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    n_ks = [10, 100, 1000]  # List of sample sizes.

    np.random.seed(1234)
    assert_error_falls_as_n_increases(calc_error, K=2, D=2, n_k_list=n_ks)
    assert_error_falls_as_n_increases(calc_error, K=100, D=100, n_k_list=n_ks)


def test_m_recovers_true_prior_mean():
    def calc_error(truth_dict):
        model = plda.Model(truth_dict['data'], truth_dict['labels'])

        expected = truth_dict['prior_mean']
        predicted = model.m

        error = calc_mean_squared_error(expected, predicted, as_log=True)

        return error

    ks = [10, 100, 1000]  # List of numbers of categories.

    np.random.seed(1234)
    assert_error_falls_as_K_increases(calc_error, n_k=50, D=2, k_list=ks)
    assert_error_falls_as_K_increases(calc_error, n_k=50, D=50, k_list=ks)


def test_calc_same_diff_log_likelihood_ratio():
    np.random.seed(1234)

    n_k = 100
    K = 7
    dim = 10

    data_dictionary = generate_data(n_k, K, dim)
    X_train = data_dictionary['data'][:500]

    Y = data_dictionary['labels'][:500]
    model = plda.Model(X_train, Y)

    X_infer_category_1 = data_dictionary['data'][500:600]
    X_infer_category_1 = model.transform(X_infer_category_1, 'D', 'U_model')
    X_infer_category_2 = data_dictionary['data'][600:]
    X_infer_category_2 = model.transform(X_infer_category_2, 'D', 'U_model')

    similarity_1v2 = model.calc_same_diff_log_likelihood_ratio(X_infer_category_1, X_infer_category_2)
    similarity_2v2 = model.calc_same_diff_log_likelihood_ratio(
        X_infer_category_2[:50],
        X_infer_category_2[50:]
    )
    assert_almost_equal(similarity_1v2, -46868.44557534719)
    assert_almost_equal(similarity_2v2, 29.917954937414834)
