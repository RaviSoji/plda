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

import unittest
from plda import PLDA
import numpy as np
from numpy.random import multivariate_normal as m_normal
from scipy.linalg import det, eigh, inv
from numpy.linalg import matrix_rank
from scipy.stats import multivariate_normal


class TestPLDA(unittest.TestCase):
    def setUp(self):
        self.dims = 3
        self.n = 100
        self.K = 5
        self.shared_cov = np.eye(self.dims)
        self.dist_bw_means = 3

        X, Y, fnames = self.gen_data(self.dims, self.n, self.K,
                                     self.shared_cov, self.dist_bw_means)
        self.model = PLDA(X, Y, fnames)
        self.X, self.Y, self.fnames = X, Y, fnames
        
    def gen_data(self, dims, n, K, shared_cov, dist_bw_means):
        np.random.seed(0)
        X = np.vstack([m_normal(np.ones(dims) * dist_bw_means * x,
                                    shared_cov, n) for x in range(K)])
        Y = np.hstack(['gaussian_{}'.format(k)] * 100 for k in range(5))
        fnames = np.asarray(['gaussian_{}_x_{}.jpg'.format(k, x) \
                            for k in range(K) for x in range(n)])

        # Do not delete these assertions.
        assert len(X.shape) == 2
        assert X.shape == (n * K, dims)
        assert Y.shape[0] == X.shape[0] == fnames.shape[0]
        assert len(Y.shape) == 1
        assert len(fnames.shape) == 1

        return X, Y, fnames

    def test_mk_data_dict(self):
        X, Y, fnames = self.gen_data(self.dims, self.n, self.K,
                                     self.shared_cov, self.dist_bw_means)
        sorting_idxs = np.argsort(fnames)
        X, Y, fnames = X[sorting_idxs], Y[sorting_idxs], fnames[sorting_idxs]

        order = np.arange(0, Y.shape[0])
        np.random.seed(0)
        np.random.shuffle(order)

        model_1 = PLDA(X[order], Y[order], fnames[order])
        model_2 = PLDA(X, Y)
        keys = list(model_1.data.keys())

        tolerance = .000000001
        for i, key in enumerate(keys):
            n_1 = model_1.data[key]['n']
            n_2 = model_2.data[key]['n']

            X_1 = model_1.data[key]['X']
            X_2 = model_2.data[key]['X']

            mean_1 = model_1.data[key]['mean']
            mean_2 = model_2.data[key]['mean']

            cov_1 = model_1.data[key]['cov']
            cov_2 = model_2.data[key]['cov']

            fnames_1 = model_1.data[key]['fnames']
            fnames_2 = model_2.data[key]['fnames']

            # Assert that data dicts in both models are equal, except fnames.
            self.assertEqual(n_1, n_2)

            X_1 = np.asarray(X_1)[np.argsort(fnames_1)]
            X_2 = np.asarray(X_2)  # X_2 is already sorted.
            self.assert_same(X_1, X_2, tolerance=tolerance)

            self.assert_same(mean_1, mean_2, tolerance=tolerance)
            self.assert_same(cov_1, cov_2, tolerance=tolerance)
            self.assertEqual(model_1.data[key].keys(),
                             model_2.data[key].keys())
            self.assertFalse(np.array_equal(fnames_1, fnames_2))

            # Assert that data dicts are storing the correct values.
            self.assertEqual(n_1, self.n)

            X_subset = X[Y == key, :]
            self.assert_same(mean_1, X_subset.mean(axis=0), tolerance=tolerance)
            self.assert_same(cov_1, np.cov(X_subset.T), tolerance=tolerance)

            truth = [None] * int(self.n)
            self.assertTrue(np.array_equal(fnames_2, np.asarray(truth)))

            fnames_1.sort()
            self.assertTrue(np.array_equal(fnames_1, fnames[Y == key]))

    def test_get_ns(self):
        ns_1, labels = self.model.get_ns(return_labels=True)
        ns_2 = self.model.get_ns()

        # Assert that function returns the same order regardless of returns.
        ns_1, ns_2, labels = np.asarray(ns_1), np.asarray(ns_2), np.asarray(labels)
        idxs = np.argsort(labels)
        ns_1, ns_2, labels = ns_1[idxs], ns_2[idxs], labels[idxs]
        self.assert_same(ns_1, ns_2)

        # Assert that returned values come from the correct data structure.
        for n, key in zip(ns_1, labels):
            self.assertEqual(self.model.data[key]['n'], n)

        # Assert that all the appropriate labels are returned.
        truth = np.unique(self.Y)
        np.sort(truth)
        self.assert_same(labels, truth)

    def test_get_means(self):
        means_1, labels = self.model.get_means(return_labels=True)
        means_2 = self.model.get_means()

        # Assert that function returns the same order regardless of returns.
        means_1, means_2, = np.asarray(means_1), np.asarray(means_2)
        labels = np.asarray(labels)
        idxs = np.argsort(labels)
        means_1, means_2, labels = means_1[idxs], means_2[idxs], labels[idxs]
        self.assert_same(means_1, means_2)

        # Assert that returned values come from the correct data structure.
        for mean, key in zip(means_1, labels):
            self.assertTrue(np.array_equal(self.model.data[key]['mean'], mean))

        # Assert that all the appropriate labels are returned.
        truth = np.unique(self.Y)
        np.sort(truth)
        self.assert_same(labels, truth)

    def test_get_covs(self):
        cov_diags_1, labels = self.model.get_covs(return_labels=True)
        cov_diags_2 = self.model.get_covs()

        # Assert that function returns the same order regardless of returns.
        cov_diags_1, cov_diags_2 = np.asarray(cov_diags_1), np.asarray(cov_diags_2)
        labels = np.asarray(labels)
        idxs = np.argsort(labels)
        cov_diags_1, cov_diags_2, labels = cov_diags_1[idxs],\
                                           cov_diags_2[idxs], labels[idxs]
        self.assert_same(cov_diags_1, cov_diags_2)

        # Assert that returned values come from the correct data structure.
        for cov, key in zip(cov_diags_1, labels):
            self.assertTrue(np.array_equal(self.model.data[key]['cov'], cov))

        # Assert that all the appropriate labels are returned.
        truth = np.unique(self.Y)
        np.sort(truth)
        self.assert_same(labels, truth)

    def test_test_calc_m(self):
        self.assert_same(self.model.m, self.X.mean(axis=0),
                         tolerance=.000000001)

    def test_fit(self):
        # Test parameter values the model computes without defined functions.
        self.assertEqual(self.model.K, self.K)
        self.assertEqual(self.model.params['K'], self.K)

        self.assertEqual(self.model.N, self.K * self.n)
        self.assertEqual(self.model.params['N'], self.K * self.n)

        self.assertEqual(self.model.n_avg, self.n)
        self.assertEqual(self.model.params['n_avg'], self.n)

        # Verify that the parameters of the  model are "fit".
        self.assert_same(self.model.m, self.model.params['m'])
        self.assert_same(self.model.S_w, self.model.params['S_w'])
        self.assert_same(self.model.S_b, self.model.params['S_b'])
        self.assert_same(self.model.W, self.model.params['W'])
        self.assert_same(self.model.Λ_b, self.model.params['Λ_b'])
        self.assert_same(self.model.Λ_w, self.model.params['Λ_w'])
        self.assert_same(self.model.A, self.model.params['A'])
        self.assert_same(self.model.Ψ, self.model.params['Ψ'])

    def test_calc_W(self):
        self.assert_invertible(self.model.W)
        vals, W = eigh(self.model.S_b, self.model.S_w)
        self.assert_same(self.model.W, W)

    def test_calc_A(self):
        dims = self.dims
        Λ_w = np.diag(np.ones(dims))
        W = np.random.randint(0, 9, self.dims ** 2).reshape(dims, dims) + \
            np.eye(dims)
        n_avg = 9

        A_model = self.model.calc_A(n_avg, Λ_w, W)

        A_truth = Λ_w * n_avg / (n_avg - 1)
        A_truth = np.sqrt(A_truth)
        A_truth = np.matmul(np.linalg.inv(W).T, A_truth)

        self.assert_same(A_model, A_truth, tolerance=None)
        self.assert_invertible(self.model.W)

    def test_calc_S_b(self):
        tolerance = .000000001
        X, Y = self.X, self.Y
        unique_labels, counts = np.unique(Y, return_counts=True)

        means = []
        for n, label in zip(counts, unique_labels):
            means.append(X[Y == label].mean(axis=0))

        m = X.mean(axis=0)
        S_b = []
        for n, mean in zip(counts, means):
            diff = mean - m
            S_b.append(n * np.outer(diff, diff))

        S_b = np.asarray(S_b).sum(axis=0) / np.sum(counts)

        self.assert_same(self.model.S_b, S_b, tolerance=tolerance)

    def test_calc_S_w(self):
        tolerance = .000000001
        X, Y = self.X, self.Y
        unique_labels = np.unique(Y)
        matrices = []
        for label in unique_labels:
            data = X[Y == label, :]
            mean = data.mean(axis=0)
            diffs = data - mean
            for row in diffs:
                matrices.append(np.outer(row, row))

        S_w = np.asarray(matrices).mean(axis=0)
        self.assert_same(self.model.S_w, S_w, tolerance=tolerance)

    def test_calc_Λ_b(self):
        tolerance = .000000001
        self.assert_diagonal(self.model.Λ_b)

        inv_W_T = np.linalg.inv(self.model.W.T)
        result = np.matmul(inv_W_T, self.model.Λ_b)
        result = np.matmul(result, inv(self.model.W))

        self.assert_same(result, self.model.S_b, tolerance=tolerance)

    def test_calc_Λ_w(self):
        tolerance = .000000001
        self.assert_diagonal(self.model.Λ_w)

        inv_W_T = np.linalg.inv(self.model.W.T)
        result = np.matmul(inv_W_T, self.model.Λ_w)
        result = np.matmul(result, inv(self.model.W))

        self.assert_same(result, self.model.S_w, tolerance=tolerance)

    def test_calc_Ψ(self):
        tolerance = .000000001

        # Recall that Λ_b, Λ_w, and Ψ are all diagonal matrices.
        n = 11
        Λ_b = np.diag(np.arange(1, self.dims + 1))
        Λ_w = np.diag(np.arange(-self.dims, 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = (n - 1) / n * (Λ_b / Λ_w) - (1 / n)
        Ψ[np.isnan(Ψ)] = 0
        Ψ[Ψ < 0] = 0

        Ψ_model = self.model.calc_Ψ(Λ_w, Λ_b, n)
        self.assert_same(Ψ_model, Ψ, tolerance=tolerance)

        self.assert_diagonal(self.model.Ψ)
        self.assertEqual(np.isnan(self.model.Ψ).sum(), 0)
        self.assertEqual(np.isinf(self.model.Ψ).sum(), 0)
        
    def test_get_relevant_dims(self):
        tolerance = .000000001
        Ψ = self.model.Ψ
        diag = Ψ.diagonal()
        relevant_dims = np.argsort(diag)[::-1][:100]
        self.assert_same(relevant_dims, self.model.get_relevant_dims(Ψ, n=100),
                         tolerance=tolerance)
        
        relevant_dims = np.squeeze(np.argwhere(diag != 0))
        self.assert_same(relevant_dims, self.model.get_relevant_dims(Ψ),
                         tolerance=tolerance)
        
    def test_add_datum(self):
        tolerance = .000000001
        old_model = self.model

        # Test adding to existing class, with fname supplied.
        new_X = np.ones(self.dims)
        existing_Y = list(self.model.data.keys())[-1]
        new_fname = 'new_fname.jpg'

        new_model = PLDA(self.X, self.Y, self.fnames)
        new_model.add_datum(new_X, existing_Y, new_fname)

        labels = set(list(self.model.data.keys()))
        unchanged = labels - set([existing_Y])
        for key in unchanged:
            self.assertEqual(old_model.data[key]['n'],
                             new_model.data[key]['n'])
            self.assert_same(np.asarray(old_model.data[key]['X']),
                             np.asarray(new_model.data[key]['X']),
                             tolerance=tolerance)
            self.assert_same(old_model.data[key]['mean'],
                             new_model.data[key]['mean'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['fnames'],
                             new_model.data[key]['fnames'])

        new_X_truth = old_model.data[existing_Y]['X'] + [new_X]
        new_n_model = new_model.data[existing_Y]['n']
        new_mean_model = new_model.data[existing_Y]['mean']
        new_cov_model = new_model.data[existing_Y]['cov']
        new_fnames_model = new_model.data[existing_Y]['fnames']

        new_n_truth = old_model.data[existing_Y]['n'] + 1
        self.assertEqual(new_n_model, new_n_truth)

        new_mean_truth = np.asarray(new_X_truth).mean(axis=0)
        self.assert_same(new_mean_model, new_mean_truth, tolerance=tolerance)

        new_cov_truth = np.cov(np.asarray(new_X_truth).T)
        self.assert_same(new_cov_model, new_cov_truth)

        new_fnames_truth = old_model.data[existing_Y]['fnames'] + [new_fname]
        new_fnames_truth = np.asarray(new_fnames_truth)
        self.assert_same(new_fnames_model.sort(), new_fnames_truth.sort())

        # Test adding to existing class without supplying fname.
        new_X = np.ones(self.dims)
        existing_Y = list(self.model.data.keys())[-1]

        new_model = PLDA(self.X, self.Y, self.fnames)
        new_model.add_datum(new_X, existing_Y)

        labels = set(list(self.model.data.keys()))
        unchanged = labels - set([existing_Y])
        for key in unchanged:
            self.assertEqual(old_model.data[key]['n'],
                             new_model.data[key]['n'])
            self.assert_same(np.asarray(old_model.data[key]['X']),
                             np.asarray(new_model.data[key]['X']),
                             tolerance=tolerance)
            self.assert_same(old_model.data[key]['mean'],
                             new_model.data[key]['mean'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['fnames'],
                             new_model.data[key]['fnames'])

        new_X_truth = old_model.data[existing_Y]['X'] + [new_X]
        new_n_model = new_model.data[existing_Y]['n']
        new_mean_model = new_model.data[existing_Y]['mean']
        new_cov_model = new_model.data[existing_Y]['cov']
        new_fnames_model = new_model.data[existing_Y]['fnames']

        new_n_truth = old_model.data[existing_Y]['n'] + 1
        self.assertEqual(new_n_model, new_n_truth)

        new_mean_truth = np.asarray(new_X_truth).mean(axis=0)
        self.assert_same(new_mean_model, new_mean_truth, tolerance=tolerance)

        new_cov_truth = np.cov(np.asarray(new_X_truth).T)
        self.assert_same(new_cov_model, new_cov_truth)

        new_fnames_truth = old_model.data[existing_Y]['fnames']
        new_fnames_model.remove(None)
        new_fnames_truth = np.asarray(new_fnames_truth)
        self.assert_same(new_fnames_model.sort(), new_fnames_truth.sort())

        # Test creating a new class with fname
        new_X = np.ones(self.dims)
        new_Y = 'new_category'
        new_fname = 'new_fname.jpg'

        new_model = PLDA(self.X, self.Y, self.fnames)
        new_model.add_datum(new_X, new_Y, new_fname)

        labels = set(list(new_model.data.keys()))
        unchanged = labels - set([new_Y])
        for key in unchanged:
            self.assertEqual(old_model.data[key]['n'],
                             new_model.data[key]['n'])
            self.assert_same(np.asarray(old_model.data[key]['X']),
                             np.asarray(new_model.data[key]['X']),
                             tolerance=tolerance)
            self.assert_same(old_model.data[key]['mean'],
                             new_model.data[key]['mean'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['fnames'],
                             new_model.data[key]['fnames'])

        new_X_truth = [new_X]
        new_n_model = new_model.data[new_Y]['n']
        new_mean_model = new_model.data[new_Y]['mean']
        new_cov_model = new_model.data[new_Y]['cov']
        new_fnames_model = new_model.data[new_Y]['fnames']

        new_n_truth = 1
        self.assertEqual(new_n_model, new_n_truth)

        new_mean_truth = new_X.copy()
        self.assert_same(new_mean_model, new_mean_truth, tolerance=tolerance)

        new_cov_truth = None
        self.assert_same(new_cov_model, new_cov_truth)

        new_fnames_truth = [new_fname]
        new_fnames_truth = np.asarray(new_fnames_truth)
        self.assert_same(new_fnames_model.sort(), new_fnames_truth.sort())

        # Test creating a new class without fname.
        new_X = np.ones(self.dims)
        new_Y = 'new_category'

        new_model = PLDA(self.X, self.Y, self.fnames)
        new_model.add_datum(new_X, new_Y)

        labels = set(list(new_model.data.keys()))
        unchanged = labels - set([new_Y])
        for key in unchanged:
            self.assertEqual(old_model.data[key]['n'],
                             new_model.data[key]['n'])
            self.assert_same(np.asarray(old_model.data[key]['X']),
                             np.asarray(new_model.data[key]['X']),
                             tolerance=tolerance)
            self.assert_same(old_model.data[key]['mean'],
                             new_model.data[key]['mean'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['cov'],
                             new_model.data[key]['cov'], tolerance=tolerance)
            self.assert_same(old_model.data[key]['fnames'],
                             new_model.data[key]['fnames'])

        new_X_truth = [new_X]
        new_n_model = new_model.data[new_Y]['n']
        new_mean_model = new_model.data[new_Y]['mean']
        new_cov_model = new_model.data[new_Y]['cov']
        new_fnames_model = new_model.data[new_Y]['fnames']

        new_n_truth = 1
        self.assertEqual(new_n_model, new_n_truth)

        new_mean_truth = new_X.copy()
        self.assert_same(new_mean_model, new_mean_truth, tolerance=tolerance)

        new_cov_truth = None
        self.assert_same(new_cov_model, new_cov_truth)

        new_fnames_truth = [None]
        new_fnames_truth = np.asarray(new_fnames_truth)
        self.assert_same(new_fnames_model.sort(), new_fnames_truth.sort())
        
#    def test_calc_marginal_likelihoods(self, data, ms=None, tau_diags=None,
#    def test_calc_posteriors(self, return_covs_as_diags=True, dims=None,
#    def test_calc_posterior_predictives(self, data, standardize_data,
    def test_whiten(self):
        tolerance = .000000001
        self.model.A = np.eye(self.dims)
        truth = self.X - self.X.mean(axis=0)
        predicted = self.model.whiten(self.X)
        self.assert_same(predicted, truth, tolerance=tolerance)

        self.model.A = np.eye(self.dims) * np.arange(1, self.dims + 1)
        truth = self.X - self.X.mean(axis=0)
        truth = np.matmul(truth, np.linalg.inv(self.model.A).T)
        predicted = self.model.whiten(self.X)
        self.assert_same(predicted, truth, tolerance=tolerance)

    def assert_same(self, a, b, tolerance=None):
        if tolerance is None:
            self.assertTrue(np.array_equal(a, b))
        else:
            self.assertTrue(np.allclose(a, b, atol=tolerance))
            
        self.assertTrue(type(a) == type(b))

    def assert_diagonal(self, A, tolerance=None):
        """ Tolerance is the number of decimals to round at. """
        diagonal = A.diagonal()
        if tolerance is not None:
            self.assert_array_equal(np.around(A, tolerance),
                                    np.around(np.diag(diagonal), tolerance))
        else:
            self.assert_same(A, np.diag(diagonal))

    def assert_invertible(self, A):
        rank = matrix_rank(A)
        is_invertible = rank != 0

        self.assertTrue(is_invertible)

    def assert_not_same(self, a, b, tolerance=None):
        if tolerance is None:
            self.assertFalse(np.array_equal(a, b))
        else:
            self.assertFalse(np.allclose(a, b, atol=tolerance))
            
        self.assertTrue(type(a) == type(b))
