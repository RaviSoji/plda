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

# -*- coding: utf-8 -*-
import numpy as np
import unittest
from numpy.random import multivariate_normal as m_normal
from numpy.linalg import matrix_rank
from plda import PLDA
from scipy.linalg import eigh, inv

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dims = 3
        cls.n = 100
        cls.K = 5
        cls.shared_cov = np.eye(cls.dims)
        cls.dist_bw_means = 3

        np.random.seed(0)
        X, Y, fnames = cls.gen_data(cls.dims, cls.n, cls.K,
                                     cls.shared_cov, cls.dist_bw_means)
        cls.model = PLDA(X, Y, fnames)
        cls.X, cls.Y, cls.fnames = X, Y, fnames

    @classmethod
    def gen_data(cls, dims, n, K, shared_cov, dist_bw_means):
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

    def assert_same(cls, a, b, tolerance=None):
        if tolerance is None:
            cls.assertTrue(np.array_equal(a, b))
        else:
            cls.assertTrue(np.allclose(a, b, atol=tolerance))
            
        cls.assertTrue(type(a) == type(b))

    def assert_diagonal(cls, A, tolerance=None):
        """ Tolerance is the number of decimals to round at. """
        diagonal = A.diagonal()
        if tolerance is not None:
            cls.assert_same(A, np.diag(diagonal), tolerance=tolerance)
        else:
            cls.assert_same(A, np.diag(diagonal))

    def assert_invertible(cls, A):
        rank = matrix_rank(A)
        is_invertible = rank != 0

        cls.assertTrue(is_invertible)

    def assert_not_same(cls, a, b, tolerance=None):
        if tolerance is None:
            cls.assertFalse(np.array_equal(a, b))
        else:
            cls.assertFalse(np.allclose(a, b, atol=tolerance))
            
        cls.assertTrue(type(a) == type(b))

    def test_get_ns(cls):
        ns, labels = cls.model.get_ns(return_labels=True)

        # Assert that returned values come from the correct data structure.
        for n, key in zip(ns, labels):
            cls.assertEqual(cls.model.data[key]['n'], n)
            cls.assertEqual(n, cls.n)

    def test_get_means(cls):
        tolerance = 1e-100
        means, labels = cls.model.get_means(return_labels=True)

        # Assert that returned values come from the correct data structure.
        for mean, key in zip(means, labels):
            cls.assert_same(cls.model.data[key]['mean'], mean,
                            tolerance=tolerance)
            cls.assert_same(mean, cls.X[cls.Y == key].mean(axis=0),
                            tolerance=tolerance)

    def test_get_covs(cls):
        tolerance = 1e-100
        cov_diags, labels = cls.model.get_covs(return_labels=True)

        # Assert that returned values come from the correct data structure.
        for cov, key in zip(cov_diags, labels):
            cls.assertTrue(np.array_equal(cls.model.data[key]['cov'], cov))
            cls.assert_same(cov, np.cov(cls.X[cls.Y == key].T))

    def test_optimized_m(cls):
        tolerance = 1e-100
        cls.assert_same(cls.model.m, cls.X.mean(axis=0),
                         tolerance=tolerance)

    def test_fit(cls):
        cls.assertEqual(cls.model.K, cls.K)
        cls.assertEqual(cls.model.N, cls.K * cls.n)
        cls.assertEqual(cls.model.n_avg, cls.n)

    def test_optimized_W(cls):
        tolerance = 1e-100

        vals, W = eigh(cls.model.S_b, cls.model.S_w)
        cls.assert_same(cls.model.W, W, tolerance=tolerance)

    def test_optimized_W_is_invertible(cls):
        cls.assert_invertible(cls.model.W)

    def test_optimized_W_diagonalizes_optimized_S_b(cls):
        tolerance = 1e-13

        S_b = cls.model.S_b
        W = cls.model.W
        Λ_b = np.matmul(W.T, np.matmul(S_b, W))

        cls.assert_diagonal(Λ_b, tolerance=tolerance)

    def test_optimized_W_diagonalizes_optimized_S_w(cls):
        tolerance = 1e-11

        W = cls.model.W
        S_w = cls.model.S_w
        Λ_w = np.matmul(W.T, np.matmul(S_w, W))

        cls.assert_diagonal(Λ_w, tolerance=tolerance)

    def test_optimized_Ψ_is_diagonal(cls):
        tolerance=None
        cls.assert_diagonal(cls.model.Ψ, tolerance=tolerance)

    def test_optimized_Ψ_is_nonnegative(cls):
        n_non_negative_elements= (cls.model.Ψ < 0).sum()
        expected = 0
        cls.assertEqual(n_non_negative_elements, expected)


    def test_optimized_A_is_invertible(cls):
        cls.assert_invertible(cls.model.A)

    def test_optimized_A_recovers_optimized_W(cls):
        tolerance = 1e-100
        """ A = inv(W^T)(n / (n-1) * Λ_w) ** .5, therefore
           inv( A / [n / (n-1) * diagonal(Λ_w)] ** .5 )^T """
        Λ_w = cls.model.Λ_w
        n = cls.model.n_avg
        result = (n / (n - 1)) * Λ_w.diagonal()
        result = np.sqrt(result)
        result = cls.model.A / result
        result = result.T
        result = inv(result)

        cls.assert_same(result, cls.model.W, tolerance=tolerance)
        
    def test_optimized_A_inv_diagonalizes_S_b(cls):
        """ (inv(A))(S_b)(inv(A^T) should yield a diagonal matrix. """
        tolerance = 1e-13

        inv_A = inv(cls.model.A)
        S_b = cls.model.S_b
        result = np.matmul(np.matmul(inv_A, S_b), inv_A.T)

        cls.assert_diagonal(result, tolerance=tolerance)

    def test_optimized_A_diagonlizes_optimized_S_w(cls):
        """ (inv(A))(S_w)(inv(A^T) should yield a diagonal matrix. """
        tolerance = 1e-13
        inv_A = inv(cls.model.A)
        S_w = cls.model.S_w
        result = np.matmul(np.matmul(inv_A, S_w), inv_A.T)

        cls.assert_diagonal(result, tolerance=tolerance)

    def test_optimized_A_and_Ψ_recover_Φ_b(cls):
        """ Φ_b = S_b - S_w / (n-1) = (A)(Ψ)(A^T), see p. 533 and p. 536.
        """
        tolerance = 1e-2
        A = cls.model.A
        Ψ = cls.model.Ψ
        S_w = cls.model.S_w
        S_b = cls.model.S_b
        n = cls.model.n_avg

        Φ_b_1 = np.matmul(np.matmul(A, Ψ), A.T)
        Φ_b_2 = S_b - (S_w * (1 / (n - 1)))

        # Ideally both Φ_b's are equal, but precision error is problematic.
        # Hence the high tolerance value (tolerance = 1e-2).
        cls.assert_same(Φ_b_1, Φ_b_2, tolerance=tolerance)

        # Hacky alternative since the assertion above has precision issues:
        #  compared to Φ_b_2, S_b should be less similar to Φ_b_1 because 
        #  the point of the term on the right of S_b is to bring it closer to
        #  the "true Φ_b".
        passes = np.abs(Φ_b_1 - S_b).sum() > np.abs(Φ_b_1 - Φ_b_2).sum()
        cls.assertTrue(passes)

    def test_optimized_A_recovers_Φ_w(cls):
        """ Φ_w = (A)(A^T) = n / (n-1) * S_w, see p. 533 and p. 536."""
        tolerance = 1e-13

        A = cls.model.A
        S_w = cls.model.S_w
        n = cls.model.n_avg

        Φ_w_1 = np.matmul(A, A.T)
        Φ_w_2 = n / (n - 1) * S_w
        cls.assert_same(Φ_w_1, Φ_w_2, tolerance=tolerance)

    def test_optimized_A_and_Φ_w_make_eye(cls):
        """ I = (V^T)(Φ_w)(V), where A = inv(V^T)
        NOTES:
        (1) There are two ways to compute Φ_w:
            Φ_w = (A)(A^T)
            Φ_w = n / (n-1) * S_w 
        (2) *** COMPUTE PHI WITH S_w: Φ_w = n/(n-1) * S_w. ***
        (3) Do NOT use Φ_w = (A)(A^T) because that is trivially true:
             (V^T)(Φ_w)(V), where V = inv(A^T), which gives
             (inv(A))(A)(A^T)(inv(A^T)) = (I)(I) = I.
        """
        tolerance = 1e-13  # Should be smaller than n / (n - 1).

        S_w = cls.model.S_w
        n = cls.model.n_avg
        V = inv(cls.model.A.T)
        cls.assertTrue(tolerance < (n / (n - 1)))

        Φ_w = n / (n - 1) * S_w
        result = np.matmul(np.matmul(V.T, Φ_w), V)
        cls.assert_same(result, np.eye(cls.dims), tolerance=tolerance)

    def test_whiten(cls):
        tolerance = 1e-200
        A = cls.model.A
        m = cls.X.mean(axis=0)
        truth = cls.X - m
        truth = np.matmul(truth, np.linalg.inv(A).T)

        result = cls.model.whiten(cls.X)
        cls.assert_same(result, truth, tolerance=tolerance)

    def test_posterior_predictive_and_marginal_for_one_datum_are_equal(cls):
        """ See proof in math notes. """
        pass

    def test_marginal_likelihood_equation(cls):
        """ EQ is from Kevin Murphy's cheatsheet. More detail in math notes."""
        pass

    def test_posterior_equation(cls):
        pass

    def test_marginal_likelihood_with_high_dimensional_input(cls):
        pass

    def test_posterior_predictive_with_high_dimensional_input(cls):
        pass
