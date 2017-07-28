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
from PLDA import PLDA
from scipy.linalg import eig
from scipy.linalg import inv
from scipy.stats import linregress


# NOTE: test_Φ_b() and test_Φ_w take 160-190 seconds to run with ~ 60 CPU cores.
# Setting 'n', 'n_classes', and n_dims to smaller values will run code quickly,
#  but will likely cause tests to fail.
class TestPLDA(unittest.TestCase):
    def setUp(self, n=100000, n_classes=5, n_dims=5):
        """ Paramters that the model should recover: Φ_w, Φ_b, m, S_b. """
        self.n_dims = n_dims = 2                    # Dimension of the data.
        self.n_classes = n_classes
        self.n = n                                  # n for each class.

        self.Ψ = self.gen_Ψ(n_dims)                 # Diagonal matrix >= 0.
        self.m = self.gen_m(n_dims)                 # ndarray [1 x n_dims]
        V = self.gen_V(self.Ψ, n_classes, n_dims)   # v ~ N(0, Ψ)
        U = self.gen_U(n_dims, n, V)                # u ~ N(v, I)
        self.A, self.S_b = self.gen_A(V, n_classes, n_dims,
                                   return_S_b=True)
        X = self.unwhiten(U, self.A, self.m)        # x = m + Au
        Y = self.gen_labels(n_classes, n)
        self.labeled_X = self.label(X, Y)

        self.model = PLDA(self.labeled_X)

    def assert_same(self, result, expected):
        are_same = np.allclose(result, expected)
        self.assertTrue(are_same)

    def gen_Ψ(self, n_dims):
        """ Diagonal matrix describing the covariance between clusters.
        """
        Ψ = np.diag(10 / np.random.sample(n_dims))

        return Ψ

    def gen_m(self, n_dims):
        """ Displacement of the mean of the data from the origin.
        """
        m = np.random.randint(-1000, 1000, n_dims).astype(float)

        return m

    def gen_V(self, Ψ, n_classes, n_dims):
        """ v ~ N(0, Ψ): Consult Equations (2) on p. 533.

        DESCRIPTION: Samples whitened class centers from a multivariate
                      Gaussian distribution centered at 0, with covariance Ψ.
                      For testing purposes, we ensure that V.sum(axis=0) = 0. 

        PARAMETERS
         n_classes      (int): Number of classes.
         n_dims         (int): Dimensionality of the data.
         Ψ        (ndarray): Covariance between whitened class centers.
                                [n_dims x n_dims] 
        RETURNS
         V          (ndarray): Whitened class centers. [n_classes x n_dims] 
        """
        assert Ψ.shape[0] == Ψ.shape[1]

        μ = np.zeros(n_dims)  # [1 x n_dims] 
        np.random.seed(0)
        V = m_normal(μ, Ψ, n_classes)
        V = V - V.mean(axis=0)  # Center means at origin to control result.


        assert np.allclose(V.sum(axis=0), 0)

        return V

    def gen_A(self, V, n_classes, n_dims, return_S_b=False):
        """ A = [B][inv(Λ ** .5)][Q.T] and assumes same number of data
             in each class v. """
        B = np.random.randint(-100, 100, (n_dims, n_dims)).astype(float)
        big_V = np.matmul(V.T, V)  # V is now a scatter matrix.
        vals, vecs = eig(big_V)
        A = B / np.sqrt(vals.real)
        A = np.matmul(A, vecs.T)

        D = np.matmul(np.matmul(vecs.T, big_V), vecs)
        assert np.allclose(D, np.diag(vals))

        if return_S_b is True:
            S_b = 1 /n_classes * np.matmul(np.matmul(A, big_V), A.T)
            x = np.matmul(A, V.T).T

            S_b_empirical = 1 / n_classes * np.matmul(x.T, x)
            assert np.allclose(S_b, S_b_empirical)

            return A, S_b
        else:
            return A

    def gen_U(self, n_dims, n, V):
        """ u ~ N(v, I). 533.
        """
        cov = np.eye(n_dims)

        U = []
        for v in V:
            μ = np.zeros(n_dims)
            U_for_class_v = m_normal(μ, cov, n)
            U_for_class_v -= U_for_class_v.mean(axis=0)
            U_for_class_v += v  # Center at v to control test results.
            U.append(U_for_class_v)

        # To control the test result, each set of u's sums to its respective v.
        for x in range(len(U)):
            are_same = np.allclose(V[x], U[x].mean(axis=0))
            assert are_same ==  True

        U = np.vstack(U)

        return U

    def unwhiten(self, U, A, m):
        """ inv(A)[x - m]. See p. 537 Fig. 2.
        """
        X = np.matmul(A, U.T).T
        X += m

        return X

    def gen_labels(self, n_classes, n):
        labels = []
        for x in range(n_classes):
            labels += [x] * n

        return labels

    def label(self, data, labels):
        labeled_data = []
        for datum, label in zip(data, labels):
            labeled_data.append((datum, label))

        return labeled_data

    def test_m(self):
        """ Mean of the data: 1 / N * Σ_i(x^i). See p. 532.
        """
        self.assert_same(self.m, self.model.m)

    def test_S_b(self):
        """ Between-scatter matrix: 1 / N * Σ_k(n_k * [m_k - m][m_k - m].T).
            See p. 532.
        """
        self.assert_same(self.S_b, self.model.S_b)

    def experiment(self, n, n_dims, n_classes):
        Ψ = self.gen_Ψ(n_dims)
        m = self.gen_m(n_dims)
        V = self.gen_V(Ψ, n_classes, n_dims)
        U = self.gen_U(n_dims, n, V)
        A, S_b = self.gen_A(V, n_classes, n_dims,
                                   return_S_b=True)
        X = self.unwhiten(U, A, m)
        Y = self.gen_labels(n_classes, n)
        labeled_X = self.label(X, Y)
 
        model = PLDA(labeled_X)

        return A, Ψ, model

    def test_Φ_w(self):
    """ Φ_w = [A][A.T]. See p. 533.
    DESCRIPTION: Since A is a free parameter, t will not necessarily recover
                  the original A. Φ_w is what really describes the covariance
                  between cluster means (see p. 533), so that is what you want
                  to test - it is "closer" to the data".
    """
        n_experiments = int(np.log10(1000000) / 2)
        n_list = [100 ** x for x in range(1, n_experiments + 1)]
        n_list = np.array(n_list).astype(float)
        n_dims = self.n_dims
        n_classes = 30 #self.n_classes
        
        Φ_w_L1_errors = []
        for n in n_list:
            A, Ψ, model = self.experiment(int(n), n_dims, n_classes)

            Φ_w = np.matmul(A, A.T)
            Φ_w_model = np.matmul(model.A, model.A.T)

            L1_error = np.abs(Φ_w - Φ_w_model).mean()
            abs_μ = (np.abs(Φ_w).mean() + np.abs(Φ_w_model).mean()) * .5
            percent_error = L1_error / abs_μ * 100
            print('Testing Φ_w with {} samples: {} percent error'.format(n,
                  percent_error))
            Φ_w_L1_errors.append(percent_error)

        Y = Φ_w_L1_errors
        X = [x for x in range(len(Φ_w_L1_errors))]
        slope_of_error_vs_N = linregress(X, Y)[0]
        self.assertTrue(slope_of_error_vs_N < 0)

    def test_Φ_b(self):
    """ Φ_b = [A][Ψ][A.T]. See p. 533.

    DESCRIPTION: Since A and Ψ are free parameters, they will not necessarily
                  recover the original A & Ψ. Φ_b is what really describes
                  the covariance between cluster means (see p. 533), so that
                  is what you want to test - they are "closer" to the data".
    """
        n_classes_list = [4 ** x for x in range(1, 6)]
        n_list = [100 * n for n in n_classes_list]
        n_list = np.array(n_list).astype(float)
        n_dims = self.n_dims

        Φ_b_L1_errors = []
        for n, n_classes in zip(n_list, n_classes_list):
            A, Ψ, model = self.experiment(int(n), n_dims, n_classes)

            Φ_b = np.matmul(np.matmul(A, Ψ), A.T)
            Φ_b_model = np.matmul(np.matmul(model.A, model.Ψ), model.A.T)

            L1_error = np.abs(Φ_b - Φ_b_model).mean()
            abs_μ = (np.abs(Φ_b).mean() + np.abs(Φ_b_model).mean()) * .5
            percent_error = L1_error / abs_μ * 100
            Φ_b_L1_errors.append(percent_error)
            print('Testing Φ_b with {} classes: {} percent error'.format(
                  n_classes, percent_error))

        Y = Φ_b_L1_errors
        X = [x for x in range(len(Φ_b_L1_errors))]
        slope_of_error_vs_N = linregress(X, Y)[0]
        self.assertTrue(slope_of_error_vs_N < 0)
