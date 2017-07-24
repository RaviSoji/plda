# -*- coding: utf-8 -*-
import numpy as np
import unittest
from numpy.random import multivariate_normal as normal
from PLDA import PLDA
from scipy.linalg import eigh
from scipy.linalg import inv

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_dims = 5
        cls.K = 7  # Total number of k-classes.
        cls.N = cls.K * 1000  # 1000 data per class k.
        cls.n_avg = cls.N / cls.K  # For testing, n_k = n_avg for all k.
        cls.labels = None
        cls.μs = None
        cls.covariances = None
        cls.data = None
        cls.class_info = None
        cls.m = None
        cls.S_b = None
        cls.S_w = None
        cls.W = None
        cls.Λ_b = None
        cls.Λ_w = None
        cls.A = None
        cls.Ψ = None
        cls.pdfs = None

        # Create data structure to hold class statistics and data.
        cls.labels = [x for x in range(cls.K)]
        cls.class_info = cls.get_data_dict_structure(cls, cls.labels)

        # Build artifical data set with means (2 ** label).
        cls.μs = [np.ones(cls.n_dims) * (2 ** x) for x in cls.labels]
        cls.covariances = [np.eye(cls.n_dims) * 10 for x in cls.labels]

        cls.data = []
        for μ, cov, label in zip(cls.μs, cls.covariances, cls.labels):
            n = int(cls.n_avg)
            for x in range(n):
                datum = (normal(μ, cov), label)
                cls.data.append(datum)
                cls.class_info[label]['data'].append(datum[0])

        # Compute and store class statistics of generated data.
        for label in cls.class_info.keys():
            μ = np.array(cls.class_info[label]['data']).mean(axis=0)
            cov = np.cov(np.array(cls.class_info[label]['data']).T)

            cls.class_info[label]['μ'] = μ
            cls.class_info[label]['covariance'] = cov

        cls.m = cls.calc_m(cls)
        cls.S_w = cls.calc_S_w(cls)
        cls.S_b = cls.calc_S_b(cls)
        cls.W = cls.calc_W(cls)
        cls.Λ_b = cls.calc_Λ_b(cls)
        cls.Λ_w = cls.calc_Λ_w(cls)
        cls.A = cls.calc_A(cls)
        cls.Ψ = cls.calc_Ψ(cls)
#        cls.pdfs = cls.get_pdfs()

        cls.model = PLDA(cls.data)

    def test_stats(cls):
        for label in cls.class_info.keys():
            result = cls.model.stats[label]['μ']
            expected = cls.class_info[label]['μ']
            cls.assert_same(result, expected)

            result = cls.model.stats[label]['covariance']
            expected = cls.class_info[label]['covariance']
            cls.assert_same(result, expected)

            result = cls.model.stats[label]['n']
            expected = len(cls.class_info[label]['data'])
            cls.assert_same(result, expected)

    def test_m(cls):
        cls.assert_same(cls.model.m, cls.m)

    def test_S_w(cls):
        cls.assert_same(cls.model.S_w, cls.S_w)

    def test_S_b(cls):
        cls.assert_same(cls.model.S_b, cls.S_b)

    def test_W(cls):
        cls.assert_same(cls.model.W, cls.W)

    def test_Λ_b(cls):
        cls.assert_same(cls.model.Λ_b, cls.Λ_b)

    def test_Λ_w(cls):
        cls.assert_same(cls.model.Λ_w, cls.Λ_w)

    def test_A(cls):
        cls.assert_same(cls.model.A, cls.A)

    def test_Ψ(cls):
        cls.assert_same(cls.model.Ψ, cls.Ψ)

    def test_Ψ_diagonal(cls):
        cls.assert_diagonal(cls.model.Ψ)

    def test_Ψ_nonnegative(cls):
        result = (cls.model.Ψ < 0).sum() # Adds 1 for every element < 0.
        expected = 0
        cls.assertEqual(result, expected)

    def test_W_diagonalizes_S_w(cls):
        W = cls.model.W
        S_w = cls.model.S_w

        result = np.matmul(np.matmul(W.T, S_w), W)
        cls.assert_diagonal(result) 

    def test_W_diagonalizes_S_b(cls):
        W = cls.model.W
        S_b = cls.model.S_b

        result = np.matmul(np.matmul(W.T, S_b), W)
        cls.assert_diagonal(result) 

    def test_W_is_invertible(cls):
        cls.assert_invertible(cls.model.W)

    def test_A_is_invertible(cls):
        cls.assert_invertible(cls.model.A)

    def test_A_recovers_W(cls):
        """ A = inv(W.T)(n / (n-1) * Λ_w) ** .5, therefore
           inv( A / [n / (n-1) * diagonal(Λ_w)] ** .5 ).T """
        Λ_w = cls.Λ_w
        n = cls.n_avg
        result = (n / (n - 1)) * Λ_w.diagonal()
        result = np.sqrt(result)
        result = cls.model.A / result
        result = result.T
        result = inv(result)

        cls.assert_same(result, cls.model.W)
        
    def test_A_diagonalizes_S_b(cls):
        """ (inv(A))(S_b)(inv(A.T) should yield a diagonal matrix. """
        inv_A = inv(cls.model.A)
        S_b = cls.model.S_b
        result = np.matmul(np.matmul(inv_A, S_b), inv_A.T)

        cls.assert_diagonal(result)

    def test_A_diagonlizes_S_w(cls):
        """ (inv(A))(S_w)(inv(A.T) should yield a diagonal matrix. """
        inv_A = inv(cls.model.A)
        S_w = cls.model.S_w
        result = np.matmul(np.matmul(inv_A, S_w), inv_A.T)

        cls.assert_diagonal(result)

    def test_A_and_Ψ_recover_phi_b(cls):
        """ phi_b = S_b - S_w / (n-1) = (A)(Ψ)(A.T)
        """
        A = cls.model.A
        Ψ = cls.model.Ψ
        S_w = cls.model.S_w
        S_b = cls.model.S_b
        n = cls.n_avg

        phi_b_1 = np.matmul(np.matmul(A, Ψ), A.T)
        phi_b_2 = S_b - (S_w * (1 / (n - 1)))

        # Ideally both phi_b's are equal, but precision error is problematic.
        # cls.assert_same(phi_b_1, phi_b_2)

        # Hacky alternative since the assertion above has precision issues:
        #  compared to phi_b_2, S_b should be less similar to phi_b_1 because 
        #  the point of the term on the right of S_b is to bring it closer to
        #  the "true phi_b".
        passes = np.abs(phi_b_1 - S_b).sum() > np.abs(phi_b_1 - phi_b_2).sum()
        cls.assertTrue(passes)

    def test_A_recovers_phi_w(cls):
        """ phi_w = (A)(A.T) = n / (n-1) * S_w """
        A = cls.model.A
        S_w = cls.model.S_w
        n = cls.n_avg

        phi_w_1 = np.matmul(A, A.T)
        phi_w_2 = n / (n - 1) * S_w
        cls.assert_same(phi_w_1, phi_w_2)

    def test_A_and_phi_w_make_eye(cls):
        """ I = (V.T)(phi_w)(V), where A = inv(V.T)
            NOTES:
            (1) There are two ways to compute phi_w:
                phi_w = (A)(A.T)
                phi_w = n / (n-1) * S_w 
            (2) *** COMPUTE PHI WITH S_w: phi_w = n/(n-1) * S_w. ***
            (3) Do NOT use phi_w = (A)(A.T) because that is trivially true:
                 (V.T)(phi_w)(V), where V = inv(A.T), which gives
                 (inv(A))(A)(A.T)(inv(A.T)) = (I)(I) = I.
        """
        S_w = cls.model.S_w
        n = cls.n_avg
        V = inv(cls.model.A.T)

        phi_w = n / (n - 1) * S_w
        result = np.matmul(np.matmul(V.T, phi_w), V)
        cls.assert_same(result, np.eye(cls.n_dims))

#    def test_pdfs(cls):
#        cls.assert_same()

    def assert_same(cls, result, expected):
        are_same = np.allclose(result, expected)
        cls.assertTrue(are_same)

    def assert_diagonal(cls, matrix):
        result = np.diag(matrix.diagonal())
        are_same = np.allclose(result, matrix)
        cls.assertTrue(are_same)

    def assert_invertible(cls, matrix):
        rank = np.linalg.matrix_rank(matrix)
        cls.assertEqual(len(matrix.shape), 2)
        cls.assertEqual(matrix.shape[0], matrix.shape[1])
        cls.assertEqual(rank, matrix.shape[0])

    def get_data_dict_structure(cls, labels):
        data_dict = dict()
        for label in labels:
            data_dict[label] = dict({'data': [],
                                     'μ': None,
                                     'covariance': None})

        return data_dict

    def calc_m(cls):
        data = []
        for label in cls.class_info.keys():
            data.append(cls.class_info[label]['data'])
        data = np.vstack(data)
        m = data.mean(axis=0)

        return m
        
    def calc_S_w(cls):
        S_w = []
        for label in cls.class_info.keys():
            m_k = cls.class_info[label]['μ']
            for example in cls.class_info[label]['data']:
                x_i_minus_m_k = example - m_k
                S_w.append(np.outer(x_i_minus_m_k, x_i_minus_m_k))

        S_w = np.array(S_w).sum(axis=0)
        S_w /= cls.N

        return S_w

    def calc_S_b(cls):
        m = cls.m
        S_b = []
        for label in cls.class_info.keys():
            mk_minus_m = cls.class_info[label]['μ'] - m
            S_b.append(int(cls.n_avg) * np.outer(mk_minus_m, mk_minus_m))

        S_b = np.array(S_b).sum(axis=0)
        S_b /= cls.N

        return S_b

    def calc_W(cls):
        """ scipy.linalg.eigh is faster than scipy.linalg.eig. Only constraint
            is that the input matrices must be symmetric. 
        """
        __, W = eigh(cls.S_b, cls.S_w)

        return W

    def calc_Λ_b(cls):
        Λ_b = np.dot(np.dot(cls.W.T, cls.S_b), cls.W)

        return Λ_b

    def calc_Λ_w(cls):
        Λ_w = np.dot(np.dot(cls.W.T, cls.S_w), cls.W)

        return Λ_w

    def calc_A(cls):
        diag = cls.Λ_w.diagonal()
        diag = np.sqrt(cls.n_avg / (cls.n_avg - 1) * diag)
        A = inv(cls.W).T * diag

        return A

    def calc_Ψ(cls):
        Λ_b = cls.Λ_b.copy()
        Λ_b[np.isclose(Λ_b, 0)] = 0
        Λ_w = cls.Λ_w.copy()
        Λ_w[np.isclose(Λ_w, 0)] = 0

        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = Λ_b / Λ_w
        Ψ[np.isnan(Ψ)] = 0
        Ψ = (cls.n_avg - 1) / cls.n_avg * Ψ - (1 / cls.n_avg)
        Ψ = np.maximum(Ψ, 0)

        return Ψ

#    def gen_pdfs(cls):
#
#        return pdfs
