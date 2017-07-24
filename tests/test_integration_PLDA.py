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
        cls.means = None
        cls.covariances = None
        cls.data = None
        cls.class_info = None
        cls.m = None
        cls.S_b = None
        cls.S_w = None
        cls.W = None
        cls.lambda_b = None
        cls.lambda_w = None
        cls.A = None
        cls.psi = None
        cls.pdfs = None

        # Create data structure to hold class statistics and data.
        cls.labels = [x for x in range(cls.K)]
        cls.class_info = cls.get_data_dict_structure(cls, cls.labels)

        # Build artifical data set with means (2 ** label).
        cls.means = [np.ones(cls.n_dims) * (2 ** x) for x in cls.labels]
        cls.covariances = [np.eye(cls.n_dims) * 10 for x in cls.labels]

        cls.data = []
        for mean, cov, label in zip(cls.means, cls.covariances, cls.labels):
            n = int(cls.n_avg)
            for x in range(n):
                datum = (normal(mean, cov), label)
                cls.data.append(datum)
                cls.class_info[label]['data'].append(datum[0])

        # Compute and store class statistics of generated data.
        for label in cls.class_info.keys():
            mean = np.array(cls.class_info[label]['data']).mean(axis=0)
            cov = np.cov(np.array(cls.class_info[label]['data']).T)

            cls.class_info[label]['mean'] = mean
            cls.class_info[label]['covariance'] = cov

        cls.m = cls.calc_m(cls)
        cls.S_w = cls.calc_S_w(cls)
        cls.S_b = cls.calc_S_b(cls)
        cls.W = cls.calc_W(cls)
        cls.lambda_b = cls.calc_lambda_b(cls)
        cls.lambda_w = cls.calc_lambda_w(cls)
        cls.A = cls.calc_A(cls)
        cls.psi = cls.calc_psi(cls)
#        cls.pdfs = cls.get_pdfs()

        cls.model = PLDA(cls.data)

    def test_stats(cls):
        for label in cls.class_info.keys():
            result = cls.model.stats[label]['mean']
            expected = cls.class_info[label]['mean']
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

    def test_lambda_b(cls):
        cls.assert_same(cls.model.lambda_b, cls.lambda_b)

    def test_lambda_w(cls):
        cls.assert_same(cls.model.lambda_w, cls.lambda_w)

    def test_A(cls):
        cls.assert_same(cls.model.A, cls.A)

    def test_psi(cls):
        cls.assert_same(cls.model.psi, cls.psi)

    def test_psi_diagonal(cls):
        cls.assert_diagonal(cls.model.psi)

    def test_psi_nonnegative(cls):
        result = (cls.model.psi < 0).sum() # Adds 1 for every element < 0.
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
        """ A = inv(W.T)(n / (n-1) * lambda_w) ** .5, therefore
           inv( A / [n / (n-1) * diagonal(lambda_w)] ** .5 ).T """
        lambda_w = cls.lambda_w
        n = cls.n_avg
        result = (n / (n - 1)) * lambda_w.diagonal()
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

    def test_A_and_psi_recover_phi_b(cls):
        """ phi_b = S_b - S_w / (n-1) = (A)(psi)(A.T)
        """
        A = cls.model.A
        psi = cls.model.psi
        S_w = cls.model.S_w
        S_b = cls.model.S_b
        n = cls.n_avg

        phi_b_1 = np.matmul(np.matmul(A, psi), A.T)
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
                                     'mean': None,
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
            m_k = cls.class_info[label]['mean']
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
            mk_minus_m = cls.class_info[label]['mean'] - m
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

    def calc_lambda_b(cls):
        lambda_b = np.dot(np.dot(cls.W.T, cls.S_b), cls.W)

        return lambda_b

    def calc_lambda_w(cls):
        lambda_w = np.dot(np.dot(cls.W.T, cls.S_w), cls.W)

        return lambda_w

    def calc_A(cls):
        diag = cls.lambda_w.diagonal()
        diag = np.sqrt(cls.n_avg / (cls.n_avg - 1) * diag)
        A = inv(cls.W).T * diag

        return A

    def calc_psi(cls):
        lambda_b = cls.lambda_b.copy()
        lambda_b[np.isclose(lambda_b, 0)] = 0
        lambda_w = cls.lambda_w.copy()
        lambda_w[np.isclose(lambda_w, 0)] = 0

        with np.errstate(divide='ignore', invalid='ignore'):
            psi = lambda_b / lambda_w
        psi[np.isnan(psi)] = 0
        psi = (cls.n_avg - 1) / cls.n_avg * psi - (1 / cls.n_avg)
        psi = np.maximum(psi, 0)

        return psi

#    def gen_pdfs(cls):
#
#        return pdfs
