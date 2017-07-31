import unittest
import PLDA
import numpy as np
from numpy.random import multivariate_normal as m_normal
from scipy.linalg import det
from scipy.linalg import eigh
from scipy.linalg import inv
from scipy.stats import multivariate_normal


class TestPLDA(unittest.TestCase):
    def setUp(self):
        self.K = 10
        self.n_dims = 2

        self.μ_list = [np.ones(self.n_dims) * x * 10 for x in range(self.K)]
        self.n_list = [100 * (x % 2 + 1) for x in range(self.K)]
        self.w_cov = self.gen_w_cov(self.n_dims)
        

        self.X = [m_normal(self.μ_list[x], self.w_cov, self.n_list[x]) \
                  for x in range(self.K)]

        self.X = np.vstack(self.X)
        self.Y = []
        for x in range(len(self.n_list)):
            self.Y += [x] * self.n_list[x]
        self.Y = np.array(self.Y)
        self.data = [(x, y) for (x, y) in zip(self.X, self.Y)]
        self.model = PLDA.PLDA(self.data)

    def test_data_list_to_data_dict(self):
        idxs = np.cumsum(np.array(self.n_list))
        labels = list(self.model.data.keys())
        for x in range(len(labels)):
            label = labels[x]
            X = self.X[idxs[x] - self.n_list[x]: idxs[x], :]
            result = np.array(self.model.data[label])

            self.assert_array_equal(result, X)

    def test_get_params_data_structure(self):
        ds = self.model.get_params_data_structure()
        n_keys = len(list(ds.keys())) 
        
        self.assertEqual(n_keys, 10 + self.K)

    def test_calc_A(self):
        """ inv( [A][(n / (n - 1) * Λ_w) ** (-.5)] ).T = W; See p. 537. """
        self.assert_invertible(self.model.A)

        expected = self.model.Λ_w.diagonal()
        expected = expected * self.model.n_avg / (self.model.n_avg - 1)
        expected = expected ** .5
        expected = 1 / expected
        expected = self.model.A * expected  # Expected should be diagonal.
        expected = np.linalg.inv(expected).T

        self.assert_array_equal(self.model.W, expected)

#    def test_calc_class_log_probs(self):
#        some_data = np.random.randint(0, 100, 10 * self.n_dims)
#        some_data = some_data.reshape(10, self.n_dims)
#
#        for label in self.model.pdfs.keys():
#            pdf = self.model.pdfs[label]
#            log_probs_model = pdf(some_data)
#
#            Ψ = self.model.Ψ
#            n_Ψ = self.model.n_avg * Ψ
#            n_Ψ_plus_eye = n_Ψ + np.eye(self.n_dims)
#            cov = Ψ + np.eye(self.n_dims)
#            cov = cov / n_Ψ_plus_eye
#            cov[np.isnan(cov)] = 0
#
#            transformation = n_Ψ / n_Ψ_plus_eye
#            transformation[np.isnan(transformation)] = 0
#
#            μ = self.model.params['v_' + str(label)]
#            μ = np.matmul(transformation, μ)
#            pdf = multivariate_normal(μ, cov).logpdf
#            log_probs = pdf(some_data)
#            print(log_probs_model)
#            print(log_probs)
#            self.assert_array_equal(log_probs_model, log_probs)
            
    def test_calc_K(self):
        K = self.model.calc_K()

        self.assertEqual(K, self.K)

    def test_calc_Λ_b(self):
        """ Λ_b = [W.T][S_b][W], so S_b = [inv([W.T])][Λ_b][inv([W])]. """
        self.assert_diagonal(self.model.Λ_b)

        inv_W_T = np.linalg.inv(self.model.W.T)
        result = np.matmul(inv_W_T, self.model.Λ_b)
        result = np.matmul(result, inv(self.model.W))

        self.assert_array_equal(result, self.model.S_b)

    def test_calc_Λ_w(self):
        self.assert_diagonal(self.model.Λ_w)

        inv_W_T = np.linalg.inv(self.model.W.T)
        result = np.matmul(inv_W_T, self.model.Λ_w)
        result = np.matmul(result, inv(self.model.W))

        self.assert_array_equal(result, self.model.S_w)

    def test_calc_m(self):
        expected = self.X.mean(axis=0)

        self.assert_array_equal(self.model.m, expected)

    def test_calc_N(self):
        result = self.model.N
        expected = np.array(self.n_list).sum()

        self.assertEqual(result, expected)

    def test_calc_n_avg(self):
        n_avg = np.array(self.n_list).mean()

        self.assertEqual(self.model.n_avg, n_avg)

    def test_calc_Ψ(self):
        """ Verify Ψ using p. 537, Fig. 2. NOTE: n is approximated as n_avg.
            max(0, (n_avg - 1) / n_avg * (Λ_b / Λ_w) - 1 / n_avg) """
        self.assert_diagonal(self.model.Ψ)
        n = self.model.n_avg

        # Recall that Λ_b, Λ_w, and Ψ are all diagonal matrices.
        diag_b = self.model.Λ_b.diagonal()
        diag_w = self.model.Λ_w.diagonal()
        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = (n - 1) / n * (diag_b / diag_w) - (1 / n)
        Ψ[np.isnan(Ψ)] = 0
        Ψ[Ψ < 0] = 0
        Ψ[np.isinf(Ψ)] = 0
        Ψ = np.diag(Ψ)

        self.assert_array_equal(self.model.Ψ, Ψ)

    def test_calc_S_b(self):
        """ Verify S_b using Equation (1) on p. 532, section 2. """
        # Assert that m is correctly computed (p. 537, Fig. 2).
        N = np.array(self.n_list).sum()
        μs = np.array(self.model.get_μs())
        weights = np.array(self.n_list) / N
        m = (μs.T * weights).T.sum(axis=0)

        self.assert_array_equal(self.model.m, m)

        # Compute S_b.
        μs = np.array(self.model.get_μs())
        diffs = μs - self.model.m
        S_b = np.matmul(diffs.T * weights, diffs)

        self.assert_array_equal(self.model.S_b, S_b)

    def test_calc_S_w(self):
        """ Veriy S_w using equation (1) (p. 532, section 2). """
        result = self.model.calc_S_w()
        N = np.array(self.n_list).sum()
        S_w = [] # List of within-class scatters for all classes.
        for label in self.model.data.keys():
            μ = self.model.stats[label]['μ']
            data = np.array(self.model.data[label])

            # Assert that the class mean is computed correctly.
            is_same = np.array_equal(data.mean(axis=0), μ)
            self.assertTrue(is_same)

            s_w = data - μ
            s_w = np.matmul(s_w.T, s_w)  # Scatter within a class.
            S_w.append(s_w)
        S_w = np.array(S_w)
        S_w = S_w.sum(axis=0)
        S_w /= N  # Weighted-mean of the within-class scatters.

        result = self.model.S_w

        self.assert_array_equal(result, S_w)
        
    def test_calc_W(self):
        self.assert_invertible(self.model.W)

        vals, W = eigh(self.model.S_b, self.model.S_w)

        self.assert_array_equal(self.model.W, W)
        #self.assert_array_equal(inv(self.model.W), self.model.W.T)

    def test_get_covariances(self):
        """ Verifies that returned covariacnes are correct and in order. """
        covs = self.model.get_covariances()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x] 
            result = covs[x]
            expected = self.model.stats[label]['covariance']
            self.assertTrue(np.array_equal(result, expected))

    def test_get_μs(self):
        """ Verifies that returned means are correct and in order. """
        μs = self.model.get_μs()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x] 
            result = μs[x]
            expected = self.model.stats[label]['μ']
            self.assertTrue(np.array_equal(result, expected))

    def test_get_sample_sizes(self):
        """ Verifies that returned sample sizes are correct and in order. """
        ns = self.model.get_sample_sizes()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x]
            result = ns[x]
            expected = self.model.stats[label]['n']

            self.assertEqual(result, expected)

    def test_get_stats_data_structure(self):
        result = self.model.get_stats_data_structure()
        self.assertIsInstance(result, dict)

        expected = {'μ': None,
                    'n': None,
                    'covariance': None}

        self.assertEqual(result, expected)

    def test_set_params(self):
        """ Not a true unit test because parameters depend on model.stats. """
        n_new_data_per_class = 10
        x = np.array([100] * self.n_dims)
        model2 = PLDA.PLDA(self.data)

        data = self.data.copy()
        for label in model2.data.keys():
             for i in range(n_new_data_per_class):
                data += [(x, label)]

        model1 = PLDA.PLDA(data)  # model1 should have updated stats or params.

        # model2 should not update stats/params until set_stats()/set_params().
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.data[label].append(x)

        model2.set_stats()  # Statistics for all classes should be updated.
        model2.set_params()

        self.assert_array_equal(model1.m, model2.m)
        self.assert_array_equal(model1.S_w, model2.S_w)
        self.assert_array_equal(model1.S_b, model2.S_b)
        self.assert_array_equal(model1.W, model2.W)
        self.assert_array_equal(model1.Λ_b, model2.Λ_b)
        self.assert_array_equal(model1.Λ_w, model2.Λ_w)
        self.assert_array_equal(model1.A, model2.A)
        self.assert_array_equal(model1.Ψ, model2.Ψ)
        self.assertEqual(model1.K, model2.K)
        self.assertEqual(model1.N, model2.N)
        self.assertEqual(model1.n_avg, model2.n_avg)

        # model.params should have a 'v_' + label key for each class.
        n_keys = len(list(self.model.params.keys()))
        self.assertEqual(n_keys, 11 + self.K)

    def test_set_pdfs(self):
        """ Not a true unit test because it depends on stats and params. """
        n_new_data_per_class = 10
        x = np.array([100] * self.n_dims)
        model2 = PLDA.PLDA(self.data)

        data = self.data.copy()
        for label in model2.data.keys():
             for i in range(n_new_data_per_class):
                data += [(x, label)]

        model1 = PLDA.PLDA(data)  # model1 should have updated stats or params.

        # model2 should not update pdfs until set_pdfs().
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.data[label].append(x)

        model2.set_stats()  # Statistics for all classes should be updated.
        model2.set_params()
        model2.set_pdfs()

        for label in model2.pdfs.keys():
            pdf1 = model1.pdfs[label]
            pdf2 = model2.pdfs[label]
            some_data = np.random.randint(0, 100, 10 * self.n_dims)
            some_data = some_data.reshape(10, self.n_dims)
            self.assert_array_equal(pdf1(some_data), pdf2(some_data))
            self.assert_array_equal(pdf1(some_data), pdf2(some_data))
        

    def test_set_stats(self):
        model1 = PLDA.PLDA(self.data)
        model2 = PLDA.PLDA(self.data)

        x = np.array([100] * self.n_dims)
        n_new_data_per_class = 10
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.data[label].append(x)

        model2.set_stats()  # Statistics for all classes should be updated.

        for label in model2.stats.keys():
            data1 = model1.data[label]
            data1 += [x] * n_new_data_per_class 
            data1 = np.array(data1)

            μ1 = data1.mean(axis=0)
            μ2 = model2.stats[label]['μ']
            self.assert_array_equal(μ1, μ2)

            cov1 = np.cov(data1.T)
            cov2 = model2.stats[label]['covariance']
            self.assert_array_equal(cov1, cov2)

            n1 = data1.shape[0]
            n2 = model2.stats[label]['n']
            self.assertEqual(n1, n2)
            
        # The 'params' dictionary should remain unchanged.
        self.assert_array_equal(model1.S_w, model2.S_w)
        self.assert_array_equal(model1.S_b, model2.S_b)
        self.assert_array_equal(model1.W, model2.W)
        self.assert_array_equal(model1.Λ_w, model2.Λ_w)
        self.assert_array_equal(model1.Λ_b, model2.Λ_b)
        self.assert_array_equal(model1.A, model2.A)
        self.assert_array_equal(model1.m, model2.m)
        self.assertEqual(model1.K, model2.K)
        self.assertEqual(model1.N, model2.N)
        self.assertEqual(model1.n_avg, model2.n_avg)

    def test_add_datum(self):
        model1 = PLDA.PLDA(self.data)
        model2 = PLDA.PLDA(self.data)

        x = np.array([100] * self.n_dims)
        n_new_data_per_class = 10
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.add_datum((x, label))

        for label in model2.data.keys():
            # Only the data should have been updated.
            # Statistics and parameters should not be updated.
            data1 = model1.data[label]
            data1 += [x] * n_new_data_per_class
            data2 = np.array(model2.data[label])
            self.assert_array_equal(data1, data2)

            μ1 = model1.stats[label]['μ']
            μ2 = model2.stats[label]['μ']
            self.assert_array_equal(μ1, μ2)

            cov1 = model1.stats[label]['covariance']
            cov2 = model2.stats[label]['covariance']
            self.assert_array_equal(cov1, cov2)

            n1 = model1.stats[label]['n']
            n2 = model2.stats[label]['n']
            self.assertEqual(n1, n2)

        self.assert_array_equal(model1.m, model2.m)
        self.assert_array_equal(model1.S_w, model2.S_w)
        self.assert_array_equal(model1.S_b, model2.S_b)
        self.assert_array_equal(model1.W, model2.W)
        self.assert_array_equal(model1.Λ_w, model2.Λ_w)
        self.assert_array_equal(model1.Λ_b, model2.Λ_b)
        self.assert_array_equal(model1.A, model2.A)
        self.assert_array_equal(model1.m, model2.m)
        self.assertEqual(model1.K, model2.K)
        self.assertEqual(model1.N, model2.N)
        self.assertEqual(model1.n_avg, model2.n_avg)

    def test_fit(self):
        n_new_data_per_class = 10
        x = np.array([100] * self.n_dims)
        model2 = PLDA.PLDA(self.data)

        data = self.data.copy()
        for label in model2.data.keys():
             for i in range(n_new_data_per_class):
                data += [(x, label)]

        model1 = PLDA.PLDA(data)  # model1 should have updated stats or params.

        # model2 should not update anything until fit() is run.
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.data[label].append(x)

        model2.fit()  # Now all the data, stats, and params should be updated.

        for label in model2.data.keys():
            # Only the data should have been updated.
            # Statistics and parameters should not be updated.
            data1 = np.array(model1.data[label])
            data2 = np.array(model2.data[label])
            self.assert_array_equal(data1, data2)

            μ1 = model1.stats[label]['μ']
            μ2 = model2.stats[label]['μ']
            self.assert_array_equal(μ1, μ2)

            cov1 = model1.stats[label]['covariance']
            cov2 = model2.stats[label]['covariance']
            self.assert_array_equal(cov1, cov2)

            n1 = model1.stats[label]['n']
            n2 = model2.stats[label]['n']
            self.assertEqual(n1, n2)

        self.assert_array_equal(model1.m, model2.m)
        self.assert_array_equal(model1.S_w, model2.S_w)
        self.assert_array_equal(model1.S_b, model2.S_b)
        self.assert_array_equal(model1.W, model2.W)
        self.assert_array_equal(model1.Λ_w, model2.Λ_w)
        self.assert_array_equal(model1.Λ_b, model2.Λ_b)
        self.assert_array_equal(model1.A, model2.A)
        self.assert_array_equal(model1.m, model2.m)
        self.assertEqual(model1.K, model2.K)
        self.assertEqual(model1.N, model2.N)
        self.assertEqual(model1.n_avg, model2.n_avg)

#    def test_equals(self):
#        pass
#    def test_predict_class(self):
#        pass

    def test_to_data_list(self):
        data = self.model.to_data_list()

        self.assertTrue(isinstance(data, list))

        for datum in data:
            self.assertTrue(isinstance(datum, tuple))

        model = PLDA.PLDA(data)
        for label in self.model.data.keys():
            result = np.array(model.data[label])
            expected = np.array(self.model.data[label])
            self.assert_array_equal(result, expected)

    def test_update_stats(self):
        # Make sure that the right/wrong things are updated/not updated.
        model1 = PLDA.PLDA(self.data)
        model2 = PLDA.PLDA(self.data)

        x = np.array([100] * self.n_dims)
        n_new_data_per_class = 10
        for label in model2.data.keys():
            for i in range(n_new_data_per_class):
                model2.data[label].append(x)

        for label in model2.data.keys():
            # Only the data should have been updated.
            # Statistics and parameters should not be updated.
            μ1 = model1.stats[label]['μ']
            μ2 = model2.stats[label]['μ']
            self.assert_array_equal(μ1, μ2)

            cov1 = model1.stats[label]['covariance']
            cov2 = model2.stats[label]['covariance']
            self.assert_array_equal(cov1, cov2)

            n1 = model1.stats[label]['n']
            n2 = model2.stats[label]['n']
            self.assertEqual(n1, n2)

            model2.update_stats(label)  # Now, class stats should be updated.

            data1 = model1.data[label]
            data1 += [x] * n_new_data_per_class 
            data1 = np.array(data1)

            μ1 = data1.mean(axis=0)
            μ2 = model2.stats[label]['μ']
            self.assert_array_equal(μ1, μ2)

            cov1 = np.cov(data1.T)
            cov2 = model2.stats[label]['covariance']
            self.assert_array_equal(cov1, cov2)

            n1 = data1.shape[0]
            n2 = model2.stats[label]['n']
            self.assertEqual(n1, n2)
            
        # The 'params' dictionary should remain unchanged.
        self.assert_array_equal(model1.S_w, model2.S_w)
        self.assert_array_equal(model1.S_b, model2.S_b)
        self.assert_array_equal(model1.W, model2.W)
        self.assert_array_equal(model1.Λ_w, model2.Λ_w)
        self.assert_array_equal(model1.Λ_b, model2.Λ_b)
        self.assert_array_equal(model1.A, model2.A)
        self.assert_array_equal(model1.m, model2.m)
        self.assertEqual(model1.K, model2.K)
        self.assertEqual(model1.N, model2.N)
        self.assertEqual(model1.n_avg, model2.n_avg)

    def test_whiten(self):
        U = self.X - self.model.m
        inv_A = np.linalg.inv(self.model.A)
        U = np.matmul(inv_A, U.T).T

        # u = inv(A)(x - m)
        result = self.model.whiten(self.X)
        self.assertTrue(np.array_equal(result, U))

        # x = [A]u + m
        result = np.matmul(self.model.A, result.T).T + self.model.m
        self.assert_array_equal(result, self.X)
        
    def gen_w_cov(self, n_dims):
        singular = True
        while singular:
            w_cov = np.eye(n_dims) * np.random.randint(-10, 10, n_dims)
            w_cov = np.matmul(w_cov, w_cov.T)
            if np.linalg.matrix_rank(w_cov) == n_dims:
                singular = False

        return w_cov

    def assert_diagonal(self, A, tolerance=None):
        """ Tolerance is the number of decimals to round at. """
        diagonal = A.diagonal()
        if tolerance is not None:
            self.assert_array_equal(np.around(A, tolerance),
                                    np.around(np.diag(diagonal), tolerance))
        else:
            self.assert_array_equal(A, np.diag(diagonal))

    def assert_invertible(self, A):
        determinant = det(A)
        is_invertible = determinant != 0

        self.assertTrue(is_invertible)

    def assert_array_equal(self, A, B, rtol=.00000000001):
        self.assertTrue(np.allclose(A, B, rtol))
