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
import operator
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.linalg import eigh
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp


class PLDA:
    def __init__(self, X, Y, fnames=None):
        self.relevant_dims = None
        self.params = dict()
        self.data = dict()

        self.N = None
        self.K = None
        self.n_avg = None
        self.m = None
        self.S_b = None
        self.S_w = None

        self.W = None
        self.Λ_b = None
        self.Λ_w = None

        self.A = None
        self.Ψ = None

        self.data = self.mk_data_dict(X, Y, fnames)
        self.fit()

    @staticmethod
    def mk_data_dict(X, Y, fnames=None):
        if fnames is None:
            fnames = len(Y) * [None]
        X, Y, fnames = np.asarray(X), np.asarray(Y), np.asarray(fnames)
        assert len(X.shape) == 2
        assert len(Y.shape) == 1 and len(fnames.shape) == 1
        assert Y.shape[0] == X.shape[0] == fnames.shape[0]

        unique_Y, counts = np.unique(Y, return_counts=True)
        data = dict()
        for y, count in zip(unique_Y, counts):
            data[y] = {'X': None,
                       'fnames': None,
                       'mean': None,
                       'cov': None,
                       'n': None}

            idxs = np.squeeze(np.argwhere(Y == y))
            X_y = X[idxs, :]
            assert count == len(idxs)

            data[y]['X'] = list(X_y)
            data[y]['mean'] = X_y.mean(axis=0)
            data[y]['n'] = count
            data[y]['cov'] = np.cov(X_y.T)
            data[y]['fnames'] = list(fnames[idxs])

        return data

    def calc_m(self, means, ns, N):
        means, ns = np.asarray(means), np.asarray(ns)
        weights = ns / N

        return (means * weights[:, None]).sum(axis=-2)

    def fit(self):
        means, labels1 = self.get_means(return_labels=True)
        ns, labels2 = self.get_ns(return_labels=True)
        covs, labels3 = self.get_covs(return_labels=True)

        assert labels1 == labels2 == labels3
        assert len(means) == len(ns) == len(covs)
        for mean, cov in zip(means, covs):
            assert mean.shape[0] == cov.shape[0]
            assert len(cov.shape) == 2

        self.params['K'] = len(self.data.keys())
        self.K = self.params['K']
        assert self.K == len(means)

        self.params['N'] = np.asarray(ns).sum()
        self.N = self.params['N']

        self.params['m'] = self.calc_m(means, ns, self.N)
        self.m = self.params['m']

        self.params['n_avg'] = self.N / self.K
        self.n_avg = self.params['n_avg']

        self.params['S_w'] = self.calc_S_w(np.asarray(covs), np.asarray(ns),
                                           self.N)
        self.S_w = self.params['S_w']

        self.params['S_b'] = self.calc_S_b(np.asarray(means), np.asarray(ns),
                                           self.m, self.N)
        self.S_b = self.params['S_b']

        self.params['W'] = self.calc_W(self.S_b, self.S_w)
        self.W = self.params['W']

        # Instead of rounding, you might want to do np.diag(Λ.diagonal())
        self.params['Λ_b'] = np.around(self.calc_Λ_b(self.S_b, self.W), 13)
        self.Λ_b = self.params['Λ_b']

        self.params['Λ_w'] = np.around(self.calc_Λ_w(self.S_w, self.W), 10)
        self.Λ_w = self.params['Λ_w']

        # Compute the parameters that maximize the model's likelihoods.
        self.params['A'] = self.calc_A(self.n_avg, self.Λ_w, self.W)
        self.A = self.params['A']

        self.params['Ψ'] = self.calc_Ψ(self.Λ_w, self.Λ_b, self.n_avg)
        self.Ψ = self.params['Ψ']

        self.relevant_dims = self.get_relevant_dims(self.Ψ)

    def calc_W(self, S_b, S_w):
        # W is the array of eigenvectors, where each column is a vector.
        eigenvalues, W = eigh(S_b, S_w)

        return W

    def calc_A(self, n_avg, Λ_w, W):
        A = n_avg / (n_avg - 1) * Λ_w.diagonal()
        A[np.isclose(A, 0)] = 0
        A = np.sqrt(A)
        inv_T_W = np.linalg.inv(W.T)
        A = inv_T_W * A

        # Assert that A is a square matrix.
        assert A.shape[0] == A.shape[1]
        assert len(A.shape) == 2

        # If A is not invertible, add Gaussian noise with SD 1/100 of min val.
        if np.linalg.matrix_rank(A) != A.shape[0]:
            min_val = np.min(A)
            SD = np.abs(.001 * min_val)
            A += np.random.normal(0, SD, A.shape)
            print('WARNING: the matrix A is singular. Gaussian noise, ' +
                  'with μ = 0 and SD = abs(.01 * np.min(A)) was added.' +
                  'Matrix rank was {}, and  A.shape[0] was {}'.format(
                   np.linalg.matrix_rank(A), A.shape[0]))

        return A

    def calc_S_b(self, mks, ns, m, N):
        assert ns.sum() == N

        weights = ns / N
        mk_minus_m = mks - m

        return np.matmul(mk_minus_m.T * weights, mk_minus_m)

    def calc_S_w(self, covs, ns, N):
        assert ns.sum() == N

        scaling_constants = (ns - 1) / N  # np.cov uses (n - 1) to norm.

        return (covs * scaling_constants[:, None, None]).sum(axis=0)

    def calc_Λ_b(self, S_b, W):
        return np.matmul(np.matmul(W.T, S_b), W)

    def calc_Λ_w(self, S_w, W):
        return np.matmul(np.matmul(W.T, S_w), W)

    def calc_Ψ(self, Λ_w, Λ_b, n_avg):
        weight = (n_avg - 1) / n_avg
        Λ_w = Λ_w.diagonal()
        Λ_b = Λ_b.diagonal()

        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = weight * Λ_b / Λ_w

        Ψ[np.isnan(Ψ)] = 0
        Ψ = Ψ - (1 / n_avg)

        Ψ[Ψ < 0] = 0
        Ψ[np.isinf(Ψ)] = 0
        Ψ = np.diag(Ψ)

        return Ψ

    def get_ns(self, return_labels=False):
        labels = list(self.data.keys())
        ns = []
        for label in labels:
            ns.append(self.data[label]['n'])

        if return_labels is False:
            return ns
        else:
            return ns, labels

    def get_means(self, return_labels=False):
        labels = list(self.data.keys())
        means = []
        for label in labels:
            means.append(self.data[label]['mean'])

        if return_labels is False:
            return means
        else:
            return means, labels

    def get_covs(self, return_labels=False):
        labels = list(self.data.keys())
        covs = []
        for label in labels:
            covs.append(self.data[label]['cov'])

        if return_labels is False:
            return covs
        else:
            return covs, labels

    def get_relevant_dims(self, Ψ, n=None):
        if n is None:
            relevant_dims = np.squeeze(np.argwhere(Ψ.diagonal() != 0))
        else:
            assert isinstance(n, int)
            relevant_dims = np.argsort(Ψ.diagonal())[::-1][:n]

        if relevant_dims.shape == ():
            relevant_dims = relevant_dims.reshape(1,)
        return relevant_dims

    def add_datum(self, datum, label, fname=None):
        existing_labels = list(self.data.keys())
        if label not in existing_labels:
            assert isinstance(datum,
                              type(self.data[existing_labels[0]]['X'][0]))
            self.data[label] = {'X': [datum],
                                'fnames': [fname],
                                'mean': datum,
                                'cov': None,
                                'n': 1}
        else:
            self.data[label]['X'].append(datum)
            X = np.asarray(self.data[label]['X'])
            self.data[label]['mean'] = X.mean(axis=0)
            self.data[label]['cov'] = np.cov(X.T)
            self.data[label]['n'] += 1
            self.data[label]['fnames'].append(fname)

    def calc_marginal_likelihoods(self, data, ms=None, tau_diags=None,
                                  standardize_data=None):
        assert isinstance(standardize_data, bool)
        if standardize_data is True:
            assert data.shape[-1] == len(self.Ψ.diagonal())

        # Handle edge cases in setting relevant_dims, ms, and tau_diags
        if tau_diags is None:
            relevant_dims = self.relevant_dims
            tau_diags = self.Ψ.diagonal()
        else:
            assert len(tau_diags.shape) == 2
            relevant_dims = np.arange(tau_diags.shape[-1])
            assert data.shape[-1] == len(relevant_dims)

        if ms is None:
            ms = np.zeros(tau_diags.shape[-1])
        else:
            assert len(ms.shape) == 2
            assert data.shape[-1] == ms.shape[-1]

        # Set up the dimensions to perform the vectorized operations.
        if len(ms.shape) < 2:
            ms = ms[None, relevant_dims]
        else:
            ms = ms[..., relevant_dims]

        if len(tau_diags.shape) < 2:
            tau_diags = tau_diags[None, relevant_dims]
        else:
            tau_diags = tau_diags[..., relevant_dims]

        if standardize_data is True:
            data = self.whiten(data)
        U = data[..., relevant_dims]

        # Test assumptions baked into the vectorized math operations below.
        assert U.shape[-1] == ms.shape[-1]
        assert ms.shape == tau_diags.shape
        assert 2 == len(ms.shape) == len(tau_diags.shape)

        n = U.shape[-2]
        mean_u = U.mean(axis=-2)
        squared_u = U ** 2
        log_probs = []
        for m, tau in zip(ms, tau_diags):
            log_constants = -.5 * n * np.log(2 * np.pi) \
                            - .5 * np.log(n * tau + 1)
            exponent_1s = -.5 * (squared_u.sum(axis=-2) + (m ** 2 / tau))

            exponent_2s = ((n ** 2) * tau * (mean_u ** 2)) + \
                          (m ** 2 / tau) + (2 * n * mean_u * m)
            exponent_2s /= 2 * (n * tau + 1)

            log_probs.append((log_constants +
                              exponent_1s +
                              exponent_2s).sum(axis=-1))

        return np.squeeze(np.stack(log_probs, axis=-1))

    def calc_posteriors(self, return_covs_as_diags=True, dims=None,
                        return_labels=False):
        assert type(dims) is np.ndarray or dims is None
        if dims is None:
            relevant_dims = np.arange(self.Ψ.diagonal().shape[0])
        else:
            relevant_dims = dims

        Ψ = self.Ψ.diagonal()[relevant_dims]
        u_bars = self.whiten(np.asarray(self.get_means()))
        u_bars = u_bars[:, relevant_dims]
        ns, labels = self.get_ns(return_labels=True)
        ns = np.asarray(ns)
        sums = (u_bars.T * ns).T
        cov_diags = []
        means = []

        for n, u_bar, sum_u in zip(ns, u_bars, sums):
            cov_diag = Ψ / (1 + n * Ψ)
            mean = sum_u * cov_diag
            cov_diags.append(cov_diag)
            means.append(mean)

        if not return_covs_as_diags:
            cov_diags = [np.diag(row) for row in cov_diags]

        if return_labels is True:
            return np.asarray(means), np.asarray(cov_diags), labels

        elif return_labels is False:
            return np.asarray(means), np.asarray(cov_diags)

    def calc_posterior_predictives(self, data, standardize_data,
                                   return_labels=False):
        assert isinstance(return_labels, bool)
        assert isinstance(standardize_data, bool)
        assert data.shape[-2] == 1

        relevant_dims = self.relevant_dims
        if standardize_data is True:
            data = self.whiten(data)

        data = data[..., None, relevant_dims]
        means, covs, labels = self.calc_posteriors(return_labels=True)
        means = means[:, relevant_dims]
        covs = covs[:, relevant_dims]

        if len(covs.shape) < 2:
            means = means[:, None]
            covs = covs[:, None]
            data = data[:, None]

        probs = self.calc_marginal_likelihoods(data, means, covs,
                                               standardize_data=False)
        if return_labels is True:
            return probs, labels
        else:
            return probs

    def whiten(self, X):
        assert X.shape[-1] == self.m.shape[0] == self.A.shape[0]

        shape = X.shape
        X = X.reshape(np.prod(shape[:-1]).astype(int), shape[-1])
        inv_A = np.linalg.inv(self.A)
        U = X - self.m
        U = np.matmul(U, inv_A.T)

        return U.reshape(shape)

    mk_data_dict.__doc__ = """
        Makes a dictionary, whose keys index class data and statistics.

        DESCRIPTION: This data structure is a dictionary of dictionaries. The
                      outer dictionary stores a dictionary for each unique
                      label (i.e. data class) in Y. The inner dictionaries
                      hold the data, file fnames, means, covariances, and
                      sample sizes for the label/data class they represent.
        """

    calc_m.__doc__ = """
        Returns the mean of the unwhitened (non-latent space) dataset.

        ARGUMENTS
         means  (ndarray), shape=(n_unique_labels, n_data_dims)
           Row-wise means of the classes in the training data.

         ns  (ndarray), shape=(n_unique_labels,)
           Sample sizes for each of the classes in the training data.

        RETURN
         m  (ndarray), shape=(n_unique_labels,)
           The vector centering the data, i.e. the mean of the training data.

        """

    fit.__doc__ = """
        Fits the plda model parameters to the data.

        DESCRIPTION: The optimization procedure follows Ioffe, 2006.
                      See p. 537, Fig. 2.
        """

    calc_W.__doc__ = """
        Computes W by solving the generalized eigenvalue problem on p. 537.

        EQUATION:
         Vector form
          \mathbf{S_b}\mathbf{w} = \lambda\mathbf{S_w}\mathbf{w}
         Matrix form
          \mathbf{S_b}\mathbf{W} = \mathbf{\lambda}\mathbf{S_w}\mathbf{W}

         Solving for W:
         (\mathbf{S_b} - \mathbf{S_w})\mathbf{W} = \mathbf{\Lambda}

        DESCRIPTION: Relies on eigh instead of eig from scipy.linalg. eigh is
                      significantly faster and only requres that the input
                      matrices be symmetric, which S_b & S_w are.
        ARGUMENT
         S_b  (ndarray), shape=(n_data_dims, n_data_dims)
           Between-class scatter matrix. [n_dims x n_dims]

         S_w  (ndarray), shape=(n_data_dims, n_data_dims)
           Within-class scatter matrix. [n_dims x n_dims]

        RETURNS
         W  (ndarray), shape=(n_data_dims, n_data_dims)
           Solution to the generalized eigenvalue problem above, where the
           columns are eigenvectors. [n_dims x n_dims]

        """

    calc_A.__doc__ = """
        Computes the matrix A, which is used to compute u: x = m + Au.

        DESCRIPTION: Note that the average class sample size is used here,
                      not the total or individual class sample sizes. See
                      p. 536 for more information on alternative ways of
                      dealing with unequal class sample sizes and p. 537
                      for equations.
        EQUATION:
         \mathbf{A} = \mathbf{W}^{-\top}
                      (\frac{\bar n}{\bar n - 1}
                       \mathbf{\Lambda_w})^{1/2}
        ARGUMENTS
         n_avg  (float)
           Mean sample size of the classes, i.e. avg. number of training data
           per class.

         Λ_w  (ndarray), shape=(n_data_dims, n_data_dims)
           Diagonalized S_w.

        RETURNS
         A  (ndarray), shape=(n_data_dims, n_data_dims)
           Matrix that transforms the latent space to the data space.

        """

    calc_S_b.__doc__ = """
        Computes the between-scatter matrix. See p.532, EQ1 in Ioffe, 2006.

        ARGUMENTS
         mks  (ndarray), shape=(n_unique_labels, n_data_dims)
           Means of the classes of data.

         ns  (ndarray), shape=(n_unique_labels,)
           Sample sizes for each class of data (i.e. for each unique Y label).

         m  (ndarray), shape=(n_data_dims,)
           Mean of the training data.

         N  (int)
           Total number of training data.

        RETURN
         S_b  (ndarray), shape=(n_data_dims, n_data_dims)
             Between-scatter matrix.
        """

    calc_S_w.__doc__ = """
        Computes the within-scatter matrix. See p.532, EQ 1 in Ioffe, 2006.

        ARGUMENTS
         covs  (ndarray), shape=(n_unique_labels, n_data_dims, n_data_dims)
           Covariance matrix for each class of data (i.e. each unique Y label).

         ns  (ndarray), shape=(n_unique_labels,)
           Sample sizes for each class of data (i.e. for each unique Y label).

         N  (int)
           Total number of training data.

        RETURNS
         S_w  (ndarray), shape=(n_data_dims, n_data_dims)
           Within-scatter matrix.

        """

    calc_Λ_b.__doc__ = """
        Diagonalized S_b - for maximizing the likelihood of the PLDA model.

        DESCRIPTION: See p. 537 to see how Λ_b is used to compute the
                      parameters that maximize the PLDA model's likelihood.
        EQUATION
         \mathbf{\Lambda_b} = \mathbf{W}^{\top}
                              \mathbf{S_w}
                              \mathbf{W}
        ARGUMENTS
         W  (ndarray), shape=(n_data_dims, n_data_dims)
           Eigenvectors solving the generalized eigenvalue problem on p. 537.

         S_b  (ndarray), shape=(n_data_dims, n_data_dims)
           The between-class scatter matrix. See p 532.

        RETURNS
         Λ_b  (ndarray): S_b diagonalized by W. [n_dims x n_dims]

        """

    calc_Λ_w.__doc__ = """
        Diagonalized S_w - for maximizing the likelihood of the PLDA model.

        DESCRIPTION: See p. 537 to see how Λ_w is used to compute the
                      parameters that maximize the PLDA model's likelihood.
        EQUATION
         \mathbf{\Lambda_b} = \mathbf{W}^{\top}
                              \mathbf{S_w}
                              \mathbf{W}
        ARGUMENTS
         W  (ndarray), shape=(n_data_dims, n_data_dims)
           Eigenvectors solving the generalized eigenvalue problem on p. 537.

         S_w  (ndarray), shape=(n_data_dims, n_data_dims)
           Within-class scatter matrix.

        RETURNS
         Λ_w  (ndarray), shape=(n_data_dims, n_data_dims)
           S_w diagonalized by W.

        """

    calc_Ψ.__doc__ = """
        Calculates the covariance of the 'whitened' cluster means.

        EQUATION:
         \mathbf{\Psi} = \max(0, \frac{n - 1}{n}
                                 \Big(\frac{\mathbf{\Lambda_b}}
                                           {\mathbf{\Lambda_w}}\Big)
                               - \frac{1}{n})

        DESCRIPTION: Following Ioffe, 2006, I use n_{avg} in place of 'n'
                      for the optimization procedure. Remember, that this is
                      just one way to deal with unequal sample sizes.
                      See p. 537, Fig 2 in Ioffe, 2006 for more details.
        ARGUMENTS
         Λ_w  (ndarray), shape=(n_data_dims, n_data_Dims)
          Should be a diagonal martix that is obtained from diagonalizing
          S_w, using W. See p. 537, Fig. 2 in Ioffe, 2006.

         Λ_b  (ndarray), shape=(n_data_dims, n_data_dims)
          Should be a diagonal martix that is obtained from diagonalizing
          S_b, using W. See p. 537, Fig. 2 in Ioffe, 2006.

         n_avg  (float)
          Average sample size of the data classes.

        RETURNS
         Ψ  (ndarray), shape=(n_data_dims, n_data_dims)
          Latent space covariance of the cluster centers.
        """

    get_ns.__doc__ = """
        Returns the number of training data in each class.

        ARGUMENT
         return_labels  (bool)
           Determines whether or not to return the labels. If they are
           returned, they are returned in the same order as the sample sizes.

        RETURNS
         ns  (list)
           A list of the sample sizes in each class of the training data.

         labels  (list)
           A list of the labels of the returned sample sizes, sorted
           in the same order.

        """

    get_means.__doc__ = """
        Returns the means of the training data in each class.

        ARGUMENT
         return_labels  (bool)
           Determines whether or not to return the labels. If they are
           returned, they are returned in the same order as the means.

        RETURNS
         means  (list)
           A list of the means for each class in the training data. The means
           are 1D np.ndarrays with shape (n_data_dims,).

         labels  (list)
           A list of the labels of the returned sample sizes, sorted
           in the same order.

        """

    get_covs.__doc__ = """
        Returns the covariances of the training data in each class.

        ARGUMENT
         return_labels  (bool)
           Determines whether or not to return the labels. If they are
           returned, they are returned in the same order as the covariances.

        RETURNS
         covs  (list)
           A list of the covariances for each class in the training data. The
           means are 2D np.ndarrays with shape (n_data_dims, n_data_dims).

         labels  (list)
           A list of the labels of the returned sample sizes, sorted
           in the same order.

        """

    get_relevant_dims.__doc__ = """
        Returns the indices of the largest elements on a matrix diagonal.

        ARGUMENT
         Ψ  (ndarray), shape=(n_dims, n_dims)
           A square matrix.

         n  (int)
           If set to None, the default setting is to return indices for ALL
           non-zero elements on the main diagonal.

        RETURN
         relevant_dims  (ndarray), shape=(number of largest diagonal elements,)
           Indices of the largest elements on Ψ's diagonal. The number of
           elements to return is determined, by the argument 'n'.
        """

    add_datum.__doc__ = """
        Adds a new datum to the dataset, but does NOT run fit()!

        ARGUMENTS
         datum  (ndarray), shape=(n_data_dims,)
           Must be the same the dimension as the training data.

         label  (string)
           This will be used as a dictionary key, so it MUST be hashable.

        PARAMETERS
         data   (dict)
           Dictionary of dictionaries that store data for a particular class.

        RETURNS
         None
        """

    calc_marginal_likelihoods.__doc__ = """
        Computes the marginal likelihood of data.

        NOTE: If you specifs ms or tau_diags, only the first len()th dims
              as used for computing the probabilities.

        DESCRIPTION: EQ 6 on p.535 is incorrect. See notes or Kevin Murphy's
                      cheat sheet on conjugate analysis of the Gaussian.
        ARGUMENTS
         data  (ndarray), shape=(..., n_data_to_compute_prob_for, n_data_dims)
           axis -1: dimension of data. If you are setting ms or tau_diags,
                     length of this axis should be same as ms and tau_diags.
           axis -2: sets of data to compute marginal likelihoods for

         ms  (ndarray), shape=(n_unique_labels, n_data_dims)
           Means of the data classes. len(ms) == 2 must be True.

         tau_diags  (ndarray), shape=(n_unique_labels, n_data_dims)
           The diagonals of the covariances of the data classes. These
           are assumed to be coming from diagonal matrices, i.e. covariance
           matrix with only diagonal entries, which indicates the dimensions
           are linearly independent. len(tau_diags) == 2 must be True.

         standardize_data  (bool)
           Must be set to true or false to indicate whether or not to transform
           data to latent space. If data is already in latent space, set this
           to false. If not, set it to True.

        RETURNS
         log_probs  (ndarray), shape=(...)

             returns an ndarray, whose last dimension corresponds to the data
             class in seld.stats.keys().
            etc.
        """

    calc_posteriors.__doc__ = """
        Returns the posterior means and covariances of the data class means.

        DESCRIPTION: The model is a conjugate multivariate Normal-Normal model.
                      The likelihood is Gaussian, N(u|v, I), the prior on the
                      mean is Gaussian (v|0, Ψ), and covariance is known to be
                      the identity matrix.
        EQUATION
         p(v^k \mid u_1^k, u_2^k, ...u_n^k) =
           N(v \mid \frac{\Psi}{n \Psi + I} \sum_i^n u_i^k,
                    \frac{\Psi}{n \Psi + I})
         where k indexes a particular class and n is the total number of data
         in that class, k.

        ARGUMENTS
         return_covs_as_diags  (bool)
           Whether to return the covariances of the posterior distributions
           on the means of the data classes as matrices, or just vectors of
           the diagonals.

         dims  (n_data_dims)
           Because the covariances are diagonal matrices, this tells us that
           the Gaussians are independent between all the dimensions. This
           argument can be set to compute the posterior for a subset of the
           dimensions. If set to None, the function computes the posterior
           for all dimensions along which the prior covariance is 0.

         return_labels  (bool)
           Whether to return the labels of the rows in means and cov_diags.

       RETURNS
        means  (ndarray), shape=(n_unique_labels, n_data_dims)
          The means of the posterior distributions on the means of the data
          classes. These are sorted in the same order as cov_diags.

        cov_diags  (ndarray), shape=(n_unique_labels, n_data_dims)
          The diagonals of the covariance matrices for the posterior
          distributions on the means of the data classes. Recall that
          these will be diagonal matrices.

        labels  (list)
          Class labels (i.e. y) of the rows in means and cov_diags. This
          is an optional return, specified by the return_labels argument.
        """

#    calc_posterior_predictives.__doc__ =

    whiten.__doc__ = """
        Standardizes the data, X. See p. 534, section 3.1 of Ioffe, 2006.

        EQUATIONS:
         Vector form
           \mathbf{A}^{-1}(\mathbf{x} - \mathbf{m})

         Matrix form
           \mathbf{A}^{-1}(\mathbf{X} - \mathbf{m}\mathbf{1}^{\top})

        ARGUMENT
         X  (ndarray), shape=(..., n_data_dims)
           Rows of unwhitened/unstandardized data.

        PARAMETERS
         A  (ndarray), shape=(n_data_dims)
           Transforms latent space to the space of the training data.

         m  (ndarray), shape=(n_data_dims,)
           Vector that centers the data (i.e. mean of the training data).

        RETURN
         U  (ndarray), shape=(..., n_data_dims)
           Rows of whitened/standardized data.
        """
