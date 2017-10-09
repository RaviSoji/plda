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

# Docstrings include page numbers that correspond to the published paper. 
# To make it easier to understand the code, mathematical notation
# (including Greek symbols) is written to be with consistent with the paper.
class PLDA:
    def __init__(self, data, save_raw=False):
        """ Data should be a list of tuples of (example, label), where
            example is an [n_dims x 1] ndarray. label can be any data type"""
        self.save_raw = save_raw
        if self.save_raw is True:
            self.raw_data = data

        # Data structure holding data for each class.
        self.data = self.data_list_to_data_dict(data)

        # Data structure holding statistics for each class.
        self.stats = dict()

        # Intermediate parameters used to compute model parameters.
        self.K = None         # (int) Number of classes in the dataset.
        self.m = None         # (ndarray) Mean of the dataset.
        self.N = None         # (int) Number of examples/data.
        self.n_avg = None     # (float) See last sentence on p. 536.
        self.S_b = None       # (ndarray) Between class scatter. [n_dims x n_dims]
        self.S_w = None       # (ndarray) Within class scatter. [n_dims x n_dims]
        self.Λ_b = None       # (ndarray) Diagonalized S_b. [n_dims x n_dims]
        self.Λ_w = None       # (ndarray) Diagonalized S_w. [n_dims x n_dims]

        # Parameters used to generate pdfs for likelihood maximization.
        self.W = None         # (ndarray) [n_dims x n_dims]
        self.A = None         # (ndarray) [n_dims x n_dims]
        self.Ψ = None         # (ndarray) [n_dims x n_dims]

        # Data structure holding all parameters.
        self.params = self.get_params_data_structure()

        # Data structure holding the PDFs necessary for classification.
        self.pdfs = dict()

        self.fit()

    @staticmethod
    def data_list_to_data_dict(data):
        """ Converts data into dictionary format, where keys are class labels.

        ARGUMENTS
         data      (list): List of data as tuples of the form (example, label).

        PARAMETERS
         None

        RETURNS
         data_dict (dict): Each label indexes a list of ndarrays (examples)
                            that belong to the same class.
        """
        assert isinstance(data, list)
        if len(data) != 0:
            assert isinstance(data[0], tuple)

        data.sort(key=operator.itemgetter(1))
        data_dict = dict()

        prev_label = None
        for example, label in data:
            if prev_label != label:
                data_dict.update({label: [example]})
                prev_label = label
            else:
                data_dict[label].append(example)

        return data_dict

    def get_params_data_structure(self):
        """ Creates a data structure that will hold model parameters.

        ARGUMENTS
         None

        PARAMETERS
         None

        RETURNS
         structure (dict): Dictionary with the keys (model parameters) indexing
                            None.
        """
        structure = dict()
        structure.update({'K': None})
        structure.update({'m': None})
        structure.update({'N': None})
        structure.update({'n_avg':None})
        structure.update({'Λ_b':None})
        structure.update({'Λ_w':None})
        structure.update({'Ψ':None})
        structure.update({'S_b': None})
        structure.update({'S_w': None})
        structure.update({'W': None})

        for label in self.data.keys():
            μ_k_key = 'v_' + str(label)
            structure.update({μ_k_key: None})

        return structure

    def calc_A(self):
        """ Computes the matrix A, which is used to compute u: x = m + Au.

        EQUATION: A = W^{-T} (n_avg/(n_avg - 1)[Λ_w]) ** .5

        DESCRIPTION: Note that the average class sample size is used here, 
                      not the total or individual class sample sizes. See
                      p. 536 for more information on alternative ways of
                      dealing with unequal class sample sizes and p. 537
                      for equations.
        ARGUMENTS
         None

        PARAMETERS
         n_avg      (float): Mean sample size of the classes.
         Λ_w      (ndarray): Diagonalized S_w (i.e. [W^T][S_w][W]).
                              [n_dims x n_dims]
        RETURNS
         A        (ndarray): The (un)whitening matrix. [n_dims x n_dims]

        """
        A = self.n_avg / (self.n_avg - 1) * self.Λ_w
        A[np.isclose(A, 0)] = 0
        A = np.sqrt(A)
        inv_T_W = np.linalg.inv(self.W.T)
        A = np.matmul(inv_T_W, A) 

        assert A.shape[0] == A.shape[1]
        assert len(A.shape) == 2

        # If A is not invertible, add Gaussian noise with SD 1/100 of min val.
        if np.linalg.matrix_rank(A) != A.shape[0]:
            min_val = np.min(A)
            SD = np.abs(.001 * min_val)
            A += np.random.normal(0, SD, A.shape)
            print('WARNING: the matrix A is singular. Gaussian noise, ' +
                  'with μ = 0 and SD = abs(.001 * np.min(A)) was added.' +
                  'Matrix rank was {}, and  A.shape[0] was {}'.format(
                   np.linalg.matrix_rank(A), A.shape[0]))

        return A

    def calc_class_log_probs(self, data):
        """ Computes the probability of each datum being in each class.

        DESCRIPTION: Probabilities are the likelihoods of the data being
                      generated by the class pdfs (multivariate Gaussians).
        ARGUMENTS
         data             (ndarray): Data for which probabilities are being
                                      computed. [n x n_dims]
        PARAMETERS
         pdfs                (dict): Holds the multivariate Gaussian pdfs for
                                      each class, where keys are the labels.
          -- pdfs[label]   (method): .logpdf method of a multivariate Gaussian,
                                      'scipy.stats._multivariate_normal_frozen'.
                                      Passing in data returns the probabilities
                                      of that pdf generating that data.
        RETURNS
         log_probs        (ndarray): Likelihood probabilities of the data 
                                      being generated by each class pdf.
                                      [n x n_dims]
        """
        data = self.whiten(data)
        log_probs = []
        for label in self.stats.keys():
            pdf = self.pdfs[label]
            log_probs.append(pdf(data))
        
        log_probs = np.array(log_probs).T

        return log_probs

    def calc_marginal_likelihoods(self, X=None, return_log=True):

        """ EQ 6 in the paper is incorrect. Refer to Kevin Murphy's cheatsheet.
        ARGUMENTS
         X        (ndarray): NON-whitened data. [n x n_dims]
         return_log  (bool): Whether to return normal or log probabilities.

        RETURNS
         log_probs          (ndarray): If return_log is True. [n_classes x 1]
         np.exp(log_probs)  (ndarray): If return_log is False. [n_classes x 1]
        """
        assert isinstance(return_log, bool)

        log_probs = []
        for label in self.stats.keys():
            if X is None:
                data = np.asarray(self.data[label])
            else:
                assert isinstance(X, list)
                data = self.data[label]
                for datum in X:
                    assert datum.shape == data[-1].shape
                    data.append(datum)
                data = np.asarray(data)
            data = self.whiten(data)
               
            n = data.shape[0]
            m = 0
            prior = self.Ψ.diagonal()
            prior
            mean = data.mean(axis=0)

            log_prob = -.5 * n * np.log(2 * np.pi)  # OK
            log_prob += -.5 * np.log(n * prior + 1)  # OK
            log_prob += -.5 * inner1d(data.T, data.T)  # OK
            log_prob += (mean ** 2) * (n ** 2) * prior / (2 * (n * prior + 1))  # OK

            log_probs.append(log_prob.sum())

        if return_log is True:
            return log_probs
        else:
            return np.exp(log_probs)
            
            
    def calc_K(self):
        """ Calculates the number of classes in the labeled data. """

        return len(self.data.keys())


    def calc_Λ_b(self):
        """ Diagonalized S_b - for maximizing the likelihood of the PLDA model.

        EQUATION: Λ_b = [W^T][S_b][W]

        DESCRIPTION: See p. 537 to see how Λ_b is used to compute the 
                      parameters that maximize the PLDA model's likelihood.
        ARGUMENTS
         None

        PARAMETERS
         W    (ndarray): Eigenvectors that solve the generalized eigenvalue
                          problem on p. 537. [n_dims x n_dims]
         S_b  (ndarray): The between-class scatter matrix. See p 532.
                          [n_dims x n_dims]
        RETURNS
         Λ_b  (ndarray): S_b diagonalized by W. [n_dims x n_dims]

        """
        W = self.W
        S_b = self.S_b
        Λ_b = np.matmul(np.matmul(W.T, S_b), W)

        return np.around(Λ_b, 13)

    def calc_Λ_w(self):
        """ Diagonalized S_w - for maximizing the likelihood of the PLDA model.
 
        EQUATION: Λ_w = [W^T][S_w][W]

        DESCRIPTION: See p. 537 to see how Λ_b is used to compute the
                      parameters that maximize the PLDA model's likelihood.
        ARGUMENTS
         None

        PARAMETERS
         W        (ndarray): Eigenvectors that solve the generalized eigenvalue
                              problem on p. 537. [n_dims x n_dims]
         S_w      (ndarray): Within-class scatter matrix. [n_dims x n_dims]

        RETURNS
         Λ_w      (ndarray): S_w, the within-class scatter matrix, diagonalized
                              by W. [n_dims x n_dims]
        """
        W = self.W
        S_w = self.S_w
        Λ_w = np.matmul(np.matmul(W.T, S_w), W)

        return np.around(Λ_w, 13)

    def calc_m(self):
        """ Computes the mean of all the examples in the dataset.

        DESCRIPTION: For efficient computation, we can take the weighted
                      average of the class means.
        ARGUMENTS
         None

        PARAMETERS
         N  (numpy.int64): Total sample size of the entire dataset.
         stats     (dict): Stores the means, covariances, and sample sizes
                            for all data classes. Keys are the class labels.
          -- stats[label]         (dict): Stores the mean, covariance, and
                                           sample size for the class 'label'.
          -- stats[label]['n']     (int): Sample size for the class 'label'.
          -- stats[label]['μ'] (ndarray): Mean for the class 'label'.
                                           [n_dims x 1]
        
        RETURNS
         m  (ndarray): Computes the mean of the data/examples. [n_dims x 1]

        """
        μs = []
        N = self.N
        for label in self.stats.keys():
            weight = self.stats[label]['n'] / N
            μs.append(weight * self.stats[label]['μ'])
        m = np.array(μs).sum(axis=0)

        return m

    def calc_N(self):
        """ Computes the sample size of the entire dataset.

        ARGUMENTS
         None

        PARAMETERS
         stats     (dict): Stores the means, covariances, and sample sizes
                            for all data classes. Keys are the class labels.
          -- stats[label]      (dict): Stores the mean, covariance, and sample
                                        size for the class 'label'.
          -- stats[label]['n']  (int): Sample size for the class 'label'.
        RETURNS
         N  (numpy.int64): Total sample size (i.e. total number of data).

        """
        n_list = []
        for label in self.stats.keys():
            n_list.append(self.stats[label]['n'])

        N = np.array(n_list).sum()

        return N

    def calc_n_avg(self):
        """ Computes the average class sample size. """
        
        return float(self.N) / float(self.K)

    def calc_Ψ(self):
        """ Calculates the covariance of the whitened cluster centers.

        EQUATION: Ψ = max(0, [(n-1) / n * [Λ_b]/[Λ_w]] - 1/n)
                  Also, note that Φ_b = [A][Ψ][A^T]

        DESCRIPTION: Remember, that for this algorithm, one way to deal with
                      unequal numbers of examples between the classes is to
                      use the average sample size in place of 'n'.
        ARGUMENTS
         None

        PARAMETERS
         n_avg    (float): Average sample size of the data classes.
         Λ_w    (ndarray): Within-class scatter matrix diagonalized by the
                              matrix W. W is the set of eigenvectors that
                              solve the generalized eigenvalue problem on
                              p. 537.
         Λ_b    (ndarray): Between-class scatter matrix diagonalized by the
                              matrix W. W is the set of eigenvectors that
                              solve the generalized eigenvalue problem on
                              p. 537.
        RETURNS
         Ψ      (ndarray): Covariance of the cluster centers in whitened
                              data space. [n_dims x n_dims]
        """
        weight = (self.n_avg - 1) / self.n_avg
        Λ_w = self.Λ_w.diagonal().copy()
        Λ_b = self.Λ_b.diagonal().copy()
  
        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = weight * Λ_b / Λ_w

        Ψ[np.isnan(Ψ)] = 0
        Ψ = Ψ - (1 / self.n_avg)

        Ψ[Ψ < 0] = 0
        Ψ[np.isinf(Ψ)] = 0
        Ψ = np.diag(Ψ)

        return Ψ

    def calc_S_b(self):
        """ Computes the "between-class" scatter matrix.

        DESCRIPTION: n used here is NOT n_avg. n_avg is used only for
                     computing (1) Λ_w, (2) Λ_b, (3) Ψ, (4) and A.

        EQUATION: S_w = Σ_k{
                          (n_k)[(m_k - m)(m_k - m)^T]
                        } / N
                  See p. 532, Equations (1).
        ARGUMENTS
         None

        PARAMETERS
         N  (numpy.int64): Total sample size (i.e. number of data).
         stats     (dict): Stores the mean, sample size, and covariance for
                            each class in a dictionary.
          -- stats[label]             (dict): Stores the mean, sample size,
                                               and covariance for the class
                                               'label'.
          -- stats[label]['μ']  (ndarray): Mean for the class 'label'.
                                               [n_dims x 1]
          -- stats[label]['n']         (int): Sample size for the class 'label'.

        RETURNS
         S_b  (ndarray): Between-class scatter matrix. [n_dims x n_dims]

        """
        n_list = np.array(self.get_sample_sizes())
         # get_sample_sizes() depends on self.stats
        weights = n_list / self.N

        μs = self.get_μs()
         # get_μs() depends on self.stats
        mk_minus_m = np.array(μs) - self.m
        S_b = np.matmul(mk_minus_m.T * weights, mk_minus_m)

        return S_b

    def calc_S_w(self):
        """ Computes the "within-class" scatter matrix.

        DESCRIPTION: Class statistics were computed using numpy functions,
                      where covariance is normalized with (n - 1), so
                       cov_k * (n-1) = Σ_{i_in_C_k}{(x^i - m_k)(x^i - m_k)^T}.

        EQUATION: S_w = Σ_k{ 
                          Σ_{i_in_C_k}{ 
                            [(x^i - m_k)(x^i - m_k)^T] 
                          }
                        } / N
                  See p. 532, Equations (1).
        ARGUMENTS
         None.

        PARAMETERS
         N  (numpy.int64): Total sample size (i.e. number of data).
         stats     (dict): Stores the mean, sample size, and covariance for
                            each class in a dictionary.
          -- stats[label]                   (dict): Stores the mean, sample
                                                     size, and covariance
                                                     for the class 'label'.
          -- stats[label]['covariance']  (ndarray): The covariance matrix for
                                                     the class 'label.
                                                     [n_dims x n_dims]
          -- stats[label]['n']               (int): Sample size for the class.
        RETURNS
         S_w  (ndarray): Within-class scatter matrix. [n_dims x n_dims]

        """
        unnormed_covariances = []
        #n_list = self.get_sample_sizes()
        for label in self.stats.keys():
            cov_n_minus_1_norm = self.stats[label]['covariance']
            n = self.stats[label]['n']
            cov_unnormed = cov_n_minus_1_norm * (n - 1)
            unnormed_covariances.append(cov_unnormed)

        unnormed_covariances = np.array(unnormed_covariances)
        S_w = unnormed_covariances.sum(axis=0) / self.N

        return S_w

    def calc_W(self):
        """ Computes W by solving the generalized eigenvalue problem on p. 537.

        EQUATION: [S_b][W] = [Λ][S_w][W]; [S_b - S_w][W] = Λ

        DESCRIPTION: Relies on eigh instead of eig from scipy.linalg. eigh is
                      significantly faster and only requres that the input
                      matrices be symmetric, which S_b & S_w are.
        ARGUMENTS
         None

        PARAMETERS
         S_b  (ndarray): Between-class scatter matrix. [n_dims x n_dims]
         S_w  (ndarray): Within-class scatter matrix. [n_dims x n_dims]

        RETURNS
         W    (ndarray): Solution to the generalized eigenvalue problem above,
                          where the Columns are eigenvectors. [n_dims x n_dims]
        """
        eigenvalues, W = eigh(self.S_b, self.S_w)
         # W is the array of eigenvectors, where each column is a vector.

        return W

    def get_covariances(self):
        """ Extracts all covariances from the self.stats data structure.

        ARGUMENTS
         None

        PARAMETERS
         stats
          -- stats.keys()              (dict_keys): The keys are the data class
                                                     labels.
          -- stats[label]                   (dict): Dictionary storing the
                                                     class mean, sample size,
                                                     and covariance.
          -- stats[label]['covariance']  (ndarray): Covariance for the class
                                                     'label'. [n_dims x n_dims]
        RETURNS
         covariances (list): Covariances for the data classes, returned in the
                              same order as self.stats.keys().
        """
        assert isinstance(self.stats, dict)

        covariances = []
        for label in self.stats.keys():
            covariance = self.stats[label]['covariance']
            covariances.append(covariance)

        return covariances

    def get_μs(self):
        """ Extracts all means from the self.stats data structure.

        ARGUMENTS
         None

        PARAMETERS
         stats
          -- stats.keys()      (dict_keys): The keys are the data class labels.
          -- stats[label]           (dict): Dictionary storing the class
                                             mean, sample size, and covariance.
          -- stats[label]['μ']     (float): Mean for the class 'label'.

        RETURNS
         μs                         (list): Means for the data classes, 
                                             returned in the same order
                                             as self.stats.keys().
        """
        assert isinstance(self.stats, dict)

        μs = []
        for label in self.stats.keys():
            μ = self.stats[label]['μ']
            μs.append(μ)

        return μs

    def get_sample_sizes(self):
        """ Extracts the sample sizes from the self.stats data structure.

        ARGUMENTS
         None

        PARAMETERS
         stats
          -- stats.keys()  (dict_keys): The keys are the data class labels.
          -- stats[label]       (dict): Dictionary storing the class
                                         mean, sample size, and covariance.
          -- stats[label]['n']   (int): Sample size for the class 'label'.

        RETURNS
         sample_sizes (list): Sample sizes for the data classes, returned
                               in the same order as self.stats.keys().
        """
        assert isinstance(self.stats, dict)

        sample_sizes = []
        for label in self.stats.keys():
            sample_size = self.stats[label]['n']
            sample_sizes.append(sample_size)

        return sample_sizes

    def get_stats_data_structure(self):
        """ Generates a data structure to hold statistics for each data class.

        ARGUMENTS
         None

        PARAMETERS
         None

        RETURNS
         structure (dict): Has 3 keys, 'μ', 'n', and 'covariance', each of
                            which point to 'None'.
        """
        structure = dict()
        structure.update({'μ': None})
        structure.update({'n': None})
        structure.update({'covariance': None})

        return structure

    def set_params(self):
        """ Computes and sets model parameters necessary for prediction.

        DESCRIPTION: See paper, individual function documentation, and
                      test code documentation.
        ARGUMENTS
         None

        PARAMETERS
         params       (dict): Stores model parameters in an easily accessible
                               dictionary format.
         data         (dict): Store a list of data ([n_dims x 1] ndarrays)
                               for each class, which can be accessed using
                               the class labels as keys.
         stats        (dict): Stores the mean, covariance, and sample_ size
                               for each class. For each class, the dictionary
                               storing these for each class can be indexed
                               using the label as they key.
         
         K             (int): Total number of labelled data classes.
         N             (int): Total number of data/examples.
         m         (ndarray): Mean of the data in unwhitened data space.
         n_avg       (float): Average number of labelled data per class.
         S_w       (ndarray): Within-class scatter matrix. [n_dims x n_dims]
         S_b       (ndarray): Between-class scatter matrix. [n_dims x n_dims]
         W         (ndarray): Solution to the eigenvalue problem on p. 537.
                               [n_dims x n_dims]
         Λ_b       (ndarray): Diagonalized S_b used - to compute the parameters
                              that maximize the model's likelihoods.
                               [n_dims x n_dims]
         Λ_w       (ndarray): Diagonalized S_w - used to compute the parameters
                              that maximize the model's likelihoods.
                               [n_dims x n_dims]
         A         (ndarray): The matrix that whitens and unwhitens the data.
                               [n_dims x n_dims]
        RETURNS
         None

        """
        # Calculate statistics for computing model parameters.
        self.params['K'] = self.calc_K()
        self.K = self.params['K']

        self.params['N'] = self.calc_N()
        self.N = self.params['N']

        self.params['m'] = self.calc_m()
        self.m = self.params['m']

        self.params['n_avg'] = self.calc_n_avg()
        self.n_avg = self.params['n_avg']

        self.params['S_w'] = self.calc_S_w()
        self.S_w = self.params['S_w']

        self.params['S_b'] = self.calc_S_b()
        self.S_b = self.params['S_b']

        # Calculate intermediate parameters.
        self.params['W'] = self.calc_W()
        self.W = self.params['W']

        self.params['Λ_b'] = np.around(self.calc_Λ_b(), 13)
        self.Λ_b = self.params['Λ_b']

        self.params['Λ_w'] = np.around(self.calc_Λ_w(), 10)
        self.Λ_w = self.params['Λ_w']
        
        # Compute the parameters that maximum the model's likelihoods.
        self.params['A'] = self.calc_A()
        self.A = self.params['A']

        self.params['Ψ'] = self.calc_Ψ()
        self.Ψ = self.params['Ψ']


        for label in self.stats.keys():
            μ = self.stats[label]['μ']
            v = self.whiten(μ)
            self.params['v_' + str(label)] = v

    def set_pdfs(self):
        """ Stores the log Gaussian pdf objects that are used for prediction.

        DESCRIPTION: The Gaussian pdf objects are used to compute the
                      probabilities of data being in an existing class.
        ARGUMENTS
         None

        PARAMETERS
         Ψ            (ndarray): Between-class covariance of the whitened
                                  class centers. [n_dims x n_dims]
         n_avg  (numpy.float64): Average sample size of the data classes.
         stats           (dict): Holds the mean, sample size, and covariance
                                  matrix for each data class. These are stored
                                  in a dictionary that can be accessed by
                                  using the label as the key: for example,
                                  stats[label]['μ'].
         params           (dict): Model parameters accessible via keys.
          -- params['v_' + str(label)]  (ndarray): Whitened class center.
         pdfs             (dict): Holds the logpdf method for each data class.
                                   They can be accessed by using class labels
                                   as keys.
          -- pdfs[label])  (method): Log multivariate normal pdf for class
                                      'label'. Passing in ndarrays of data
                                      returns their likelihood porbabilities.
        RETURNS
         None

        """
        assert np.array_equal(np.diag(self.Ψ.diagonal()), self.Ψ)
         # Verify that Ψ is diagonal.

        Ψ = self.Ψ.diagonal()
        n_Ψ = self.n_avg * Ψ
        n_Ψ_plus_eye = n_Ψ + 1

        cov = np.diag(1 + Ψ / n_Ψ_plus_eye)
        transformation = np.diag(n_Ψ / n_Ψ_plus_eye)
        for label in self.stats.keys():
            μ = self.params['v_' + str(label)]
            μ = np.matmul(transformation, μ)
            m_normal = multivariate_normal(μ, cov)
            self.pdfs[label] = m_normal.logpdf

    def set_stats(self):
        """ Computes & sets mean, covariance, and sample size for each class.
        """
        for label in self.data.keys():
            self.update_stats(label)

    def add_datum(self, datum):
        """ Adds a new datum to the dataset, but does NOT run fit() on data!

        DESCRIPTION: This function also appends raw_data if save_raw
                      was set to true when the model was initialized.
        ARGUMENTS
         datum (tuple): An example-label pair in the form (example, label).

        PARAMETERS
        save_raw (bool): Set during model initalization, and determines whether
                          raw data (tuples) should be saved.
        raw_data (list): List of raw data/tuples (example, label). This 
                          variable only exists if save_raw was set to True
                          during model initialization.
        data     (dict): Stores lists of class data/examples, where each
                          example is a flattened ndarray. Data for each class
                          is accessed using the label as the key.
        RETURNS
         None

        """
        assert isinstance(datum, tuple)

        if self.save_raw is True:
            self.raw_data.append(datum)

        example, label = datum
        if label not in self.data:
            self.data.update({label: [example]})
        else:
            self.data[label].append(example)

    def fit(self):
        """ Fits model to the data by computing class stats and model params.

        DESCRIPTION: See (1) documentation in set_stats(), set_params(),
                      and set_pdfs(), (2) equations on p. 532, 533, 535, and
                      537 in the published paper, (3) documentation in 
                      the test code, test_integration_PLDA.py in particular.
        ARGUMENTS
         None

        PARAMETERS
         None

        RETURNS
         None

        """
        self.set_stats()
        self.set_params()
        self.set_pdfs()

    def equals(self, model):
        """ Determines whether two models are equivalent (including data).

        DESCRIPTION: Two models are equal if they have the same data,
                      class statistics, and parameters.
        ARGUMENTS
         model     (PLDA): A probabilistic linear discriminant analysis object.

        PARAMETERS
         data      (dict): Holds data for each class, where keys are the class
                            labels.
           -- data[label]: List of [n_dims x 1] ndarrays (data examples)..
         stats     (dict): Holds the means, covariances, and sample sizes
                            for all of the data classes. Keys are the class
                            labels.
          -- stats[label]: Dictionary with the keys 'μ', 'n', and
                            'covariance' indexing the statistics for that
                            class. Sample size ('n') is type (int), μ
                            is type (numpy.ndarary) with shape (n_dims,),
                            covariance is type (numpy.ndarray) with shape
                            (n_dims, n_dims). 
        RETURNS
         equal     (bool): Whether the two models are equal in terms of the
                            classes, class statistics, and model parameters.
        """
        has_same_data = self.data.keys() == model.data.keys()
        for label in self.data.keys():
            has_same_data &= self.data[label] == model.data[label]

        has_same_stats = self.stats.keys() == model.stats.keys()
        for label in self.stats.keys():
            for stat in self.stats[label]:
                # Some statistics are arrays, so you need to take product.
                is_same = np.prod(self.stats[label][stat] == \
                                  model.stats[label][stat])
                has_same_stats &= bool(is_same)

        has_same_params = self.params.keys() == model.params.keys()
        #for param in self.params.keys():
        #    is_same = np.prod(self.params[param] == model.params[param]) 
        #    print(param, is_same)
        #    has_same_params &= bool(is_same)
        #print(has_same_params)

        equal = has_same_data & has_same_stats & has_same_params

        return equal

    def predict_class(self, data, MAP_estimate=True, return_probs=False):
        """ Classifies data into an existing or new class.

        DESCRIPTION: If MAP_estimate is set to false, the classification is
                      done probabilitically using the normed vector of class
                      likelihood probabilities.
        ARGUMENTS
         data             (ndarray): Flattened examples/data. [n x n_dims]
         MAP_estimate        (bool): Whether the labels should be selected
                                      probabilistically or via MAP estimation.
         return_probs        (bool): Whether the unnormed probabilities of the
                                      data being in each class should be
                                      returned.
        PARAMETERS
         pdfs               (dict): Stores the log multivariate Gaussian pdfs
                                     for each class. Keys are the class labels.
          -- pdfs[label]  (method): logpdf method for the multivariate Gaussian
                                     fit to the whitened data in class 'label'.
        RETURNS
         MAP_predictions    (list): List of class label predictions for the
                                     data, where the labels are MAP estimates.
                                     Order of the list is the same as the data.
         prob_predictions   (list): List of class label predictions for the
                                     data, where the labels are selected
                                     probabilistically. Order of the list is
                                     the same as the data.
         unnormed_probs  (ndarray): Unnormed probabilities of the data being in
                                     each class. [n x n_dims]
        """
        assert isinstance(data, np.ndarray)
        n_data = data.shape[0]
        if n_data == 1:
            data = np.squeeze(data)
            data = np.asarray([data, data])

        unnormed_logprobs = self.calc_class_log_probs(data)
        probs = np.exp(unnormed_logprobs.T - logsumexp(unnormed_logprobs, 
                       axis=1)).T
        if MAP_estimate == False:
            for i, row in enumerate(probs):
                if row.sum() > 1:
                    probs[i] = probs[i] / row.sum()
        labels = [label for label in self.stats.keys()]

        if MAP_estimate is True:
            label_idxs = np.argmax(unnormed_logprobs, axis=1)
            MAP_predictions = [labels[idx] for idx in label_idxs]
            
            if return_probs is False:
                return MAP_predictions[:n_data]
            else:
                unnormed_probs = np.exp(unnormed_logprobs)
                return MAP_predictions[:n_data], unnormed_probs[:n_data]

        elif MAP_estimate is False:
            assert data.shape[0] == probs.shape[0]

            predictions = []
            for idx in range(n_data):
                predictions.append(np.random.multinomial(1, probs[idx, :]))
            predictions = np.array(predictions)
            label_idxs = np.argmax(predictions, axis=1)
            prob_predictions = [labels[idx] for idx in label_idxs]

            if return_probs is False:
                return prob_predictions
            else:
                unnormed_probs = np.exp(unnormed_logprobs)
                return prob_predictions, unnormed_probs
        else:
            raise ValueError('Invalid parameters for the function.')


    def to_data_list(self):
        """ Returns data as a list of tuples in the form (example, label).

        ARGUMENTS
         None

        PARAMETERS
         data                 (dict): Dictionary of lists of data.
          -- data.keys() (dict_keys): Labels for the classes of data.
          -- data[label]      (list): List of ndarrays of data in the class
                                       'label'.
        RETURNS
         data_list            (list): List of labeled data as tuples,
                                       (example, label).
        """
        data_list = []
        for label in self.data.keys():
            for example in self.data[label]:
                data_list.append((example, label))

        return data_list

    def update_stats(self, label):
        """ Updates mean, covariance, and sample size of given data class.

        DESCRIPTION: Mean and covariance are computed using numpy, thus
                      the covariances' norms are N - 1. Sample size (i.e. 'n')
                      is computed by getting the length of the list of data.
        ARGUMENTS
         label    (user defined): Dictionary key for a particular data class.

        PARAMETERS
         data             (dict): Holds lists of data/ndarrays for each class.
           -- data[label] (list): List of [1 x n_dims] ndarrays (data).
         stats            (dict): Holds the mean, sample size, and covariance
                                   for each data class.
         stats[label]     (dict): Holds statistics for the data class 'label'.
           -- stats[label]['n']               (int): class sample size
           -- stats[label]['μ']           (ndarray): [1 x n_dims]
           -- stats[label]['covariance']  (ndarray): [n_dims x n_dims]

        RETURNS
         None 

        """
        examples_list = self.data[label]
        ds = self.get_stats_data_structure()
        self.stats.update({label: ds})
        examples_arr = np.array(examples_list)

        n = len(examples_list)
        μ = examples_arr.mean(axis=0)
        cov = np.cov(examples_arr.T)

        self.stats[label]['n'] = n
        self.stats[label]['μ'] = μ
        self.stats[label]['covariance'] = cov

    def whiten(self, X):
        """ Standardizes the data, X. 

        EQUATION: u = inv(A)(x - m)
            
        DESCRIPTION: See p. 534, section 3.1 for more details on working
                      in the latent space, u. 

        ARGUMENTS
         X (ndarray): Rows of unwhitened/unstandardized data. [N x n_dims]

        PARAMETERS
         A (ndarray): Transformation that whitens the data. [n_dims x n_dims]
         m (ndarray): Vector that centers the data. [n_dims x 1]

        RETURNS
         U (ndarray): Rows of whitened/standardized data. [N x n_dims]
        """
        inv_A = np.linalg.inv(self.A)
        U = X - self.m
        U = np.matmul(inv_A, U.T).T

        return U
