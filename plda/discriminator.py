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

import os
import warnings
import numpy as np
from .plda import PLDA
from numpy.core.umath_tests import inner1d
from itertools import combinations_with_replacement
from scipy.misc import logsumexp


class Discriminator:
    """ X, Y, and fnames are assumed to be sorted in the same order. """
    def __init__(self, X, Y, fnames):
        assert X.shape[0] == Y.shape[0] == fnames.shape[0]
        assert len(X.shape) == 2
        assert len(fnames) == np.unique(fnames).shape[0]

        self.X = X
        self.Y = Y
        self.fnames = fnames
        self.model = None

    def fit_model(self, X, Y, fnames):
        self.model = PLDA(X, Y, fnames)

    def fnames_to_idxs(self, fname_ndarray):
        idx_array = np.zeros(fname_ndarray.shape)
        for idx, fname in np.ndenumerate(fname_ndarray):
            idx_array[idx] = np.argwhere(self.fnames == fname)
    
        return idx_array.astype(int)

    def get_unique_idx_pairs(self, idxs):
        idx_pairs = []
        for pair in combinations_with_replacement(idxs, 2):
            idx_pairs.append(list(pair))
    
        return np.asarray(idx_pairs)

    def gen_pairs(self, idxs_1, idxs_2):
        X1 = self.X[idxs_1, :]
        X2 = self.X[idxs_2, :]
        Y1 = self.Y[idxs_1]
        Y2 = self.Y[idxs_2]
        fnames1 = self.fnames[idxs_1]
        fnames2 = self.fnames[idxs_2]
        
        return np.stack((X1, X2), axis=-2), np.stack((Y1, Y2), axis=-1), \
               np.stack((fnames1, fnames2), axis=-1)

    def cross_validate(self, n=None, num_shuffles=1, leave_out=True,
                       return_2d_array=False):
        assert n > 0 and isinstance(n, int)
        assert num_shuffles > 0 and isinstance(num_shuffles, int)

        warnings.warn('run_leave_out() deletes the existing model.')
        if n == 1:
            warnings.warn('n was set to 1. This means all the pairs are' +
                           '\"same\" trials.')

        all_results = []
        for curr_shuffle in range(num_shuffles):
            idxs = np.arange(self.Y.shape[-1])
            np.random.shuffle(idxs)
            n_iters_per_shuffle = self.Y.shape[0] - n + 1
            shuffle_results = []
            for iteration in range(n_iters_per_shuffle):
                print('Shuffle {} of {}, iteration {} of {}'.format(
                      curr_shuffle, num_shuffles - 1, iteration,
                      n_iters_per_shuffle - 1))
                start = iteration
                end = start + n
                test_idxs = idxs[start:end]
                test_pair_idxs = self.get_unique_idx_pairs(test_idxs)
                results, col_titles = self.run(test_pair_idxs[:, 0],
                                              test_pair_idxs[:, 1],
                                              leave_out=leave_out)
                shuffle_results.append(results)
            all_results.append(np.asarray(shuffle_results))

        all_results = np.asarray(all_results)
        if return_2d_array is True:
            all_results = all_results.reshape(-1, all_results.shape[-1])

        return all_results, col_titles


    def run(self, idxs1, idxs2, leave_out, return_log=True):
        assert isinstance(leave_out, bool)

        if leave_out is False:
            self.fit_model(self.X, self.Y, self.fnames)
        else:
            idxs = np.arange(self.Y.shape[0])
            train_idxs = np.asarray(list(set(idxs) - set(idxs1) - set(idxs2)))
            self.fit_model(self.X[train_idxs, :],
                           self.Y[train_idxs],
                           self.fnames[train_idxs])
        
        data_pairs, lbl_pairs, fname_pairs = self.gen_pairs(idxs1, idxs2)
        truth = lbl_pairs[:, 0] == lbl_pairs[:, 1]
        predictions, log_probs_same  = self.predict_same_diff(data_pairs)

        results, col_titles = self.tidy_data(predictions, truth,
                                             log_probs_same,
                                             lbl_pairs, fname_pairs,
                                             prob_type='same',
                                             leave_out=leave_out,
                                             return_log=return_log)

        return results, col_titles

    def tidy_data(self, predictions, truth, log_probs, lbl_pairs, fname_pairs,
                  prob_type='same', leave_out=False, return_log=False):
        assert predictions.shape == truth.shape
        assert lbl_pairs.shape == fname_pairs.shape
        assert predictions.shape[0] == lbl_pairs.shape[0]
        assert lbl_pairs.shape[1] == 2
        assert prob_type == 'same' or prob_type == 'diff'
        assert isinstance(leave_out, bool), isinstance(return_log, bool)

        truth = np.squeeze(truth)
        lbl_pairs = np.squeeze(lbl_pairs)
        fname_pairs = np.squeeze(fname_pairs)

        col_titles = ['prediction', 'truth',
                      'stimulus_label_1','stimulus_label_2',
                      'stimulus_filename_1','stimulus_filename_2']
        results = [predictions, truth,
                   lbl_pairs[..., 0], lbl_pairs[..., 1],
                   fname_pairs[..., 0], fname_pairs[..., 1]]

        col_titles.append('leave_out')
        if leave_out is True:
            results.append([True] * len(predictions))
        elif leave_out is False:
            results.append([False] * len(predictions))

        if return_log is True and prob_type == 'same':
            col_titles.append('log_prob_same')
            results.append(log_probs)
        elif return_log is True and prob_type =='diff':
            col_titles.append('log_prob_diff')
            results.append(log_probs)
        elif return_log is False and prob_type == 'same':
            col_titles.append('prob_same')
            results.append(np.exp(log_probs))
        elif return_log is False and prob_type == 'diff':
            col_titles.append('prob_diff')
            results.append(np.exp(log_probs))
        else:
            raise ValueError

        return np.stack(results, axis=-1), np.asarray(col_titles)

    def calc_probs_diff(self, data_pairs):
        assert data_pairs.shape[-2] == 2
        assert len(data_pairs.shape) > 1

        relevant_dims = self.model.relevant_dims
        u_pairs = self.model.whiten(data_pairs)[..., relevant_dims]
        means, cov_diags = self.model.calc_posteriors(dims=relevant_dims)

        u_pairs = u_pairs[..., None, :]
        log_pps = self.model.calc_marginal_likelihoods(u_pairs, means,
                                                       cov_diags,
                                                       standardize_data=False)
        idxs_K = np.arange(self.model.K)
        idxs_1, idxs_2 = np.meshgrid(idxs_K, idxs_K)

        log_ps_diff = np.add(log_pps[..., 0, idxs_1], log_pps[..., 1, idxs_2])
        log_ps_diff[..., idxs_K, idxs_K] = - np.inf

        return log_ps_diff

    def calc_probs_same(self, data_pairs):
        assert len(data_pairs.shape) > 1
        assert data_pairs.shape[-2] == 2

        relevant_dims = self.model.relevant_dims
        u_pairs = self.model.whiten(data_pairs)[..., relevant_dims]
        means, cov_diags = self.model.calc_posteriors(dims=relevant_dims)

        log_mls = self.model.calc_marginal_likelihoods(u_pairs, means,
                                                       cov_diags,
                                                       standardize_data=False)
        return log_mls

    def calc_prob_same_diff(self, data_pairs, return_log=True, norm_probs=True,
                            return_prob='same'):
        assert data_pairs.shape[-2] == 2
        assert isinstance(return_log, bool)
        assert return_prob == 'same' or return_prob == 'diff'

        log_ps_diff = self.calc_probs_diff(data_pairs)
        log_ps_same = self.calc_probs_same(data_pairs)
        assert log_ps_diff.shape[-2] == log_ps_diff.shape[-1] == log_ps_same.shape[-1]

        log_prob_diff = logsumexp(log_ps_diff, axis=(-1, -2))
        log_prob_same = logsumexp(log_ps_same, axis=-1) + np.log(6)
         # Since there are 42 "different probabilities" and 7 " same probabilities.
         # Multiplying by six makes the prior 50/50 on the same/diff task.

        if return_prob == 'same':
            log_probs = log_prob_same
        else:
            log_probs = log_prob_diff

        if norm_probs is True:
            norms = logsumexp([log_prob_diff, log_prob_same], axis=0)
            log_probs = log_probs - norms

        if return_log is True:
            return log_probs
        else:
            return np.exp(log_probs)

    def predict_same_diff(self, data_pairs, return_log=True, MAP_estimate=True):
        assert isinstance(return_log, bool)
        assert isinstance(MAP_estimate, bool)
        assert len(data_pairs.shape) > 1

        log_chance = np.log(.5)
        log_ps_same = self.calc_prob_same_diff(data_pairs, return_prob='same')

        if MAP_estimate is False:
            predictions = log_ps_same.shape
            for idx, log_p_same in np.ndenumerate(log_ps_same):
                p_same = np.exp(log_p_same)
                prediction = bool(np.random.multinomial(1, [p_same, 1 - p_same])[0])
                predictions[idx] = prediction
        else:
            predictions = np.zeros(log_ps_same.shape).astype(bool)
            predictions[log_ps_same > log_chance] = True
            predictions[log_ps_same < log_chance] = False
            bool_idxs = log_ps_same == log_chance
            sz = bool_idxs.sum()
            guesses = np.random.multinomial(1, [.5, .5], size=sz)[:,0].astype(bool)
            predictions[bool_idxs] = guesses
            
        if return_log is True:
            return predictions, log_ps_same
        else:
            return predictions, np.exp(log_ps_same)

    fit_model.__doc__ = """
        Fits the plda model to the task, using X and Y.
    
        ARGUMENTS
         X  (ndarray), shape=(n_data, n_data_dims)
           The data, sorted row-wise. That is, each column is a datum,
           and the columns are the values at those dimensions.
    
         Y  (ndarray), shape=(n_data,)
           Labels of the data, sorted in the same order as X.
    
         fnames  (ndarray), shape=(n_data,), optional
           Filenames of the data, sorted in the same order as X.
    
        RETURNS
         None
        """

    fnames_to_idxs.__doc__ = """
         Converts an ndarray of filenames to indices indexing the model's data.
    
        ARGUMENT
         fname_ndarray  (ndarray), shape=(...)
           Contains filenames that may or may not be repeated.
    
        RETURN
         idx_array  (ndarray):
           Has same shape as fname_ndarray, but with integer entries. These
           integers replace the string filenames with indices that
           index the filenames in self.fnames.
    
        EXAMPLE:
         fname_ndarray[0] == self.fnames[idx_array[0]]
         fname_ndarray[0]  # 'img_name.jpg'
         idx_array[0]   # 54
         self.fnames[54]  # 'img_name.jpg'
        """

    get_unique_idx_pairs.__doc__ = """
        Takes a vector of numbers and generates a list of all unique pairs.

        DESCRIPTION: By unique I mean that if [1, 5] and [5, 1] are in one
                     list, this function makes sure only one of them is used.

        DESCRIPTION: More specifically, this function takes a vector of values,
                      and then essentially finds all the possible ways to
                      "choose 2" WITH replacement.
                     Order within pairs doesn't matter, so a vector of length N
                      will result in pairings with unique pairs of elements.
                     Also, note that some of these pairs will be two copies of
                      the same datum.
        ARGUMENTS
         idxs  (ndarray), shape=(n_unique_idxs,)
           Unique indices, indexing the data in self.fnames.
         
        RETURNS
         idx_pairs  (ndarray), shape=(n_combos, 2)
           All the possible (unique) ways one could select two data.
        """

    gen_pairs.__doc__ = """
        Generates pairs of data, labels, and filenames using indices.

        ARGUMENTS
         idxs_1  (ndarray), shape=(...)
           An array of unique integers.

         idxs_2  (ndarray), shape=(...)
           An array of unique integers.
         
        RETURNS
          X_pairs  (ndarray), shape=(..., 2, n_data_dims)
            Pairs of data.

          Y_pairs  (ndarray), shape=(..., 2)
            Pairs of labels, in the same order as X_pairs.

          fname_pairs  (ndarray), shape=(..., 2)
            Pairs of filenames, in the same order as X_pairs.
        """

    cross_validate.__doc__ = """
        Performs cross validation with plda via the same-different task.

        DESCRIPTION: During any particular "shuffle" the order of the data is
                      essentially randomized.
                     Then, the first n data in this shuffled list are selected
                      to be the test set. If leave_out is set to True, these
                      are ommitted during model training. After this, the
                      function increments by 1 and does the same thing, but
                      this time with the 1st to (n + 1)'th data. This continues
                      until there are no more data left, i.e. (n_data - n + 1)
                      runs.
                     If num_shuffles is greater than zero, the data are
                      re-shuffled, and the process is run that number of times.
        ARGUMENTS
         n  (int)
           Number of data to test on.

         num_shuffles  (int)
           Number of times to re-run the cross-validation on newly shuffled
           data. This matters only when your dataset is small..
           
         leave_out  (bool), optional
           Whether or not to leave the 'n' data out of the training set before
           testing on the 'n' data. Default value is True (recommended).

         return_2d_array  (bool), optional
           Whether to return the results as a 2D ndarray.

        RETURNS
         results  (ndarray), shape=SEE BELOW.
           If return_2d_array is False,
               shape=(num_shuffles + 1, n_data - n + 1, n_combos, 3 + n_classes)
            First dimension corresponds to a particular shuffle.
            Second dimension corresponds to runs with training data left
            Third dimension corresponds to the number of unique data combos.
             See the get_unique_idx_pairs() method for details.
            The fourth dimension represents the actual data, whose column
             labels are returned as col_titles.
           If return_2d_array is True,
            shape=(num_shuffles + 1) * (n_data - n + 1) * n, 3 + n_classes

         col_titles  (ndarray), shape=(8,)
           Titles of the "columns" (i.e. last dimension)
        """

    run.__doc__ = """
        Uses plda to evaluate whether pairs of data are "same" or "different".

        ARGUMENTS
         idxs1  (ndarray), shape=(n_pairs,)
           Indices indexing one of the two data to be compared on the
           same-different task.

         idxs2  (ndarray), shape=(n_pairs,)
           Indices indexing one of the two data to be compared on the
           same-different task.

         leave_out  (bool)
           Whether or not to leave out the test data during training. Set to
           False by default.

         return_log  (bool)
           Whether or not to return probabilities of the two stimuli being
           the "same" as log probabilities or not.
     
        RETURNS
         results  (ndarray), shape=(n_pairs, 8)
           Results of the plda model being run on the indexed pairs of data.

         col_titles (ndarray), shape=(8,)
           Titles for each of the columns in the second dimension of results.
        """

    calc_probs_diff.__doc__ = """
        Computes the probs of two data being generated by different classes.

        DESCRIPTION: Uses the marginal likelihood equation. See jupyter
                     notebook for mathematical details
        ARGUMENTS
         data_pairs  (ndarray), shape=(..., 2, n_data_dims)
           Pairs of data to for which the model will make "same" and
           "different" evaluations.
         
        RETURNS
         log_ps_diff  (ndarray), shape=(..., n_unique_labels, n_unique_labels)
           Log joint probabilities of the two data being generated by different
           classes. The diagonals are set to -inf because the probabilities of
           "same" are computed differently -- see calc_probs_same(). 
           SEE ALSO jupyter notebook notes on the mathematical details.
        """

    calc_probs_same.__doc__ = """
        Computes the probs of two data being generated by the same class.

        DESCRIPTION: Uses the marginal likelihood equation. See jupyter
                     notebook for mathematical details
        ARGUMENTS
         data_pairs  (ndarray), shape=(..., 2, n_data_dims)
           Pairs of data to for which the model will make "same" and
           "different" evaluations.
         
        RETURNS
         log_ps_same  (ndarray), shape=(..., n_unique_classes)
          Probabilities that both data were generated from the same
          distribution.
        """

    calc_prob_same_diff.__doc__ = """
        Calculates the probabilities of data pairs being in the same category.

        DESCRIPTION: See the jupyter notebook for mathematical details.

        ARGUMENTS
         data_pairs  (ndarray), shape=(..., 2, n_data_dims)
           Pairs of data to for which the model will make "same" and
           "different" evaluations.

         return_log  (bool)
           Whether or not return the logarithm of probabilities.

         norm_probs  (bool)
           Whether or not to normalize the probabilities.

         return_prob  (str)
           Must be set to either "same" or "diff". This determines whether
           the function returns the probabilities of the two data being
           generated by the same or different distributions.
    
        RETURNS
          probabilities  (ndarray), shape=(...)
            Model certainty about "same" and "different" judgements.
            If return_log is set to True, these are log probabilities.
            If norm_probs is set to True, these are also normalized.
            Whether to return probabilities of "same" or to return
            probabilities of "different" is determined by the return_prob
            argument.
        """

    predict_same_diff.__doc__ = """
        Uses the plda to predict whether two data are in the same category.

        DESCRIPTION: See calc_prob_same_diff(), calc_probs_same()
                      calc_probs_diff() for more detail.
                     See also the jupyter notebook for notes on the math.

        ARGUMENTS
         data_pairs  (ndarray), shape=(..., 2, n_data_dims)
           Pairs of data to for which the model will make "same" and
           "different" evaluations.

         return_log  (bool)
           Whether or not return the logarithm of probabilities.

         MAP_estimate  (bool)
           Whether or not to predict using the MAP estimate or predict
           probabilistically.

         return_prob  (str)
           Must be set to either "same" or "diff". This determines whether
           the function returns the probabilities of the two data being
           generated by the same or different distributions.
     
        RETURNS
         predictions  (ndarray), shape=(...)
          Model evaluations, predicting whether the pairs of data are from
          the same data generating distributions or not.

         probabilities  (ndarray), shape=(...)
          Probabilities associated with the predictions. These are log values
          if return_log is set to True.
        """

def main():
    raise NotImplementedError

    # Example of cross validation
    # Example of running selected examples with leave out
    # Example of running selected examples with leave in
    # Take the result of cross validation, and run those test pairs with leave in.
