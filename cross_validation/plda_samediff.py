import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from plda import PLDA
import warnings
import numpy as np
from numpy.core.umath_tests import inner1d
from itertools import combinations_with_replacement
from scipy.misc import logsumexp


class SameDiffTask:
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
        """ E.g. If [1, 5] and [5, 1] are in one list,
            this function makes sure one of them is used.
        """
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

    def cross_validate(self, n=None, num_shuffles=1, return_2d_array=True):
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
                results = self.run(test_pair_idxs[:, 0], test_pair_idxs[:, 1],
                                   leave_out=True)
                shuffle_results.append(results)
            all_results.append(np.asarray(shuffle_results))

        all_results = np.asarray(all_results)
        if return_2d_array is True:
            all_results = all_results.reshape(-1, all_results.shape[-1])

        return all_results


    def run(self, idxs1, idxs2, leave_out=False):
        assert isinstance(leave_out, bool)

        if leave_out is False:
            self.fit_model(self.X, self.Y, self.fnames)
        else:
            idxs = np.arange(self.Y.shape[0])
            train_idxs = np.asarray(list(set(idxs) - set(idxs1) - set(idxs2)))
            train_X = self.X[train_idxs, :]
            train_Y = self.Y[train_idxs]
            train_fnames = self.fnames[train_idxs]
            self.fit_model(train_X, train_Y, train_fnames)
        
        data_pairs, lbl_pairs, fname_pairs = self.gen_pairs(idxs1, idxs2)
        truth = lbl_pairs[:, 0] == lbl_pairs[:, 1]
        predictions, log_probs_same  = self.predict_same_diff(data_pairs)

        truth = np.squeeze(truth)
        lbl_pairs = np.squeeze(lbl_pairs)
        fname_pairs = np.squeeze(fname_pairs)

        return np.stack([truth, predictions, log_probs_same,
                         lbl_pairs[..., 0], lbl_pairs[..., 1],
                         fname_pairs[..., 0], fname_pairs[..., 1]], axis=-1)

    def calc_probs_diff(self, data_pairs):
        """
        Array columns and rows are ordered the same way the labels are in
         self.model.stats.keys()
        """
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

    def predict_same_diff(self, img_pairs, return_log=True, MAP_estimate=True):
        assert isinstance(return_log, bool)
        assert isinstance(MAP_estimate, bool)
        assert len(img_pairs.shape) > 1

        log_chance = np.log(.5)
        log_ps_same = self.calc_prob_same_diff(img_pairs, return_prob='same')

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

    fnames_to_idxs.__doc__ = """
        ARGUMENT
         fname_ndarray  (ndarray): Contains filenames that may or may not be
                                    repeated.
        RETURN
         idx_array      (ndarray): Has same shape as fname_ndarray, but with
                                    integer entries. These integers replace
                                    the string filenames with indices that
                                    index the filenames in self.fnames.
        EXAMPLE:
         fname_ndarray[0] == self.fnames[idx_array[0]]  # True
         print(fname_ndarray[0])  # 'img_name.jpg'
         print(idx_array[0])   # 54
         (self.fnames[54])  # 'img_name.jpg'
        """

def main():
    raise NotImplementedError

    # Example of cross validation
    # Example of running selected examples with leave out
    # Example of running selected examples with leave in
    # Take the result of cross validation, and run those test pairs with leave in.
