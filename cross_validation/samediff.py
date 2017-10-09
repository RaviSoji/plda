import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import PLDA
import warnings
import numpy as np
from numpy.core.umath_tests import inner1d
from itertools import combinations_with_replacement
from preprocess import fnames_to_idxs
from scipy.misc import logsumexp


class SameDiffTask:
    """ X, Y, and fnames are assumed to be sorted in the same order. """
    def __init__(self, model_class, X, Y, fnames):
        self.model_class = model_class
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.fnames = np.asarray(fnames)
        self.model = None

        assert self.X.shape[0] == self.Y.shape[0] == self.fnames.shape[0]
        assert len(self.X.shape) == 2
        assert len(fnames) == np.unique(fnames).shape[0]


    def fit_model(self, X, Y):
        training_data = [(x, y) for (x, y) in zip(X, Y)]
        self.model = self.model_class(training_data)


    def to_idxs(self, fname_ndarray):
        """
        ARGUMENT
         fname_ndarray  (ndarray): Contains filenames that may or may not be
                                    repeated.
        RETURN
         idx_array      (ndarray): Has same shape as fname_ndarray, but with
                                    integer entries. These integers index the
                                    filenames in unique_fnames.
        """
        idx_array = np.zeros(fname_ndarray.shape)
        for idx, fname in np.ndenumerate(fname_ndarray):
            idx_array[idx] = np.argwhere(self.fnames == fname)
    
        return idx_array.astype(int)
    

    def get_unique_idx_pairs(self, idxs):
        """ E.g. If [1, 5] and [5, 1] are in one list,
            this function drops one of the two pairs.
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
        
        for x1, x2, y1, y2 in zip(X1, X2, Y1, Y2):
            image_pair = np.asarray([x1, x2])
            label_pair = (y1, y2)
    
            yield image_pair, label_pair


    def cross_validate(self, n=None, num_shuffles=1, return_2d_array=True):
        assert n >= 0 and isinstance(n, int)
        assert num_shuffles > 0 and isinstance(num_shuffles, int)

        warnings.warn('run_leave_out() deletes the existing model.')

        all_results = []
        for curr_shuffle in range(num_shuffles):
            idxs = np.arange(self.Y.shape[0])
            np.random.shuffle(idxs)
            n_iters_per_shuffle = self.Y.shape[0] - n + 1
    
            shuffle_results = []
            for iteration in range(n_iters_per_shuffle):
                print(iteration)
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
            self.fit_model(self.X, self.Y)

        else:
            idxs = np.arange(self.Y.shape[0])
            train_idxs = np.asarray(list(set(idxs) - set(idxs1) - set(idxs2)))
            train_X = self.X[train_idxs, :]
            train_Y = self.Y[train_idxs]
            self.fit_model(train_X, train_Y)
        
        test_pairs = self.gen_pairs(idxs1, idxs2)
        truth = []
        predictions = []
        log_probs_same = []
        for img_pair, lbl_pair in test_pairs:
            truth.append(lbl_pair[0] == lbl_pair[1])
            prediction, log_prob_same = self.predict_same_diff(img_pair)
            predictions.append(prediction)
            log_probs_same.append(log_prob_same)

        return self.format_results(truth, predictions, log_probs_same, idxs1,
                                   idxs2).T


    def format_results(self, truth, predictions, log_probs_same, idxs1, idxs2):
        truth = np.asarray(truth)
        predictions = np.asarray(predictions)
        log_probs_same = np.asarray(log_probs_same)
        lbls1 = np.asarray(self.Y[idxs1])
        lbls2 = np.asarray(self.Y[idxs2])

        return np.asarray([truth, predictions, log_probs_same, lbls1, lbls2])



    def calc_posterior_predictives(self, data, return_log=False):
        assert data.shape[0] == 2
        assert data.shape[1] > 1
        assert isinstance(return_log, bool)

        log_probs = self.model.calc_class_log_probs(data)
        log_probs = (log_probs.T - logsumexp(log_probs, axis=1)).T
    
        if return_log is True:
            return log_probs
        else: return np.exp(log_probs)


    def calc_prob_same(self, log_posterior_predictive_ps, return_log=True,
                       sum_over_classes=True):
        assert log_posterior_predictive_ps.shape[0] == 2
        assert log_posterior_predictive_ps.shape[1] > 1
    
        log_ps_same = np.sum(log_posterior_predictive_ps, axis=0)
    
        if return_log is True and sum_over_classes is True:
            log_p_same = logsumexp(log_ps_same)
            return log_p_same
    
        elif return_log is False and sum_over_classes is True:
            log_p_same = logsumexp(log_ps_same)
            return np.exp(log_p_same)
    
        elif return_log is True and sum_over_classes is False:
            return log_ps_same
    
        else:
            return np.exp(log_probs_same)
    
    
    def calc_prob_diff(self, log_posterior_predictive_ps, return_log=True,
                       sum_over_classes=True):
        log_ps = log_posterior_predictive_ps
        n_classes = log_ps.shape[1]
        assert log_ps.shape[0] == 2
    
        x1_log_ps = log_ps[0, :]
        log_ps_diff = np.zeros(n_classes)
    
        idxs = np.ones(n_classes).astype(bool)
        for class_i in range(n_classes):
            temp_idxs = idxs.copy()
            temp_idxs[class_i] = False
            temp_log_ps = log_ps[1, temp_idxs]
            assert temp_log_ps.shape[0] == log_ps.shape[1] - 1
            log_p_x2_not_class_i = logsumexp(temp_log_ps)
            log_ps_diff[class_i] = x1_log_ps[class_i] + log_p_x2_not_class_i
    
        if return_log is True and sum_over_classes is True:
            log_p_diff = logsumexp(log_ps_diff)
            return log_p_diff
    
        elif return_log is False and sum_over_classes is True:
            log_p_diff = logsumexp(log_ps_diff)
            return np.exp(log_p_diff)
    
        elif return_log is True and sum_over_classes is False:
            return log_ps_diff
    
        else:
            log_ps_diff = np.exp(log_ps_diff)
            return np.exp(log_ps_diff)

    def predict_same_diff(self, img_pair, return_log_prob_same=True,
                          MAP_estimate=True, same_probs='marginal_likelihood'):
        assert same_probs == 'marginal_likelihood' or \
               same_probs == 'posterior_predictive'
        if MAP_estimate is False:
            raise NotImplementedError
    
        log_chance = np.log(.5)
    
        log_pps = self.calc_posterior_predictives(img_pair, return_log=True)
        log_prob_same = self.calc_prob_same(log_pps)
    
        if log_prob_same > log_chance:
            prediction = True
    
        elif log_prob_same < log_chance:
            prediction = False
    
        else:
            prediction = np.random.multinomial(1, [.5, .5])
    
            if prediction[0] == 1:
                prediction = True
            else:
                prediction = False
            
        if return_log_prob_same is True:
            return prediction, log_prob_same
    
        elif return_log_prob_same is False:
            return prediction
    
        else: raise ValueError


def main():
    X = np.load('preprocessed_imgs.npy')
    Y = np.load('labels.npy')
    fnames = np.load('fnames.npy')
    mturk_fname_pairs = np.load('mturk_trials.npy')

    same_diff_plda = SameDiffTask(PLDA.PLDA, X, Y, fnames)
    idx_pairs = same_diff_plda.to_idxs(mturk_fname_pairs)
    idx_pairs = idx_pairs.reshape(-1, idx_pairs.shape[-1])
    idxs0 = idx_pairs[:, 0]
    idxs1 = idx_pairs[:, 1]
    mturk_leave_out= same_diff_plda.run(idxs0, idxs1, leave_out=True)
    mturk_leave_in= same_diff_plda.run(idxs0, idxs1, leave_out=False)

    results_leave_2 = same_diff_plda.run_leave_out()
    #results_leave_4 = 
