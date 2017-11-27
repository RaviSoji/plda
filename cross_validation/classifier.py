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
import sys
import warnings
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from plda import PLDA


class Classifier:
    def __init__(self, X, Y, fnames=None):
        assert len(X.shape) == 2
        assert X.shape[0] == Y.shape[0]
        if fnames is not None:
            assert X.shape[0] == fnames.shape[0]

        self.X = X
        self.Y = Y
        self.fnames = fnames
        self.model = None

    def fit_model(self, X, Y, fnames=None):
        self.model = PLDA(X, Y, fnames)

    def cross_validate(self, n=1, num_shuffles=1, leave_out=True):
        assert n > 0 and isinstance(n, int)
        assert num_shuffles > 0 and isinstance(num_shuffles, int)
        assert isinstance(leave_out, bool)
        warnings.warn('Deleting the existing model to run cross validation.')

        all_results = []
        for curr_shuffle in range(num_shuffles):
            all_idxs = np.arange(self.Y.shape[-1])
            np.random.shuffle(all_idxs)
            n_iters_per_shuffle = self.Y.shape[0] - n + 1
            shuffle_results = []
            for iteration in range(n_iters_per_shuffle):
                print('Shuffle {} of {}, iteration {} of {}'.format(
                      curr_shuffle, num_shuffles - 1, iteration,
                      n_iters_per_shuffle - 1))
                start = iteration
                end = start + n
                test_idxs = list(all_idxs[start:end])
                results, col_titles = self.run(test_idxs, leave_out=leave_out)
                shuffle_results.append(results)
            all_results.append(np.asarray(shuffle_results))

        all_results = np.asarray(all_results)

        return all_results, col_titles

    def run(self, test_idxs, leave_out, return_probs=True, return_log=True,
            return_fnames=True):
        assert isinstance(leave_out, bool)

        if leave_out is True:
            all_idxs = np.arange(self.Y.shape[0])
            train_idxs = np.asarray(list(set(all_idxs) - set(test_idxs)))
            train_X, train_Y = self.X[train_idxs, :], self.Y[train_idxs]
            if self.fnames is not None:
                train_fnames = self.fnames[train_idxs]
            else:
                train_fnames = None
        else:
            train_X, train_Y, train_fnames = self.X, self.Y, self.fnames

        test_X, test_Y = self.X[test_idxs, :], self.Y[test_idxs]

        self.fit_model(train_X, train_Y, train_fnames)

        predictions, log_pps = self.predict(test_X, self.model,
                                            standardize_data=True)

        results, col_titles = self.tidy_data(predictions, log_pps, test_idxs,
                                             return_probs, return_log,
                                             return_fnames, leave_out)

        return results, col_titles

    def tidy_data(self, predictions, log_probs, test_idxs,
                  return_probs, return_log, return_fnames, leave_out):
        assert isinstance(return_log, bool)

        col_titles = ['prediction', 'truth']
        results = [predictions[:, None], self.Y[test_idxs][:, None]]

        col_titles.append('leave_out')
        if leave_out is True:
            results.append(np.asarray([True] * len(predictions))[:, None])
        elif leave_out is False:
            results.append(np.asarray([False] * len(predictions))[:, None])

        if return_fnames is True and self.fnames is None:
            fnames = np.asarray([None] * len(test_idxs))
            col_titles.append('filename')
            results.append(fnames[:, None])

        elif return_fnames is True and self.fnames is not None:
            col_titles.append('filename')
            results.append(self.fnames[test_idxs][:, None])

        if return_probs is True and return_log is False:
            probs = np.exp(log_probs)
            col_titles += ['prob_{}'.format(key) \
                           for key in list(self.model.data.keys())]
            results.append(probs)
                       
        elif return_probs is True and return_log is True:
            probs = log_probs
            col_titles += ['log_prob_{}'.format(key) \
                           for key in list(self.model.data.keys())]
            results.append(probs)

        return np.hstack(results), np.asarray(col_titles)

    def predict(self, X, model, standardize_data):
        assert isinstance(model, PLDA)
        assert isinstance(standardize_data, bool)

        log_pps, \
        labels = model.calc_posterior_predictives(X[..., None, :],
                                                  standardize_data,
                                                  return_labels=True)
        axis = len(np.squeeze(X).shape) - 1
        idxs = np.argmax(log_pps, axis=axis)
        predictions = np.asarray(labels)[[idxs]]
        if len(log_pps.shape) == 1:
            log_pps = log_pps[None, :]

        return predictions, log_pps

    def fnames_to_idxs(self, fname_ndarray):
        idx_array = np.zeros(fname_ndarray.shape)
        for idx, fname in np.ndenumerate(fname_ndarray):
            idx_array[idx] = np.argwhere(self.fnames == fname)
    
        return idx_array.astype(int)
  
    def get_confusion_matrix(self, results, as_ndarray=False):
        labels = np.unique(results[..., :2])
        row_dict = {label: 0 for label in labels}
        matrix_dict = {label: row_dict.copy() for label in labels}
        actual_Y = results[..., 1].flatten()
        predicted_Y = results[..., 0].flatten()

        for actual, predicted in zip(actual_Y, predicted_Y):
            matrix_dict[actual][predicted] += 1

        if as_ndarray is True:
            shape = (len(matrix_dict.keys()), len(matrix_dict.keys()))
            arr = np.zeros(shape)

            row = 0
            for row_key in matrix_dict.keys():
                col = 0
                for col_key in matrix_dict[row_key].keys():
                    arr[row, col] = matrix_dict[row_key][col_key]
                    col += 1
                row += 1

            matrix = arr
        else:
            matrix = matrix_dict

        return matrix

    fit_model.__doc__ = """
        Fits the model to the data, labels, and filenames (if given).

        ARGUMENTS
         X  (ndarray or list), shape=(n_data, n_data_dims)
           Each row is a datum and each column is a dimension of the datum.

         Y  (ndarray or list), shape=(n_data,)
           Labels that are sorted in the same order as X.

         fnames  (ndarray or list), shape=(n_data,), optional
           Filenames of the data, sorted in the same order as X.

        RETURN
         None
        """

    cross_validate.__doc__ = """
        Performs cross validation on plda via a classification task.

        DESCRIPTION: During any particular "shuffle", the order of the data
                      is essentially randomized.
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
           data. This matters only when your dataset is small.
           
         leave_out  (bool), optional
           Whether or not to leave the 'n' data out of the training set before
           testing on the 'n' data. Default value is True (recommended).

        RETURNS
         results  (ndarray), shape=(num_shuffles, n_data - n, n, 3 + n_classes)
           First dimension corresponds to a particular shuffle.
           Second dimension corresponds to runs with training data left
           Third dimension corresponds to test trials.
           The fourth dimension represents the actual data, whose column labels
            are returned as col_titles

         col_titles (ndarray), shape=results.shape[-1]
           Titles of the columns (last dimension) in the results.
        """

    run.__doc__ = """
        Runs the fitted plda model on data specified by the input indices.

        ARGUMENTS
         test_idxs  (list of ints)
           Indices that index the data in self.fnames. These data will comprise
           the test set.

         leave_out  (bool)
           Whether or not to leave the test data out of the training set.

         return_probs  (bool)
           Whether to return the probabilities of the test image being
           classified to each of the labels.

         return_log  (bool)
           Whether or not to return log probabiltiies.

         return_fnames  (bool)
           Whether or not to return the filenames of the test images.
         
        RETURNS
         results  (ndarray), shape=(*test_idxs.shape, >2)
           The last dimension always returns the prediction in the first column
           and the truth in the second column. If return_fnames is set to
           True, this goes in the second column, etc. See col_titles.

         col_titles  (ndarray), shape=results.shape[-1]
           Titles of the columns (last dimension) in the results.
        """

    tidy_data.__doc__ = """
        Formats the inputs into "tidy data" format. See data science blogs.

        ARGUMENTS
         predictions  (ndarray)
          Shape depends on the input to the .predict() function. These are the
          model's predicted classifications of the test data.

         log_probs  (ndarray), shape=(*predictions.shape, n_unique_lables)
          Log probabilities of the image being assigned to each class in
          the training data.

         test_idxs  (list), length=len(predictions)
          The indices used to generate predictions and log_probs.

         return_log  (bool)
          Whether or not to return log probabilities of the test image
          being assigned to each label.
         
        RETURNS
         tidied_data  (ndarray), shape=(*predictions.shape, 9)
        """

    predict.__doc__ = """
        Generates predictions and their log probabilities via the plda model.

        ARGUMENTS
         X  (ndarray), shape=(..., n_data, n_data_dims)
          The data that is to be classified.

         model  (PLDA)
          A trained probabilistic linear discriminant analysis model. That is,
          its "fit" method should have been run to fit the parameters to data.

         standardize_data  (bool)
          Whether or not to transform the data, X, to latent space.

        RETURNS
         predictions  (ndarray), shape=X.shape[:-1]
           Model predictions about the labels of the rows in X. These are
           generated by indexing the labels used to initialze the object.

         log_pps  (ndarray), shape=(*X.shape[:-1], num_unique_labels)
           The log probabilities the model used to generate the predictions.
           These are sorted in the same order as 'predictions'.
        """

    to_idxs.__doc__ = """
        Converts an ndarray of filenames to indices indexing the model's data.

        ARGUMENT
         fname_ndarray  (ndarray)
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

    get_confusion_matrix.__doc__ = """
        Generates a confusion matrix from model predictions and true labels.

        ARGUMENTS
         results  (ndarray), shape=(..., >1)
           Must have at least two columns with the first being model predictions
           and the second being the truth.

         as_ndarray  (bool), optional
           Returns the matrix as an ndarray

        RETURNS
         confusion_matrix  (dictionary of dictionaries OR ndarray)
           Variable type depends on what as_array is set to. 
        """

def main():
    idxs = np.load('mouth_open_idxs.npy')
    X = np.load('X.npy')[idxs]
    Y = np.load('Y.npy')[idxs]
    fnames = np.load('fnames.npy')[idxs]

    plda_classification = Classifier(X, Y, fnames)
    results_leave_in = plda_classification.cross_validate(n=1)
    results_leave_out = plda_classification.cross_validate(n=1)

    experiment_img_names = np.load('')
    img_idx_arr = plda_classification.to_idxs(experiment_img_names)
    results_experiment = plda_classification.run(img_idx_arr)
