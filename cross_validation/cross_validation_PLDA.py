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
sys.path.append(os.getcwd() + '/../')
import numpy as np
from PLDA import PLDA
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.transform import resize


def build_google_faces_dataset(load_dir, resized_shape):
    data_shape=resized_shape
    imgs = []
    lbls = []
    for fname in os.listdir(load_dir):
        if fname[-4:] == '.jpg':
            label = ''.join([i for i in fname if not i.isdigit()])
            label = label[:-4]
            if label != 'neutral':  # Ignore the 2 'neutral' face images.
                img = imread(load_dir + fname)
                img = resize(img, data_shape)
                img = img.flatten()
                imgs.append(img)
                lbls.append(label)

    return imgs, lbls

def check_equal_folds(*args, **kwargs):
    n = args[0].shape[0]
    k = kwargs['k']
    warn = n % k != 0
    if warn:
        print('\nWARNING')
        print('\'k\' did not divide the total number of data \'n\': ' +
              'n % k = {}.'.format(n % k))
        print('{} subsamples of {} data '.format(k - 1, int(n / k)) + 
              'each are being tested during each k-folds iteration.')
       
def rerun(n_iterations):
    def wrap_CV(CV_func):
        k_folds_scores = []
        def wrapper(*args, **kwargs):
            check_equal_folds(*args, **kwargs)

            for x in range(n_iterations):
                scores = CV_func(*args, **kwargs)
                k_folds_scores.append(scores)
                scores = []

            return k_folds_scores
        return wrapper
    return wrap_CV

@rerun(n_iterations=10)  # Number of times to run k-folds functions.
def k_folds_CV_PLDA(X, Y, k=5, MAP_estimate=True):
    """ Cross validation on Google Face emotions dataset, using k-folds.

    DESCRIPTION: If MAP_estimate is set to False, n_runs number of runs
                  are run for each of the k-subsamples runs as well. 
                  This function runs several iterations of k-folders such that
                  the k sets of data are randomly selected.
    ARGUMENTS
     X         (ndarray): Rows of data examaples. [n x n_dims]
     Y         (ndarray): Labels corresponding to the rows of X. [n x 1]
     k             (int): k in k-folds. The number of subsets/subsamples to
                           to break the data into. During each run, all but
                           one set is used as training data, and the remaining
                           is used to test the model's performance. Number
                           of data in the test set is (n // k).
     MAP_estimate (bool): True makes the PLDA model use the MAP estimate for
                           classifciation. If this is set to false, the model
                           makes classifications probabilitically and as such
                           n_runs are run for each 'k'/subsample.
    RETURNS
     None

    """
    n = X.shape[0]  # Total number of data.
    assert k <= n
    n_test = int(n / k)  # Number of data to reserve for testing.
    n_runs = 1 # Runs per subsample. Only used if MAP_estimate is False.

    # Make sure you don't the test on a dataset smaller than 'n // k'.
    if n % k == 0:
        k_runs = k
    else:
        k_runs = k - 1

    scores = []
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    for k_run in range(k_runs):
        # Create training and test data sets.
        start =  k_run * n_test
        end = k_run * n_test + n_test

        test_idxs = idxs[start:end]
        test_X = X[test_idxs,:]
        test_Y = Y[test_idxs]

        data_idxs = set(idxs) - set(test_idxs)  # Set subtraction.
        data_idxs = np.array(list(data_idxs))
        data_X = X[data_idxs, :]
        data_Y = Y[data_idxs]
    
        # Format the data and fit the model.
        data = []
        for x, y in zip(data_X, data_Y):
            data.append((x, y))
    
        model = PLDA(data)
    
        # Evaluate the model.
        if MAP_estimate is False:
            classifications = model.predict_class(test_X,
                                              MAP_estimate=MAP_estimate)
            correct = 0
            for classification, label in zip(classifications, test_Y):
                correct += classification == label
            scores.append(correct / len(test_Y))
        else:
            for run in range(n_runs):
                classifications = model.predict_class(test_X,
                                                  MAP_estimate=MAP_estimate)
                correct = 0
                for classification, label in zip(classifications, test_Y):
                    correct += classification == label
                scores.append(correct / len(test_Y))
    
    scores = np.array(scores)
    return scores

def main():
    load_dir = os.getcwd() + '/Google_Faces/'
    imgs, lbls = build_google_faces_dataset(load_dir, resized_shape=(100,100))

    X = np.array(imgs)
    Y = np.array(lbls)
    del imgs, lbls

    standardized_X = X - X.mean(axis=0)
    standardized_X = standardized_X / np.std(standardized_X, axis=0)

    pca = PCA()
    pca.fit(standardized_X)
    V = pca.components_[:175, :].T
    pca_standardized_X = np.matmul(standardized_X, V)

    scores = k_folds_CV_PLDA(pca_standardized_X, Y, k=50, MAP_estimate=False)
    scores = np.array(scores)
    print('Mean Accuracy: {}'.format(scores.mean()))
    print('Std. Dev.: {}'.format(np.std(scores)))

if __name__ == '__main__':
    main()
