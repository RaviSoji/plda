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
import numpy as np
from skimage.io import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA


def load_jpgs_and_lbls(directory, dataset, as_grey=True):
    """ Reads '.jpg' files from a directory as grayscaled images.

    DESCRIPTION
     Grayscaling weights: 0.2125 RED + 0.7154 GREEN + 0.0721 BLUE.
     cafe and google data file names should begin with the emotion name.

    ARGUMENT
     directory    (str): Directory containing the files to be read.
     dataset      (str): Must be set to either 'cafe' or 'google'.

    RETURNS
     images      (list): List of 2D ndarrays (grayscaled images).
     labels      (list): List of strings in the same order as images.
    """
    assert isinstance(directory, str)
    assert dataset == 'cafe' or dataset == 'google'
    assert as_grey == True

    images = []
    labels = []
    for fname in os.listdir(directory):
        if fname.endswith('.jpg'):
            img = imread(directory + '/' + fname, as_grey=as_grey)
            images.append(img)
            label = get_label(fname, dataset=dataset)
            labels.append(label)

    return images, labels

def get_label(filename, dataset):
    if dataset == 'cafe':
        return filename[:3]

    elif dataset == 'google':
        label = ''.join([i for i in filename if not i.isdigit()])
        label = label[:-4]

        return label

    else:
        raise ValueError('dataset must be set to either \"cafe\" or \"google\".')

def get_smallest_shape(images):
    smallest_size = -1
    for img in images:
        assert isinstance(img, np.ndarray)
        if img.shape[0] < smallest_size or smallest_size == -1:
            smallest_size = img.shape[0]

    return (smallest_size, smallest_size)
                

def get_principal_components(flattened_images, n_components='default',
                             default_pct_variance_explained=.96):
    """ Standardizes the data and gets the principal components.
    """
    for img in flattened_images:
        assert isinstance(img, np.ndarray)
        assert img.shape == flattened_images[-1].shape
        assert len(img.shape) == 1
    X = np.asarray(flattened_images)
    X -= X.mean(axis=0)  # Center all of the data around the origin.
    X /= np.std(X, axis=0)

    pca = PCA()
    pca.fit(X)

    if n_components == 'default':
        sorted_eig_vals = pca.explained_variance_
        cum_pct_variance = (sorted_eig_vals / sorted_eig_vals.sum()).cumsum()
        idxs = np.argwhere(cum_pct_variance >= default_pct_variance_explained)
        n_components = np.squeeze(idxs)[0]
        
    V = pca.components_[:n_components + 1, :].T
    principal_components = np.matmul(X, V)

    return principal_components

def preprocess_faces(images, n_components='default', resize_shape='default'):
    """ Notes: Images are all square, but not the same size. We resize all the
                images to be the same sized square, thereby preserving the
                aspect ratios.
    """
    for img in images:
        assert img.shape[0] == img.shape[1]

    if resize_shape == 'default':
        resize_shape = get_smallest_shape(images)

    preprocessed_images = []
    for img in images:
        prepped_img = imresize(img, resize_shape).astype(float)
        prepped_img = prepped_img.flatten()
        preprocessed_images.append(prepped_img)
    preprocessed = get_principal_components(preprocessed_images, n_components)

    return preprocessed

def leave_n_out(X, Y, n, max_runs='default'):
    """ n is the number to leave out for the test set(s). 
        Note that this code uses set subtraction to get the training indices,
        so this will mean that ordering of the training sets will be similar.
    """
    if max_runs == 'default':
        max_runs = X.shape[0] - n

    idxs = np.arange(0, X.shape[0])
    np.random.shuffle(idxs)

    for x in range(X.shape[0] - n):
        if x < max_runs:
            test_idxs = idxs[x: x + n]
            training_idxs = set(idxs) - set(test_idxs)
            training_idxs = np.asarray(list(training_idxs))
    
            train_X = X[training_idxs, :]
            train_Y = Y[training_idxs]
            test_X = X[test_idxs, :]
            test_Y = Y[test_idxs]
    
            yield (train_X, train_Y), (test_X, test_Y)

        else:
            print('Warning: max_runs was not set to default, so estimate of
                   performance will be more biased.')
            break

def predict_leave_n_out(X, Y, n=10, MAP_estimate=True, max_runs='default'):
    predicted_ys = []
    actual_ys = []
    for (train_x, train_y), (test_x, test_y) in leave_n_out(X, Y, n):
        training_data = [(x, y) for (x, y) zip(train_X, train_Y)]
        model = PLDA(training_data)

        predictions = model.predict_class(test_X, MAP_estimate=MAP_estimate)

        predicted_ys += [y for y in predictions]
        actual_ys += [y for y in test_y]

    return predicted_ys, actual_ys

def get_scores(actual_Y, predicted_Y, return_confusion_matrix=True):
    assert len(actual_Y) == len(predicted_Y)
    classes = set(actual_Y)
    n_total = len(actualY)
    model_predictions = {y: 0 for y in classes}
    matrix = {y: model_predictions for y in classes}}

    for actual, predicted in zip(actual_Y, predicted_Y):
        matrix[actual][predicted] += 1

    score = 0
    for actual in matrix.keys():
        number_correct = matrix[actual][actual]
        score += number_correct
    score /= n_total

    if return_confusion_matrix is False:
        return score
    else:
        return score, confusion_matrix
    
#def cross_validate(X, Y, return_class_scores):
#    fit
#    predict
#    return scores

def main():
    img_dir = ''
    X, Y = load_jpgs_and_lbls(img_dir, as_grey=True)
    X = preprocess_faces(X)

