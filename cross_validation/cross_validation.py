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
import numpy as np
from skimage.io import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA
sys.path.insert(0, '../')
from PLDA import PLDA


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
    assert as_grey is True

    images = []
    labels = []
    for fname in os.listdir(directory):
        if fname.endswith('.jpg'):
            print('Loading: {}'.format(fname))
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
        raise ValueError('Dataset must be set to either ' +
                         '\"cafe\" or \"google\".')


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
            training_idxs = np.asarray(list(training_idxs)).astype(int)

            train_X = X[training_idxs, :]
            train_Y = [Y[idx] for idx in training_idxs]
            test_X = X[test_idxs, :]
            test_Y = [Y[idx] for idx in test_idxs]

            yield (train_X, train_Y), (test_X, test_Y)

        else:
            print('Warning: max_runs was not set to default, so estimate of' +
                  'performance will be more biased.')
            break


def predict_leave_n_out(X, Y, model_class, n=10, MAP_estimate=True,
                        max_runs='default'):
    predicted_ys = []
    actual_ys = []
    for (train_x, train_y), (test_x, test_y) in leave_n_out(X, Y, n):
        training_data = [(x, y) for (x, y) in zip(train_x, train_y)]
        model = model_class(training_data)

        predictions = model.predict_class(test_x, MAP_estimate=MAP_estimate)

        predicted_ys += [y for y in predictions]
        actual_ys += [y for y in test_y]

    return predicted_ys, actual_ys


def get_scores(predicted_Y, actual_Y, return_confusion_matrix=True):
    assert len(actual_Y) == len(predicted_Y)
    classes = set(actual_Y)
    n_total = len(actual_Y)
    model_predictions = {y: 0 for y in classes}
    matrix = {y: model_predictions.copy() for y in classes}

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
        confusion_matrix = np.zeros((len(classes), len(classes)))
        classes = list(classes)
        for j, class_j in enumerate(classes):
            for k, class_k in enumerate(classes):
                confusion_matrix[j, k] = matrix[class_j][class_k]

        confusion_matrix = confusion_matrix.T / confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix.T
        return score, (confusion_matrix, classes)


def remove_neutral_faces(images, labels):
    for i, label in enumerate(labels):
        if label == 'neutral':
            del images[i]
            del labels[i]

    return images, labels

def cv(X, Y, model_class, train_idxs, test_idxs):
    data = [(x, y) for (x, y) in zip(X, Y)]
    training_pairs = []
    test_imgs = []
    for i, (x, y) in enumerate(data):
        if i in train_idxs:
            training_pairs.append((x, y))
        if i in test_idxs:
            test_imgs.append(x)
    test_imgs = np.asarray(test_imgs)

    model = model_class(training_pairs)
    predictions = model.predict_class(test_imgs)

    return predictions
    


def cross_validate_plda():
    X = np.load('preprocessed.npy')
    Y = np.load('labels.npy')
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    test_idxs = list(idxs[:10])
    predictions_leave_none = []
    predictions_leave_one = []

    for test_idx in test_idxs:
        print(test_idx)
        predictions_leave_none += cv(X, Y, PLDA, idxs, [test_idx])
        leave_one_training_idxs = [idx for idx in idxs if idx != test_idx]
        predictions_leave_one += cv(X, Y, PLDA, leave_one_training_idxs, [test_idx])

    percent_correct = (np.asarray(predictions_leave_none) == \
                       np.asarray(predictions_leave_one)).sum() / len(test_idxs)


    print(percent_correct)
#    """ ---------- GOOGLE FACES DATASET --------- """
#    google_faces_dir = os.getcwd() + '/Google_Faces/'
#    X, Y = load_jpgs_and_lbls(google_faces_dir, dataset='google', as_grey=True)
#    X, Y = remove_neutral_faces(X, Y)
#    # Remove the 2 neutral images in the Google dataset.
#    X = preprocess_faces(X)
#
#    # Leave n=1 out.
#    predictions, true_labels = predict_leave_n_out(X, Y, PLDA, n=1)
#    scores, (confusion_matrix, classes) = get_scores(predictions, true_labels)
#
#    print(scores, confusion_matrix, classes)
#    # Leave 0 out.
#    training_data = [(x, y) for (x, y) in zip(X, Y)]
#    model = PLDA(training_data)
#    predictions = model.predict_class(X)
#    scores, (confusion_matrix, classes) = get_scores(predictions, Y)
#
    """ ---------- CAFE DATASET --------- """
#    cafe_faces_dir = os.getcwd() + '/../sessions_imgs/'
#    X, Y = load_jpgs_and_lbls(cafe_faces_dir, dataset='cafe', as_grey=True)
#    X = preprocess_faces(X)
#    X = np.load('preprocessed.npy')
#    Y = np.load('labels.npy')
#
#    overall_scores = []
#    confusion_matrices = []
#    class_lists = []
#    # Leave n=1 out.
#
#    # Leave 0 out.
#    print('Leave out number: 0')
#    training_data = [(x, y) for (x, y) in zip(X, Y)]
#    model = PLDA(training_data)
#    predictions = model.predict_class(X)
#    score, (confusion_matrix, classes) = get_scores(predictions, Y)
#
#    ns_leave_out = [1, 2, 5, 10, 25, 50, 100, 250, 500]
#    for number in ns_leave_out:
#        print('Leave {} out.'.format(number))
#        predictions, true_labels = predict_leave_n_out(X, Y, PLDA, n=number)
#        score, (confusion_matrix, classes) = get_scores(predictions, true_labels)
#        overall_scores.append(score)
#        confusion_matrices.append(confusion_matrix)
#        class_lists.append(classes)
#
#    return overall_scores, confusion_matrices, class_lists
