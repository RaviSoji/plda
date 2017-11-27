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


def load_jpgs_and_lbls(directory, dataset, as_grey=True, return_fnames=False):
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
    fnames = []
    for fname in os.listdir(directory):
        if fname.endswith('.jpg'):
            print('Loading: {}'.format(fname))
            img = imread(directory + '/' + fname, as_grey=as_grey)
            images.append(img)
            label = get_label(fname, dataset=dataset)
            labels.append(label)
            fnames.append(fname)

    if return_fnames is True:
        return np.asarray(images), np.asarray(labels), np.asarray(fnames)
    elif return_fnames is False:
        return np.asarray(images), np.asarray(labels)
    else:
        raise ValueError


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


def fnames_to_idxs(fname_ndarray, unique_fnames=None):
    """
    ARGUMENTS
     fname_ndarray  (ndarray): Contains filenames that may or may not be
                                repeated.
     unique_fnames  (ndarray): Must contain every filename in fname_ndarray
                                and possibly others. Each filename should
                                only occur once.
                      Example: np.unique(fname_ndarray)
    RETURNS
     idx_array      (ndarray): Has same shape as fname_ndarray, but with
                                integer entries. These integers index the
                                filenames in unique_fnames.
     unique_frames  (ndarray): The unique filenames that were passed in as
                                an argument. If set to None, then this is set
                                to np.unique(fname_ndarray).
    """
    if unique_fnames is not None:
        if not isinstance(unique_fnames, np.ndarray):
            unique_fnames = np.asarray(unique_fnames)
    else:
        unique_fnames = np.unique(fname_ndarray)

    idx_array = np.zeros(np.asarray(fname_ndarray).shape)
    for idx, fname in np.ndenumerate(fname_ndarray):
        idx_array[idx] = np.argwhere(unique_fnames == fname)

    return idx_array.astype(int), unique_fnames


def preprocess_cafe(dir_raw_cafe_faces, test_fnames_path=None):
    imgs, lbls, fnames = load_jpgs_and_lbls(dir_raw_cafe_faces, dataset='cafe',
                                            return_fnames=True)
    if test_fnames_path is not None:
        test_fnames = np.load(test_fnames_path)
    else:
        test_fnames = []
    
    preprocessed = preprocess_faces(imgs, n_components='default',
                                    resize_shape=(400, 400))

    idxs = np.arange(len(imgs))
    test_idxs, _  = fnames_to_idxs(test_fnames, fnames)
    test_idxs = np.unique(test_idxs)
    train_idxs = np.asarray(list(set(idxs) - set(test_idxs)))

    test_X = preprocessed[test_idxs, :]
    test_Y = lbls[test_idxs]
    test_fnames = fnames[test_idxs]

    train_X = preprocessed[train_idxs, :]
    train_Y = lbls[train_idxs]
    train_fnames = fnames[train_idxs]

    return (train_X, train_Y, train_fnames), (test_X, test_Y, test_fnames)
