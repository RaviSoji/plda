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
import pandas as pd
from skimage.io import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA
sys.path.insert(0, '../')


def get_verbose_print_function(v):
    assert isinstance(v, bool)

    func = print if v is True else lambda *a, **k: None

    return func


def get_resize_function(shape):
    if isinstance(shape, tuple):
        assert isinstance(shape[0], int)
        assert isinstance(shape[1], int)
        assert len(shape) == 2
        
        func = lambda img: imresize(img, shape)
        
    elif shape is None:
        func = lambda img: img
        
    else:
        raise ValueError

    return func


def read_cafe(directory, resize_shape=None, verbose=False):
    """ Reads the Child Affective Facial Expressions dataset as gray images.

    DESCRIPTION
     Grayscaling weights: 0.2125 RED + 0.7154 GREEN + 0.0721 BLUE.
     This function assumes the filenames begin with the emtion name
      when obtaining the image label (first 3 characters of the filename).

     RGB > Grayscale > Resize (optional)

    ARGUMENTS
     directory      (str): Directory containing the files to be read.
     resize_shape   (int): Optional parameter for reshaping as images are read.
                            This is useful when images are large and/or
                            when the dataset is large.
     verbose       (bool): Whether or not to print file names as they are read.

    RETURNS
     images      (list): List of 2D ndarrays (grayscaled images).
     labels      (list): List of strings in the same order as images.
     fnames      (list): List of string file names in the same order as images.
    """
    assert isinstance(directory, str)
    assert directory.endswith('/')

    verbose_print = get_verbose_print_function(verbose)
    resize = get_resize_function(resize_shape)

    images = []
    labels = []
    fnames = []
    verbose_message = 'Loading: {}'
    for fname in os.listdir(directory):
        if fname.endswith('.jpg'):
            verbose_print(verbose_message.format(fname))

            img = imread(directory + fname, as_grey=True)
            images.append(resize(img))

            label = fname[:3]
            labels.append(label)

            fnames.append(fname)

    return images, labels, fnames


def preprocess_faces(images, resize_shape=None,
                     n_components_pca=None, pct_variance_explained_pca=None,
                     verbose=False):
    """ Grayscale > resize (optional) > flatten > PCA (optional). """
    for img in images:
        assert img.shape[0] == img.shape[1]

    verbose_print = get_verbose_print_function(v=verbose)
    resize = get_resize_function(resize_shape)

    preprocessed_images = []
    verbose_message = 'Preparing the {}th image for PCA.'
    for i, img in enumerate(images):
        prepped_img = resize(img).astype(float)
        prepped_img = prepped_img.flatten()
        preprocessed_images.append(prepped_img)

        verbose_print(verbose_message.format(i))

    if n_components_pca is None and pct_variance_explained_pca is None:
        return np.asarray(preprocessed_images)
    else:
        return get_principal_components(preprocessed_images,
                                        n_components_pca,
                                        pct_variance_explained_pca)
        

def get_principal_components(flattened_images, n_components=None,
                             pct_variance_explained=None):
    """ Gets the principal components of flattened_images.

    DESCRIPTION: Choose either the number of components to use or the
                  lower bound on the percent variance to be captured.
    
    ARGUMENTS
     preprocessed_imgs        (list): Must be a list of 1D ndarrays.
     n_components_pca          (int): Number of pca components to use.
     pct_variance_explained  (float): Must be in the half-open interval (0, 1].

    RETURNS
     principal_components  (ndarray): shape=(len(flattened_imgs), n_components)
    """
    for img in flattened_images:
        assert isinstance(img, np.ndarray)
        assert img.shape == flattened_images[-1].shape
        assert len(img.shape) == 1
    assert pct_variance_explained is not None or n_components is not None

    X = np.asarray(flattened_images)
    X -= X.mean(axis=0)  # Center all of the data around the origin.
    X /= np.std(X, axis=0)  # Standardize all the variables/features/columns.

    pca = PCA()
    pca.fit(X)
    sorted_eig_vals = pca.explained_variance_
    cum_pct_variance = (sorted_eig_vals / sorted_eig_vals.sum()).cumsum()

    if pct_variance_explained is not None:
        idxs = np.argwhere(cum_pct_variance >= pct_variance_explained)
        n_components = np.squeeze(idxs)[0] + 1  # Add 1 to ensure >=.

    V = pca.components_[:n_components, :].T
    principal_components = np.matmul(X, V)

    return principal_components, cum_pct_variance[:n_components]

def main():
    # Preprocess only the mouth open images for the experiment.
    imgs, lbls, fnames = read_cafe('cafe_faces/mouth_open/',
                                   resize_shape=(400, 400), verbose=True)
    imgs, lbls, fnames = np.asarray(imgs), np.asarray(lbls), np.asarray(fnames)
    # 50 principal components explain >79.2% of the variance.
    pc, cum_pct_var = preprocess_faces(imgs, n_components_pca=50, verbose=True)
