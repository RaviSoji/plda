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


def load_imgs_and_labels(directory, as_grey=True):
    """ Reads '.jpg' files from a directory as grayscaled images.

     DESCRIPTION
     Grayscaling weights: 0.2125 RED + 0.7154 GREEN + 0.0721 BLUE

    ARGUMENT
    directory  (string): Directory containing the files to be read.

    RETURN
    images      (list): List of 2D ndarrays (grayscaled images).
    """
    images = []
    labels = []
    for fname in os.listdir(directory):
        if fname.endswith('.jpg'):
            img = imread(directory + '/' + fname, as_grey=as_grey)
            images.append(img)
            labels.append(fname[:3])
    return images, labels

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

def preprocess_cafe_faces(images, n_components='default', resize_shape='default'):
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
    preprocessed_images = get_principal_components(preprocessed_images, n_components)

    return preprocessed_images

def preprocess_google_faces(images, reszie_shape='default'):
    """
    """
    raise NotImplementedError

def main():
    img_dir = ''
    X, Y = load_imgs_and_labels(img_dir, as_grey=True)
    X = preprocess_cafe_faces(X)
    
