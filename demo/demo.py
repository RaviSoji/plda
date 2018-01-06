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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from numpy.random import multivariate_normal as m_normal
from scipy.stats import norm, chi2
sys.path.append(os.getcwd() + '/../')
import plda


# Define a bunch of helper plotting functions.
def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """ Code is slightly modified, but essentially borrowed from:
         https://stackoverflow.com/questions/18764814/make-contour-of-scatter
    """
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('Either `q` or `nsig` should be specified')

    r2 = chi2.ppf(q, 2)
    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_scatter(ax, x, y, s=5, c='black', label='', plot_training_cov=False,
                 model=None):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    if plot_training_cov is True:
        assert model is not None

    ax.scatter(x, y, c=c, s=s, label=label)

    if plot_training_cov is True:
        cov = model.data[label]['cov']
        mean_x, mean_y = model.data[label]['mean']
        w, h, deg = cov_ellipse(cov, nsig=2)
        ell = Ellipse(xy=(mean_x, mean_y),
                      width=w, height=h,
                      angle=deg, linewidth=2)
        ell.set_facecolor('none')
        ell.set_edgecolor('black')
        ax.add_patch(ell)
    ax.set_aspect('equal')

    return ax


def lbls_to_clrs(lbls, lbl_clr_pairs):
    assert len(np.unique(lbls)) == len(lbl_clr_pairs)
    assert isinstance(lbls, np.ndarray)
    assert isinstance(lbl_clr_pairs, list)
    assert isinstance(lbl_clr_pairs[0], tuple)

    colors = np.empty((*lbls.shape, lbl_clr_pairs[0][1].shape[0]),
                      dtype=lbl_clr_pairs[0][1].dtype)
    for lbl, clr in lbl_clr_pairs:
        colors[lbls == lbl] = clr

    return colors


def gen_training_set(n_gaussians, sample_size, n_dims):
    cov = np.random.randint(-10, 10, (n_dims, n_dims))
    cov = np.matmul(cov, cov.T) + np.eye(n_dims) * np.random.rand(n_dims)

    pts = np.vstack([m_normal(np.ones(n_dims) *
                              np.random.randint(-100, 100, 1),
                              cov, sample_size)
                     for x in range(n_gaussians)])
    lbls = np.hstack([['gaussian_{}'.format(x)] * sample_size
                      for x in range(n_gaussians)])

    return pts, lbls

if __name__ == '__main__':
    n_gaussians = 5
    sample_size = 100
    n_dims = 2
    n_test = 5000

    # Initialize training and test data.
    np.random.seed(0)
    train_X, train_Y = gen_training_set(n_gaussians, sample_size, n_dims)

    margin = np.sqrt(np.cov(train_X.T).diagonal().sum()) * .1
    (min_x, min_y) = np.min(train_X, axis=0) - margin
    (max_x, max_y) = np.max(train_X, axis=0) + margin
    test = np.asarray([np.random.uniform(min_x, max_x, n_test),
                       np.random.uniform(min_y, max_y, n_test)]).T

    # Use plda to classify test data.
    classifier = plda.Classifier(train_X, train_Y)
    predictions, log_probs = classifier.predict(test, standardize_data=True)

    # Plot classified data.
    colors = cm.rainbow(np.linspace(0, 1, n_gaussians))
    unique = np.unique(predictions)
    c = lbls_to_clrs(predictions, [pair for pair in zip(unique, colors)])

    fig, ax_arr = plt.subplots(1)
    for label in unique:
        idxs = predictions == label
        plot_scatter(ax_arr, test[idxs, 0], test[idxs, 1],
                     label=label, c=c[idxs, :], plot_training_cov=True,
                     model=model)

    (min_x, min_y), (max_x, max_y) = np.min(test, axis=0), np.max(test, axis=0)
    ax_arr.set_xlim(min_x, max_x)
    ax_arr.set_ylim(min_y, max_y)
    fig.set_size_inches(10, 10)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Demo', fontsize=20)

    plt.show()
