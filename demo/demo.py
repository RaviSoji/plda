import os
import sys
sys.path.append(os.getcwd() + '/../')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.random import randint
from numpy.random import multivariate_normal as m_normal
from PLDA import PLDA
from scipy.stats import norm, chi2


def gen_artificial_data(n_classes, n_list, n_dims):
    """ Generates a data set from multi-variate Gaussians. """
    assert len(n_list) == n_classes
    np.random.seed(1)  # Ensures that the same plot is generated every time.

    # Generate the within and between class covariance matrices.
    w_class_cov = np.random.randint(-10, 10, n_dims ** 2)
    w_class_cov = w_class_cov.reshape(n_dims, n_dims)
    w_class_cov = np.matmul(w_class_cov, w_class_cov.T)  # Make symmetric.
    
    bw_class_cov = randint(100, 10000, n_dims)
    bw_class_cov = np.diag(bw_class_cov)
     # LDA assumes within class covariance to be the same for all clusters,
     #  so do not generate a new one for each mean, unless you want to see
     #  how the model performs when the data does not meet LDA assumptions.

    # Generate cluster means.
    means = m_normal(np.zeros(n_dims), bw_class_cov, n_classes)

    # Generate points from each of the clusters.
    points = []
    for x in range(len(n_list)):
        points.append(m_normal(means[x, :], w_class_cov, n_list[x]))
    points = np.vstack(points)

    # Generate labels.
    labels = []
    for x in range(len(means)):
        labels += [x] * n_list[x]

    return points, labels

def plot_model_results(PLDA_model, ax, MAP_estimate=True):
    n_test = 50000  # Number of test data.

    data = np.array([[x, y] for ((x, y), label) in PLDA_model.raw_data])
    std = np.std(data, axis=0)
    x = data[:, 0]
    y = data[:, 1]
    x_min, x_max = np.amin(x) - .25 * std[0], np.amax(x) + .25 * std[0]
    y_min, y_max = np.amin(y) - .25 * std[1], np.amax(y) + .25 * std[1]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    n_classes = len(PLDA_model.stats)
    colors = cm.rainbow(np.linspace(0, 1, n_classes))

    test_x = np.random.randint(x_min, x_max, n_test)
    test_y = np.random.randint(y_min, y_max, n_test)
    classifications = PLDA_model.predict_class(np.array([test_x, test_y]).T,
                                               MAP_estimate=MAP_estimate)

    idxs = np.argsort(np.array(classifications))
    classifications = np.array(classifications)[idxs]
    test_x = test_x[idxs]
    test_y = test_y[idxs]
    ax.scatter(test_x, test_y, color=colors[classifications], s=.5)

    return ax

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

def plot_contours(PLDA_model, ax, nsig):
    """ Plots contour of the 95% CI of each multivariate Gaussian in the model.
    """
    ells = []
    for label in PLDA_model.stats.keys():
        mean = PLDA_model.stats[label]['μ']
        cov = PLDA_model.stats[label]['covariance']
        w, h, t = cov_ellipse(cov, nsig=2)
        ells.append(Ellipse(mean, width=w, height=h, angle=t))

    for e in ells: 
        ax.add_artist(e)
        e.set_facecolor('none')

    return ax1

def main():
    n_classes = 10  # Number of clusters (i.e. multivariate Gaussians)
    n_list = [300 * (x % 3 + 1) for x in range(1, n_classes + 1)]  # Sample size
    n_dims = 2  # Dimensionality of the data.

    points, labels = gen_artificial_data(n_classes, n_list, n_dims)

    data = []
    for x, y in zip(points, labels):
        data.append((x, y))

    model = PLDA(data, save_raw=True)

    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax = plot_contours(model, ax, nsig=2)
    plot_model_results(model, ax, MAP_estimate=True)
    plt.show()
    
    # (4) Plots 95% CI level contours of Gaussians fit to the TRAINING data.
    #      Colors of points represent model classifications. If the model
    #      is accurate, points inside the counters should be colored the same
    #      as long as there isn't much overlap between the distributions (i.e.
    #      countours).
