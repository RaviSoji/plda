import os
import numpy as np
from PLDA import PLDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import multivariate_normal as m_normal
from numpy.random import randint
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse


def gen_artificial_data(n_classes, n_list, n_dims):
    """ Generates a data set from multi-variate Gaussians. """
    assert len(n_list) == n_classes
    np.random.seed(1)  # Ensures that the same plot is generated every time.

    # Generate the within and between class covariance matrices.
    w_class_cov = np.random.randint(-10, 10, n_dims ** 2)
    w_class_cov = w_class_cov.reshape(n_dims, n_dims)
    w_class_cov = np.matmul(w_class_cov, w_class_cov.T)  # Make symmetric.
    print(w_class_cov)
    bw_class_cov = randint(100, 10000, n_dims)

    bw_class_cov = np.diag(bw_class_cov)
    w_class_cov = np.eye(n_dims) * w_class_cov
     # LDA assumes within class covariance to be the same for all clusters,
     #  so do not generate a new one for each mean, unless you want to see
     #  how the model performs when the data does not meet LDA assumptions.

    # Generate cluster means.
    means = m_normal(np.zeros(n_dims), bw_class_cov, n_classes)

    # Generate points inside each of the clusters.
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

    n_classes = len(PLDA_model.stats)
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    for i in range(len(PLDA_model.stats)):
        label = list(PLDA_model.stats.keys())[i]
        data = np.array(PLDA_model.data[label])
        x = data[:, 0]
        y = data[:, 1]
        c = colors[i]

        ax.scatter(x, y, color=c, s=.5)
    return ax

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """ Code is slightly modified, essentially borrowed from: 
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

def plot_contours(PLDA_model, nsig):
    """ Code is modified, but essentially borrowed from:
         https://stackoverflow.com/questions/12301071/
                  multidimensional-confidence-intervals
    """
    ells = []
    for label in PLDA_model.stats.keys():
        mean = PLDA_model.stats[label]['μ']
        cov = PLDA_model.stats[label]['covariance']
        w, h, t = cov_ellipse(cov, nsig=2)
        ells.append(Ellipse(mean, width=w, height=h, angle=t))
    fig = plt.figure(0)
    ax1 = fig.add_subplot(111)
    for e in ells: 
        ax1.add_artist(e)
        e.set_facecolor('none')
    #ax1.set_xlim(-100, 100)
    #ax1.set_ylim(-100, 100)

    return ax1

def main():
    # (1) Generate Artifical data.
    n_classes = 10  # Number of clusters (i.e. multivariate Gaussians)
    n_list = [300 * (x % 3 + 1) for x in range(1, n_classes + 1)]  # Sample size
    n_dims = 2  # Dimensionality of the data.

    points, labels = gen_artificial_data(n_classes, n_list, n_dims)

    # (2) Format data for PLDA model.
    data = []
    for x, y in zip(points, labels):
        data.append((x, y))

    # (3) Build model.
    model = PLDA(data)

    # (4) Plot 95% CI level contours of Gaussians fit to the TRAINING data
    #      and color TEST points based on the model's prediction. If the model
    #      is accurate, points inside the counters should be colored the same
    #      as long as there isn't much overlap between the distributions (i.e.
    #      countours).
    ax = plot_contours(model, nsig=2)
    plot_model_results(model, ax, MAP_estimate=True)
    plt.show()