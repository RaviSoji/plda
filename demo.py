import os
import numpy as np
from PLDA import PLDA
from sklearn.decomposition import PCA


def gen_artificial_data(n_classes, n_list, n_dims):
    """ Generates a data set from multi-variate Gaussians. """
    assert len(n_list) == n_classes
    from numpy.random import multivariate_normal as m_normal
    from numpy.random import randint

    # Generate the within and between class covariance matrices.
    w_class_cov = 10
    bw_class_cov = randint(100, 10000, n_dims)

    bw_class_cov = np.diag(bw_class_cov)
    w_class_cov = np.eye(n_dims) * w_class_cov
     # LDA assumes within class covariance to be the same for all clusters,
     #  so do not generate a new one for each mean, unless you want to see
     #  how the model performs when the data does not meet LDA assumptions.

    # Generate cluster means.
    means = m_normal(np.zeros(n_dims), bw_class_cov, n_classes)

    # Generate points in the clustesr.
    points = []
    for x in range(len(n_list)):
        points.append(m_normal(means[x, :], w_class_cov, n_list[x]))
    points = np.vstack(points)

    # Generate labels.
    labels = []
    for x in range(len(means)):
        labels += [x] * n_list[x]

    return points, labels

def plot_model_results(PLDA_model):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_classes = len(model.stats)
    colors = cm.rainbow(np.linspace(0, 1, n_classes)
    for i in range(len(model.stats)):
        label = list(model.stats.keys())[i]
        x = model.data[label][:, 0]
        y = model.data[label][:, 1]
        c = colors[x]

        plt.scatter(x, y, color=c)
        

def main():
    n_classes = 5
    n_list = [10, 15, 20, 35, 40]
    n_dims = 2
    points, labels = gen_artificial_data(n_classes, n_list, n_dims)

    # Format points and their labels for PLDA model.
    data = []
    for x, y in zip(points, labels):
        data.append((x, y))

    # Build model.
    model = PLDA(data)

    #plot_results(model)
