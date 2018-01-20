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
from numpy.random import multivariate_normal as m_normal
sys.path.append(os.getcwd() + '/../')
import plda


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
    classifier.fit_model()
    predictions, log_probs = classifier.predict(test, standardize_data=True)

    print('Prediction\t{}'.format(['log_p ' + str(key)
                                   for key in classifier.model.data.keys()]))
    for pred, log_p in zip(predictions[:5], log_probs[:5]):
        print('{}\t{}'.format(pred, [np.around(p, 5) for p in log_p]))
