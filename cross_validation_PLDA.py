import os
import numpy as np
from PLDA import PLDA
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA


def build_google_faces_dataset(load_dir, resized_shape=(100,100)):
    data_shape=resized_shape
    imgs = []
    lbls = []
    for fname in os.listdir(load_dir):
        if fname[-4:] == '.jpg':
            label = ''.join([i for i in fname if not i.isdigit()])
            label = label[:-4]
            if label != 'neutral':
                img = imread(load_dir + fname)
                img = resize(img, data_shape)
                img = img.flatten()
                imgs.append(img)
                lbls.append(label)

    return imgs, lbls

def k_folds_CV_PLDA(X, Y, k=5, MAP_estimate=True):
    """ Cross validation on Google Face emotions dataset, using k-folds.

    DESCRIPTION: If MAP_estimate is set to False, n_runs number of runs
                  are run for each of the k-folds runs as well. This function
                  runs several iterations of k-folders such that the k sets
                  of data are randomly selected.
    ARGUMENTS
     X         (ndarray): Rows of data examaples. [n x n_dims]
     Y         (ndarray): Labels corresponding to the rows of X. [n_dims x 1]
     k             (int): k in k-folds. That is, the number of subsets to
                           to break up the data into. During each run, all but
                           one set is used as training data, and the remainder
                           is used to test the model's performance. Number
                           of data in the test set is int(n / k).
     MAP_estimate (bool): True makes the PLDA model use the MAP estimate for
                           prediction. If this is set to false, the model
                           makes classifications probabilitically and as such
                           several runs are made for each k-fold.
    RETURNS
     None

    """
    n = X.shape[0]  # Total number of data.
    assert k <= n
    n_test = int(n / k)  # Number of data to reserve for testing.
    n_runs = 1  
     # Number of probabilistic runs over which to average performance.

    scores = []
    if n % k != 0:
        k_runs = k - 1
    else:
        k_runs = k

    for k_run in range(k_runs):
        # Create training and test data sets.
        idxs = np.arange(n)
        np.random.shuffle(idxs)

        start =  k_run * n_test
        end = k_run * n_test + n_test

        test_idxs = idxs[start:end]
        test_X = X[test_idxs,:]
        test_Y = Y[test_idxs]

        data_idxs = set(idxs) - set(test_idxs)
        data_idxs = np.array(list(data_idxs))
        data_X = X[data_idxs, :]
        data_Y = Y[data_idxs]
    
    
        # Format the data and fit the model.
        data = []
        for x, y in zip(data_X, data_Y):
            data.append((x, y))
    
        model = PLDA(data)
    
        # Evaluate the model.
        for run in range(n_runs):
            predictions = model.predict_class(test_X,
                                              MAP_estimate=MAP_estimate)
            correct = 0
            for classification, label in zip(predictions, test_Y):
                correct += classification == label
            scores.append(correct / len(test_Y))
    
    scores = np.array(scores)
    print('Scores: {}'.format(scores))
    print('Mean score: {}'.format(scores.mean()))
    print('Standard deviation: {}'.format(np.std(scores)))

def main():
    load_dir = os.getcwd() + '/Google_Faces/'
    imgs, lbls = build_google_faces_dataset(load_dir)

    X = np.array(imgs)
    Y = np.array(lbls)
    del imgs, lbls

    standardized_X = X - X.mean(axis=0)
    standardized_X = standardized_X / np.std(standardized_X, axis=0)

    pca = PCA()
    pca.fit(standardized_X)
    V = pca.components_[:175, :].T
    pca_standardized_X = np.matmul(standardized_X, V)

    k_folds_CV_PLDA(pca_standardized_X, Y, k=50, MAP_estimate=False)
