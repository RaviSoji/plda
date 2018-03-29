# Probabilistic Linear Discriminant Analysis

### Disclaimer
This model was written for
 an [Explainable Artificial Intelligence (XAI) project](
     http://shaftolab.com/people.html), 
 so it keeps a bunch of parameters and
 data in memory that you will not need for simple classification and
 discrimination problems.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. ECCV 2006.](https://link.springer.com/chapter/10.1007/11744085_41)

### Model Dependencies
* Python3.5
* numpy 1.13.1
* scipy 0.19.1

### Demo Dependency
* matplotlib 2.0.2  (for Demo) 

### Preprocessing Dependencies (not necessary)
* sklearn 0.18.2  (for preprocess.py)
* It is important that data fed into the model has full rank covariance.
  You can check the data for this with `np.linalg.matrix_rank()` and
   `np.cov()`.

### Testing
First, cd to the 'plda.py' file.
``` shell
cd /Documents/folder/path/into/plda/ # README.md should be in this folder.
```

To run ALL tests:
``` shell
python3.5 -m unittest discover
```

To run only one of the test files in the tests folder, 
 run the corresponding line/command, below:
``` shell
python3.5 -m unittest tests.test_units_plda # Runs quickly.
python3.5 -m unittest tests.test_integration_plda  # Runs quickly.
python3.5 -m unittest tests.test_inference_plda 
# Takes a little under 3 minutes to run for me, even with ~60 CPU cores.

python3.5 -m unittest tests.test_units_discriminator.py  # Not implemented
python3.5 -m unittest tests.test_integration_discriminator.py  # Not implemented
python3.5 -m unittest tests.test_inference_discriminator.py  # Not implemented

python3.5 -m unittest tests.test_units_classifier.py  # Not implemented
python3.5 -m unittest tests.test_integration_classifier.py  # Not implemented
python3.5 -m unittest tests.test_inference_classifier.py  # Not implemented
```

## Classification Demos

### MNIST Handwritten Digits Data

### Artificial 2D Gaussian Data
See `plda/demos/gaussian_data/gaussian_demo.ipynb`.

Training data are randomly generated from 5 2D Gaussians with the same covariance matrix. Test data are generated from a uniform distribution, and the colors of those points represent model classifications. The contours depict 95% confidence intervals of the labeled training data. Note that the demo training data were generated with the same covariance because this a fundamental assumption of linear discriminant analysis models.

![Figure 1-1](/demos/gaussian_data/classification_demo.jpg?raw=True)
