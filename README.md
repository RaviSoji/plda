# Probabilistic Linear Discriminant Analysis

### Disclaimer
This model was written for
 an [Explainable Artificial Intelligence (XAI) project](
     http://shaftolab.com/people.html), 
 so it stores a bunch of parameters and data in memory that 
 are not necessary for simple classification and discrimination problems.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. 
 In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. 
 ECCV 2006.](
 https://link.springer.com/chapter/10.1007/11744085_41)

### Dependencies
If you already have Anaconda or Miniconda installed,
 you can automatically download all dependencies to a conda environment 
 called `plda` with the following. 

```conda env create -f environment.yml -n plda```

__Model dependencies__
- Python3.5
- numpy 1.13.1
- scipy 0.19.1

__Demo Dependency__
- matplotlib 2.0.2  (for demos) 

__Preprocessing Dependency__
- sklearn 0.18.2
  - See the MNIST tutorial for an example: 
     `demos/mnist_data/mnist_demo.ipynb`.
- Data fed into the model must have full rank covariance because of the
   implemented optimization algorithm.
  You can check the data for this with `np.linalg.matrix_rank()` and
   `np.cov()`.
- If your data does not have full rank covariance, 
   you will need to preprocess your data with Principal Components Analysis
   to obtain the linearly independent principal components. 
  If you are not sure how to do this, check out my MNIST demo: 
   `demos/mnist_data/mnist_demo.ipynb`.

### Testing
Note that I changed the model interface, 
 so the tests below will probably fail until I get a chance to update them.

After cloning this repository, cd into it.
``` shell
cd plda/ # README.md and LICENSE should be in this folder.
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
See [demos/mnist_data/mnist_demo.ipynb](
     ./demos/mnist_data/mnist_demo.ipynb).
- This demo will show you how to preprocess your data so the model's
   optimization algorithm can run.

### Visualization of classification in 2D space.
See [demos/gaussian_data/gaussian_demo.ipynb](
     ./demos/gaussian_data/gaussian_demo.ipynb).

Training data are randomly generated from 5 2D Gaussians with 
 the same covariance (a fundamental assumption of
 probabilistic linear discriminant analysis).
Test data are generated from a uniform distribution, 
 and the colors of those points represent model classifications; 
 the contours depict 95% confidence intervals based on the original
 training data (not model certainty).

![Figure 1-1](/demos/gaussian_data/classification_demo.jpg?raw=True)
