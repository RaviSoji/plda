# Probabilistic Linear Discriminant Analysis

### Disclaimer
This model was written for
 an [Explainable Artificial Intelligence (XAI) project](
     http://shaftolab.com/people.html), 
 so it stores a bunch of parameters in memory that 
 are not necessary for simple classification problems.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. 
 In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. 
 ECCV 2006.](
 https://link.springer.com/chapter/10.1007/11744085_41)

### Dependencies
If you already have 
 [Anaconda or Miniconda](
  https://conda.io/docs/user-guide/install/index.html),
 you can automatically download all dependencies to a conda environment 
 called `plda` with the following. 

```conda env create -f environment.yml -n plda```

### Testing
__Running all tests__ (~120 seconds with ~60 CPU cores)

1. If you created the Conda environment with the name `plda`, 
    activate it with the following.
   ``` shell
   source activate plda
   ```

2. Run all the tests with the following.
   ``` shell
   python3.6 pytest plda/tests/
   ```

3. Clean up the `__pycache__/` folders with the following.
   ``` shell
   py3clean plda/
   ```

__Running a particular test file__
Follow steps 1-3 from above, replacing step 2 with one of the following.

``` shell
pytest plda/tests/test_model/test_model_units.py  # ~.66s for me.
pytest plda/tests/test_model/test_model_integration.py  # ~1.0s for me.
pytest plda/tests/test_model/test_model_inference.py  #  ~80.6s for me.

pytest plda/tests/test_optimizer/test_optimizer_units.pa  # ~.59s for me..
pytest plda/tests/test_optimizer/test_optimizer_integration.py  # ~.78s.
pytest plda/tests/test_optimizer/test_optimizer_inference.py  # 25.3s for me.

python3.6 plda/tests/test_classifier/test_classifier_integration.py  #.69s.
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
