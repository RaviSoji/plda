# Probabilistic Linear Discriminant Analysis

__Paper Citation__

[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. 
 In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. 
 ECCV 2006.](
 https://link.springer.com/chapter/10.1007/11744085_41)

__Disclaimers__

1. Parameters are estimated via empirical Bayes.
2. This code was originally written for an Explainable Artificial Intelligence 
    (XAI) project at the [CoDaS Laboratory](http://shaftolab.com/people.html), 
    so it keeps parameters in memory that are unnecessary for simple 
    classification problems.

__Thanks__!

Special thanks to 
 [@seandickert](https://github.com/seandickert) and 
 [@matiaslindgren](https://github.com/matiaslindgren) for pushing for and 
 implementing the same-different discrimination and the pip install, 
 respectively!

## Usage Demo with MNIST Handwritten Digits Data

Outline for [mnist_demo/mnist_demo.ipynb](./mnist_demo/mnist_demo.ipynb).

0. If you installed this package in a virtual environment, 
    activate that virtual environment first.
1. Import plda and other convenient packages.
2. Load data.
3. Preprocess data and fit model.
4. __How to classify datapoints: Overfit classifier__.
5. __How to classify datapoints: Better-fit classifier__.
6. Extracting LDA-features.
7. __How to classify datapoints: "same-or-different category" discrimination__.
8. Extracting preprocessing information.
9. Extracting model parameters.

## Dependencies

If you are new to code or research in general,
 check out the following download and installation instructions: 
 [`git`](https://git-scm.com/downloads) and 
 [`conda`](https://github.com/conda/conda).

__To add this repository as a dependency in your own conda environment 
 `yml` file__, 
 add the following to the end of your dependencies
 (e.g. this repository's [environment.yml](./environment.yml) file).
  ```
  - python>=3.5
  - numpy~=1.14.2
  - scipy~=1.0.1
  - scikit-learn~=0.19.1
  - pip=20.2.1
  - pip:
    - git+git://github.com/RaviSoji/plda@master
  ```

__Alternatively, you can install dependencies in 3 steps using Terminal__.

1. `cd` into your favorite directory.
2. `git clone https://github.com/RaviSoji/plda.git`

Now, you have 3 options.
- To globally install on your machine: 
   `pip install plda`.
  This is easiest, but it is bad practice.
- To install within an existing virtual environment named `myenv` that has 
   `pip` installed,
   activate that virtual environment first (e.g. `conda activate myenv`), 
   and then run `pip install plda`.
- To automatically create a conda environment named `myenv`, 
   with this package and its dependencies,
   run `conda env create -f plda/environment.yml -n myenv`.

## Testing

If you installed this package in an environment, activate it.
For example, if you created the Conda environment with the name `myenv`, 
 activate it with the following.
``` shell
conda activate myenv
```

To run all tests (~120 seconds with ~60 CPU cores), use the following.
``` shell
pytest plda/  # README.md should be in this directory.
```

To run a particular test file, run one of the following.
``` shell
pytest plda/tests/test_model/test_model_units.py  # ~.66s for me.
pytest plda/tests/test_model/test_model_integration.py  # ~1.0s for me.
pytest plda/tests/test_model/test_model_inference.py  #  ~80.6s for me.

pytest plda/tests/test_optimizer/test_optimizer_units.py  # ~.59s for me.
pytest plda/tests/test_optimizer/test_optimizer_integration.py  # ~.78s.
pytest plda/tests/test_optimizer/test_optimizer_inference.py  # ~25.3s for me.

pytest plda/tests/test_classifier/test_classifier_integration.py  # ~.69s.
```

Once you finish running the tests, 
 remove all the `__pycache__/` folders generated by pytest with the following.
``` shell
py3clean plda/*  # This README.md should be in here.
```

Finally, if you are done working with the model and test code, 
 deactivate the Conda environment.
``` shell
conda deactivate  # You can run this from any directory.
```
