# Probabilistic Linear Discriminant Analysis

__Paper Citation__

[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. 
 In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. 
 ECCV 2006.](ioffe2006plda.pdf)

__Disclaimers__

1. Parameters are estimated via empirical Bayes.
2. I wrote this code while working on an Explainable Artificial Intelligence 
    (XAI) project at the 
    [CoDaS Laboratory](http://shaftolab.com/people.html), 
    so it keeps parameters in memory that are unnecessary for simple 
    classification problems.
   It's intended to be readable to researchers.

__Thanks__!

[@seandickert](https://github.com/seandickert) and 
 [@matiaslindgren](https://github.com/matiaslindgren) pushed for and 
 implemented the same-different discrimination and the pip install, 
 respectively!

## Demo with MNIST Handwritten Digits Data: [mnist_demo/mnist_demo.ipynb](./mnist_demo/mnist_demo.ipynb).

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

## Installation

If you are new to programming, research, or sharing remote machines, 
 you will save yourself a lot of headache by installing the following software:
 [`git`](https://git-scm.com/downloads) and 
 [`conda`](https://github.com/conda/conda).

__Installing with `pip install`__

1. `cd` into your favorite directory.
2. `git clone https://github.com/RaviSoji/plda.git`
3. Activate your virtual environmental if you have one.
   E.g. `conda activate myenv`.
   If you want to make one called `myenv` with the python dependencies, 
    run `conda env create -f plda/environment.yml -n myenv`, 
    and then run `conda activate myenv`.
4. Run either `pip install plda/` or `pip install ./plda`. Either will work.

__Installing using your own conda environment.yml file.__

__To add this repository as a dependency in your own conda environment 
 `yml` file__, 
 add the following to the end of your dependencies.
  ```
  - python>=3.5
  - numpy~=1.14.2
  - scipy~=1.0.1
  - scikit-learn~=0.19.1
  - pip=20.2.1
  - pip:
    - git+git://github.com/RaviSoji/plda@master
  ```
Here is an example: [environment.yml](./environment.yml) file).

## Tests

See [./tests/README.md](./tests/README.md)
