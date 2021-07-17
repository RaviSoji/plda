# Probabilistic Linear Discriminant Analysis

### Demo with MNIST Handwritten Digits Data

If you installed this package in a virtual environment, 
 remember to activate that virtual environment first.
Link: [mnist_demo/mnist_demo.ipynb](./mnist_demo/mnist_demo.ipynb).

## Installation

If you are new to programming, research, or sharing remote machines, 
 you will save yourself a lot of headache by installing the following software:
 [`git`](https://git-scm.com/downloads) and 
 [`conda`](https://github.com/conda/conda).

__Easy install__.

You can make a new conda environment called `myenv` with both this package and 
 its python dependencies automatically installed with the following steps. 

1. `cd` into your favorite directory.
2. `git clone https://github.com/RaviSoji/plda.git`
3. `conda env create -f plda/environment.yml -n myenv`

__Installing with `pip install`__.

1. `cd` into your favorite directory.
2. `git clone https://github.com/RaviSoji/plda.git`
3. If you have one, activate your virtual environment.
4. Run either `pip install plda/` or `pip install ./plda`.

__Installing using your own conda environment.yml file__.

1. Add the following to the end of your dependencies.
   Here is an example: [environment.yml](./environment.yml).
  ```
  - python>=3.5
  - numpy~=1.14.2
  - scipy~=1.0.1
  - scikit-learn~=0.19.1
  - pip=20.2.1
  - pip:
    - git+git://github.com/RaviSoji/plda@master
  ```

## Testing the software

See [tests/README.md](./tests/README.md).

## Credit and disclaimers

__Paper Citation__

[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. 
 In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. 
 ECCV 2006.](ioffe2006plda.pdf)

__More thanks__!

[@seandickert](https://github.com/seandickert) and 
 [@matiaslindgren](https://github.com/matiaslindgren) pushed for and 
 implemented the same-different discrimination and the pip install, 
 respectively!

__Disclaimers__

1. Parameters are estimated via empirical Bayes.
2. I wrote this code while working on an Explainable Artificial Intelligence 
    (XAI) project at the 
    [CoDaS Laboratory](http://shaftolab.com/people.html), 
    so it keeps parameters in memory that are unnecessary for simple 
    classification problems.
   It's intended to be readable to researchers.
