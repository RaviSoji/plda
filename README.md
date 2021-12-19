# Probabilistic Linear Discriminant Analysis

## Install instructions

__Option 1: `pip install` without dependencies__.

Use the following if you already have the necessary 
 [dependencies](./environment.yml).

```bash
pip install https://github.com/RaviSoji/plda/tarball/master
```

__Option 2: Easy install with all dependencies__.

If you are new to programming, research, or sharing remote machines, 
 you will save yourself a lot of headache by installing the following software:
 [`git`](https://git-scm.com/downloads) and 
 [`conda`](https://github.com/conda/conda).

With the following, 
 you will install this package and its dependencies in a conda 
 environment called `myenv`.

1. `cd` into your favorite directory.
2. `git clone https://github.com/RaviSoji/plda.git`
3. `conda env create -f plda/environment.yml -n myenv`

## Uninstall instructions

- To uninstall `plda` only: `pip uninstall plda`.
- To remove the `myenv` conda environment:`conda env remove -n myenv`.

## Demo with MNIST Handwritten Digits Data

See [mnist_demo/mnist_demo.ipynb](./mnist_demo/mnist_demo.ipynb).

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
