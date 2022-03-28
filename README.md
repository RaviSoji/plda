# Probabilistic Linear Discriminant Analysis

## Demo with MNIST Handwritten Digits Data

<div align="center">
<a href="https://colab.research.google.com/github/RaviSoji/plda/blob/master/mnist_demo/mnist_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Demo"/>
</a>
</div>

See [mnist_demo/mnist_demo.ipynb](./mnist_demo/mnist_demo.ipynb).

## Install instructions

__Option 1: `pip install` without dependencies__.
Use this after installing necessary [dependencies](./environment.yml).

```bash
pip install https://github.com/RaviSoji/plda/tarball/master
```

__Option 2: `conda install` with all dependencies__.
This requires [`conda`](https://github.com/conda/conda).

- Via [`git`](https://git-scm.com/downloads):

    ```bash
    git clone https://github.com/RaviSoji/plda.git
    conda env create -f plda/environment.yml -n myenv
    ```

- Alternatively, via `wget`:

    ```bash
    wget https://raw.githubusercontent.com/RaviSoji/plda/master/environment.yml
    conda env create -f environment.yml -n myenv
    ```

## Uninstall instructions

- To uninstall `plda` only: `pip uninstall plda`.
- To remove the `myenv` conda environment: `conda env remove -n myenv`.

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
