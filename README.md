# probabilistic_LDA 

### Disclaimer
UPDATE: I've merged all the updates in the revamp branch, but I still haven't had time to rewrite test_inference.py. Check back Dec. 15, 2017.

If all that you want to do is apply this model to a dataset and measure its
 performance, you will find the code to be bulky in that it stores parameters,
 variables, and data that you do not need. This code was written for
 an [Explainable Artificial Intelligence (XAI) project](http://shaftolab.com/people.html)
 (and [machine teaching](http://shaftolab.com/publications.html), in particular), whose
 objective is model explainability (not necessarily speed or performance),
 which requires saving such information.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. ECCV 2006.](https://link.springer.com/chapter/10.1007/11744085_41)

### Model Dependencies
* numpy 1.13.1
* scipy 0.19.1

### Cross-validation and Demo Dependencies
* matplotlib 2.0.2  (for Demo) 

### Preprocessing Dependencies (not necessary)
* For convenience, I supply the preprocessed data in the cross_validation directory.
* skimage 0.13.0  (for preprocess.py)
* sklearn 0.18.2  (for preprocess.py)

### Testing
First, cd to the 'plda.py' file.
```
cd /Documents/folder/path/to/probabilistic_LDA/ # PLDA.py should be in this folder.
```

To run ALL tests:
```
python3.5 -m unittest discover
```

To run only one of the three test files in the tests folder, run the corresponding line/command, below:
```
python3.5 -m unittest tests.test_units_plda # Runs quickly.
python3.5 -m unittest tests.test_integration_plda  # Runs quickly.
python3.5 -m unittest tests.test_inference_plda  # Takes a little under 3 minutes to run for me, even with ~60 CPU cores.

python3.5 -m unittest tests.test_units_discriminator.py
python3.5 -m unittest tests.test_integration_discriminator.py
python3.5 -m unittest tests.test_inference_discriminator.py

python3.5 -m unittest tests.test_units_classifier.py
python3.5 -m unittest tests.test_integration_classifier.py
python3.5 -m unittest tests.test_inference_classifier.py
```

### Demo with Artificial 2D Gaussian Data
The demo folder contains (1) a demo.py file, (2) a jupyter notebook generating the illustration below, (3) a .jpg file of the output figure, and (4) a jupyter notebook of some simple cross validation tests (see next section).

If you want to quickly visualize the model at work, cd to demo.py, and run the file:
```
cd Documents/folder/path/to/probabilistic_LDA/demo/ # demo.py should be in this folder.
python3.5 demo.py
```
Training data are randomly generated from 5 2D Gaussians with the same covariance matrix. Test data are generated from a uniform distribution, and the colors of those points represent model classifications. The contours depict 95% confidence intervals of the labeled training data. Note that the demo training data were generated with the same covariance because this a fundamental assumption of linear discriminant analysis models.

![Figure 1-1](/demo/classification_demo.jpg?raw=True)


### Cross-validation
I have uploaded a simple jupyter notebook demo of leave n out cross-validation in the demo folder. It uses the Child Affective Facial Expressions dataset.
```
cd Documents/folder/path/to/probabilistic_LDA/demo/ # cross_validation_demo.ipynb should be in this folder.
```
