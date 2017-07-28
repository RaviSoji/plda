# probabilistic_LDA

### Disclaimer
If all that you want to do is apply this model to a dataset and measure its
 performance, you will find the code to be bulky in that it stores parameters,
 variables, and data that you do not need. This code was written for
 an [Explainable Artificial Intelligence (XAI) project](http://shaftolab.com/people.html)
 ([machine teaching, in particular](http://shaftolab.com/publications.html)), whose
 objective is model explainability (not necessarily speed or performance),
 which requires saving such information.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. ECCV 2006.](https://link.springer.com/chapter/10.1007/11744085_41)

### Model Dependencies
* numpy 1.13.1
* scipy 0.19.1

### Cross-validation and Demo Dependencies
* skimage 0.13.0  (for Cross-validation)
* sklearn 0.18.2  (for Cross-validation)
* matplotlib 2.0.2  (for Demo) 

### Testing (Requires decent computing power! I had access to ~60 CPU cores.)
First, cd to the repository (i.e. the directory containing the 'PLDA.py' file).
```
cd /Documents/folder/path/to/probabilistic_LDA/ # PLDA.py should be in this folder.
```

To run all tests:
```
python3.5 -m unittest discover
```

To run only one of the three test files in the tests folder, run the corresponding line/command:
```
python3.5 -m unittest tests/test_inference_PLDA.py
python3.5 -m unittest tests/test_integration_PLDA.py
python3.5 -m unittest tests/test_units_PLDA.py
```

### Demo with Artificial 2D Gaussian Data
The demo folder contains (1) a demo.py file that will generate the illustration below
and (2) an IPython notebook that uses the demo to illustrate various attributes
of the PLDA object.

If you want to quickly visualize the model at work, cd to demo.py, and run the file:
```
cd Documents/folder/path/to/probabilistic_LDA/demo/ # demo.py should be in this folder.
python3.5 demo.py
```
Training data is randomly generated from 10 2D Gaussians, which are also
 randomly generated. Colors of the points represent the model classifications 
 of random test points whereas the contours depict 95% confidence intervals 
 (based on 300-900 points per class) of the labeled training data. Note that
 training data were generated with the same covariance because this is a 
 central model assumption.

![Figure 1-1](https://github.com/RaviSoji/probabilistic_LDA/blob/master/demo/2D_example.png?raw=True)

### k-Folds Cross-validation on the Google_Faces Emotions Dataset
To my knowledge, K-folds on LDA for facial recognition (where classes are 
 specific people) generally gets about 40-60% accuracy. Here, we are 
 trying to do emotion classification based on very little and uncontrolled
 data (23-86 samples of DIFFERENT faces for each of the 7 emotions) and 
 are finding around 33% accuracy, which is bad, but also above chance level.
 
To the the code, cd to cross_validation.py and run the file:
```
cd Documents/folder/path/to/probabilistic_LDA/cross_validation/ # cross_validation.py should be in this folder.
python3.5 cross_validation_PLDA.py
```
