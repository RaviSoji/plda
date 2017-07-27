# probabilistic_LDA

### Disclaimer
If all that you want to do is apply this model to a dataset and measure its
 performance, you will find the code to be bulky in that it stores parameters,
 variables, and data that you do not need. This code was written for
 an [Explainable Artificial Intelligence project](http://shaftolab.com/people.html)
 ([machine teaching, in particular](http://shaftolab.com/publications.html)) in mind
 , whose objective is model explainability (not necessarily speed or performance),
 which requires saving parameters and training data.

### Paper Citation
[Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. ECCV 2006.](https://link.springer.com/chapter/10.1007/11744085_41)

### Python Dependencies
* numpy 1.13.1
* scipy 0.19.1

### Demo and Cross-validation Dependencies
* skimage 0.13.0
```
pip install scikit-image
```

* sklearn 0.18.2
```
pip install -U scikit-learn
```

### Testing (Requires decent computing power! I had access to ~60 CPU cores.)
cd to the directory containing the respository, the 'PLDA.py' file in particular.
```
cd /Documents/folder/path/to/probabilistic_LDA/ # PLDA.py should be in this folder.
```

To run all tests:
```
python3.5 -m unittest discover
```

To run one of the three sets of tests in the tests folder, run one of the following lines/commands:
```
python3.5 -m unittest tests/test_inference_PLDA.py
python3.5 -m unittest tests/test_integration_PLDA.py
python3.5 -m unittest tests/test_units_PLDA.py
```

### k-Folds Cross-validation on the Google_Faces Emotions Dataset
K-folds on facial recognition (where classes are specific people),
 traditionally gets about 40-60% accuracy. Here, we are trying to do emotion
 classification based on very little and uncontrolled data (23-86 samples of
 DIFFERENT faces for each of the 7 emotions) and are finding around 33%
 accuracy, which is bad, but also above chance level.
```
python3.5 cross_validation_PLDA.py
```
