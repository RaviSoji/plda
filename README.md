# probabilistic_LDA

### Paper citation
Ioffe S. (2006) Probabilistic Linear Discriminant Analysis. In: Leonardis A., Bischof H., Pinz A. (eds) Computer Vision – ECCV 2006. ECCV 2006.

### Python Dependencies
numpy
scipy

### Demo and cross-validation dependencies
* skimage 0.13.0
```
pip install scikit-image
```

* sklearn 0.18.2
```
pip install -U scikit-learn or conda install scikit-learn
```

### Cross-validation on the Google_Faces Emotions dataset
```
python3.5 cross_validation_PLDA.py
```

### Testing (Requres decent computing power. I had access to ~ 60ish CPU cores)
cd to the directory containing the respository, the 'PLDA.py' file in particular.
```
cd /Documents/folder/path/to/probabilistic_LDA/
```

To run all tests:
```
python3.5 -m unittest discover
```

To run one of the three tests in the tests folder, use one of the following:
```
python3.5 -m unittest tests/test_inference_PLDA.py
```
```
python3.5 -m unittest tests/test_integration_PLDA.py
```
```
python3.5 -m unittest tests/test_units_PLDA.py
```
