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
