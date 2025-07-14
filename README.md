# SBTPredictors
Testing of the prediction components

# Install

- Setup python 3.12 virtual environment, activate it
- pip install -r requirements.txt

- Need to modify the virtual environment tsfresh code under:

```
$VENV_ROOT/src/tsfresh/tsfresh/feature_extraction/feature_calculators.py
```

add the 2 functions from feature_calculators_extra.py

Example of predictor training
```
python3 ./run_experiments.py
```

Example of decision node testing
```
python3 ./analyse_pareto_fronts.py
```
