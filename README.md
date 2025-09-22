# ASSIST SBT Prediction Python Code

Python code for the testing of the ASSIST SBT Prediction components

# Installation

- Create a new Python 3.12 virtual environment, activate it:
Example paths are given below:

```
python3.12 -m venv ~/python-venvs/tsfresh_test_repo
. ~/python-venvs/tsfresh_test_repo/bin/activate
```

In this case the VENV_ROOT is ~/python-venvs/tsfresh_test_repo,
so set this variable:
```
export VENV_ROOT=/home/jharbin/python-venvs/tsfresh_test_repo/
```

Install the required Python dependencies:
```
pip install -r requirements.txt
```

Because the ASSIST code uses a custom feature within the TSFresh library, 
this needs to be added to the TSFresh library in the virtual environment. 
Use the following command to add this feature into the virtual environment
copy of TSFresh:

```
cat custom_tsfresh_features/feature_calculators_extra.py >> $VENV_ROOT/src/tsfresh/tsfresh/feature_extraction/feature_calculators.py
```

# Predictor training

Example of predictor training for RQ1 and RQ2:
```
python3 ./run_experiments.py
```

When executed, predictors, regression graphs and CSV result data will
be placed under *./results/* with the current data stamp, named for
specific use cases.

# Decision node testing

Example of decision node testing for RQ3
```
python3 ./analyse_pareto_fronts.py
```
