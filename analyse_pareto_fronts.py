from abc import ABC, ABCMeta, abstractmethod
from sklearn.model_selection import KFold

import structlog
import pymoo
import pandas as pd
import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import paretoset
import data_loader

log = structlog.get_logger()

# TODO: metrics_test dataframe needs to include all metrics for the test - change Java code correspondingly
def compute_front_from_tests(metric_tests, metric_columns_direction):
    # columns_direction is a hashtable of column name to either 'max' or 'min'
    metric_cols_chosen = list(metric_columns_direction.keys)
    metric_tests_metricsonly = metric_tests[metric_cols_chosen]
    # Computes the front from the given metric_test, inverting if the columns chosen are flipped
    mask = paretoset.paretoset(metric_tests_metricsonly, sense=metric_columns_direction[metric_cols_chosen])
    selected_tests = metric_tests[mask]
    return selected_tests

def compute_front_quality_indications(front):
    pass

def max_point_from_front(front, metric_column_direction):
    # TODO: check that this is using the max_point properly
    metric_names = metric_column_direction.keys
    point_coords = np.array([])
    for n in metric_names:
        point_coords = np.append(point_coords, np.max(front[n]))
    return np.array(point_coords)


def max_point_from_all_fronts(front_list):
    max_points_2d = np.column_stack(list(map(lambda pf: max_point_from_front(pf), front_list)))
    max_of_all = np.amax(max_points_2d, axis=1)
    return max_of_all


def front_to_numpy_array(front, metric_columns_direction):
    selected_cols = list(metric_columns_direction.keys)
    # TODO: invert cols according to the direction value
    return front[selected_cols].to_numpy()


def indicators_for_front(front, ref_front, metric_columns_direction):
    # Compute the IGD and reference front
    # Reference point is obtained as the max point from all fronts

    # compute the reference point and reference front from all the front contents?

    ref_point = max_point_from_all_fronts([front, ref_front])
    all_fronts_indicators = {}

    # Get front as numpy array
    front_np = front_to_numpy_array(front, metric_columns_direction)
    hv_ind = HV(ref_point=ref_point)
    igd_ind = IGD(ref_front)
    hv_val = hv_ind(front)
    igd_val = igd_ind(front)
    print("HV = " + str(hv_val) + "IGD = " + str(igd_val))
    front_info = {"hypervolume": hv_val, "igd": igd_val}
    return front_info


def indicators_for_multiple_fronts(fronts, selected_cols):
    """Compute the IGD and reference front"""
    #Reference point is obtained as the max point from all fronts

    # compute the reference point and reference front from all the front contents?

    ref_point = max_point_from_all_fronts(fronts)
    # Assumes the reference front is the first
    ref_front = fronts[0]
    all_fronts_indicators = {}

    for front_pandas in fronts:
        all_fronts_indicators[front_pandas] = indicators_for_front(front_pandas, ref_front, selected_cols)
    return all_fronts_indicators

class MissingMetric(LookupError):
    """Raised when a metric is not present - because a predictor was not available for it"""

class MissingThreshold(LookupError):
        """Raised when a metric is not present - because a predictor was not available for it"""

class FrontCalculationFailed(Exception):
    """Calculation of a front failed"""

class DecisionNode(metaclass=ABCMeta):
    @abstractmethod
    def execute_or_not(self, test_id, test_metrics):
        pass

# Null decision node just includes everything
class NullDecisionNode(DecisionNode):
    def execute_or_not(self, test_id, test_metrics):
        return True

# Uses a single fixed value for each metric for the decision threshold
class FixedThresholdBasedDecisionNode(DecisionNode):
    def __init__(self, target_metric_ids, thresholds, greater_than):
        self.target_metric_ids = target_metric_ids
        self.thresholds = thresholds
        self.greater_than = greater_than

    # Other decision nodes
    # simulated annealing type for the population management
    # based on indicators - e.g. would the hypervolume increase for selecting this test?

    def execute_or_not(self, test_id, test_metrics):
        found_count = 0
        for m in self.target_metric_ids:
            if test_metrics[m] is None:
                raise MissingMetric(m)

            if self.thresholds[m] is None:
                raise MissingThreshold(m)

            if self.greater_than:
                if test_metrics[m] > self.thresholds[m]:
                    found_count += 1
            else:
                if test_metrics[m] < self.thresholds[m]:
                    found_count += 1
        return found_count

def load_test_definition(test_id, needed_operations):
    return data_loader.load_individual_instance(needed_operations)

def predict_for_test_id(test_id, needed_operations, predictors):
    # TODO: call the predictors and get the results
    metric_results = {}
    for col_name, pipeline_gen_func in predictors:
        print("Running predictor for ", test_id, " and column", col_name)
        # load the specific test definition here and make the prediction
        test_definition = load_test_definition(test_id, needed_operations)
        metric_results[col_name] = pipeline_gen_func.predict([test_definition])
    return metric_results

def front_from_decision_node(test_ids, actual_test_metrics, metric_columns_direction, needed_operations, predictors, decision_node):
    tests_chosen = pd.DataFrame()
    try:
        for test_id, test_metrics in zip(test_ids, actual_test_metrics):
            # get the test prediction for test N
            predicted_metrics = predict_for_test_id(test_id, needed_operations, predictors)
            should_execute = decision_node.execute_or_not(test_id, predicted_metrics)
            if should_execute:
                # Log the decision to choose this test
                tests_chosen.append(test_metrics)
        return compute_front_from_tests(tests_chosen, metric_columns_direction)
    except MissingMetric as mmetric:
        log.info("Metric", mmetric, " is missing")
        raise FrontCalculationFailed(mmetric)

def evaluate_decision_nodes_front_quality(test_ids, metrics_test_df, metric_columns_direction, predictors, decision_nodes):
    # Assess the front from all tests
    front_all_tests = compute_front_from_tests(metrics_test_df, metric_columns_direction)
    quality_indicators = indicators_for_front(front_all_tests, front_all_tests, metric_columns_direction)
    log.info("Front for all tests:", front_all_tests)
    log.info("Quality indicators:" , str(quality_indicators))
    # Then test the possible decision nodes
    for decision_node in decision_nodes:
        front_with_decisions = front_from_decision_node(test_ids, metrics_test_df, metric_columns_direction, predictors, decision_node, metric_columns_direction)
        quality_indicators_for_front = indicators_for_front(front_with_decisions, front_all_tests, metric_columns_direction)
        log.info("Front with the decision nodes:", front_with_decisions)

def test_predictor_on_fronts(expt_config, human1_predfile, statichumans_predfile, path_predfile):
    # Load predictors from files  - for all 3 metrics
    predictors_for_cols = { "distanceToHuman1" : data_loader.load_predictor_from_file(human1_predfile),
                            "distanceToStaticHumans" : data_loader.load_predictor_from_file(statichumans_predfile),
                            "pathCompletion": data_loader.load_predictor_from_file(path_predfile) }

    metric_columns_direction = { "distanceToHuman1" : "min",
                                 "distanceToStaticHumans" : "min",
                                 "pathCompletion": "min" }

    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = data_loader.read_data(expt_config["data_dir_base"], mfile)

    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        metrics_test_df = metrics.iloc[test_index]
        test_ids = None

        metric_columns_direction = { "distanceToHuman1" : "min",
                                     "distanceToStaticHumans" : "min",
                                     "pathCompletion": "min" }

        thresholds = { "distanceToHuman1" : 4.0,
                       "distanceToStaticHumans" : 2.5,
                       "pathCompletion": 0.7 }

        target_metric_ids = metric_columns_direction.keys

        decision_node = FixedThresholdBasedDecisionNode(target_metric_ids, thresholds, False)
        decision_nodes = [decision_node]
        evaluate_decision_nodes_front_quality(test_ids, metrics_test_df, metric_columns_direction, predictors_for_cols, decision_nodes)
