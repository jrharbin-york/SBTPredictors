from sklearn.model_selection import KFold
import structlog
import pandas as pd
import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import paretoset
from tabulate import tabulate

import data_loader
import datasets
from decision_node import MissingMetric, NullDecisionNode, RandomDecisionNode
from fixed_threshold_decision_node import FixedThresholdBased
from indicator_based_decision_node import IndicatorBasedDecisions
from simulated_annealing_decision_node import SimulatedAnnealingThreshold, SimulatedAnnealingThresholdSingleDimensional, \
    SimulatedAnnealingThresholdMultiDimensional

log = structlog.get_logger()

class FrontCalculationFailed(Exception):
    """Calculation of a front failed"""

class DecisionNodeAnalysis:
    def __init__(self, base_files_dir, metrics_test_df, metric_columns_direction, needed_operations, PREDICT_ALL_TEST_SIMULTANEOUSLY = True):
        # Hash of the column name and direction - either "max" or "min"
        self.base_files_dir = base_files_dir
        self.metrics_test_df = metrics_test_df
        self.metric_columns_direction = metric_columns_direction
        self.needed_operations = needed_operations
        self.metric_cols_chosen = list(metric_columns_direction.keys())
        self.PREDICT_ALL_TEST_SIMULTANEOUSLY = PREDICT_ALL_TEST_SIMULTANEOUSLY

    def compute_front_from_tests(self, metric_tests):
        # columns_direction is a hashtable of column name to either 'max' or 'min'
        print(self.metric_columns_direction)
        directions = [self.metric_columns_direction[k] for k in self.metric_cols_chosen]

        if len(metric_tests) > 0:
            print(metric_tests)
            metric_tests_metricsonly = metric_tests[self.metric_cols_chosen]
            # Computes the front from the given metric_test, inverting if the columns chosen are flipped
            mask = paretoset.paretoset(metric_tests_metricsonly, sense=directions)
            selected_tests = metric_tests[mask]
            return selected_tests
        else:
            return pd.DataFrame([], columns=self.metric_cols_chosen)

    def max_point_from_front(self, front):
        # TODO: check that this is using the max_point properly
        metric_names = self.metric_cols_chosen
        point_coords = np.array([])
        for i in range(0,len(metric_names)):
            point_coords = np.append(point_coords, np.max(front[:,i]))
        return np.array(point_coords)

    def max_point_from_all_fronts(self, front_list):
        max_points_2d = np.column_stack(list(map(lambda pf: self.max_point_from_front(pf), front_list)))
        max_of_all = np.amax(max_points_2d, axis=1)
        return max_of_all

    def front_to_numpy_array(self,front):
        # TODO: invert cols according to the direction value
        return front[self.metric_cols_chosen].to_numpy()

    def indicators_for_front(self, front_df, ref_front_df):
        # Compute the IGD and reference front
        # Reference point is obtained as the max point from all fronts
        # compute the reference point and reference front from all the front contents?

        # Get front as numpy array
        front = self.front_to_numpy_array(front_df)
        ref_front = self.front_to_numpy_array(ref_front_df)

        if len(front > 0):
            ref_point = self.max_point_from_all_fronts([front, ref_front])
        else:
            ref_point = self.max_point_from_all_fronts([ref_front])

        hv_ind = HV(ref_point=ref_point)
        igd_ind = IGD(ref_front)
        hv_val = hv_ind(front)
        igd_val = igd_ind(front)
        print(f"HV = {hv_val},IGD = {igd_val}")
        front_info = {"hypervolume": hv_val, "igd": igd_val}
        return front_info

    def load_test_definition(self, test_id):
        filepath = self.base_files_dir + "/" + test_id
        return data_loader.load_individual_instance(filepath, self.needed_operations)

    def predict_for_test_id(self, test_id, predictors):
        metric_results = {}
        for col_name, pipeline_gen_func in predictors.items():
            print("Running predictor for ", test_id, " and column", col_name)
            # load the specific test definition here and make the prediction
            # TODO: handle multiple ones here
            test_definition = self.load_test_definition(test_id)
            metric_results[col_name] = pipeline_gen_func.predict([test_definition])
        return metric_results

    def predict_all_tests(self, test_ids, predictors):
        metric_results = {}
        for col_name, pipeline_gen_func in predictors.items():
            log.info(f"Running predictor for {col_name} on all tests")
            # load the specific test definition here and make the prediction
            test_definitions = data_loader.create_combined_data(self.base_files_dir, test_ids, self.needed_operations)
            metric_results[col_name] = pipeline_gen_func.predict(test_definitions)
        return metric_results

    def choose_tests_from_decision_node(self, actual_test_metrics, predictors, decision_node):
        tests_chosen_rows = []
        #tests_chosen = pd.DataFrame([], columns=actual_test_metrics.columns)
        try:
            for test_index, test_metrics in actual_test_metrics.iterrows():
                # get the test prediction for test N
                test_id = test_metrics["testID"]
                predicted_metrics = self.predict_for_test_id(test_id, predictors)
                # can also compute intermediate indicators here from tests_chosen
                should_execute = decision_node.execute_or_not(test_id, predicted_metrics)
                if should_execute:
                    # Log the decision to choose this test
                    tests_chosen_rows.append(test_metrics)
                    decision_node.accept_test(test_id, test_metrics)
            tests_chosen = pd.DataFrame(tests_chosen_rows)
            return tests_chosen
        except MissingMetric as mmetric:
            log.info("Metric", mmetric, " is missing")
            raise FrontCalculationFailed(mmetric)

    def choose_tests_from_decision_node_predict_all_first(self, actual_test_metrics, predictors, decision_node):
        tests_chosen_rows = []
        try:
            test_ids = actual_test_metrics["testID"]
            predicted_metrics_all = self.predict_all_tests(test_ids, predictors)

            metric_row_id = 0
            for (test_index, test_metrics) in actual_test_metrics.iterrows():
                test_id = test_metrics["testID"]
                # get the test prediction for test N - needs to be formatted as a dict of 1-element arrays, selecting
                # just the element for that particular test
                predicted_metrics = { key: value[metric_row_id] for key, value in predicted_metrics_all.items() }
                metric_row_id += 1
                should_execute = decision_node.execute_or_not(test_id, predicted_metrics)
                if should_execute:
                    # Log the decision to choose this test
                    tests_chosen_rows.append(test_metrics)
                    decision_node.accept_test(test_id, test_metrics)
            tests_chosen = pd.DataFrame(tests_chosen_rows)
            return tests_chosen
        except MissingMetric as mmetric:
            log.info("Metric", mmetric, " is missing")
            raise FrontCalculationFailed(mmetric)

    def evaluate_decision_nodes_front_quality(self, predictors, decision_node):
        # Assess the front from all tests
        front_all_tests = self.compute_front_from_tests(self.metrics_test_df)
        log.debug(f"Front from all tests:\n {front_all_tests}")
        quality_indicators_all_tests = self.indicators_for_front(front_all_tests, front_all_tests)
        log.debug(f"Quality indicators: {quality_indicators_all_tests}")

        # Then test the possible decision nodes
        decision_node_results = {}

        decision_node.register_front_all_tests(front_all_tests)
        if self.PREDICT_ALL_TEST_SIMULTANEOUSLY:
            chosen_by_decision_node = self.choose_tests_from_decision_node_predict_all_first(self.metrics_test_df, predictors, decision_node)
        else:
            chosen_by_decision_node = self.choose_tests_from_decision_node(self.metrics_test_df, predictors, decision_node)
        tests_chosen_count = len(chosen_by_decision_node)
        front_with_decisions = self.compute_front_from_tests(chosen_by_decision_node)
        quality_indicators_for_front = self.indicators_for_front(front_with_decisions, front_all_tests)
        log.info(f"Front generate with the decision node {decision_node}:\n {front_with_decisions}")
        # TODO: log front_all_tests and front_with_decision_nodes
        decision_node_results = {"decision_node" : decision_node.description(),
                                 "front_all_tests_size" : len(front_all_tests),
                                 "tests_chosen_count" : tests_chosen_count,
                                 "all_tests_count" : len(self.metrics_test_df),
                                 "quality_indicators_all_tests" : str(quality_indicators_all_tests),
                                 "front_from_decision_node_size": len(front_with_decisions),
                                 "quality_indicators_for_front" : str(quality_indicators_for_front) }
        return decision_node_results

def test_evaluate_predictor_decisions_for_experiment(expt_config):
    # Load predictors from files - separate predictor for all 3 metrics
    human1_predfile = "./temp-saved-predictors/eterry-15files/human1.predictor"
    statichumans_predfile = "./temp-saved-predictors/eterry-15files/statichumans.predictor"
    path_predfile = "./temp-saved-predictors/eterry-15files/pathcompletion.predictor"

    predictors_for_cols = { "distanceToHuman1" : data_loader.load_predictor_from_file(human1_predfile),
                            "distanceToStaticHumans" : data_loader.load_predictor_from_file(statichumans_predfile),
                            "pathCompletion": data_loader.load_predictor_from_file(path_predfile) }

    metric_columns_direction = { "distanceToHuman1" : "min",
                                 "distanceToStaticHumans" : "min",
                                 "pathCompletion": "min" }

    decision_metrics_file = "./temp-saved-predictors/eterry-15files/decisionMetrics.csv"
    decision_data_files, decision_metrics = data_loader.read_data(expt_config["data_dir_base"], decision_metrics_file)
    decision_nodes_info_for_splits = {}

    thresholds = { "distanceToHuman1" : 3.0,
                   "distanceToStaticHumans" : 2.0,
                   "pathCompletion": 0.6 }

    target_metric_ids = metric_columns_direction.keys()

    needed_operations = expt_config["needed_columns"]
    base_files_dir = expt_config["data_dir_base"]
    analyser_fast = DecisionNodeAnalysis(base_files_dir, decision_metrics, metric_columns_direction, needed_operations, True)
    analyser_slow = DecisionNodeAnalysis(base_files_dir, decision_metrics, metric_columns_direction, needed_operations, False)
    null_decision_node = NullDecisionNode()

    random_decision_node_prob = 0.1
    random_decision_node = RandomDecisionNode(random_decision_node_prob)

    distance_divisor_per_metric = {
        "distanceToHuman1": 10,
        "distanceToStaticHumans": 4,
        "pathCompletion": 1
    }

    metric_weights = {
        "distanceToHuman1": 1.0,
        "distanceToStaticHumans": 1.0,
        "pathCompletion": 1.0
    }

    fixed_threshold_decision_node_1 = FixedThresholdBased(target_metric_ids, 1, thresholds, False)
    fixed_threshold_decision_node_2 = FixedThresholdBased(target_metric_ids, 2, thresholds, False)
    fixed_threshold_decision_node_3 = FixedThresholdBased(target_metric_ids, 3, thresholds, False)

    # TODO: set single node
    sim_annealing_node_1 = SimulatedAnnealingThresholdSingleDimensional(target_metric_ids, distance_divisor_per_metric, metric_weights, initial_temperature=10.0)
    sim_annealing_node_2 = SimulatedAnnealingThresholdSingleDimensional(target_metric_ids, distance_divisor_per_metric, metric_weights, initial_temperature=100.0)
    sim_annealing_node_3 = SimulatedAnnealingThresholdSingleDimensional(target_metric_ids, distance_divisor_per_metric, metric_weights, initial_temperature=1000.0)

    hypervolume_based = IndicatorBasedDecisions("hypervolume", analyser_slow, 1.0)

    normal_decision_nodes = [
        null_decision_node,
        random_decision_node,
        fixed_threshold_decision_node_1,
        fixed_threshold_decision_node_2,
        fixed_threshold_decision_node_3,
    ]

    min_temperature = 10
    max_temperature = 520
    step_temperature = 20
    temperature_ranges = range(min_temperature,max_temperature,step_temperature)
    decision_nodes_single_range = list(map (lambda temp: SimulatedAnnealingThresholdSingleDimensional(target_metric_ids, distance_divisor_per_metric, metric_weights, initial_temperature=temp), temperature_ranges))
    decision_nodes_multi_range = list(map (lambda temp: SimulatedAnnealingThresholdMultiDimensional(target_metric_ids, distance_divisor_per_metric, initial_temperature=temp), temperature_ranges))

    decision_nodes_fast = decision_nodes_single_range + decision_nodes_multi_range + normal_decision_nodes
    decision_nodes_slow = [hypervolume_based]

    all_decision_node_results = {}

    num = 0
    for decision_node in decision_nodes_fast:
        num += 1
        decision_node_name = decision_node.__class__.__name__ + "_" + str(num)
        decision_node_info = analyser_fast.evaluate_decision_nodes_front_quality(predictors_for_cols, decision_node)
        log.info(f"RES: decision_node_name={decision_node_name}, all_tests_count={decision_node_info["all_tests_count"]}, tests_chosen_count={decision_node_info["tests_chosen_count"]}, front_all_tests_size={decision_node_info["front_all_tests_size"]}, quality_indicators_all_tests={decision_node_info["quality_indicators_all_tests"]}, front_from_decision_node_size={decision_node_info["front_from_decision_node_size"]}, quality_indicators_for_front={decision_node_info["quality_indicators_for_front"]}")
        all_decision_node_results[decision_node_name] = decision_node_info

    for decision_node in decision_nodes_slow:
        num += 1
        decision_node_name = decision_node.__class__.__name__ + "_" + str(num)
        decision_node_info = analyser_slow.evaluate_decision_nodes_front_quality(predictors_for_cols, decision_node)
        log.info(f"RES: decision_node_name={decision_node_name}, all_tests_count={decision_node_info["all_tests_count"]}, tests_chosen_count={decision_node_info["tests_chosen_count"]}, front_all_tests_size={decision_node_info["front_all_tests_size"]}, quality_indicators_all_tests={decision_node_info["quality_indicators_all_tests"]}, front_from_decision_node_size={decision_node_info["front_from_decision_node_size"]}, quality_indicators_for_front={decision_node_info["quality_indicators_for_front"]}")
        all_decision_node_results[decision_node_name] = decision_node_info

    results_df = pd.DataFrame(all_decision_node_results.values())
    res_filename = "all_decision_node_results.csv"
    results_df.to_csv(res_filename)
    print(tabulate(results_df, headers="keys"))

if __name__ == '__main__':
    test_evaluate_predictor_decisions_for_experiment(datasets.expt_config_eterry_human1_15files)