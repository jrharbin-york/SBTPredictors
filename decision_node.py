import math
from abc import ABCMeta, abstractmethod
import random

import pandas as pd
import structlog

log = structlog.get_logger()

class MissingMetric(LookupError):
    """Raised when a metric is not present - because a predictor was not available for it"""

class MissingThreshold(LookupError):
        """Raised when a metric is not present - because a predictor was not available for it"""

class DecisionNode(metaclass=ABCMeta):
    def __init__(self):
        self.front_all_tests = None

    @abstractmethod
    def execute_or_not(self, test_id, predicted_metrics):
        pass

    def accept_test(self, test_id, test_metrics):
        pass

    def register_front_all_tests(self, front_all_tests):
        self.front_all_tests = front_all_tests

class RandomDecisionNode(DecisionNode):
    def __init__(self, include_test_prob):
        super().__init__()
        self.include_test_prob = include_test_prob

    def execute_or_not(self, test_id, predicted_metrics):
        v = random.uniform(0.0,1.0)
        return v < self.include_test_prob

class NullDecisionNode(DecisionNode):
    """Null decision node just includes every test"""

    def execute_or_not(self, test_id, predicted_metrics):
        return True

class FixedThresholdBased(DecisionNode):
    """Uses a single fixed value for each metric for the decision threshold"""
    def __init__(self, target_metric_ids, metrics_needed, thresholds, greater_than):
        super().__init__()
        self.target_metric_ids = target_metric_ids
        self.thresholds = thresholds
        self.greater_than = greater_than
        self.metrics_needed = metrics_needed

    # Other decision nodes
    # simulated annealing type for the population management
    # based on indicators - e.g. would the hypervolume increase for selecting this test?

    def execute_or_not(self, test_id, predicted_metrics):
        found_count = 0
        for m in self.target_metric_ids:
            if predicted_metrics[m] is None:
                raise MissingMetric(m)

            if self.thresholds[m] is None:
                raise MissingThreshold(m)

            if self.greater_than:
                if predicted_metrics[m] > self.thresholds[m]:
                    found_count += 1
            else:
                if predicted_metrics[m] < self.thresholds[m]:
                    found_count += 1
        return found_count >= self.metrics_needed

# TODO: simulated annealing decision node
# Acceptance probability of accepting a worse solution than the current best - decays over
# time exponentially
# example here: https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/

class SimulatedAnnealingThreshold(DecisionNode):
    """Simulated annealing has a probability of accepting a worst solution than the current best -
    computes the distances between predicted and best actual and uses the diagonal distance to calibrate
    the probability based upon temperature"""

    # TODO: currently assumes minimisation only
    def __init__(self, target_metrics_ids, distance_divisor_per_metric, initial_temperature = 100.0):
        super().__init__()
        self.target_metric_ids = target_metrics_ids
        self.initial_temperature = 100.0
        self.current_best_predictions = {}
        self.tests_per_epoch_increment = 1.0
        self.distance_divisor_per_metric = distance_divisor_per_metric
        self.epoch = 0

    def execute_or_not(self, test_id, predicted_metrics):
        execute = False
        diff_m_squared = 0.0
        for m in self.target_metric_ids:
            prediction_for_m = predicted_metrics[m]

            # Execute if there is no value known for a particular metric
            if not (m in self.current_best_predictions):
                execute = True
            else:
                # Sum the distance squared scaled by the divisor for that metric
                diff_m = (prediction_for_m - self.current_best_predictions[m]) / self.distance_divisor_per_metric[m]
                diff_m_squared += diff_m ** 2.0

            diff_all_metrics = math.sqrt(diff_m_squared)
            t_now = self.initial_temperature / float(self.epoch + 1)
            metropolis = math.exp(-diff_all_metrics / t_now)
            log.debug(f"epoch={self.epoch}, t_now={t_now}, metropolis={metropolis}")
            if diff_all_metrics < 0 or random.uniform(0.0, 1.0) < metropolis:
                execute = True

        return execute

    def accept_test(self, test_id, actual_test_metrics):
        self.epoch += 1.0 / float(self.tests_per_epoch_increment)
        for m in self.target_metric_ids:
            if not (m in self.current_best_predictions):
                self.current_best_predictions[m] = actual_test_metrics[m]
            else:
                # TODO: assumes minimisation is better
                if actual_test_metrics[m] < self.current_best_predictions[m]:
                    self.current_best_predictions[m] = actual_test_metrics[m]

# Indicator-based decision node
class IndicatorBasedDecisions(DecisionNode):
    def __init__(self, indicator_id, decision_analysis, improvement_relative_factor = 1.0):
        super().__init__()
        self.current_best_front = None
        self.indicator_id = indicator_id
        self.best_indicator_value = None
        self.decision_analysis = decision_analysis
        self.improvement_relative_factor = improvement_relative_factor

    def execute_or_not(self, test_id, predicted_test_metrics):
        if self.best_indicator_value is None:
            return True
        else:
            # Create front including the new test's predicted values
            # Compute the indicator speculative value - is it better?
            # If so, accept it
            potential_front_plus_new_test = self.current_best_front.append(predicted_test_metrics)
            hypothetical_front = self.decision_analysis.compute_front_from_tests(potential_front_plus_new_test)
            quality_indicators_with_new_test = self.decision_analysis.indicators_for_front(hypothetical_front, self.front_all_tests)
            quality_for_indicator = quality_indicators_with_new_test[self.indicator_id]
            # TODO: need a threshold here for this... if it improves the indicator value by X/%
            include = (quality_for_indicator > self.best_indicator_value * self.improvement_relative_factor)
            return include

    def accept_test(self, test_id, actual_test_metrics):
        if self.current_best_front is None:
            self.current_best_front = pd.DataFrame([])

        potential_front_plus_new_test = pd.concat([self.current_best_front, actual_test_metrics])
        self.current_best_front = self.decision_analysis.compute_front_from_tests(potential_front_plus_new_test)
        self.best_indicator_value = self.decision_analysis.indicators_for_front(self.current_best_front, self.front_all_tests)
