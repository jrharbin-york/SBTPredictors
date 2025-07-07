from abc import ABC, ABCMeta, abstractmethod

class MissingMetric(LookupError):
    """Raised when a metric is not present - because a predictor was not available for it"""

class MissingThreshold(LookupError):
        """Raised when a metric is not present - because a predictor was not available for it"""

class DecisionNode(metaclass=ABCMeta):
    @abstractmethod
    def execute_or_not(self, test_id, predicted_metrics):
        pass

    def accept_test(self, test_id, test_metrics):
        pass

class NullDecisionNode(DecisionNode):
    """Null decision node just includes every test"""
    def execute_or_not(self, test_id, predicted_metrics):
        return True

class FixedThresholdBased(DecisionNode):
    """Uses a single fixed value for each metric for the decision threshold"""
    def __init__(self, target_metric_ids, metrics_needed, thresholds, greater_than):
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

# TODO: can we split these up into multiple and a decision voting?

# TODO: simulated annealing decision node
class SimulatedAnnealingThreshold(DecisionNode):
    pass

# Indicator-based decision node
class IndicatorBasedDecisions(DecisionNode):
    def __init__(self, target_indicator, decision_analysis, front_all_tests):
        self.target_indicator = target_indicator
        self.current_best_front = None
        self.indicator_id = "HV"
        self.best_indicator_value = None
        self.decision_analysis = decision_analysis
        # TODO: need to know the front for all tests here!
        self.front_all_tests = self.front_all_tests

    def execute_or_not(self, test_id, predicted_test_metrics):
        if self.best_indicator_value is None:
            return True
        else:
            # Create front including the new test's predicted values
            # Compute the indicator speculative value - is it better?
            # If so, accept it
            potential_front_plus_new_test = self.current_best_front.append(predicted_test_metrics)
            hypothetical_front = self.decision_analysis.compute_front_from_tests(potential_front_plus_new_test)
            quality_indicators_with_new_test = self.decision_analysis.indicators_for_front(hypothetical_front, front_all_tests)
            quality_for_indicator = quality_indicators_with_new_test[self.indicator_id]
            if quality_for_indicator > self.best_indicator_value:
                # TODO: do we change current_best_front here or in accept_test?
                self.current_best_front = hypothetical_front
                self.best_indicator_value = quality_for_indicator
                return True
            else:
                return False

    def accept_test(self, test_id, actual_test_metrics):
        pass
