# Indicator-based decision node
import pandas as pd
from decision_node import DecisionNode, log

class IndicatorBasedDecisions(DecisionNode):
    def __init__(self, indicator_id, decision_analysis, min_tests_to_accept_first, improvement_relative_factor = 1.0, ):
        super().__init__()
        # TODO: need to supply the sign and flip if the indicator should be maximised
        self.current_best_front = None
        self.indicator_id = indicator_id
        self.best_indicator_value = None
        self.decision_analysis = decision_analysis
        self.improvement_relative_factor = improvement_relative_factor
        self.accepted_tests = []
        self.min_tests_to_accept_first = min_tests_to_accept_first

    def description(self):
        return f"IndicatorBased({self.indicator_id})"

    def execute_or_not(self, test_id, predicted_test_metrics):
        # TODO: min_tests should be supplied as parameter

        if (self.best_indicator_value is None) or (len(self.accepted_tests) < self.min_tests_to_accept_first):
            # Always accept the first test
            return True
        else:
            hypothetical_front_plus_prediction = pd.DataFrame(self.current_best_front)
            new_prediction_df = pd.DataFrame(predicted_test_metrics)
            hypothetical_front_plus_prediction = pd.concat([hypothetical_front_plus_prediction, new_prediction_df], ignore_index=True)
            hypothetical_front = self.decision_analysis.compute_front_from_tests(hypothetical_front_plus_prediction)

            predicted_quality_indicators_with_new_test = self.decision_analysis.indicators_for_front(hypothetical_front, self.front_all_tests)
            predicted_quality_for_indicator = predicted_quality_indicators_with_new_test[self.indicator_id]
            log.debug(f"point on hypothetical front={len(hypothetical_front)},predicted_quality_for_indicator={predicted_quality_for_indicator}")
            # improvement_relative_factor of 1.0 will accept any increase
            include = (predicted_quality_for_indicator > (self.best_indicator_value * self.improvement_relative_factor))
            return include

    def accept_test(self, test_id, actual_test_metrics):
        self.accepted_tests.append(actual_test_metrics)
        accepted_tests_df = pd.DataFrame(self.accepted_tests)
        self.current_best_front = self.decision_analysis.compute_front_from_tests(accepted_tests_df)
        new_indicators = self.decision_analysis.indicators_for_front(self.current_best_front, self.front_all_tests)
        self.best_indicator_value = new_indicators[self.indicator_id]