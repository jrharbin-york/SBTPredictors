# Indicator-based decision node
import random
from abc import ABCMeta, abstractmethod

import pandas as pd
from decision_node import DecisionNode, log

class IndicatorBasedDecisions(DecisionNode, metaclass=ABCMeta):
    def __init__(self, indicator_id, decision_analysis, min_tests_to_accept_first, improvement_min_factor = 1.0, execute_lower_prob=0.5):
        super().__init__()
        # TODO: need to supply the sign and flip if the indicator should be maximised
        self.current_best_front = None
        self.indicator_id = indicator_id
        self.best_indicator_value = None
        self.decision_analysis = decision_analysis
        # TODO: could maybe use improvement relative factor like simulated annealing
        #self.improvement_min_factor = improvement_min_factor
        self.execute_lower_prob = execute_lower_prob

        # prob of acceptance is linear shift from 0 (at improvement_rel_Factor e.g. 0.8) to 1 at 1.0 or higher

        self.accepted_tests = []
        self.min_tests_to_accept_first = min_tests_to_accept_first

    def description(self):
        return f"IndicatorBased({self.indicator_id}, {self.execute_lower_prob})"

    @abstractmethod
    def relative_quality(self, predicted_test_metrics):
        pass

    def execute_or_not(self, test_id, predicted_test_metrics):
        if (self.best_indicator_value is None) or (len(self.accepted_tests) < self.min_tests_to_accept_first):
            # Always accept the first test
            return True
        else:
            hypothetical_front_plus_prediction = pd.DataFrame(self.current_best_front)
            new_prediction_df = pd.DataFrame(predicted_test_metrics)
            hypothetical_front_plus_prediction = pd.concat([hypothetical_front_plus_prediction, new_prediction_df], ignore_index=True)
            hypothetical_front = self.decision_analysis.compute_front_from_tests(hypothetical_front_plus_prediction)

            predicted_quality_indicators_with_new_test = self.decision_analysis.indicators_for_front(hypothetical_front, self.front_all_tests)
            predicted_hv_for_indicator = predicted_quality_indicators_with_new_test[self.indicator_id]
            log.debug(f"point on hypothetical front={len(hypothetical_front)},predicted_hv_for_indicator={predicted_hv_for_indicator}")
            # improvement_relative_factor of 1.0 will accept any increase
            
            if predicted_hv_for_indicator > self.best_indicator_value:
                # always accept a better value
                return True
            else:
                relative_quality = self.relative_quality(predicted_test_metrics)
                # compute a probability that goes to zero for self.improvement_relative_factor

                # disable the improvment_min_factor to simplify everything - self.improvement_min_factor always 0
                #prob_for_quality = self.execute_lower_prob * (relative_quality - self.improvement_min_factor) / (1 - self.improvement_min_factor)

                prob_for_quality = self.execute_lower_prob * relative_quality

                log.debug(f"relative_quality={relative_quality},prob_for_quality={prob_for_quality}")
                # if the prediction is worse, accept with given probability
                return random.uniform(0.0, 1.0) < prob_for_quality

    def accept_test(self, test_id, actual_test_metrics):
        self.accepted_tests.append(actual_test_metrics)
        accepted_tests_df = pd.DataFrame(self.accepted_tests)
        self.current_best_front = self.decision_analysis.compute_front_from_tests(accepted_tests_df)
        new_indicators = self.decision_analysis.indicators_for_front(self.current_best_front, self.front_all_tests)
        self.best_indicator_value = new_indicators[self.indicator_id]