# Indicator-based decision node
import random
from pymoo.indicators.gd import GD
import pandas as pd
from decision_node import DecisionNode, log
from indicator_based_decision_node import IndicatorBasedDecisions

class HypervolumeWithGDRelativeQuality(IndicatorBasedDecisions):
    def relative_quality(self, predicted_test_metrics):
        current_best_front_df = pd.DataFrame(self.current_best_front)
        predicted_test_metrics_df = pd.DataFrame(predicted_test_metrics)

        gd_for_new_point = self.decision_analysis.gd_for_point(current_best_front_df, predicted_test_metrics_df)
        max_possible_gd = max(max(abs(self.decision_analysis.front_to_numpy_array(predicted_test_metrics_df))))
        rq = 1 - (gd_for_new_point / max_possible_gd)
        print(f"predicted_test_metrics={predicted_test_metrics},gd_for_new_point={gd_for_new_point}, rq={rq}")
        return rq