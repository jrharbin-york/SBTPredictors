# Indicator-based decision node
import random

import numpy as np
from pymoo.indicators.gd import GD
import pandas as pd
from decision_node import DecisionNode, log
from indicator_based_decision_node import IndicatorBasedDecisions

class HypervolumeWithGDRelativeQuality(IndicatorBasedDecisions):
    def extract_point_from_front_df(self, front_df, point_index):
        if point_index < len(front_df):
            extracted_point_df = front_df.iloc[[point_index]]
            front_df_copy = front_df.copy()
            front_without_point_df = front_df_copy.drop(point_index).reset_index(drop=True)
            return extracted_point_df, front_without_point_df
        else:
            return None, None

    def get_all_gds_for_existing_front_points(self, front_df):
        gds = []
        for point_index, _ in front_df.iterrows():
            point_df, front_without_point_df = self.extract_point_from_front_df(front_df, point_index)
            # if the front is empty, there will be no GD for this point
            if point_df is not None:
                gd_with_removed_point = self.decision_analysis.gd_for_point(front_without_point_df, point_df)
                gds.append(gd_with_removed_point)
        return gds

    def relative_quality(self, predicted_test_metrics):
        current_best_front_df = pd.DataFrame(self.current_best_front)
        predicted_test_metrics_df = pd.DataFrame(predicted_test_metrics)

        gd_for_hypothetical_point = self.decision_analysis.gd_for_point(current_best_front_df, predicted_test_metrics_df)
        other_gds = self.get_all_gds_for_existing_front_points(current_best_front_df)

        if len(other_gds) > 0:
            rq = 1.0 - gd_for_hypothetical_point / np.max(other_gds)
        else:
            rq = 1.0
        print(f"predicted_test_metrics={predicted_test_metrics},gd_for_new_point={gd_for_hypothetical_point}, other_gds={other_gds}, rq={rq}")
        return rq