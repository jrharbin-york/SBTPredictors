from decision_node import DecisionNode, MissingMetric, MissingThreshold

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

    def description(self):
        return f"FixedThreshold({self.metrics_needed})"

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