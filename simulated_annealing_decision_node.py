import math
import random
from abc import ABCMeta

from decision_node import DecisionNode, log

class SimulatedAnnealingThreshold(DecisionNode, metaclass=ABCMeta):
    pass

class SimulatedAnnealingThresholdMultiDimensional(SimulatedAnnealingThreshold):
    """Simulated annealing has a probability of accepting a worst solution than the current best -
    computes the distances between predicted and best actual and uses the diagonal distance to calibrate
    the probability based upon temperature"""

    # TODO: currently assumes minimisation only
    def __init__(self, target_metrics_ids, distance_divisor_per_metric, initial_temperature = 100.0):
        super().__init__()
        self.target_metric_ids = target_metrics_ids
        self.initial_temperature = initial_temperature
        self.current_best_per_metric = {}
        self.tests_per_epoch_increment = 1.0
        self.distance_divisor_per_metric = distance_divisor_per_metric
        self.epoch = 0

    def description(self):
        return f"SimAnnealingMulti({self.initial_temperature})"

    def execute_or_not(self, test_id, predicted_metrics):
        execute = False
        diff_m_squared = 0.0

        self.epoch += 1.0 / float(self.tests_per_epoch_increment)

        for m in self.target_metric_ids:
            prediction_for_m = predicted_metrics[m]

            # Execute if there is no value known for a particular metric
            if not (m in self.current_best_per_metric):
                execute = True
            else:
                # Sum the distance squared scaled by the divisor for that metric
                diff_m = (prediction_for_m - self.current_best_per_metric[m]) / self.distance_divisor_per_metric[m]
                if diff_m < 0:
                    execute = True
                diff_m_squared += diff_m ** 2.0

        diff_all_metrics = math.sqrt(diff_m_squared)
        t_now = self.initial_temperature / float(self.epoch + 1)
        metropolis = math.exp(-diff_all_metrics / t_now)

        # TODO: diff_all_metrics < 0 will never be executed
        #if diff_all_metrics < 0 or (random.uniform(0.0, 1.0) < metropolis):
        if random.uniform(0.0, 1.0) < metropolis:
            execute = True

        log.debug(f"diff_all_metrics={diff_all_metrics}, epoch={self.epoch}, t_now={t_now}, metropolis={metropolis}, execute={execute}")
        return execute

    def accept_test(self, test_id, actual_test_metrics):
        for m in self.target_metric_ids:
            if not (m in self.current_best_per_metric):
                # TODO: rename current_best_prediction to current_best_value
                self.current_best_per_metric[m] = actual_test_metrics[m]
            else:
                # TODO: assumes minimisation is better
                if actual_test_metrics[m] < self.current_best_per_metric[m]:
                    self.current_best_per_metric[m] = actual_test_metrics[m]

class SimulatedAnnealingThresholdSingleDimensional(SimulatedAnnealingThreshold):
    """Simulated annealing has a probability of accepting a worst solution than the current best -
    computes the distances between predicted and best actual, using a function to reduce them to
    a single dimension. Then uses a probability based upon temperature"""

    # TODO: currently assumes minimisation only
    def __init__(self, target_metrics_ids, distance_divisor_per_metric, metric_weights, initial_temperature = 100.0):
        super().__init__()
        self.target_metric_ids = target_metrics_ids
        self.initial_temperature = initial_temperature
        self.current_best = None
        self.tests_per_epoch_increment = 1.0
        self.metric_weights = metric_weights
        self.distance_divisor_per_metric = distance_divisor_per_metric
        self.epoch = 0

    def description(self):
        return f"SimAnnealingSingle({self.initial_temperature})"

    def reduction_function(self, metric_hash):
        total = 0.0
        for metric_id in self.target_metric_ids:
            w = self.metric_weights[metric_id]
            print(f"metric_hash={metric_hash}")
            v = metric_hash[metric_id]
            divisor = self.distance_divisor_per_metric[metric_id]
            log.info(f"metric_id={metric_id}, w={w},v={v}, type(v)={type(v)} divisor={divisor}")
            total += w * (v / divisor)
        return total

    def execute_or_not(self, test_id, predicted_metrics):
        execute = False
        self.epoch += 1.0 / float(self.tests_per_epoch_increment)
        prediction_for_m = self.reduction_function(predicted_metrics)

        # Execute if there is no value known for a particular metric
        if self.current_best is None:
            log.debug(f"Executing test since no current best defined")
            execute = True
        else:
            dist = (prediction_for_m - self.current_best)
            t_now = self.initial_temperature / float(self.epoch + 1)
            metropolis = math.exp(-dist / t_now)
            log.debug(f"dist={dist}, epoch={self.epoch}, t_now={t_now}, metropolis={metropolis}, execute={execute}")
            if (dist < 0) or random.uniform(0.0, 1.0) < metropolis:
                execute = True

        return execute

    def accept_test(self, test_id, actual_test_metrics):
        if self.current_best is None:
            # TODO: rename current_best_prediction to current_best_value
            self.current_best = self.reduction_function(actual_test_metrics)
        else:
            # TODO: assumes minimisation is better
            reduced = self.reduction_function(actual_test_metrics)
            if reduced < self.current_best:
                self.current_best = reduced