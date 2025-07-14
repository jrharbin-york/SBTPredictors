from abc import ABCMeta, abstractmethod
import random
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

    def description(self):
        return self.__class__.__name__

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




