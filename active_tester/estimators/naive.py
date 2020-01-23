import numpy as np
import unittest
from .base import BaseMetricEstimationStrategy
from sklearn.metrics import accuracy_score


class Naive(BaseMetricEstimationStrategy):

    def estimate(self, y_vetted):
        y_groundtruth_estimated = np.asarray(y_vetted)
        # insert the noisy labels at any place where there is no vetted label
        unlabeled = y_groundtruth_estimated == -1
        y_groundtruth_estimated[unlabeled] = self.y_noisy[unlabeled, 0]
        estimated_metric_value = self.metric(y_groundtruth_estimated, self.y_classifier)

        return {'tester_labels': y_groundtruth_estimated, 'tester_metric': estimated_metric_value,
                'tester_prob': None}
