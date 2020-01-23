import numpy as np
from active_tester.util import estimate_expectation
from active_tester.estimators.base import BaseMetricEstimationStrategy
from sklearn.metrics import accuracy_score
from active_tester.label_estimation.methods import oracle_one_label, no_oracle
import unittest


class Learned(BaseMetricEstimationStrategy):

    def __init__(self, metric, estimation_method, use_features=False, X=None, y_noisy=None, p_classifier=None):
        super().__init__(metric, y_noisy, p_classifier)
        self.estimate_z = estimation_method
        self.X = X
        self.use_features=use_features
        if X is not None:
            sizes = []
            for i in range(1, np.ndim(self.X)):
                sizes.append(np.size(self.X, axis=i))
            total_size = np.product(sizes)
            self.X = np.reshape(self.X, (np.size(self.X, axis=0), total_size))

    def set_args(self, y_noisy, p_classifier, X):
        super().set_args(y_noisy, p_classifier)
        self.X = X
        if X is not None:
            sizes = []
            for i in range(1, np.ndim(self.X)):
                sizes.append(np.size(self.X, axis=i))
            total_size = np.product(sizes)
            self.X = np.reshape(self.X, (np.size(self.X, axis=0), total_size))



    def estimate(self, y_vetted):
        y_vetted = np.asarray(y_vetted)
        if np.all(y_vetted == -1):
            p_groundtruth_estimated = no_oracle(self.y_noisy, self.p_classifier, y_vetted)
        elif self.use_features:
            p_groundtruth_estimated = self.estimate_z(self.y_noisy, self.p_classifier, y_vetted, X=self.X)
        else:
            p_groundtruth_estimated = self.estimate_z(self.y_noisy, self.p_classifier, y_vetted, X=None)
        y_groundtruth_estimated = np.argmax(p_groundtruth_estimated, axis=1)
        estimated_metric_value = estimate_expectation(self.metric, self.p_classifier, p_groundtruth_estimated)
        return {'tester_labels': y_groundtruth_estimated, 'tester_metric': estimated_metric_value,
                'tester_prob': p_groundtruth_estimated}
