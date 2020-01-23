import numpy as np
from active_tester.query_strategy.base import BaseQueryStrategy
import unittest


class Random(BaseQueryStrategy):

    def get_dependencies(self):
        return ['X', 'y_noisy', 'p_classifier']

    def choose_indices(self, y_vetted, k):
        y_vetted = np.asarray(y_vetted)
        available_indices = np.arange(len(self.y_noisy))[y_vetted == -1]
        sampled_indices = np.random.choice(available_indices, size=k, replace=False)
        return sampled_indices
