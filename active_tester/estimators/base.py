import numpy as np
from abc import ABC, abstractmethod


class BaseMetricEstimationStrategy(ABC):

    def __init__(self, metric, y_noisy=None, p_classifier=None):
        """
        :param metric: the metric we want to estimate (function which takes two arguments, true labels and predicted)
        :param y_noisy: the noisy labels (vector)
        :param p_classifier: classifier predicted probabilities (array, num_items by num_classes)
        """
        self.metric = metric
        self.y_noisy = np.asarray(y_noisy)
        if self.y_noisy.ndim == 1:
            self.y_noisy = self.y_noisy[:, None]
        if p_classifier is not None:
            self.p_classifier = np.asarray(p_classifier)
            self.y_classifier = np.argmax(self.p_classifier, axis=1)
            self.num_classes = np.size(p_classifier, axis=1)

    def set_args(self, y_noisy, p_classifier):
        self.y_noisy = y_noisy
        self.p_classifier = p_classifier
        self.y_classifier = np.argmax(self.p_classifier, axis=1)
        self.num_classes = np.size(p_classifier, axis=1)

    @abstractmethod
    def estimate(self, y_vetted):
        """
        Perform the metric estimation
        :param y_vetted: vetted labels, -1 if unvetted (vector)
        :return: estimate to the metric
        """
        pass
