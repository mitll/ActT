import numpy as np
from abc import ABC, abstractmethod


class BaseQueryStrategy(ABC):
    """
    Base class for active testing query strategies
    """

    def __init__(self, X=None, y_noisy=None, p_classifier=None):
        """
        Initialize the query strategy object
        :param X: Features of items, can be None (num_items by num_features)
        :param y_noisy: noisy label of each item (vector)
        :param p_classifier: classifier predicted probabilities (array, num_items by num_classes)
        """
        if X is not None:
            self.X = np.asarray(X)
        else:
            self.X = X
        self.y_noisy = np.asarray(y_noisy)
        self.p_classifier = np.asarray(p_classifier)

    def set_args(self, X, y_noisy, p_classifier):
        """
        Set args for query_strategy; used so user initialize blank constructor and
        query_vetted can populate query_strategy when metrics are calculated for a model
        :param X: Features of items, can be None (num_items by num_features)
        :param y_noisy: noisy label of each item (vector)
        :param p_classifier: classifier predicted probabilities (array, num_items by num_classes)
        """
        self.X = X
        self.y_noisy = y_noisy
        self.p_classifier = p_classifier

    @abstractmethod
    def get_dependencies(self):
        """
        Implements Return parameters needed to initialize and call choose_indices.
        :return: list of strings of arguments required; commmon keys defined in ActiveTesting
        """
        # Default return of get_dependencies; should be the minimum parameters that every query-strategy would need?
        return ['X', 'y_noisy', 'p_classifier']

    @abstractmethod
    def choose_indices(self, y_vetted, k):
        """
        Implements item selection strategy.  Depending on the strategy, it may select 1 item or more than 1 item.
        Should return the selected indices
        :param y_vetted: current vetted labels, -1 indicates unvetted
        :param k: number of samples to take
        :return: the sampled indices
        """
        pass
