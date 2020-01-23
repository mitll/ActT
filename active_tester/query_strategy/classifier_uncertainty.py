import numpy as np
from active_tester.query_strategy.base import BaseQueryStrategy
import unittest
from scipy.stats import entropy


def compute_entropy_distribution(y_vetted, y_noisy, p_classifier):
    """
    Computes entropy of a set of distributions (rows of p_classifier)
    :param y_vetted: labels for vetted items
    :param y_noisy: noisy labels
    :param p_classifier: classifier predicted probabilities
    :return:
    """
    y_vetted = np.asarray(y_vetted)
    unlabeled = y_vetted == -1
    available_indices = np.arange(len(y_noisy))[unlabeled]
    classifier_entropy = entropy(np.transpose(p_classifier[unlabeled]))
    classifier_entropy = normalize(classifier_entropy)
    return available_indices, classifier_entropy


def normalize(distribution):
    return distribution / np.sum(distribution)


class ClassifierUncertainty(BaseQueryStrategy):

    def __init__(self, X=None, y_noisy=None, p_classifier=None, option='smoothed'):
        """
        Initialize the query strategy object
        :param X: Features of items, can be None (num_items by num_features)
        :param y_noisy: noisy label of each item (vector)
        :param p_classifier: classifier predicted probabilities (array, num_items by num_classes)
        :param option: Can be greedy, sample, or otherwise defaults to smoothed.  Greedy items with the largest
        probability.  Sample chooses according to the distribution defined by the entropy of the classifier
        predicted probabilities.  Smoothed combines the above distribution with a uniform distribution.
        """
        if X is not None:
            self.X = np.asarray(X)
        else:
            self.X = X
        self.y_noisy = np.asarray(y_noisy)
        self.p_classifier = np.asarray(p_classifier)
        self.option = option

    def get_dependencies(self):
        return ['X', 'y_noisy', 'p_classifier']

    def choose_indices(self, y_vetted, k):
        """
        See init comments for explanation of the three strategies
        :param y_vetted: vetted items
        :param k: number of samples
        :return: list of sampled indices
        """
        if self.option == 'greedy':
            available_indices, classifier_entropy = compute_entropy_distribution(y_vetted, self.y_noisy,
                                                                                 self.p_classifier)
            return available_indices[np.argsort(classifier_entropy)[::-1][:k]]
        elif self.option == 'sample':
            available_indices, classifier_entropy = compute_entropy_distribution(y_vetted, self.y_noisy,
                                                                                 self.p_classifier)
            sampled_indices = np.random.choice(available_indices, size=k, replace=False, p=classifier_entropy)
            return sampled_indices
        else:
            available_indices, classifier_entropy = compute_entropy_distribution(y_vetted, self.y_noisy,
                                                                                 self.p_classifier)
            hybrid_distribution = 0.5 * classifier_entropy + \
                                  0.5 * (1.0/len(classifier_entropy)) * np.ones(len(classifier_entropy))
            hybrid_distribution = normalize(hybrid_distribution)
            sampled_indices = np.random.choice(available_indices, size=k, replace=False, p=hybrid_distribution)
            return sampled_indices
