import numpy as np
from active_tester.query_strategy.base import BaseQueryStrategy
import unittest
from scipy.stats import entropy


def compute_vote_entropy(y_vetted, p_classifier, y_noisy):
    """
    Compute the entropy of the distribution induced by the votes, for each item
    :param y_vetted: vetted items
    :param p_classifier: classifier predicted probabilities
    :param y_noisy: noisy votes
    :return: the indices that are available and the selected items
    """
    # Compute votes
    num_items = np.size(p_classifier, axis=0)
    num_classes = np.size(p_classifier, axis=1)
    votes = np.zeros((num_items, num_classes))
    for j in range(num_classes):
        votes[:, j] = np.sum(y_noisy == j, axis=1)
    # Compute entropy
    y_vetted = np.asarray(y_vetted)
    unlabeled = y_vetted == -1
    available_indices = np.arange(len(y_noisy))[unlabeled]
    num_votes = np.sum(votes, axis=1)
    vote_distribution = votes / num_votes[:, None]
    vote_entropy = entropy(np.transpose(vote_distribution[unlabeled]))
    vote_entropy = vote_entropy / np.sum(vote_entropy)
    return available_indices, vote_entropy


def normalize(distribution):
    return distribution / np.sum(distribution)


class LabelUncertainty(BaseQueryStrategy):

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
        if self.option == 'greedy':
            available_indices, vote_entropy = compute_vote_entropy(y_vetted, self.p_classifier, self.y_noisy)
            return available_indices[np.argsort(vote_entropy)[::-1][:k]]
        elif self.option == 'sample':
            available_indices, vote_entropy = compute_vote_entropy(y_vetted, self.p_classifier, self.y_noisy)
            sampled_indices = np.random.choice(available_indices, size=k, replace=False, p=vote_entropy)
            return sampled_indices
        else:
            available_indices, vote_entropy = compute_vote_entropy(y_vetted, self.p_classifier, self.y_noisy)
            hybrid_distribution = 0.5*vote_entropy+0.5*(1.0/len(vote_entropy))*np.ones(len(vote_entropy))
            hybrid_distribution = normalize(hybrid_distribution)
            sampled_indices = np.random.choice(available_indices, size=k, replace=False, p=hybrid_distribution)
            return sampled_indices
