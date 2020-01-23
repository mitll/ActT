import numpy as np
from active_tester.query_strategy.base import BaseQueryStrategy
from active_tester.query_strategy.random import Random
from active_tester.label_estimation.methods import oracle_one_label, oracle_multiple_labels
from active_tester.util import estimate_expectation, estimate_expectation_fixed, draw_samples
import unittest
from sklearn.metrics import accuracy_score


def compute_uncertainty_distribution(p_classifier, y_noisy, num_classes, unlabeled):
    label_probs = np.zeros((len(y_noisy), num_classes))
    for i in range(len(y_noisy)):
        label_probs[i, y_noisy[i]] = 1.0
    measure = np.sum(np.abs(label_probs[unlabeled, :] - p_classifier[unlabeled, :]), axis=1)
    measure = measure / np.sum(measure)
    return measure


class MCM(BaseQueryStrategy):

    def __init__(self, estimation_method, X=None, y_noisy=None, p_classifier=None, option='smoothed'):
        """
        Choose items related to the difference between the classifier predictions
        and the current ground truth estimates
        :param estimation_method: pass None when using the naive estimator
        :param X: matrix of features
        :param y_noisy: noisy labels
        :param p_classifier: classifier predicted probabilities
        :param option: Can be greedy, sample, or otherwise defaults to smoothed.  Greedy items with the largest
        probability.  Sample chooses according to the distribution defined by the entropy of the classifier
        predicted probabilities.  Smoothed combines the above distribution with a uniform distribution.
        """
        super().__init__(X, y_noisy, p_classifier)
        self.X = X
        if X is not None:
            sizes = []
            for i in range(1, np.ndim(self.X)):
                sizes.append(np.size(self.X, axis=i))
            total_size = np.product(sizes)
            self.X = np.reshape(self.X, (np.size(self.X, axis=0), total_size))
        self.estimate_z = estimation_method
        if p_classifier:
            self.num_classes = np.size(p_classifier, axis=1)
        self.option = option

    def set_args(self, X, y_noisy, p_classifier):
        self.X = X
        if X is not None:
            sizes = []
            for i in range(1, np.ndim(self.X)):
                sizes.append(np.size(self.X, axis=i))
            total_size = np.product(sizes)
            self.X = np.reshape(self.X, (np.size(self.X, axis=0), total_size))
        self.y_noisy = y_noisy
        self.p_classifier = p_classifier
        self.num_classes = np.size(p_classifier, axis=1)

    def get_dependencies(self):
        return ['X', 'y_noisy', 'p_classifier', 'estimation_method']

    def choose_indices(self, y_vetted, k):
        """
        Choose items to vet
        :param y_vetted: currently vetted items
        :param k: number of items to select
        :return: list of select items
        """

        y_vetted = np.asarray(y_vetted)
        unlabeled = y_vetted == -1
        available_indices = np.arange(len(self.y_noisy))[unlabeled]

        if self.estimate_z is not None:
            # if no items have been selected, we can't run estimate_z
            if np.sum(y_vetted > -1) == 0:
                return Random(self.X, self.y_noisy, self.p_classifier).choose_indices(y_vetted, k)
            else:
                label_probs = self.estimate_z(self.y_noisy, self.p_classifier, y_vetted, self.X)
                # 1-norm between probability distributions
                measure = np.sum(np.abs(label_probs[unlabeled, :] - self.p_classifier[unlabeled, :]), axis=1)
                measure = measure / np.sum(measure)
                if self.option == 'greedy':
                    return available_indices[np.argsort(measure)[::-1][:k]]
                elif self.option == 'sample':
                    return np.random.choice(available_indices, size=k, replace=False, p=measure)
                else:
                    hybrid_distribution = 0.5 * measure + 0.5 * (1.0 / len(measure)) * np.ones(len(measure))
                    return np.random.choice(available_indices, size=k, replace=False, p=hybrid_distribution)
        else:
            # computes uncertainty between noisy labels vote distribution and classifier label probabilties
            measure = compute_uncertainty_distribution(self.p_classifier, self.y_noisy, self.num_classes, unlabeled)
            measure = measure / np.sum(measure)
            if self.option == 'greedy':
                return available_indices[np.argsort(measure)[::-1][:k]]
            elif self.option == 'sample':
                return np.random.choice(available_indices, size=k, replace=False, p=measure)
            else:
                hybrid_distribution = 0.5 * measure + 0.5 * (1.0 / len(measure)) * np.ones(len(measure))
                return np.random.choice(available_indices, size=k, replace=False, p=hybrid_distribution)

