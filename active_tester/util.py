import numpy as np
import unittest
from sklearn.metrics import accuracy_score


def draw_samples(p, num_samples=10000):
    """
    Draw samples from each row of p
    :param p: each row is the probability distribution of a particular item (num_items x num_classes)
    :param num_samples: number of samples to draw for each in total
    :return: num_items x num_samples array of samples
    """
    num_classes = np.size(p, axis=1)
    # Draw samples from the distribution
    samples_from_all = np.zeros((np.size(p,axis=0),num_samples))
    for i in range(len(p)):
        # This draws num_samples samples from the distribution of the first item
        samples_from_all[i,:] = np.random.choice(np.arange(num_classes), size=num_samples, p=p[i, :])

    return samples_from_all


def estimate_expectation(Q, p_classifier, p_truth, num_samples=10000):
    """
    Estimate the expected value of a function that takes a set of predicted labels and a set of true labels as input,
    where the expectation is taken over the probability distribution of the true labels
    :param Q: the function to estimate the expected value of
    :param p_classifier: classifier probabilities for each item (num_items x num_classes)
    :param p_truth: probability distribution of the true labels (num_items x num_classes)
    :param num_samples: number of samples to use to estimate the expectation
    :return: the estimate to the expected value of Q
    """

    y_classifier = np.argmax(p_classifier, axis=1)

    samples_from_all = draw_samples(p_truth, num_samples=num_samples)

    # Compute metric for each sample
    metric_evaluations = []
    for i in range(num_samples):
        metric_evaluations.append(Q(y_classifier, samples_from_all[:, i]))

    # return the mean of the metric evaluations
    return np.mean(metric_evaluations)

def estimate_expectation_fixed(Q, p_classifier, samples, i, c):
    """
    Estimate the expected value of a function that takes a set of predicted labels and a set of true labels as input.
    Here, we assume we are given the samples to use and only need to adjust item i to class c
    :param Q: the function to estimate the expected value of
    :param p_classifier: classifier probabilities for each item (num_items x num_classes)
    :param samples: num_items x num_samples array of categorical labels
    :param i: item to change
    :param c: class to change item i to
    :return: the estimate to the expected value of Q
    """

    y_classifier = np.argmax(p_classifier, axis=1)
    num_samples = np.size(samples, axis=1)

    # adjust row i to c
    samples[i, :] = c

    # Compute metric for each sample
    metric_evaluations = []
    for n in range(num_samples):
        metric_evaluations.append(Q(y_classifier, samples[:, n]))

    # return the mean of the metric evaluations
    return np.mean(metric_evaluations)

