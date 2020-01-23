import numpy as np
from active_tester.query_strategy.base import BaseQueryStrategy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from scipy.linalg import qr
import unittest


def house(x):
    """
    Code to return info necessary to perform householder transform.
    Translated from Alg 5.1.1 in Matrix Computations 4th edition
    :param x: sub-vector of matrix column
    :return: vector and coefficient
    """
    sigma = np.linalg.norm(x[1:])**2
    v = x.copy()
    v[0] = 1
    if sigma == 0 and x[0] >= 0:
        beta = 0
    elif sigma == 0 and x[0] < 0:
        beta = -2
    else:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2*v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]
    return v, beta


def update(A_incoming, r):
    """
    Updates one column of matrix using householder transformation
    :param A_incoming: matrix to be modified
    :param r: only entries below and to the right of (r,r) will be changed
    :return: modified matrix
    """
    A = A_incoming
    v, beta = house(A[r:, r])
    v = v[:, np.newaxis]
    A[r:, r:] = A[r:, r:] - beta * v @ (np.transpose(v) @ A[r:, r:])
    A[(r + 1):, [r]] = v[1:]
    return A


def QRCP_reorthogonalize(A_incoming, k, y_vetted):
    """
    QR with column pivoting, with some columns orthoganalize first and truncates after k steps
    Translated from Algorithm 5.4.1 in Matrix Computations 4th edition
    We orthogonalize against vetted items first to encourage diversity
    between batches as well as within batches
    :param A_incoming: matrix to apply QRCP to
    :param k: number of items to select
    :param y_vetted: vector of current item labels
    :return: indices to vet and modified matrix
    """

    A = A_incoming
    m = np.size(A, axis=0)
    n = np.size(A, axis=1)

    permutation = np.arange(n)

    column_norms = np.linalg.norm(A, axis=0)**2
    num_vetted = np.sum(y_vetted != -1)
    vetted_indices = np.arange(len(y_vetted))[y_vetted != -1]
    # orthogonalize against these first
    r = -1
    for index in vetted_indices:
        r = r + 1

        # swap columns
        temp = A[:, r]
        A[:, r] = A[:, index]
        A[:, index] = temp
        temp = permutation[r]
        permutation[r] = permutation[index]
        permutation[index] = temp

        A = update(A, r)

        # update column norms
        for i in range((r+1), n):
            column_norms[i] = column_norms[i] - A[r, i]**2

    tau = max(column_norms[(r+1):])
    while tau > 0 and r < num_vetted + k-1:
        r = r + 1
        norm_is_equal = np.isclose(column_norms, tau) * (np.arange(n) >= r)
        index = np.nonzero(norm_is_equal)[0][0]

        # swap columns
        temp = A[:, r]
        A[:, r] = A[:, index]
        A[:, index] = temp
        temp = column_norms[r]
        column_norms[r] = column_norms[index]
        column_norms[index] = temp
        temp = permutation[r]
        permutation[r] = permutation[index]
        permutation[index] = temp

        A = update(A, r)

        # update column norms
        for i in range((r+1), n):
            column_norms[i] = column_norms[i] - A[r, i]**2
        tau = max(column_norms[(r+1):])
    return np.asarray(permutation), A


def QRCP(A_incoming, k, y_vetted, permutation):
    """
    QR with column pivoting, with some columns orthoganalize first and truncates after k steps
    Translated from Algorithm 5.4.1 in Matrix Computations 4th edition
    We orthogonalize against vetted items first to encourage diversity
    between batches as well as within batches
    :param A_incoming: matrix to apply QRCP to
    :param k: number of items to select
    :param y_vetted: vector of current item labels
    :param permutation: current ordering of the columns
    :return: indices to vet and modified matrix
    """

    A = A_incoming
    m = np.size(A, axis=0)
    n = np.size(A, axis=1)

    column_norms = np.linalg.norm(A, axis=0)**2
    num_vetted = np.sum(y_vetted != -1)

    r = num_vetted - 1
    tau = max(column_norms[(r+1):])
    while tau > 0 and r < num_vetted + k-1:
        r = r + 1
        norm_is_equal = np.isclose(column_norms, tau) * (np.arange(n) >= r)
        index = np.nonzero(norm_is_equal)[0][0]

        # swap columns
        temp = A[:, r]
        A[:, r] = A[:, index]
        A[:, index] = temp
        temp = column_norms[r]
        column_norms[r] = column_norms[index]
        column_norms[index] = temp
        temp = permutation[r]
        permutation[r] = permutation[index]
        permutation[index] = temp

        A = update(A, r)

        # update column norms
        for i in range((r+1), n):
            column_norms[i] = column_norms[i] - A[r, i]**2
        # This will fail if we have already selected everything
        if r < n-1:
            tau = max(column_norms[(r+1):])
    return np.asarray(permutation), A


class DPP(BaseQueryStrategy):

    def __init__(self, X=None, y_noisy=None, p_classifier=None, epsilon=1, sigma=1, gamma=1):
        """

        :param X: matrix of features
        :param y_noisy: noisy labels
        :param p_classifier: classifier predicted probabilities
        :param epsilon: currently unused
        :param sigma: kernel scaling parameter
        :param gamma: power to raise item importance scores by
        """
        super().__init__(X, y_noisy, p_classifier)
        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.L = None
        self.permutation = None
        if p_classifier:
            self.num_classes = np.size(p_classifier, axis=1)

    def set_args(self, X, y_noisy, p_classifier):
        self.X = X
        # adjust X to 2D
        sizes = []
        for i in range(1, np.ndim(self.X)):
            sizes.append(np.size(self.X, axis=i))
        total_size = np.product(sizes)
        self.X = np.reshape(self.X, (np.size(self.X, axis=0), total_size))
        self.y_noisy = y_noisy
        self.p_classifier = p_classifier
        self.num_classes = np.size(p_classifier, axis=1)
        self.permutation = np.arange(len(y_noisy))

    def get_dependencies(self):
        return ['X', 'y_noisy', 'p_classifier', 'epsilon', 'sigma', 'gamma']

    def choose_indices(self, y_vetted, k):
        """
        Approximate the mode of the DPP and return indices
        :param y_vetted: currently vetted items
        :param k: number of items to select
        :return: list of select items
        """

        # Only do this the first time choose_indices is called
        if self.L is None:
            # Compute kernel
            S = squareform(pdist(X=self.X))
            S = np.exp(-S**2 / (2.0 * self.sigma))
            q = entropy(np.transpose(self.p_classifier))
            q = q**self.gamma
            self.L = q[None, :] * S * q[:, None]
            y_vetted = np.asarray(y_vetted)
            # Apply this version to orthogonalize against what was vetted in the initial steps
            self.permutation, self.L = QRCP_reorthogonalize(self.L, 0, y_vetted)

        y_vetted = np.asarray(y_vetted)
        # Greedy algorithm to select new items
        self.permutation, self.L = QRCP(self.L, k, y_vetted, self.permutation)

        num_vetted = np.sum(y_vetted != -1)

        return self.permutation[num_vetted:(num_vetted+k)]
