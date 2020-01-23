import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import unittest
from statswag.estimators import MLEOneParameterPerLabeler


def create_expert_probabilities(true, predicted, num_classes):
    """
    Create a smoothed matrix of confusion matrix probabilities
    :param true: true labels
    :param predicted: predicted labels
    :param num_classes: number of classes
    :return: smoothed matrix of confusion matrix probabilities
    """

    # Compute confusion matrix and then normalize to get probabilities
    # Rows of the confusion matrix are the true labels and columns are the predicted
    noisy_labels_confusion_matrix = confusion_matrix(true, predicted, labels=np.arange(num_classes))
    # Smoothing the confusion matrix to prevent 0 values
    noisy_labels_confusion_matrix = noisy_labels_confusion_matrix + 0.5
    row_sums = np.sum(noisy_labels_confusion_matrix, axis=1)
    noisy_labels_confusion_matrix = noisy_labels_confusion_matrix / row_sums[:, None]
    return noisy_labels_confusion_matrix


def compute_label_probabilities(confusion, expert_predictions, probabilities, X=None):
    """
    Compute label probabilities using approach in the active testing paper
    :param confusion: list of confusion matrices (normalized to hold probabilities), num_experts in length
    :param expert_predictions: num_items x num_experts matrix (or vector) of noisy labels
    :param probabilities: learned probabilities of labels given classifier scores
    :return: matrix (num_items x num_classes) of label probabilities
    """

    num_items = np.size(expert_predictions, axis=0)
    num_classes = np.size(probabilities, axis=1)
    num_experts = len(confusion)

    # Compute probability distribution over the potential labels, for each item
    label_distribution = np.zeros((num_items, num_classes))
    if num_experts > 1:
        for i in range(num_items):
            prod = 1.0*np.ones(num_classes)
            for j in range(num_experts):
                if expert_predictions[i, j] != -1:
                    prod = prod*confusion[j][:, expert_predictions[i, j]]
            label_distribution[i, :] = prod * probabilities[i, :]
        row_sums = np.sum(label_distribution, axis=1)
        label_distribution = label_distribution / row_sums[:, None]
    else:
        for i in range(num_items):
            # Entry c of row i is Prob(y_noisy_i|y_vetted_i=c)*Prob(y_vetted_i=c|p_classifier_i)
            label_distribution[i, :] = confusion[0][:, expert_predictions[i, 0]] * probabilities[i, :]
        row_sums = np.sum(label_distribution, axis=1)
        label_distribution = label_distribution / row_sums[:, None]

    return label_distribution


def oracle_one_label(y_noisy, p_classifier, y_vetted, X=None):
    """
    Estimate the label distribution when we assume the vetter is an oracle and there is one noisy label
    :param y_noisy: noisy labels, should be vector of length num_items, should be between 0 and num_classes - 1
    :param p_classifier: prediction, should be a matrix of num_items x num_classes, all values between 0 and 1,
    all rows sum to 1
    :param y_vetted: vetted labels, should be either -1 if no vetted label exists, or between 0 and num_classes - 1
    :return: probability distribution for each item, matrix of num_items x num_classes, all values between 0 and 1
    """

    num_classes = np.size(p_classifier, axis=1)
    # Determine vetted portion of the dataset
    V = (y_vetted != -1)

    # Determine vetted subsets of y_noisy, p_classifier, and y_vetted
    y_noisy_V = y_noisy[V]
    p_classifier_V = p_classifier[V]
    y_vetted_V = y_vetted[V]

    noisy_labels_confusion_matrix = create_expert_probabilities(y_vetted_V, y_noisy_V, num_classes)

    # Train a logistic regression classifier on the vetted data
    classifier = LogisticRegression(solver='lbfgs', multi_class='auto')
    if X is not None:
        X_V = X[V, :]
        classifier.fit(np.hstack((p_classifier_V, X_V)), y_vetted_V)
        # Apply classifier to all data (only the prediction on unvetted data will be used)
        predicted_probabilities = classifier.predict_proba(np.hstack((p_classifier, X)))
    else:
        classifier.fit(p_classifier_V, y_vetted_V)
        # Apply classifier to all data (only the prediction on unvetted data will be used)
        predicted_probabilities = classifier.predict_proba(p_classifier)

    label_distribution = compute_label_probabilities([noisy_labels_confusion_matrix], y_noisy, predicted_probabilities)

    # For items where we have a vetted label, replace the distribution one where all weight is on the vetted label
    label_distribution[V == 1, :] = 0.0
    label_distribution[V == 1, y_vetted[V == 1]] = 1.0

    return label_distribution


def oracle_multiple_labels(y_noisy, p_classifier, y_vetted, X):
    """
    Estimate the label distribution when we assume the vetter is an oracle and there is more than one noisy label
    :param y_noisy: noisy labels, should be matrix of num_items x num_experts, should be between 0 and num_classes - 1
    :param p_classifier: prediction, should be a matrix of num_items x num_classes, all values between 0 and 1,
    all rows sum to 1
    :param y_vetted: vetted labels, should be either -1 if no vetted label exists, or between 0 and num_classes - 1
    :return: probability distribution for each item, matrix of num_items x num_classes, all values between 0 and 1
    """

    num_classes = np.size(p_classifier, axis=1)
    num_experts = np.size(y_noisy, axis=1)

    # Compute confusion matrix for each expert and then normalize to get probabilities
    # Rows of the confusion matrix are the true labels and columns are the predicted
    noisy_labels_confusion_matrices_list = []
    for j in range(num_experts):
        # Determine subsets of the dataset labeled by each expert AND vetted
        subset = (y_noisy[:, j] != -1) * (y_vetted != -1)
        noisy_labels_confusion_matrices_list.append(create_expert_probabilities(y_vetted[subset],
                                                                                y_noisy[subset, j], num_classes))

    # Train a logistic regression classifier on the vetted data
    classifier = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=200)
    subset = y_vetted != -1
    if X is not None:
        classifier.fit(np.hstack((p_classifier[subset, :], X[subset, :])), y_vetted[subset])
        # Apply classifier to all data (only the prediction on unvetted data will be used)
        predicted_probabilities = classifier.predict_proba(np.hstack((p_classifier, X)))
    else:
        classifier.fit(p_classifier[subset, :], y_vetted[subset])
        predicted_probabilities = classifier.predict_proba(p_classifier)

    label_distribution = compute_label_probabilities(noisy_labels_confusion_matrices_list, y_noisy,
                                                     predicted_probabilities)

    # For items where we have a vetted label, replace the distribution one where all weight is on the vetted label
    subset = y_vetted != -1
    label_distribution[subset, :] = 0.0
    label_distribution[subset, y_vetted[subset]] = 1.0

    return label_distribution


def no_oracle(y_noisy, p_classifier, y_vetted, X=None):

    y_classifier = np.argmax(p_classifier, axis=1)[:, np.newaxis]
    if y_noisy.ndim == 1:
        y_noisy = y_noisy.reshape(-1, 1)
    y_vetted = y_vetted.reshape(-1, 1)
    all_labels = np.hstack((y_classifier, y_noisy, y_vetted))
    all_labels = np.asarray(all_labels, dtype=np.object)
    all_labels[all_labels == -1] = np.nan

    results = MLEOneParameterPerLabeler(n_classes=np.size(p_classifier, axis=1)).fit(all_labels)
    return results['probs']
