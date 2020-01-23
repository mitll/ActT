import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
import active_tester
from sklearn.metrics import accuracy_score
from active_tester.query_strategy.random import Random
from active_tester.query_strategy.dpp import DPP
from active_tester.query_strategy.classifier_uncertainty import ClassifierUncertainty
from active_tester.query_strategy.noisy_label_uncertainty import LabelUncertainty
from active_tester.query_strategy.MCM import MCM
from active_tester.estimators.naive import Naive
from active_tester.estimators.learned import Learned
from active_tester.label_estimation.methods import oracle_one_label, no_oracle, oracle_multiple_labels
import pandas as pd

#SimulateDataset class
#Description: Creates a data object that contains the training and test data,
#noisy labels, and the model to be evaluatedself.
#Expected uer flow as follows:
#       obj = TestDataset() -> obj.genTestDataset() -> obj.genModel()
#Each of the attibutes of interest may be accessed through the obj object


def perturb_gt(labels,acc, num_classes):
    for i in range(len(labels)):
            if np.random.rand(1) > acc:
                labels[i] = np.random.choice(np.delete(np.arange(num_classes),
                                                                labels[i]))
    return labels


class TestDataset(object):
    """
    Description: constructor that takes optional parameters.
    :param num_samples: collective number of data samples in the testing
     and training datasets
    :param num_features: number of columns in the input array
    :param num_informative_features: number of informative features per
     cluster
    :param num_classes: number of columns in the output array
    :param num_clusters_per_class: number of clusters per class
    """
    def __init__(self, num_samples=None, confidence_level=-1, num_features=None,
                 num_informative_features=None, num_redundant_features=None, num_classes=None,
                 num_clusters_per_class = None):
        self.X = None
        self.y_true = None
        self.y_noisy = None
        self.X_train = None
        self.y_train = None
        self.model = None
        self.size = 0
        self.confidence = None
        self.confidence_level = confidence_level

        # If the optional parameters are not included in the function call
        # then set them to the default sklearn values
        if num_samples is None:
            self.num_samples = 100
        else:
            self.num_samples = num_samples

        if num_features is None:
            self.num_features = 20
        else:
            self.num_features = num_features

        if num_informative_features is None:
            self.num_informative_features = 2
        else:
            self.num_informative_features = num_informative_features

        if num_redundant_features is None:
            self.num_redundant_features = 0
        else:
            self.num_redundant_features = num_redundant_features

        if num_classes is None:
            self.num_classes = 2
        else:
            self.num_classes = num_classes

        if num_clusters_per_class is None:
            self.num_clusters_per_class = 2
        else:
            self.num_clusters_per_class = num_clusters_per_class

    # Getter and setter methods for model attributes
    def get_num_samples(self):
        return self.num_samples

    def get_num_features(self):
        return self.num_features

    def get_num_informative_features(self):
        return self.num_informative_features

    def get_num_redundant_features(self):
        return self.num_redundant_features

    def get_num_classes(self):
        return self.num_classes

    def get_num_clusters_per_class(self):
        return self.num_clusters_per_class

    def gen_test_dataset(self, num_experts=1, expert_accuracy=0.75):
        """
        Description: generate a classification dataset using sklearn's
         make_classification. Using the 'true' label generated from
         make_classification, inject noise into a copy of the labels to create noisy
         labels. Use a fraction of the training data as test data. Use the noisy
         label as the test data for the model.
        """

        X, y = datasets.make_classification(n_samples=self.get_num_samples(), n_features=self.get_num_features(),
                                            n_informative=self.get_num_informative_features(),
                                            n_redundant=self.get_num_redundant_features(),
                                            n_classes=self.get_num_classes(),
                                            n_clusters_per_class=self.get_num_clusters_per_class())

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

        self.X = X_test
        self.y_true = y_test
        self.size = len(self.X)

        self.X_train = X_train
        self.y_train = y_train

        # Transpose the noisy labels matrix to account for multiple experts
        self.y_noisy = np.array(list(y_test))
        self.y_noisy = np.reshape(self.y_noisy, (len(self.y_noisy), 1))
        self.y_noisy = np.repeat(self.y_noisy, num_experts, axis=1)

        for i in range(len(self.y_noisy)):
            for j in range(num_experts):
                if np.random.rand(1) > expert_accuracy:
                    self.y_noisy[i, j] = np.random.choice(np.delete(np.arange(self.get_num_classes()),
                                                                    self.y_noisy[i, j]))

    def gen_model(self):
        """
        Description: generate an MLP Classifier using the training data generated
        from genTestData()
        """
        clf = MLPClassifier(alpha=1, max_iter=1000)
        clf.fit(self.X_train, self.y_train)
        self.model = clf

    def test(self, num_experts=1, expert_accuracy=0.75):
        """
        Description: test the random query strategy and naive estimator methods by
        generating a dataset object and an ActiveTester object.
        :param budget: number of alloted queries
        """
        # Generate a sample dataset and model
        self.gen_test_dataset(num_experts=num_experts, expert_accuracy=expert_accuracy)
        self.gen_model()

    def compare_estimator_query(self, estimators=None, query_strategies=None,
                                sample_sizes=[100, 200, 300, 400, 500],
                                estimation_method=['No Oracle'],
                                vetter_acc=[1.0],
                                useX=True,
                                num_iterations=10):
        """
        Run active testing with various parameter combinations to estimate classifier accuracy.
        :param estimators: list of estimators to use
        :param query_strategies: list of query strategies to use
        :param sample_sizes: list of sample sizes to use
        :param estimation_method: list of estimation methods to use
        :param oracle_acc: list of oracle accuracies to use
        :param useX: boolean, True to use features, False to ignore
        :param num_iterations: how many times to run each combination
        :return:
        """

        true_metric = accuracy_score(self.model.predict(self.X), self.y_true)
        metric = accuracy_score

        estimator_list = {}
        query_strategy_list = {}
        estimation_method_list = {}

        estimation_method_options = {'No Oracle': no_oracle,
                                     'Oracle One Label': oracle_one_label,
                                     'Oracle Multiple Labels': oracle_multiple_labels}

        if estimation_method is not None:
            for l in estimation_method:
                estimation_method_list[l] = estimation_method_options[l]

        results_for_plotting = []
        for l_k, l_v in estimation_method_list.items():
            # Initialize possible query strategies
            learned = Learned(metric=metric, estimation_method=l_v, use_features=useX)
            naive = Naive(metric=metric)

            # Possible query stategies
            rand = Random()
            dpp = DPP()
            classifier_uncertainty_max = ClassifierUncertainty(option="greedy")
            classifier_uncertainty_sample = ClassifierUncertainty(option="sample")
            classifier_uncertainty_uniform_sample = ClassifierUncertainty(option="smoothed")
            label_uncertainty_max = LabelUncertainty(option="greedy")
            label_uncertainty_sample = LabelUncertainty(option="sample")
            label_uncertainty_uniform_sample = LabelUncertainty(option="smoothed")
            mcm_max = MCM(option="greedy", estimation_method=l_v)
            mcm_sample = MCM(option="sample", estimation_method=l_v)
            mcm_uniform_sample = MCM(option="smoothed", estimation_method=l_v)

            estimators_options = {'Naive': naive, 'Learned': learned}
            if estimators is None:
                estimator_list = estimators_options

            query_strategies_options = {'Random': rand,
                                        'Classifier Uncertainty Greedy': classifier_uncertainty_max,
                                        'Classifier Uncertainty Sample': classifier_uncertainty_sample,
                                        'Classifier Uncertainty Smoothed': classifier_uncertainty_uniform_sample,
                                        'Noisy Label Uncertainty Greedy': label_uncertainty_max,
                                        'Noisy Label Uncertainty Sample': label_uncertainty_sample,
                                        'Noisy Label Uncertainty Smoothed': label_uncertainty_uniform_sample,
                                        'MCM Greedy': mcm_max,
                                        'MCM Sample': mcm_sample,
                                        'MCM Smoothed': mcm_uniform_sample,
                                        'DPP': dpp}
            if query_strategies is None:
                query_strategy_list = query_strategies_options

            if estimators is not None:
                for est in estimators:
                    if est in estimators:
                        estimator_list[est] = estimators_options[est]
                    else:
                        print("One of the estimators is not a valid option.  Select from " +
                              '[%s]' % ', '.join(map(str, list(estimators_options.keys()))))

            if query_strategies is not None:
                for query in query_strategies:
                    if query in query_strategies_options:
                        query_strategy_list[query] = query_strategies_options[query]
                    else:
                        print("One of the estimators is not a valid option.  Select from " +
                              '[%s]' % ', '.join(map(str, list(query_strategies_options.keys()))))

            for est_k, est_v in estimator_list.items():
                for query_k, query_v in query_strategy_list.items():
                    for i in sample_sizes:
                        for acc in vetter_acc:
                            for iteration in range(num_iterations):

                                at = active_tester.ActiveTester(est_v, query_v)

                                temp_gt = perturb_gt(np.copy(self.y_true), acc, self.get_num_classes())

                                #Set dataset and model values in the active tester object
                                at.standardize_data(num=self.size, X=self.X,
                                                    classes=[str(i) for i in np.arange(0, self.num_classes)],
                                                    Y_ground_truth=temp_gt, Y_noisy=self.y_noisy)
                                at.gen_model_predictions(self.model)
                                at.query_vetted(False, i, 10)
                                at.test()

                                results = at.get_test_results()
                                results_for_plotting.append({'Estimator': est_k,
                                                             'Query Strategy': query_k,
                                                             'Number of Vetted Items': i,
                                                             'Vetter Error': np.round(1.0-acc, 2),
                                                             'Error': abs(results['tester_metric'] - true_metric),
                                                             'Estimation Method': l_k})

        results_for_plotting = pd.DataFrame(results_for_plotting)
        return results_for_plotting
