import numpy as np
import imghdr
from enum import Enum
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import csv

from .dataset import createDataset
from .model_interface import ModelInterface
from .query_strategy.prototypical import Prototypical
from active_tester.query_strategy.noisy_label_uncertainty import LabelUncertainty
from active_tester.query_strategy.classifier_uncertainty import ClassifierUncertainty
from .estimators.learned import Learned

class InteractiveType(Enum):
	PLOTLIB = 1
	OPENFILE = 2
	VISUALIZER = 3
	RAWX = 4
	IMGBYTEX = 5

class ActiveTester:
	'''
	ActiveTester class
	Description: the main interface for users to perform their active
	testing.  Expected userflow is to call gen_data -> gen_model_predictions()
	-> query_oracle(optional) -> test() .  From there, user can call various
	getter methods to grab metrics of importance or return the result map to
	and access metrics directly from the dictionary.
	'''

	# Set in constructor or setters
	estimator = None		# function
	query_strategy = None	# function

	# Set in standardize_data
	X = np.asarray([]) 		# np.array()
	X_is_img = False
	X_feature_label = np.asarray([])	# np.array()
	Y_ground_truth = np.asarray([])		# np.array()
	Y_noisy = np.asarray([])			# np.array()
	classes = {}						# dict {'class': int}
	dataset = None						# dataset._Dataset object


	# Set in gen_model_predictions()
	model_results = {}	# map {'model_labels': np.array(), 'Y_prime_prob': np.array(), ...(unsure)}

	# Set in gen_data or query_oracle
	Y_vetted = np.asarray([])			# np.array()

	# Set in test()
	test_results = {}		# map {'tester_labels': np.array(), tester_prob: np.array(), tester_metric: np.array()}


	def __init__(self, estimator, query_strategy):
		'''
		Description: constructor that takes estimator and query_strategy as input
		:param estimator: function used to estimate ground truth labels
		:param query_strategy: function used to specify samples for querying oracle
		:return: None
		'''
		self.estimator = estimator
		self.query_strategy = query_strategy
		self.rearrange=None


	# Getters and setters
	def get_X(self):
		return self.X

	def set_X(self, X):
		self.X = X

	def set_Y_noisy(self, Y_noisy):
		self.Y_noisy = Y_noisy

	def set_confidence(self, confidence):
		self.confidence = confidence

	def get_model_results(self):
		return self.model_results

	def set_model_results(self, model_results):
		self.model_results = model_results

	def set_prob_array(self, model_results):
		self.model_results['probabilities'] = model_results

	def get_test_results(self):
		return self.test_results

	def get_Y_vetted(self):
		return self.Y_vetted

	def set_Y_vetted(self, Y_vetted):
		self.Y_vetted = Y_vetted


	def _check_ground_truth(self):
		check = True

		if self.Y_ground_truth.size != 0:
			for i in self.Y_ground_truth:
				if i == -1:
					check = False
					break
		else:
			check = False

		return check


	def _check_Y_vetted(self):
		num_classes = len(self.classes)
		class_found = [False for x in range(num_classes)]
		found_count = 0

		for _class in self.Y_vetted:
			if _class != -1:
				if not class_found[_class]:
					class_found[_class] = True

					found_count += 1
					if found_count == num_classes:
						return True

		return False

	def _query_vetted_index(self, indices, interactive, interactive_type, raw=None, visualizer=None, class_labels=None):
		for index in indices:
			# Check if indices are integers
			if not isinstance(index, (int, np.integer)):
				print('Error: indices are not int values')
				return

			# Condition if interactive; display and query user input
			current_vetted_label = 0
			if interactive:
				self._visualize_row(interactive_type, index, raw)
				print("The available labels are: %s" % list(self.classes.keys()))

				# Query for label
				vetted_label = None
				valid = False
				while not valid:
					try:
						vetted_label = input("Label the provided item: ").lower()
						self.Y_vetted[index] = self.classes[vetted_label]
						current_vetted_label = self.classes[vetted_label]
						valid = True
						print("\n")
					except KeyError:
						print("Only accept the following labels: %s" % list(self.classes.keys()))
			# Query ground truth if interactive is disabled
			else:
				self.Y_vetted[index] = self.Y_ground_truth[index]
				current_vetted_label = self.Y_ground_truth[index]

			# Class_labels only used if calling _query_vetted_index from pre-processing step
			if class_labels is not None:
				class_labels[current_vetted_label] = True
		return class_labels

	def _visualize_row(self, interactive_type, index, raw):
		'''
		Description: visualize row to user
		:param interactive_type: Enum type of how to visualize row to user
		:param index: <INT> index of row to display
		:param raw: <STRING> file to display to user
		:return: None
		'''
		if interactive_type == InteractiveType.PLOTLIB:
			img = cv2.imread(np.array(raw)[index], 0)
			plt.figure()
			plt.imshow(img, cmap='gray')
			plt.colorbar()
			plt.grid(False)
			plt.show()

		elif interactive_type == InteractiveType.OPENFILE:
			print("\"")
			with open(np.array(raw)[index]) as f:
				for line in f:
					print(line)
			print("\"")
			print("\n")

		elif interactive_type == InteractiveType.VISUALIZER:
			# TODO: Implement function call to print only map specified
			pass

		elif interactive_type == InteractiveType.IMGBYTEX:
			plt.figure()
			plt.imshow(np.squeeze(self.X[index]), cmap='gray')
			plt.colorbar()
			plt.grid(False)
			plt.show()

		else:
			print(self.get_X()[index])

		return

	def standardize_data(self, rearrange=False, is_img_byte=False, num=-1, X=None, classes=[],
						 Y_ground_truth=None,Y_vetted=None,Y_noisy=None):
		"""
		Description: takes data from various ML libraries, parses it to numpy arrays,
		samples it using a given sample strategy, and set them as attributes in ActiveTester
		:param rearrange: whether to shuffle data
		:param is_img_byte: whether raw data can be displayed as an image
		:param num: number of samples to draw from the dataset (set to -1 for all)
		:param X: array of features for items
		:param classes: list of string classes to include as choices for querying vetted user;
		must be in order of class numeric value
		:param Y_ground_truth: array or dataframe from various ML frameworks of known ground truth labels
		:param Y_vetted: array or dataframe from various ML frameworks of noisy labels
		:param Y_noisy: array or dataframe from various ML frameworks of oracle's labels
		:return:
		"""

		self.rearrange=rearrange
		self.dataset = createDataset(features_array=X, classes=classes, y_dim=1, Y_ground_truth=Y_ground_truth,
									 Y_vetted=Y_vetted, Y_noisy=Y_noisy)

		# Create labels
		labels = ["noisy", "vetted"]

		# Check if ground truth exists
		if Y_ground_truth is not None:
			labels.append("ground_truth")

		sample = self.dataset.sample(num=num, rearrange=rearrange, labels=labels)

		self.Y_vetted = sample["vetted"]
		self.Y_noisy = sample["noisy"]
		if Y_ground_truth is not None:
			self.Y_ground_truth = sample["ground_truth"]
		self.X_is_img = is_img_byte
		self.X = sample["features"]

		# Create classes dictionary
		self.classes = {}
		for i, cl in enumerate(classes):
			self.classes[cl.lower()] = i

		return

	def gen_model_predictions(self, model, outputs=None):
		""""
		Description: wraps model to a shared format, runs it on parsed dataset, and
		calculates sets of metrics from model and stores as a large dictionary attribute model_results
		:param model: model from various ML libraries (supports sklearn)
		:param outputs: list of tensorflow outputs
		:return: None
		"""
		# Tenserflow case
		if isinstance(model, tf.keras.Model):
			if outputs:
				model_wrapper = ModelInterface(model, len(self.classes), 'tf', outputs=outputs)
			else:
				model_wrapper = ModelInterface(model, len(self.classes), 'tf',
											   outputs=[('probabilities',len(self.classes))])

		# Assume SKLEARN
		else:
			model_wrapper = ModelInterface(model, len(self.classes))

		self.model_results = model_wrapper.predict(np.copy(self.X))
		return

	def query_vetted(self, interactive, budget, batch_size=1, raw=None, visualizer=None):
		"""
		Description: uses given query strategy to query user (oracle), label data, and
		store as attribute
		:param interactive: <Boolean> if querying user or will attempt to find vetted labels in Dataset
		:param budget: number of samples to label
		:param batch_size: how many items to select at a time
		:param raw: <NUMPY array> of file resources; if none show raw X
		:param visualizer: <FUNCTION> generic function that takes a X row as arg and returns dictionary to print
		to user; if none show all X; does not apply if raw has value
		:return: None
		"""

		if (raw is not None) and (self.rearrange == True):
			print("There may be a mismatch between the ordering of the vetted labels and the items. Please set rearragne to False")

		if budget < len(self.classes.keys()):
			print("budget needs to be greater than or equal to the number of classes: %d" % len(self.classes.keys()))
			print ("Exiting query_vetted...")
			return

		if budget % batch_size != 0:
			print("budget must be divisible by batch_size")
			print("Exiting query_vetted...")
			return

		if 'probabilities' in self.model_results:
			# Line checks if there is a 1.0 in every row of probabilities
			if np.all(np.apply_along_axis(lambda x: np.any(x==1),axis=1,arr=self.model_results['probabilities']))\
					and isinstance(self.query_strategy, (ClassifierUncertainty)):
				print("Classfier Uncertainty requires uncertain models")
				print("Current models contain 1.0 as a probability for every row.")
				print("Exiting query_vetted...")
				return

		# For label uncertainty query strat check if more than 1 expert
		if self.Y_noisy.shape[1] <= 1 and isinstance(self.query_strategy, (LabelUncertainty)):
			print("Label Uncertainty requires more than 1 expert for noisy labels")
			print("Current models contains %d expert" % self.Y_noisy.shape[1])
			print("Exiting query_vetted...")
			return

		# Check if image
		# Assumes all images if first raw is an image
		image = False
		if raw:
			img_check = imghdr.what(raw[0])
			if img_check:
				image = True

		# Check if raw is txt, json, or csv file type
		json_check = False
		csv_check = False
		if raw and not image:
			with open(raw[0]) as f:
				# JSON File
				if f.read(1) in '{[':
					json_check = True
				else:
					f.seek(0)
					reader = csv.reader(f)
					try:
						# CSV File
						if len(next(reader)) == len(next(reader)) > 1:
							csv_check = True
					except StopIteration:
						pass

		# Determine how to visualize data to user
		# 5 Cases;
		# image=0, raw=0, visualizer=0; Returns each full X row
		interactive_type = InteractiveType.RAWX
		# X is byte array of image
		if self.X_is_img:
			interactive_type = InteractiveType.IMGBYTEX
		# image=1, raw=1; Use Matplotlib to visualize image
		elif image and raw:
			interactive_type = InteractiveType.PLOTLIB
		# image=0, raw=1; Opens file, unsure implementation because security issues
		elif raw:
			interactive_type = InteractiveType.OPENFILE
		# image=0, raw=0, visualizer=1; Uses visualizer function and X as input to return map that should be visualized
		elif visualizer:
			interactive_type = InteractiveType.VISUALIZER

		# Run query strategy to get indices to query

		if isinstance(self.query_strategy,Prototypical):
			self.query_strategy.set_args(np.copy(self.model_results['embeddings']),np.copy(self.model_results['probabilities']))
		# Every other default case
		else:
			self.query_strategy.set_args(np.copy(self.X), np.copy(self.Y_noisy), np.copy(self.model_results['probabilities']))

		# If non interactive check if ground truth value exists
		if not interactive:
			ground_truth_check = self._check_ground_truth()
			if not ground_truth_check:
				print("Querying ground truth wihtout interaction requires complete ground truth set")
				print("Setting to interactive mode")
				interactive = True

		# Preprocessing to ensure vetted label from each class before using query strategy
		# Create found class list

		if not isinstance(self.query_strategy, Prototypical):
			print('Beginning preprocessing to find vetted labels of each class...')
			new_budget = budget

			class_labels = []
			num_classes = len(self.classes.keys())
			for _ in range(num_classes):
				class_labels.append(False)
			class_labels = np.asarray(class_labels)

			# Initialize probabilites to search through and loop till label from all classes found
			prob_search = np.copy(self.model_results['probabilities'])
			while not all(class_labels) and new_budget > 0:
				# Find max probability in each class
				max_prob_class = np.argmax(prob_search, axis=0)

				indices = []
				# Loop through each max, add to query vetted queue if class not found, and set probabilities to
				# -1 if the current class_label was not found
				for i in range(len(class_labels)):
					if not class_labels[i]:
						indices.append(max_prob_class[i])
						new_budget -= 1
					prob_search[max_prob_class[i]][i] = -1

				# Run query vetted on the set of indices
				class_labels = self._query_vetted_index(indices, interactive, interactive_type, raw, visualizer, class_labels=class_labels)
			print('Completed preprocessing')
			print('Budget reduced from \"%d\" to \"%d\"' % (budget, new_budget))
			budget = new_budget

		# Loop through find index and query user input
		for _ in range(int(budget/batch_size)):
			indices = self.query_strategy.choose_indices(np.copy(self.Y_vetted), batch_size)

			self._query_vetted_index(indices, interactive, interactive_type, raw, visualizer)

			#if isinstance(self.query_strategy,Prototypical):
			#	if batch_size==1:
			#		self.prototypes.update_prototype(self.model_results['embeddings'][indices[0]],self.Y_vetted[indices[0]])
			#	else:
			#		self.prototypes.update_prototypes_batch(self.model_results['embeddings'][indices],self.Y_vetted[indices])


		# If budget size is no longer divisible by batch_size after preprocessing, choose indices for the remainder
		remainder = budget % batch_size
		if remainder != 0:
			indices = self.query_strategy.choose_indices(np.copy(self.Y_vetted), remainder)
			self._query_vetted_index(indices, interactive, interactive_type, raw, visualizer)

		return

	def test(self):
		'''
		Description: uses given estimator to estimate ground truth labels
		store as attribute
		:param None
		:return: None
		'''
		# Check if Y_vetted has a class from each sample
		if isinstance(self.estimator, Learned):
			if self.estimator.estimate_z.__name__ == 'oracle_one_label' or self.estimator.estimate_z.__name__ == 'oracle_multiple_labels':
				if not self._check_Y_vetted():
					print("With oracle_one_label and oracle_multiple_labels, test requires vetted samples from each class")
					print("Query_vetted more samples before running test again...")
					return

		# Set arguments for estimator and run estimate
		if isinstance(self.estimator, Learned):
			if isinstance(self.query_strategy, Prototypical):
				self.estimator.set_args(np.copy(self.Y_noisy), np.copy(self.model_results['probabilities']), np.copy(self.model_results['embeddings']))
			else:
				self.estimator.set_args(np.copy(self.Y_noisy), np.copy(self.model_results['probabilities']), np.copy(self.X))
		else:
			self.estimator.set_args(np.copy(self.Y_noisy), np.copy(self.model_results['probabilities']))
		self.test_results = self.estimator.estimate(np.copy(self.Y_vetted))

		return
