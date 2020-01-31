# ActT
Active Testing package for python

Authors: Jensen Dempsey, John Holodnak, Carmen Stowe, Adam Tse

The basic idea of active testing (see __[Active Testing: An Efficient and Robust Framework for Estimating Accuracy](https://icml.cc/Conferences/2018/Schedule?showEvent=2681)__) is to intelligently update the labels of a noisily-labeled test set to improve estimation of the performance metric for a particular system.  The noisy labels may come, for example, from crowdsourced workers and as a result are not expected to have high accuracy.  The required inputs to use active testing are:
* A system under test,
* A performance metric (accuracy, precision, recall, etc.) of interest,
* A test dataset where each item has at least one noisy label and a score from the system under test,
* Access to a vetter that can provide (hopefully) high quality labels,

Active testing has two main steps.  In the first step, items from the test dataset are queried (according to some query_strategy) and sent to the vetter to receive a label.  In the second step, some combination of the system scores, the noisy labels, the vetted labels, and the features are used to estimate the performance metric of interest.

This package implements a variety of query strategies and two metric estimation strategies.  See the notebooks folder for  examples and more detailed explanations of how the package works.  We recommend looking at them in this order
* Introduction and Basic Use
* Metric Estimators
* Query Strategies
* DPP
* Prototypical Vetting
* Text Data
* Image Data

See below for examples of basic usage.


## Installation

To install ActT, download the package, navigate to the ActT directory and run
```bash
pip install .
```
Or to directly install without cloning
```bash
pip install git+https://github.com/mitll/ActT
```
## Basic Usage
```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from active_tester import ActiveTester
from active_tester.estimators.naive import Naive
from active_tester.query_strategy.random import Random
import numpy as np

# load a simple dataset and build a simple classifier to evaluate
iris = load_iris()
X = iris.data
y = iris.target
clf = LogisticRegression()
clf = clf.fit(X, y)

# generate some noisy labels (in this case, we just re-use y)
Y_noisy = np.reshape(y,(len(y),1))

# initialize active tester object and format the dataset
active_test = ActiveTester(Naive(metric=accuracy_score), Random())
active_test.standardize_data(rearrange=False, X=X, classes=iris.target_names, 
                             Y_ground_truth=None, Y_vetted=None, Y_noisy=Y_noisy)

# run the model on the dataset
active_test.gen_model_predictions(clf)

# query 5 labels from the vetter (you!)
active_test.query_vetted(True, budget=5, raw=None, visualizer=None)

# use input to estimate classifier performance and print the result
active_test.test()
result = active_test.get_test_results()
print(result)
```

If the model is not available, but probablities predicted by the model are, we can call `set_prob_array` and send in the predictions directly.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from active_tester import ActiveTester
from active_tester.estimators.naive import Naive
from active_tester.query_strategy.random import Random
import numpy as np

# load a simple dataset and build a simple classifier to evaluate
iris = load_iris()
X = iris.data
y = iris.target
clf = LogisticRegression()
clf = clf.fit(X, y)

# generate some noisy labels (in this case, we just re-use y)
Y_noisy = np.reshape(y,(len(y),1))

# initialize active tester object and format the dataset
active_test = ActiveTester(Naive(metric=accuracy_score), Random())
active_test.standardize_data(rearrange=False, X=X, classes=iris.target_names, 
                             Y_ground_truth=None, Y_vetted=None, Y_noisy=Y_noisy)

# run the model on the dataset
y_predictions = clf.predict_proba(X)
active_test.set_prob_array(y_predictions)

# query 5 labels from the vetter (you!)
active_test.query_vetted(True, budget=5, raw=None, visualizer=None)

# use input to estimate classifier performance and print the result
active_test.test()
result = active_test.get_test_results()
print(result)
```
## Acknowledgements

This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via Air Force Life Cycle Management Center Contract FA8702-15-D-0001. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government.
