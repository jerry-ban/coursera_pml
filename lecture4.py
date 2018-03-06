### Model evaluation and selection
# accuracy, others including: satisfaction, revenue, survival rate increase...
# select evaluation, compute multiple models, select best model

# test positive/negative identify(imbalanced classes), also classification like fraud detection
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset


cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

### Naive Bayes Classifiers
"""
Pros:   Easy to understand;   Simple/efficient parameter estimation;    works well with high-dimensional data;
        useful as a baseline comparison against more sophisticated methods;
Cons:   Assumption that features are condition ally independent;     as a result, other classifier types often have better generalization performance
        Their confidence estimates for predictions are not very accurate
"""
from sklearn.naive_bayes import GaussianNB
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
clf = GaussianNB.fit(X_train,y_train)
plot_class_regions_for_classifier(clf, X_train,y_train, X_test, y_test, "Gaussian N.B. Classifier")


### Ensembles of Decision Trees

###Random Forest
from sklearn.ensemble import RandomForestClassifier

"""
Pros:   Widely used, excellent prediction performance on many problems
        Doesn't require normalization of features or extensive parameter turning
        Handles a mixture of feature types, like decision trees
        Easily paralleled across multiple CPUs
Cons:   The resulting models are often difficult for humans to interpret
        may not be good for VERY HIGH-Dimensional tasks(text classifiers) compared to linear models(faster/accurate)
"""


###Gradient Boosted Decision Trees(GBDT)
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=2)
clf = GradientBoostingClassifier().fit(X-train, y_train)
"""
training a series of small decision trees, each tree attempts to correct errors from the previous stage,
learning rate: high: more complex trees, smaller: simpler trees.
Pros:   Often best off-the-shelf accuracy on many problems
        Using model for prediction requires only modelt memory and is fast
        Doesn't require normalization of features to perform well
        handles a mixture of feature types(like decision trees)
Cons:   Difficult to explain(like random forests)
        Requires careful tuning of learning rate and other parameters
        Traing can require significant computation
        (like decision trees)Not recommended for txt classification and other problems with very high dimensional sparse features for accuracy and computational cost
"""

### Nerual Networks

from sklearn.neural_network import MLPClassifier # multi-layer perceptron

nnclf = MLPClassifier(hidden_layer_sizxes=[100], solver = "lbfgs", random_state = 0).fit(X_train, y_train)
nnclf = MLPClassifier(hidden_layer_sizxes=[10, 10], solver = "lbfgs", random_state = 0).fit(X_train, y_train) // 2 hidden layers
nnclf = MLPClassifier(hidden_layer_sizxes=[100, 100], solver = "lbfgs", activation="tanh", alpha=0.1,  random_state = 0).fit(X_train, y_train) // 2 hidden layers
"""

Pros:   Form the basis of complex models and can be formed into advanced architectures to capture complex features given enough data and computation
Cons:   Larger, more complex models require significant training time, data, customization
        preprocessing  data required
        less good choice if features are different types
"""

# NN example: http://playground.tensorflow.org

### Deep Learning(Optional)
"""
Pros:   powerful: has achieved significant gains over other ML approaches on many difficult learning tasks, great performance across domains
        automatic feature extraction
        current software provides flexible architectures that can be adapted for new domains fairly easily
Cons:   Can require huge training data;   huge computing power;
        architectures can be complex and often tailored to specific application;      difficult to explain
"""
# Deep learning software for python:(using GPU for parallel)
# Keras https://keras.io/
# Lasagne Https://lasagne.readthedocs.io/en/latest/
# TensorFlow https://www.tensorflow.org/
# Theano http://deeplearning.net/software/theano/

# Deep Learning in a Nutshell: Core Concepts. (2016, September 08). Retrieved May 10, 2017
# part-1: https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/


### Data leakage