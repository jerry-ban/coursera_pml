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

###Naive Bayes Classifiers
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


###Random Forest
"""
Pros:   Widely used, excellent prediction performance on many problems
        Doesn't require normalization of features or extensive parameter turning
        Handles a mixture of feature types, like decision trees
        Easily parammelized across multiple CPUs
Cons:   The resultingmodels are often difficult for humans to interpret
        may not be good for VERY HIGH-DImensional tasks(text classifiers) compared to linear models(faster/accurate)
"""


###Gradient Boosted Decision Trees(GBDT)
"""
Pros:   Widely used, excellent prediction performance on many problems

"""
