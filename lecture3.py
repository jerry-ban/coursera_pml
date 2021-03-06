### Model evaluation and selection
# accuracy, others including: satisfaction, revenue, survival rate increase...
# select evaluation, compute multiple models, select best model

# test positive/negative identify(imbalanced classes), also classification like fraud detection

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.dummy import DummyClassifier

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dataset=load_digits()
X,y = dataset.data, dataset.target
for class_name,class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name, class_count)
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced !=1] = 0
print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])
np.bincount(y_binary_imbalanced)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
svm = SVC(kernel="rbf", C=1).fit(X_train,y_train)
svm.score(X_test,y_test)

# Dummy classifiers server as a sanity check purpose, not used for real classifier
from sklearn.dummy import DummyClassifier
dummy_majority= DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
# most_frequent, stratified(random from training set class distribution), uniform(uniform random), constant(when positive class is minority)

y_dummy_pred=dummy_majority.predict(X_test)
y_dummy_pred
dummy_majority.score(X_test,y_test)
### dummy classifier: for sanity check, if
### dummy regressors
## strategy: mean, median, quantile, constant

# confusion matrix
from sklearn.metrics import confusion_matrix
dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_majority_pred = dummy_majority.predict(X_test)
confusion_dummy =confusion_matrix(y_test, y_majority_pred)
confusion_dummy # [ [TN,FN], [FP,TP] ]

dummy_classprop = DummyClassifier(strategy="stratified").fit(X_train, y_train)
y_classprop_pred = dummy_classprop.predict(X_test)
confusion_dummy =confusion_matrix(y_test, y_classprop_pred)
confusion_dummy # [ [TN,FN], [FP,TP] ]

svm = SVC(kernel="linear", C=1).fit(X_train,y_train)
svm_pred= svm.predict(X_test)
confusion_svm =confusion_matrix(y_test, svm_pred)
confusion_svm

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)
"""
error: 1-accuracy
recall: TP/(TP+FN) = TP/Real_P
precision: TP/(TP+FP) =TP/ Forecased_P
"""
##to increase precision, we must:
#either increase the number of true positives the classifier predicts
# or reduce the number of errors where the classifier incorrectly predicts that a negative instance is in the positive
"""
Specificity: = TN/(FP+TN)
there is often tradeoff between predcision and recall
high precision, low recall
low precision, high recall: for medical case, tumor positive
Recalled oriented task often paired with human expret to filter false positive:
    search and information extraction in legal discovery; Tumor detection;
Precision oriented task:
    search engine ranking,  query suggestion; Document classification; many customer facing tasks(users remember failures so not good)

F1-score:
 F1= 2* Precision* Recall/(Rrecision+Recall)
 F_beta = (1+Betw**2) * Precision * Recall/(Beta**2 * Precision + Recall)
    Beta: allow to adjustment the emphasis on recall vs precision:
    precision oriented: beta=0.5, :false positive hurt performance more than false negatives
    recall oriented: beta = 2, :false negative hurt performance more than false positives
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
ds = pd.Series(name = "score",  data =  [ accuracy_score(y_test, svm_pred), precision_score(y_test, svm_pred),
               recall_score(y_test, svm_pred), f1_score(y_test, svm_pred)], index = ["accuracy", "precision", "recall", "f1"])
ds

print( classification_report(y_test, svm_pred, target_names = ["not 1", "1"]))

##### Decision functions
# each classifier score value per test point indicates how confidently the classifier predicts the P/N (large magnitude positive/negative values)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))
y_score_list
# usually class I if threshold (probability) > 0.5, change the threshold value to check precision/recall
y_probability_lr = lr.fit(X_train, y_train).predict(X_test)
y_probability_list = list(zip(y_test[0:20], y_probability_lr[0:20]))
y_probability_list

#Precision-Recall & ROC Curves
# ROC: Receiver Operating Characteristic Curve: widely used to illustrate the performance of a binary classifier.
## ROC curve: TPR~FPR: TP/(TP+FN) ~ FP/(FP+TN) = 1- specificity
# ROC-X: False Positive Rate, ROC-Y: True positive rate, ideal point is upper left.
# AUC(area under curve) bigger, the better


#Scikit-learn using scoring parameter for cross-validation
from sklearn.metrics.scorer import SCORERS
print ( sorted(list(SCORERS.keys())) )

###CV example
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

dataset = load_digits()
# again, making this a binary problem with 'digit 1' as positive class
# and 'not 1' as negative class
X, y = dataset.data, dataset.target == 1
clf = SVC(kernel='linear', C=1)

# accuracy is the default scoring metric
print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))
# use AUC as scoring metric
print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))
# use recall as scoring metric
print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall'))

#grid search example
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)


m= LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)