##machineing package: scikit-learn: scikit-learn.org, user guides and API reference

# scikit-learn 0.18.1
# scipy 0.19.0
# numpy 1.12.1
# pandas 0.20.1
# matplotlib 2.0.1
# seaborn 0.7.1

#import os
#os.getcwd()
#os.chdir(path)

import scipy as sp  # www.scipy.org, sci computing, statistics, optimization, linear algebra, math func
import numpy as np # www.numpy.org, data structures used by skl, array is mostly used
import pandas as pd # pandas.pydata.org key data structure: DataFrame, data manipulation

from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib as mpl  #matplotlib.org
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sometime use seaborn visualizatio package: seaborn.pydata.org
import seaborn as sn

# adspy_shared_utilities is file coming with the class, should be under the same folder
from adspy_shared_utilities import load_crime_dataset
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
from adspy_shared_utilities import(plot_class_regions_for_classifier)
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances

# Supervised learning methods overview

# KNN make few assupmtions, can make potentially accurate but sometimes unstable predictions
#   in KNN,  less K means more variance, and overfitting

# Linear  models makes strong assumptions about data structure and give stable but potentially inaccurate predictions

# model accuracy vs. model complexity from training & testing : always increasing vs. increasing then decreasing (normally)

# overfitting and underfitting
# model is too complex(variables) called overfit
# model is too simple (less variable) called underfit


#make datasets
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
## regression
fruit_file_txt = r"C:\_research\coursera_pml\fruit_data_with_colors.txt"
cmap_bold =ListedColormap(["height", "width" "mass", "color_score"])
fruits = pd.read_table(fruit_file_txt)
feature_names_fruits = ["height", "width", "mass", "color_score"]
X_fruits = fruits[feature_names_fruits]
y = fruits["fruit_label"]
target_names_fruits = ["apple", "mandarin", "mass", "lemon"]

X_fruits_2d = fruits[["height", "width"]]
y_fruits_2d = fruits[["fruit_label"]]

# one variable dataset generation
from sklearn.datasets import make_regression
plt.figure()
plt.title("Sample regression with one feature")
X_R1,y_R1=make_regression(n_samples=100, n_features=1, n_informative=1, bias=150, noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker="o", s=50)
# default 75-25 split for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state provide seed
plt.show()

# complex regression
from sklearn.datasets import make_friedman1
plt.figure()
plt.title("complex regression with one feature")
X_F1,y_F1=make_friedman1(n_samples=100, n_features=7, random_state=0)
plt.scatter(X_F1[:,2], y_F1, marker="o", s=50)

from sklearn.datasets import make_classification
plt.figure()
plt.title("complex regression with more features")
X_C2,y_C2= make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1,
                            flip_y=0.1, class_sep=0.5,random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2, marker="o", s=50, cmap=cmap_bold)
plt.show()

# more difficult synthetic dataset for classification (binary)
# with classes that are not linearly separable
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2  # change 8 clusters to 2 clusters by reassign lable to 0/1
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()


# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_cancer.shape

# Communities and Crime dataset
(X_crime, y_crime) = load_crime_dataset()
X_crime.shape
y_crime
# default 75-25 split for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state provide seed
plt.show()

#Nearest neighbors classfication(k=1) is overfitting
#KNN for regression
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'.format(knnreg.score(X_test, y_test)))
# the R2 score here sometimes called "coefficient of determination", = 1- RSS/Total_sum_of_square, bigger betterR
# SS_res(Rss) = sum(y^_i-y_i)**2, SS_tot = sum(y_i - y_bar)**2
# SS_reg = sum(y^_i-y_bar)**2, also called explained sum of squares

fig, subaxes = plt.subplots(1, 2, figsize=(8,4))
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

for thisaxis, K in zip(subaxes, [1, 3]):
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    thisaxis.set_xlim([-2.5, 0.75])
    thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10,
                 label='Predicted', alpha=0.8)
    thisaxis.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Target value')
    thisaxis.set_title('KNN regression (K={})'.format(K))
    thisaxis.legend()
plt.tight_layout()
plt.show()

#Regressionmodel complexity as a function of K
# plot k-NN regression on sample dataset for different values of K
fig, subaxes = plt.subplots(5, 1, figsize=(5,20))
X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,random_state = 0)

for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]):
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    train_score = knnreg.score(X_train, y_train)
    test_score = knnreg.score(X_test, y_test)
    thisaxis.plot(X_predict_input, y_predict_output)
    thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
    thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Target value')
    thisaxis.set_title('KNN Regression (K={})\n\
Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'
                      .format(K, train _score, test_score))
    thisaxis.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)



###*** Linear Regression-Least-Squares, a example of a Linear Model
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,random_state = 0)
linreg= LinearRegression().fit(X_train, y_train)
linreg.intercept_  # _ means derived from training data
linreg.coef_
linreg.score(X_train, y_train)
linreg.score(X_test, y_test)

#plot
plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker = "o", s=50, alpha=0.5)
plt.plot(X_R1, linreg.predict(X_R1), "r")
plt.title("Least-squares Linear Regression")
plt.xlabel("Feature value(x)")
plt.ylabel("Target value (y)")
plt.show()#

### feature normalization important to KNN, regularized regression,SVM, Neural networks...
### Ridge Regression, need feature normalization
# linear regression, added a penalty factor, fit with all parameters, prediction only ordinary regression factors
# this is called regularization, to prevent overfitting, by restricting the model, typically to reduce its complexity
# retularization term is controlled by Alpha factor, higher alpha means more regularization and simpler model
# Ridge regression use L2 regularization
### some normalization method: minMax: (x-x_min)/(x_max-x_min); standard norm: (x-x_mean)/std_x

from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state = 0)
linridge=Ridge(alpha=20.0).fit(X_train, y_train)
linridge.intercept_
linridge.coef_
linridge.score(X_train, y_train)
linridge.score(X_test, y_test)
np.sum(linridge.coef_!=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) # compute min&max
X_train_scaled = scaler.transform(X_train) #scal data according to range
X_test_scaled = scaler.transform(X_test)
clf = Ridge().fit(X_train_scaled, y_train)
r2_score = clf.score(X_test_scaled, y_test)
r2_score

# more efficient by do fitting and transforming together
scaler = MinMaxScaler()
X_train_Scaled = scaler.fit_transform(X_train)
linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
X_test_scaled =scaler.transform(X_test)
linridge.score(X_train_scaled, y_train)
linridge.score(X_test_scaled, y_test)

### different scale for training and test dataset will cause random skew in the data
### fit the scaler using any part of the test data will cause data leakage( info leaked from test to trainning)

#Ridge regression with regularization parameter: alpha
print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))


### Lasso regression is also regularized LR, with L1 penality,
# it has effect to set least influential variables coefficient to 0,
# this is called a sparse solution, a kind of feature selection
## many small/medium sized effects: use ridge
## only a few variables with medium/large effect: use lasso

from sklearn.linear_model import Lasso
scaler=MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state = 0)
X_train_scaled = scaler.fit_transform(X_train) #scal data according to range
X_test_scaled = scaler.transform(X_test)
linlasso=Lasso(alpha=2.0, max_iter=10000).fit(X_train_scaled, y_train)
np.sum(linlasso.coef_!=0)
np.sum(linridge.coef_!=0)
linlasso.score(X_train_scaled, y_train)
linlasso.score(X_test_scaled, y_test)
for e in sorted(list(zip(list(X_crime), linlasso.coef_)), key = lambda e: -abs(e[1])):
    if e[1] != 0:  print('\t{}, {:.3f}'.format(e[0], e[1]))

print('Lasso regression: effect of alpha regularization\n\
parameter on number of features kept in final model\n')

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)

    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))


### polynomial Features(transformation) with Linear Regression: (x0, x1)=> (x0, x1, x0^2, x0_x1, x1^2)
# to capture interactions/ to make a classification problem easier
# may cause overfit, so usually combined with regularized method, like ridge or lasso
from sklearn.preprocessing import PolynomialFeatures
X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1, random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
["linreg", linreg.score(X_train, y_train), linreg.score(X_test, y_test)]

poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state = 0)
linregpol = LinearRegression().fit(X_train, y_train)
["linreg", linregpol.score(X_train, y_train), linregpol.score(X_test, y_test)]

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state = 0)
linridge = Ridge().fit(X_train, y_train)
["linreg", linridge.score(X_train, y_train), linridge.score(X_test, y_test)]



### Logistic Regression
# ## Linear models for classification
# ### Logistic regression
# #### Logistic regression for binary classification on fruits dataset using height, width features (positive class: apple, negative class: others)

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
y_fruits_apple = y_fruits_2d == 1   # make into a binary problem: apples vs everything else
X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.as_matrix(), y_fruits_apple.as_matrix(), random_state = 0))

# C is regularization, default is on, value =1.0, higher values means less regularizatio, and mormalization important
clf = LogisticRegression(C=100).fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, 'Logistic regression for binary classification\nFruit dataset: Apple vs others', subaxes)

h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'
     .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

h = 10
w = 7
print('A fruit with height {} and width {} is predicted to be: {}'
     .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')

print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# Classifier Margin: split 2 class, bigger(between 2 classes) the better
# Max margin linear classifier: Linear Support Vector Machines,
# decision boundary with max margin between classes using a (linear) classifier in the original/transformed feature space

from sklearn.svm import SVC
from adspy_shared_utilities import(plot_class_regions_for_classifier_subplot)
X_train, X_test, y_train, y_test = (train_test_split(X_C2, y_C2, random_state = 0))
fig, subaxes = plt.subplots(1,1,figsize=(7,5))
this_C=1.0 # regularization factor, larger C, less regularization(to fit training data as well as possible, indivisual data is important)
clf=SVC(kernel="linear", C=this_C).fit(X_train, y_train)
title="Linear SVC, C={:.3f}".format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

""" Linear Models: Pros and Cons
Pros: Simple and easy to train;
        Fast to prediction
        Scales well to very large datasets
        works well with sparse data
        Reasons for prediction are relatively easy to interpret
Cons: for lower-dimensional data, other models may have superior generalization performance
        for classification, data may not be linearly separable(more on this in SVM with no linear kernels)
"""

### multi-class classfication
### 1 to all, namely if class1 or all else, if class 2 or all else, if class3 or all else,...choose the biggest chance


### Kernelized support vector machines
# transform current data to higher dimentional data and may easier to group
""" radial basis function kernel(Gausian kerneal)
the kernelized SVM can compute these more complex decision boundaries just in terms of similarity
calculations between pairs of points in the high dimensional space where the transformed feature
representation is implicit. This similarity function which mathematically is a kind of dot product is the kernel in kernelized
"""

from sklearn.svm import SVC
from adspy_shared_utilities import(plot_class_regions_for_classifier)
X_train, X_test, y_train, y_test = (train_test_split(X_D2, y_D2, random_state = 0))
plot_class_regions_for_classifier(SVC().fit(X_train, y_train), X_train, y_train, None, None, "SV CLassifier: RBF kernel")

plot_class_regions_for_classifier(SVC(kernel="poly", degree =3).fit(X_train, y_train), X_train, y_train, None, None, "SV CLassifier: Poly(3) kernel")

# RBF kernel has gamma parameters, exp(-gamma(x-x')2), Gamma controls how far the influence of
# a single trending example reaches, which in turn affects how tightly the decision boundaries
#  end up surrounding points in the input space.
# Small gamma means a larger similarity radius. So that points farther apart are considered similar.
# Which results in more points being group together and smoother decision boundaries.
###***large gamma: more tight, complex boundary, each training point importance
###***small C: more regularization, get larger margin for DB, even more point wrong labeled

fig, subaxes = plt.subplots(1, 3, figsize=(11,4))

for this_gamma, subplot in zip([0.01, 1.0,10.0], subaxes):
    clf = SVC(kernel = "rbf", gamma = this_gamma).fit(X_train, y_train)
    title = "Support Vector Classifier: RBF, gamma={:.2f}".format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subplot)

plt.tight_layout()

# SVMs also have a regularization parameter, C, to control the tradeoff between satisfying the max margin criterion
#  to find the simple decision boundary, and avoiding misclassification errors on the training set.
# The C parameter is also an important one for kernelized SVMs, and it interacts with the gamma parameter.

# if gammma large, C little effect; gamma smaller and means more(tight) strict, C would be more alike linear classfier.
# need tuned at the same time
# kernelized kernel sensitive to gamma
ig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):

    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# kernelized SVM: summary
# Pros: Can perform well on a range of datasets
#   versatile: different kernel functions can be specified, or custom kernels can be defined for specific data types
# works well for both low and high-dimensional data

# Cons: Efficiency(time/memory)  decreases as training size increase
#   needs normalization of input data and parameter tuning
#   does not provide direct probability estimates
#   difficult to interpret why a prediction was made


### cross validation
# K-fold: commonly K=3/5/10; data split into 5 equal size, each is a fold, one fold is test, another is training set
# default is 3, Stratified cross-validation, namely each fold contain a proportion of class the same as overall dataset.
# thus all class would be fairly represented in test set.

# sklearn uses regular k-fold CV since class proportion usually is not relevant to common regression problems.
# one extreme is leave-one-out-CV, means k=sample size of dataset, so each sample is a fold.
# from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors=5)
X=X_fruits_2d.as_matrix()
y= list(y_fruits_2d["fruit_label"])
cv_scores=cross_val_score(clf, X,y)
np.mean(cv_scores)

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

# validation curves show sensitivity to changes in an important parameter, this is to evaluate the model
param_range = np.logspace(-3, 3, 4)  #[0.001, 0.1, 10, 1000]
train_scores, test_scores = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()


### Decision Tree
# a number of mathematical ways to compute the best split, one criterion widely used for DT is called Information Game
# leaf node can be pure/mixed
# to prevent overfitting: stopping tree growth early(pre-pruning); build complete tree with pure leaves and prune back to a simplier form(Post Pruning)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier  #implemented by pre-pruning only, pars: max_depth/max_leaf_nodes, min_sample_leaf
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf2.score(X_test, y_test)))
plot_decision_tree(clf, iris.feature_names, iris.target_names)
plot_decision_tree(clf2, iris.feature_names, iris.target_names)
plt.show()

# Feature Importance: how important is a feature to ovaerall prediction accuracy?
# A number between 0/1 assigned to each feature
# Feature importance of 0 => the feature was not used in prediction
# Feature importance of 1=> the feature predictions the target perfectly
# All feature importances are normallized to sum to 1

# a classifer should have feature_importances_ property
from adspy_shared_utilities import plot_feature_importances
plt.figure(figsize=(10,4))
plot_feature_importances(clf, iris.feature_names)

#less value of feature importance does not mean feature not important, jus not shown in early stage of tree, may identical/correlated with other informative feature
# model specific, so use mean of multi train/test splits.

from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train

    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             iris.target_names)

    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])

plt.tight_layout()

#Pros & Cons:
"""
Pros: easily visualized and interpreted
      No feature normalization or scaling typically needed
      Work well with datasets using a mixture of feature types(continus, categorical, binary)
Cons: Even after tuning, DT can often still overfit
      usually need an ensemble of trees for better generalization performance.
"""