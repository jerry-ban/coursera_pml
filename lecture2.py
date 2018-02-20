##machineing package: scikit-learn: scikit-learn.org, user guides and API reference

# scikit-learn 0.18.1
# scipy 0.19.0
# numpy 1.12.1
# pandas 0.20.1
# matplotlib 2.0.1
# seaborn 0.7.1

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import scipy as sp  # www.scipy.org, sci computing, statistics, optimization, linear algebra, math func
import numpy as np # www.numpy.org, data structures used by skl, array is mostly used
import pandas as pd # pandas.pydata.org key data structure: DataFrame, data manipulation

import matplotlib as mpl  #matplotlib.org
import matplotlib.pyplot as plt

# sometime use seaborn visualizatio package: seaborn.pydata.org
import seaborn as sn

#also graphviz plotting library
import graphviz

fruit_file_txt = r"C:\_research\coursera_pml\fruit_data_with_colors.txt"
fruits = pd.read_table(fruit_file_txt)
fruits.head()
fruits.shape

# create train-test split
X = fruits[["mass", "width", "height"]]
y = fruits["fruit_label"]

# default 75-25 split for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state provide seed

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
lookup_fruit_name

# examing the data
from matplotlib import cm

X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(12,12), cmap=cmap)

# 3-Dim feature scatterplot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax=fig.add_subplot(111, projection="3d")
ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c=y_train, marker="o", s=100)
ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_zlabel("color_score")
plt.show()

# K-Nearest Neighbors classification KNN

