# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Classification
# * Evaluating classification models

# %% [markdown]
# # Classification
#
# With regression we are predicting *continuous* labels, basically floating point numbers. However, some problems have *categorical* labels, which correspond to discrete groups rather than numbers.
#
# <img src="https://cdn-coiao.nitrocdn.com/CYHudqJZsSxQpAPzLkHFOkuzFKDpEHGF/assets/static/optimized/rev-85bf93c/wp-content/uploads/2021/04/regression-vs-classification_simple-comparison-image_v3.png" width=600>

# %% [markdown]
# At first glance you might think we could just do regression with `int` in place of `float`. For instance, consider data falling in 3 groups: `Apple, Banana, Orange`. If we convert these to `int` so they are discrete, we would get `0, 1, 2`. Technically we can then use a regressor to predict the values.
#
# Aside from having to round off the outputs, this simple strategy makes a HUGE assumption in the math: that `Apple` is closer to `Banana` than it is to `Orange`, since they are represented by `0, 1, 2`. This will lead to systematic bias in the predictions and reward the wrong types of predictions, while having no basis in reality for the problem.

# %% [markdown]
# ## Dataset
#
# Let's dive in and see some examples of this in action. We need to switch to a dataset that has categorical labels:

# %%
# import requests
import pandas as pd
import os

# Set the path to the data file
filename = 'steels.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)

data                            # show a view of the data file

# %% [markdown]
# In this dataset, we have elemental compositions at the left and then some mechanical properties at the right.
# Let's try to use the data to predict the `Alloy code`, which is categorical.
#
# > **Note:** Some column names in this dataset have leading spaces (e.g., `' C'`, `' 0.2% Proof Stress (MPa)'`). We include these in the code below.
#
# We can start by looking at the values of `Alloy code`:

# %%
data['Alloy code'].unique()

# %% [markdown]
# This is too many categories for us to keep track of.
# Let's simplify things by taking only the first letter of each code.
# We can call this an `Alloy family`:

# %%
data['Alloy family'] = [x[0] for x in data['Alloy code']]
data.head()

# %% [markdown]
# Now we have only a few categories:

# %%
data['Alloy family'].unique()

# %% [markdown]
# We can prepare the train/test data by including all the composition columns in our $X$ and the `Alloy family` as our $y$:

# %%
from sklearn.model_selection import train_test_split

x = data.loc[:, ' C':'Nb + Ta']
y = data['Alloy family']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
# note: we make sure to all get the same answer with random_state=0
print(xtrain.shape, xtest.shape)

# %% [markdown]
# ## Logistic regression
#
# Despite its name, logistic regression is a linear classification algorithm.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/400px-Exam_pass_logistic_curve.svg.png" width=400>

# %%
from sklearn import linear_model

model = linear_model.LogisticRegression(random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

# %% [markdown]
# It looks like the `LogisticRegression` gets a perfect score on the test data.
# What does this look like?
# As a reminder, the model is predicting one of 4 classes, `C, L, M, V`:

# %%
model.predict(xtest)

# %% [markdown]
# How does it work?
# This is a linear model that assigns a class label based on the class with the largest predicted probability.
# For a binary problem, the probability is computed based on a softmax function (from `scikit-learn` documentation):
#
# $\hat{p}(X_i) = \mathrm{expit}(X_i w + w_0) = [1 + \mathrm{exp}(-X_i w - w_0)]^{-1}$
#
# For a multinomial problem, we need something more complicated (that we won't bother computing here).
#
# Just to show you how they are stored, here are the actual intercept and coefficients from the fitted model:

# %%
print( model.intercept_ )
print( model.coef_ )

# %% [markdown]
# And here are the raw probabilities predicted by the model:

# %%
model.predict_proba(xtest)

# %% [markdown]
# We can manually convert this to class labels using the `numpy.argmax` function, which returns the index of the greatest value (with `axis=1` it's the greatest value in each row):

# %%
import numpy as np

p = model.predict_proba(xtest)
pred_label = np.argmax(p, axis=1)
print(pred_label)

# %% [markdown]
# Finally we can convert this back to labels manually using the `numpy.unique` function:

# %%
np.unique(y)[pred_label]

# %% [markdown]
# Finally, just to prove it to you:

# %%
np.unique(y)[pred_label] == model.predict(xtest)

# %% [markdown]
# You can note here how `scikit-learn` conveniently handles conversion between integer labels and categorical label codes for us.

# %% [markdown]
# ## Support Vector Machines
#
# Support Vector Machine (SVM) is a ML method that identifies an *optimal boundary between different classes in the feature space*.
#
# step 1: the boundary is identified
#
# step 2: new observations are easily classified by checking which side of the boundary they fall on.
#
# While this is a very common method for classification, it can also be extended to do regression.
#
# Here is a simple classification example in 2D with only 2 classes:
#
# <img src="../lectures/assets/lecture04_svm_boundary_linear.jpg" width=500>
#
# This approach can be extended to multiple classes and higher dimensions.
# It can also be used with nonlinear decision boundaries (illustrated in the `sklearn` documentation [here](https://scikit-learn.org/stable/modules/svm.html#classification)):
#
# <img src="../lectures/assets/lecture04_svm_boundary_nonlinear.jpg">
#
# The nonlinearity is introduced through a function called a **kernel**.
#
# From [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method):
#
# > Kernel functions enable a method to operate in a   high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space.
# >
# >This operation is often computationally cheaper than the explicit computation of the coordinates.
# >
# > This approach is called the "kernel trick".
#
# We will discuss this further in our lesson on unsupervised learning.
#
#

# %% [markdown]
# For now, let's test the performance of the SVM

# %%
from sklearn import svm

model = svm.SVC(kernel='linear', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

# %%
model = svm.SVC(kernel='rbf', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

# %%
model = svm.SVC(kernel='poly', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

# %% [markdown]
# From these scores, it looks like adding nonlinearity to the decision boundary actually reduces performance.
#
# We can't tell from this simple analysis if this is because:
#
# *  the decision boundary is really linear, or
#
# * if we don't have enough training data to fit the boundary, or
#
# * if the boundary is nonlinear in a different way from any of these kernels.
#
# For more information on choosing kernels: https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html
#

# %% [markdown]
# ## Naive Bayes
#
# The Naive Bayes method assumes conditional independence between pairs of features.
# This permits a relatively simple formula for making class predictions:
#
# $\hat{y} = \mathrm{arg} \mathrm{max}_y P(y) \prod_{i=1}^n P(x_i | y)$
#
# Different likelihood probabilities can be selected.
# For instance, we can try a Gaussian:
#
# $P(x_i | y) = (2 \pi \sigma^2_y)^{-1/2} \exp \left( - \frac{(x_i-\mu_y)^2}{2\sigma^2_y} \right)$
#
# where $\sigma_y$ and $\mu_y$ are **parameters** of the model estimated by maximum likelihood estimation.

# %%
from sklearn import naive_bayes

model = naive_bayes.GaussianNB().fit(xtrain, ytrain)

model.score(xtest, ytest)

# %% [markdown]
# We see here that while the accuracy is quite high, two points in the test set are misclassified using Gaussian Naive Bayes whereas there were no errors using Logistic Regression or Support Vector Machines.

# %% [markdown]
# ## [Check your understanding]
#
# Using the Naive Bayes classifier above...
# * Identify the test cases that are misclassified (e.g., using `numpy.argwhere`)
# * Compute the class probabilities for these cases -- are they on the boundary or does the model confidently predict the wrong class?

# %%
y_pred = model.predict(xtest)
# np.argwhere( y_pred == ytest )
# print( y_pred != ytest )
not_equal = np.array(y_pred != ytest)
indices = np.argwhere( not_equal )

print(np.unique(ytest))
print(ytest.iloc[indices.flatten()])
model.predict_proba(xtest)[indices]

# %% [markdown]
# ## Decision tree
#
# Now let's try with a Decision Tree.
# The interface is very similar to the code we gave you for regression, we just use a `DecisionTreeClassifier` instead of a `DecisionTreeRegressor`:

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=0).fit(xtrain, ytrain)

model.score(xtest, ytest)  # this gives the accuracy of the classifier

# %% [markdown]
# Our simple model scored a perfect 100% accuracy on the test set, which it hadn't seen before! Let's explore this by drawing the tree itself:

# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(24, 8))
_ = tree.plot_tree(model, ax=ax, fontsize=10, label='root', class_names=['C', 'L', 'M', 'V'],
                   impurity=False, precision=2, proportion=True)

# %% [markdown]
# As you can see, the tree is very shallow and requires few decisions.
# This is actually because the `Alloy code` is defined by the composition of the alloy -- our decision tree is actually learning the real definition of the codes.
# In cases such as this, decision tree is actually a great option!
#
# We can also use the fitted tree to visualize the compositions -- we see that features 7, 9, and 6 completely define the tree!

# %%
from plotly import express as px
from sklearn import preprocessing

# can do this in one go with `fit_transform` if we don't need the encoder object
labels = preprocessing.LabelEncoder().fit_transform(y)

px.scatter_3d(x=x.iloc[:, 6], y=x.iloc[:, 7], z=x.iloc[:, 9], color=labels)

# %% [markdown]
# So another way to think about the decision tree algorithm is to imagine it dividing up the space into boxes which belong to a single class.

# %% [markdown]
# ## K Neighbors
#
# Based on these results, we can imagine that K Neighbors will also work fairly well. Let's try it:

# %%
from sklearn import neighbors

model = neighbors.KNeighborsClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)

# %% [markdown]
# Indeed, K Neighbors correctly votes 100% of the time (*on unseen test data!*).
# This means our problem is too easy -- we'll now make it harder to learn more about how the models work.

# %% [markdown]
# ## A more challenging dataset
#
# Instead of using the alloy compositions, let's try to predict the family based on the properties. This corresponds to the columns `0.2% Proof Stress (MPa)` through `Reduction in Area (%)`.

# %%
x = data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']
y = data['Alloy family']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
print(xtrain.shape)

# %% [markdown]
# Now we can test our `DecisionTreeClassifier` again.

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)

# %% [markdown]
# We see here the model accuracy is less than perfect. Does this hold up for the `KNeighborsClassifier`?

# %%
from sklearn import neighbors

model = neighbors.KNeighborsClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)

# %% [markdown]
# Yes -- we are no longer dealing with a "definition" of the `Alloy code`.

# %% [markdown]
# ## [Check your understanding]
#
# Let's see if the mighty Neural Network has trouble with this problem.
# Train a Neural Network for this problem and evaluate its performance.
#
# > Hint: look at the `sklearn.neural_network` documentation to find the name of the model
#
# > Note: neural networks have a random initialization so the results will change every time you run the cell. You can freeze the init with the keyword argument `random_state=0`
#
# > Bonus: try modifying the hyperparameter `hidden_layer_sizes` to improve the performance

# %%

# %% [markdown]
# # Classification Metrics
#
# Let's take a moment to discuss metrics for classification problems. Because the labels are not numerical, we can't use RMSE anymore.
# Basically, if the true label is `0`, we should penalize `1` or `2` the same amount.

# %% [markdown]
# ![image](../lectures/assets/lecture04_categorical_vs_continuous.jpg)

# %% [markdown]
# **Accuracy** is the most obvious metric for classification. Accuracy is simply the number of correct predictions divided by the total predictions; when we use `model.score`, it is telling us the fraction of correct predictions. We can also compute accuracy using `sklearn.metrics.accuracy_score`:
#
#

# %%
from sklearn import metrics

ypred = model.predict(xtest)
print(f'accuracy = {metrics.accuracy_score(ytest, ypred)}')

# %% [markdown]
# **Precision** and **Recall** are the next most common metrics.
#
# Precision is the fraction of times the predicted label is correct, out of all the times that label is predicted.
#
# Recall is the fraction of times the correct label is predicted out of all the true labels of that type.
#
# It's possible for a model to have a poor accuracy with high precision, or poor accuracy with high recall. These values can also vary significantly by class - we'll see this in a bit.
#
#
#

# %% [markdown]
# Finally, we can get what's called the **F1 Score**, which is the harmonic mean of the Precision and Recall:
#
# $F_1 = 2 \frac{\mathrm{precision} \cdot \mathrm{recall}}{ \mathrm{precision} + \mathrm{recall} }$
#
# This is basically just a way to balance the two measures in a single, combined measure that is more discriminating than accuracy.

# %% [markdown]
# We can get all of this information from `sklearn` using `metrics.classification_report`:

# %%
print(metrics.classification_report(ytest, ypred))

# %% [markdown]
# You will see a bunch of numbers here. Basically we get the precision, recall, and f1-score for each class individually, with the number of labels in the class (called "support"). We also get the overall accuracy underneath, and the precision, recall, and f1-scores again with different weighting schemes (macro and weighted avg).
#
# > If I ask you for the precision or recall score of a model, you can pick either of the two weighting schemes. I'll always mean the overall score for the model rather than for an individual class.

# %% [markdown]
# ## Confusion matrix
#
# We can also evaluate the performance of the model graphically using a confusion matrix.
#
# The confusion matrix shows the number of labels in each category for both the predictions and the true observations.
#
# It can communicate a lot of information quickly without resorting to technical definitions such as the Accuracy, Precision, and Recall.
#
# Here's a reference:
#
# <img src="https://www.researchgate.net/profile/Rune-Jacobsen/publication/334840641/figure/fig3/AS:794222751928321@1566368868347/Confusion-matrix-and-evaluation-metrics.png" height=300>

# %% [markdown]
# We can create this confusion matrix using the convenient `sklearn.metrics.ConfusionMatrixDisplay` function:

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, xtest, ytest)

# %% [markdown]
# We can see from this chart that the `V` family has relatively few mistakes, while the `L` family seems to be very difficult to distinguish from `C` and `M`.
#
# At this moment it's difficult for us to determine which features are responsible for this.
#
# In the following lessons we will learn several tools that could help with this problem:
#
# * **representation learning** can help us understand how high-dimensional observations differ from each other
#
# * **feature importance** can help us quantify which features are responsible for certain decisions

# %% [markdown]
# ## [Check your understanding]
#
# * Train a Random Forest to predict these same labels.
# > Make sure you use a classifier model, not a regressor!
# * Print the metrics and identify the model average precision, recall, and overall accuracy.
# * Draw a confusion matrix showing the model performance on the test set.

# %%

# %% [markdown]
# # Additional remarks

# %% [markdown]
# ## Important reminders about classification vs regression
#
# Classification predictive modeling problems are different from regression predictive modeling problems.
# * Classification is the task of predicting a discrete class label.
# * Regression is the task of predicting a continuous quantity.
#
# There is some overlap between the algorithms for classification and regression; for example:
# * A classification algorithm may predict a continuous value, but the continuous value is in the form of a probability for a class label.
# * A regression algorithm may predict a discrete value, but the discrete value is in the form of an integer quantity.
#
# Some algorithms can be used for both classification and regression with small modifications, such as decision trees and artificial neural networks.
# Some algorithms cannot, or cannot easily be used for both problem types, such as linear regression for regression predictive modeling and logistic regression for classification predictive modeling.
#
# Importantly, the way that we evaluate classification and regression predictions varies and does not overlap, for instance:
# * Classification predictions can be evaluated using accuracy, whereas regression predictions cannot.
# * Regression predictions can be evaluated using root mean squared error, whereas classification predictions cannot.

# %% [markdown]
# ## Additional reading
#
# I **highly** suggest reading Chapter 2 of the textbook, "Introduction to Machine Learning with Python: A Guide for Data Scientists."
# Many detailed examples are given with more exposition about the algorithms and ways to analyze them.
