# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# ---
# id: Lecture04_logistic_regression
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## Logistic regression
#
# Despite its name, logistic regression is a linear classification algorithm.
#
# <img src="../lectures/assets/lecture04_logistic_regression.jpg" alt="Illustration of a logistic regression curve" width=400>
#
# It looks like the `LogisticRegression` gets a perfect score on the test data.
# What does this look like?
# As a reminder, the model is predicting one of 4 classes, `C, L, M, V`:
#
# How does it work?
# This is a linear model that assigns a class label based on the class with the largest predicted probability.
# For a binary problem, the probability is computed based on a softmax function (from `scikit-learn` documentation):
#
# $\hat{p}(X_i) = \mathrm{expit}(X_i w + w_0) = [1 + \mathrm{exp}(-X_i w - w_0)]^{-1}$
#
# For a multinomial problem, we need something more complicated (that we won't bother computing here).
#
# Just to show you how they are stored, here are the actual intercept and coefficients from the fitted model:
#
# And here are the raw probabilities predicted by the model:
#
# We can manually convert this to class labels using the `numpy.argmax` function, which returns the index of the greatest value (with `axis=1` it's the greatest value in each row):
#
# Finally we can convert this back to labels manually using the `numpy.unique` function:
#
# Finally, just to prove it to you:
#
# You can note here how `scikit-learn` conveniently handles conversion between integer labels and categorical label codes for us.

# %%
from sklearn import linear_model

model = linear_model.LogisticRegression(random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

model.predict(xtest)

print( model.intercept_ )
print( model.coef_ )

model.predict_proba(xtest)

import numpy as np

p = model.predict_proba(xtest)
pred_label = np.argmax(p, axis=1)
print(pred_label)

np.unique(y)[pred_label]

np.unique(y)[pred_label] == model.predict(xtest)
