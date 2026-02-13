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
# id: Lecture04_naive_bayes
# type: Foundational
# parent_lecture: Lecture04
# ---
#
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
#
# We see here that while the accuracy is quite high, two points in the test set are misclassified using Gaussian Naive Bayes whereas there were no errors using Logistic Regression or Support Vector Machines.

# %%
from sklearn import naive_bayes

model = naive_bayes.GaussianNB().fit(xtrain, ytrain)

model.score(xtest, ytest)
