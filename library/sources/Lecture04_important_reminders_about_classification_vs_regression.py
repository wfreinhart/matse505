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
# id: Lecture04_important_reminders_about_classification_vs_regression
# type: Foundational
# parent_lecture: Lecture04
# ---
#
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
