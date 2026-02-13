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
# id: Lecture03_random_forest
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Random Forest
#
# In practical applications, Decision Trees tend to be somewhat weak. There may be too many features to evaluate, or too much variation to provide a strong prediction from a single path through the tree. Decision Trees can be made more robust through aggregation of multiple trees into an **ensemble**. One of the most common forms of ensemble learning is the Random Forest, so named because it uses a collection of Decision Trees (i.e., a Forest):
#
# <img src="../lectures/assets/lecture03_random_forest.jpg" alt="Schematic of a Random Forest ensemble learning architecture" width=600>
#
# Using this collection of many trees with consensus voting can give much stronger results than the single Decision Tree. This method can actually provide very complex mappings and as a result there is a big risk of overfitting. To see how this might happen, you can imagine creating leaf nodes for each individual outcome -- this might be similar to K-Neighbors with only one neighbor.
#
# This model has also suffers from overfitting, like the Decision Trees (since the Forest is made up of Trees).
# From the training data, you would think it predicts the outcome nearly perfectly. However, we get 2-3x increase in RMSE in testing. This is stil the best model we have so far, but it's very important not to believe the training result.
#
# We can greatly reduce overfitting by limiting the depth of the decision trees using the `max_depth` keyword argument:
#
# This model still outperforms the polynomial LinearRegression while not suffering greatly from overfitting. It is substantially more robust than a single Decision Tree. However, it is no longer even traceable, let alone interpretable -- there is no way we can parse the decisions made by 100+ individual trees!

# %%
from sklearn import ensemble

model = ensemble.RandomForestRegressor().fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)

model = ensemble.RandomForestRegressor(max_depth=5).fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)
