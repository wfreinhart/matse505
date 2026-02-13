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
# id: Lecture08_sequential_feature_selection
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## Sequential feature selection
#
# Sequential Feature Selector (SFS) is a greedy search algorithm that evaluates the feature combinations in a forward direction and adds the most significant features one by one.
# This type of algorithm is best suited for feature selection in instance-based algorithms, where the greedy search can be applied directly to the feature space.
#
# Read the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) to try deploying this on the regression problem above.

# %%
from sklearn.feature_selection import SequentialFeatureSelector

model = linear_model.LinearRegression().fit(xtrain, ytrain)
selector = SequentialFeatureSelector(model, n_features_to_select=5)
