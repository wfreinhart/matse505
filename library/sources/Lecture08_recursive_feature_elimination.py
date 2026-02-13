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
# id: Lecture08_recursive_feature_elimination
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## Recursive feature elimination
#
# Recursive Feature Elimination (RFE) is a backward elimination algorithm that removes the least important features recursively.
# This type of algorithm is best for parametric models where features can interact.
#
# Read the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) to try deploying this on the regression problem above.

# %%
from sklearn.feature_selection import RFE

model = linear_model.LinearRegression()
selector = RFE(model, n_features_to_select=5, step=1)
