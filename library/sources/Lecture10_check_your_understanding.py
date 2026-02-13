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
# id: Lecture10_check_your_understanding
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## [Check your understanding]
#
# (a) Implement cross-fold validation inside the GA fitness function and use the mean validation $R^2$ as the feature fitness.
#
# (b) Alternatively, add an option to perform hyperparameter tuning for the `n_neighbors` and `weighting` of `KNeighborsRegressor` (together with feature selection).
# > You will need to split apart your `features` array into different parts when you get inside the `fitness` function. For instance,
# ```
# k = features[0]
# w = features[1]
# sel = features[2:]
# ```
# > You may also need to manipulate the values such as rounding to `int`, mapping to discrete options for `str`, etc.
#
# (c) Alternatively, add an option to select both which features will be used and which model.
# > You will need to map an integer index to a model definition. One way is to use a predefined list like so:
#
# ```
# # before the fitness function:
# possible_models = [linear_model.LinearRegression(), ensemble.RandomForestRegressor(), neighbors.KNeighborsRegressor(), ...]
#
# # then inside the fitness function:
# model = possible_models[i]
# ```
