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
# id: Lecture03_training_versus_testing
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# # Training versus Testing
#
# One of the big differences between ML and statistical modeling is the intended use. We have used linear regression to evaluate how variables are related to each other. With ML we often want to make *predictions*, or *extrapolations*. This requires a change to our workflow.
#
# In order to evaluate the performance of the model on unseen data (i.e., a prediction), we need to *hold out* some data to emulate the effect of the model encountering data it hasn't been fitted to. This will be called the **test** data.
# The data used to fit will be called the **training** data.
#
# As we start using more sophisticated models which are less grounded in physical reality and more on statistical trends, it is vitally important that we evaluate our models in this way to avoid *overfitting*.
# Remember that a model with enough degrees of freedom can perfectly fit any data, but it won't have any predictive power!
