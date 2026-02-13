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
# id: Lecture09_hyperparameters
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# # Hyperparameters
#
# To reiterate:
# Hyperparameters are the settings of a model that are not fitted.
# The "hyper" is in contrast to the "regular" parameters of the model (which are fitted).
# These are things like the number of neighbors to use in K-Neighbors, maximum tree depth for tree models, and the max iterations in Neural Networks.
#
# So far we have mostly been using the default hyperparameters of the models from `sklearn`, with a few exceptions.
# However, there is nothing special about the default parameters and in most cases we will want to select different parameters.
# Here we'll discuss how to select optimal hyperparameters.
