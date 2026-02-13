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
# id: Lecture09_in_theory
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## In theory
#
# The solution is to use "cross-fold validation."
# In this scheme, we separate the data into some $k$ number of "folds" and then train $k$ different models using different sub-splits:
#
# <img src="../lectures/assets/hyperparameter_validation_set.jpg" width=600 alt="Diagram of the Train, Validation, and Test set split strategy">
#
# This has the great advantage of not needing to remove any more data from the training set while averaging out some of the variance in the test performance.
