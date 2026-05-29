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
# id: Lecture04_exercise_3
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## [Check your understanding]
#
# * Train a Random Forest to predict these same labels.
# > Make sure you use a classifier model, not a regressor!
# * Print the metrics and identify the model average precision, recall, and overall accuracy.
# * Draw a confusion matrix showing the model performance on the test set.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
