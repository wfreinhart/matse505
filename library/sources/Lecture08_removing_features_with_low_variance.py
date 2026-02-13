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
# id: Lecture08_removing_features_with_low_variance
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## Removing features with low variance
#
# The most obvious thing to try is removing features that don't vary much.
# You can imagine a corner case where a feature has zero variance (all the same value), where it obviously would not affect the output.
#
# Read the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html) to try deploying this on the regression problem above.

# %%
from sklearn.feature_selection import GenericUnivariateSelect

selector = GenericUnivariateSelect(score_func='f_regression', mode='k_best', param=5)
