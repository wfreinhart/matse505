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
# id: Lecture10_interaction_between_feature_selection_and_hyperparameter_tuning
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Interaction between feature selection and hyperparameter tuning
#
# This feature selection scheme is only half of a complete workflow -- each of these feature subsets will have different optimal hyperparameters!
# For instance, imagine the extreme cases where we prune all but one feature.
# Of course the optimal $k$ could be different here than in the case where we include all the features.
# Likewise with distance weighting.
# To be sure you have the best possible model, you need to optimize features and hyperparameters together.
# This often requires training thousands of models!
#
# For this reason, there is always a balance between practicality and performance.
# If you will spend hundreds or thousands of compute hours and gain only 1% improved performance you are probably wasting your time and resources.
# As a result, it's best to start small and work your way up in complexity if you are seeing improvement.
