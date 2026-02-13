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
# id: Lecture06_feature_scaling
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# # Feature scaling
#
# One thing to watch out for when doing unsupervised representation learning is features with different units. If one of the features has magnitudes of $10^6$ and another has $10^1$, the big one will completely dominate the embedding. We can get around this using **feature scaling**.
#
# <img src="../lectures/assets/feature_scaling.jpg" width=600 alt="Comparison of data distribution before and after different scaling methods: StandardScaler, MinMaxScaler, and RobustScaler">
#
# Making sure the data are roughly isotropic really helps us capture more information in the embedding.
