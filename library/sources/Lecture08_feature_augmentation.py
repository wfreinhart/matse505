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
# id: Lecture08_feature_augmentation
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# # Feature augmentation
#
# Sometimes we may benefit from including additional features beyond those in the original dataset.
# There are a few possible mechanisms that can lead to this being helpful:
# * **Nonlinearity.** This is the main one. We already saw how linear models can learn nonlinear trends through the use of polynomial expansions. Introducing interactions such as products or quotients can lead to nonlinear behaviors in the original space being captured by simple linear relationships in the expanded feature space. We trade feature complexity for algorithmic simplicity.
# * **Reduced dimensionality.** While nonlinear models can learn nonlinear relationships, it may be challenging on small datasets. If we know that the product of two features is an important variable, manually introducing a single feature that is the product of two others can reduce the dimensionality of the input features and therefore reduce the number of parameters that must be learned.
