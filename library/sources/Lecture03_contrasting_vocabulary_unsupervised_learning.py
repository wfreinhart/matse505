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
# id: Lecture03_contrasting_vocabulary_unsupervised_learning
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Contrasting vocabulary: Unsupervised learning
#
# Now consider that we have only a collection of $(X)_i$ without corresponding $(y)_i$.
# In other words, we have **unlabeled** data.
# However, we may still want to learn something about the data.
# For instance, do any of the data stick out as outliers?
# Are there discrete groups which can be discerned?
# This is the objective of unsupervised learning.
# The problem can be stated mathematically as $f(X) = \ell$, where $\ell$ is a categorical label belonging to a **class** or **cluster**. Graphically, the problem looks like this:
#
# <img src="../lectures/assets/lecture03_classification_groups.jpg" height=400>
#
# We'll do some unsupervised learning in a future lecture.
