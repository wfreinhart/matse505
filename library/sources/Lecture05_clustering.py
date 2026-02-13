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
# id: Lecture05_clustering
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Clustering
#
# As discussed earlier, clustering is a ML scheme where we try to identify discrete groups of data points without having any labels associated.
# This stands in contrast to both flavors of supervised learning (regression and classification) which always have target labels.
#
# <img src="../lectures/assets/kmeans_diagram.jpg" height=300 alt="Schematic diagram of the K-Means clustering algorithm steps">
#
# This can be helpful for understanding:
# * patterns in the data,
# * outlier detection, or
# * even generating classification labels for a supervised classification task.
#
# **Note: clustering is not classification!**
#
# At first glance clustering seems a lot like classification.
# It's worth taking a moment to make sure you understand the difference -- there are no predefined labels here!
# This graphic may help make the point:
#
# <img src="../lectures/assets/classification_vs_clustering.jpg" width=600 alt="Comparison between supervised classification (with labels) and unsupervised clustering (without labels)">
