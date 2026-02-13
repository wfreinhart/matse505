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
# id: Lecture05_gaussian_mixtures
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Gaussian mixtures
#
# The first clustering model we consider is the Gaussian Mixture Model (GMM).
# This name is pretty literal, as you can see in this image:
#
# <img src="../lectures/assets/gmm_distribution_mixture.jpg" width=500 alt="A Gaussian Mixture Model representing a probability distribution as a sum of multiple Gaussian components">
#
# The idea is that each cluster of data represents a distribution with different parameters.
# We can fit the mixture of Gaussians to represent these clusters, perhaps accounting for some overlapping at the edges.
# It will look like this in 2D:
#
# <img src="../lectures/assets/gmm_2d_clusters.jpg" width=500 alt="2D visualization of clusters identified by a Gaussian Mixture Model">
#
# The GMM is a parametric model with each distribution having a mean $\mu_i$ and vector of covariance $\sigma_{ij}$ (i.e., each pair of distributions has a covariance).
# Depending on how the assumptions, the model can behave quite differently:
#
# <img src="../lectures/assets/gmm_covariances.jpg" width=500 alt="Visualization of different Gaussian Mixture Model covariance types: spherical, tied, diag, and full">
#
# The implementation in `scikit-learn` is very straightforward to use.
# The interface is pretty similar to the supervised models: `fit()` and `predict()`.
# The only difference is `fit` takes only one argument: `x` (since again, there are no labels).
#
# This output is similar to the classification problems, but obviously we have no target labels to compare to.
# We can visualize this result in a 3D space:
#
# Some of the clusters intersect in this view.
# It's important to remember this is only 3 of 8 dimensions in the space -- and all 8 are considered by the GMM.
# We can view it in another slice to see a clearer picture:

# %%
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4).fit(x)  # no y labels!
labels = gmm.predict(x)
print(labels)

from plotly import express as px

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)
