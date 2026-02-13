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
# id: Lecture06_manifold_learning
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# # Manifold learning
#
# Manifold learning is another term for nonlinear dimensionality reduction.
# Roughly speaking, a [manifold](https://en.wikipedia.org/wiki/Manifold) is an $n$-dimensional surface that is locally smooth.
# The term "manifold learning" therefore means searching for a smooth, low-dimensional surface in a high-dimensional space.
#
# Here's a nice motivating example that shows the difference between a **linear projection** and a **nonlinear manifold**:
#
# <img src="../lectures/assets/manifold_learning_spiral.jpg" height=400 alt="A 3D spiral dataset representing a manifold that can be unrolled into 2D">
#
# If we plot the data just based on the $x$ coordinate, we end up with the picture on the left. But if we find patterns in the data using unsupervised learning, we can "unroll" the spiral and get a new representation like the one on the right.
#
# Let's try this with a toy dataset before we move to our alloy compositions.
#
# If we try to apply PCA on this data, we will not get what we want:
#
# Why?
# Because PCA is a linear projection and linear methods can never reproduce nonlinear behavior.
# We need to introduce a nonlinearity in our learning pipeline...

# %%
from sklearn import datasets
from plotly import express as px

S, t = datasets.make_swiss_roll(n_samples=400)

px.scatter_3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], color=t)

from sklearn import decomposition
from matplotlib import pyplot as plt

# fit the model
pca = decomposition.PCA().fit(S)
St = pca.transform(S)

fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
